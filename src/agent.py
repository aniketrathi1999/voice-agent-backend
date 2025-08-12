import logging
import os
import base64
import time, asyncio
from dotenv import load_dotenv
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.agents.telemetry import set_tracer_provider
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def setup_langfuse(
    session_id: str,
    host: str | None = None,
    public_key: str | None = None,
    secret_key: str | None = None,
):
    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    host = host or os.getenv("LANGFUSE_HOST")

    if not public_key or not secret_key or not host:
        raise ValueError("LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST must be set")

    langfuse_auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host.rstrip('/')}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"

    resource = Resource.create({"session.id": session_id})
    trace_provider = TracerProvider(resource=resource)
    trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    set_tracer_provider(trace_provider)

logger = logging.getLogger("agent")

# thresholds (tune if you like)
MISSED_ENDPOINT_GRACE_S = 2.0     # grace beyond your max_endpointing_delay
MISSED_ENDPOINT_IDLE_S  = 6.0     # idle time before giving up
load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


class SessionHandler:
    def __init__(self, ctx: JobContext, session: AgentSession):
        self.ctx = ctx
        self.session = session
        self.last_user_end = None
        self.last_partial = None
        self.turn_open = False
        self.false_triggers = 0
        self.missed_endpoints = 0
        self.watchdog_task = None
        self.current_turn_span = {"span": None}
        self.usage_collector = metrics.UsageCollector()

    def register_callbacks(self):
        self.session.on("user_turn_started", self.on_user_turn_started)
        self.session.on("stt_partial", self.on_stt_partial)
        self.session.on("user_turn_ended", self.on_user_turn_ended)
        self.session.on("agent_speech_started", self.on_agent_speech_started)
        self.session.on("agent_false_interruption", self.on_agent_false_interruption)
        self.session.on("metrics_collected", self.on_metrics_collected)
        self.ctx.add_shutdown_callback(self.finalize_traces)
        self.ctx.add_shutdown_callback(self.log_usage)

    def on_user_turn_started(self, ev):
        tracer = trace.get_tracer("agent")
        self.false_triggers = 0
        self.missed_endpoints = 0
        self.current_turn_span["span"] = tracer.start_span(
            "user_turn",
            links=[trace.Link(self.session_span.get_context())],
            attributes={
                "livekit.participant.identity": ev.participant.identity,
                "livekit.participant.name": ev.participant.name,
            },
        )

        self.turn_open = True
        loop = asyncio.get_event_loop()
        if self.watchdog_task and not self.watchdog_task.done():
            self.watchdog_task.cancel()
        self.watchdog_task = loop.create_task(self._watchdog(self.current_turn_span["span"]))

    def on_stt_partial(self, ev):
        self.last_partial = time.time()
        if self.current_turn_span["span"]:
            self.current_turn_span["span"].add_event("stt_partial", {"len": len(ev.text or ""), "confidence": getattr(ev, "confidence", None)})

    def on_user_turn_ended(self, ev):
        self.last_user_end = time.time()
        self.turn_open = False
        if self.current_turn_span["span"]:
            self.current_turn_span["span"].add_event("user_turn_ended")

    def on_agent_speech_started(self, ev):
        if self.last_user_end:
            e2e_ms = int((time.time() - self.last_user_end) * 1000)
            logger.info(f"[LAT] end-user → TTS start: {e2e_ms} ms")
            if self.current_turn_span["span"]:
                self.current_turn_span["span"].set_attribute("latency.e2e_ms", e2e_ms)
                self.current_turn_span["span"].add_event("response_latency", {"e2e_ms": e2e_ms})
            self.session_span.add_event("response_latency", {"e2e_ms": e2e_ms})

    def on_agent_false_interruption(self, ev):
        self.false_triggers += 1
        s = self.current_turn_span["span"]
        s.set_attribute("lk.false_triggers_total", self.false_triggers)
        s.add_event("false_trigger", {"count": self.false_triggers})
        self.session_span.add_event("false_trigger", {"count": self.false_triggers})
        logger.info("[CNT] false trigger +1")
        self.session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    async def _watchdog(self, turn_span):
        try:
            max_delay = float(getattr(self.session, "max_endpointing_delay", 6.0)) + MISSED_ENDPOINT_GRACE_S
            deadline = time.time() + max_delay
            while self.turn_open:
                await asyncio.sleep(0.2)
                now = time.time()
                if self.last_partial and (now - self.last_partial) > MISSED_ENDPOINT_IDLE_S:
                    break
                if now >= deadline:
                    break
            if self.turn_open:
                self.turn_open = False
                self.missed_endpoints += 1
                if turn_span:
                    turn_span.add_event("missed_endpoint", {"count": self.missed_endpoints})
                self.session_span.add_event("missed_endpoint", {"count": self.missed_endpoints})
                logger.info("[CNT] missed endpoint +1")
        except asyncio.CancelledError:
            pass

    def on_metrics_collected(self, ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        self.usage_collector.collect(ev.metrics)

    async def log_usage(self):
        summary = self.usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    async def finalize_traces(self):
        if self.current_turn_span["span"]:
            self.current_turn_span["span"].set_attribute("counts.false_triggers", self.false_triggers)
            self.current_turn_span["span"].set_attribute("counts.missed_endpoints", self.missed_endpoints)
            self.current_turn_span["span"].end()
        self.session_span.set_attribute("counts.false_triggers", self.false_triggers)
        self.session_span.set_attribute("counts.missed_endpoints", self.missed_endpoints)
        self.session_span.end()

async def entrypoint(ctx: JobContext):
    session = AgentSession(
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(model="nova-3", language="multi"),
        tts=cartesia.TTS(voice="6f84f4b8-58a2-430c-8c79-688dad597532"),
        turn_detection="vad",
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        min_endpointing_delay=0.5,
        max_endpointing_delay=6.0,
        allow_interruptions=True
    )
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    await ctx.connect()

    room_sid = await ctx.room.sid
    setup_langfuse(session_id= room_sid)
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "sessionId": room_sid,
    }



    tracer = trace.get_tracer("agent")
    session_span = tracer.start_span(
        "voice_session",
        attributes={
            "session.id": room_sid,
            "room.name": ctx.room.name,
            "turn_detection": session.turn_detection,
            "min_endpointing_delay": getattr(session, "min_endpointing_delay", None),
            "max_endpointing_delay": getattr(session, "max_endpointing_delay", None),
        },
    )

    handler = SessionHandler(ctx, session)
    handler.session_span = session_span
    handler.register_callbacks()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
