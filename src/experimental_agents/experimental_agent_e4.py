import logging
import os
import base64
import asyncio  # NEW
import time     # NEW
from typing import AsyncIterable, Optional  # NEW

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    ModelSettings,      # NEW
)
from livekit.agents.metrics import TTSMetrics
from livekit.agents.stt import SpeechEventType, SpeechEvent  # NEW
from livekit import rtc                                     # NEW
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins import deepgram, noise_cancellation, openai, silero
from livekit.agents.telemetry import set_tracer_provider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.util.types import AttributeValue

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# -------- NEW: small metrics helpers --------
def _levenshtein(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = min(prev, dp[j], dp[j - 1]) + 1
            prev = cur
    return dp[n]

def word_error_rate(ref: str, hyp: str) -> float:
    ref_w = ref.strip().split()
    hyp_w = hyp.strip().split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    return _levenshtein(ref_w, hyp_w) / max(1, len(ref_w))

def lcs_ratio(a: str, b: str) -> float:
    # Quick longest-common-subsequence-ish overlap via dynamic programming
    A, B = a.split(), b.split()
    m, n = len(A), len(B)
    if m == 0 or n == 0:
        return 0.0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            tmp = dp[j]
            if A[i - 1] == B[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    lcs = dp[n]
    return lcs / max(m, n)
# --------------------------------------------

def setup_langfuse(
    metadata: dict[str, AttributeValue] | None = None,
    *,
    host: str | None = None,
    public_key: str | None = None,
    secret_key: str | None = None,
):
    """Setup Langfuse tracing with proper authentication"""
    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    host = host or os.getenv("LANGFUSE_HOST")

    if not public_key or not secret_key:
        raise ValueError("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set")

    langfuse_auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host.rstrip('/')}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"
    
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    set_tracer_provider(trace_provider, metadata=metadata)
    
    logger.info(f"Langfuse setup complete for session")
    return trace_provider

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )
        # -------- NEW: early-prompting controls --------
        self._early_conf_threshold = float(os.getenv("EARLY_CONF_THRESHOLD", "0.70"))
        self._early_min_chars = int(os.getenv("EARLY_MIN_CHARS", "12"))
        self._early_mode = os.getenv("EARLY_MODE", "threshold")  # off | threshold
        self._early_sent = False
        self._early_text = ""
        self._early_t0: Optional[float] = None
        self._last_final_text = ""
        # -----------------------------------------------

    # -------- NEW: inspect STT stream for partials & confidence and prompt early --------
    async def stt_node(
        self,
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings: ModelSettings,
    ) -> Optional[AsyncIterable[SpeechEvent]]:
        # Use the default STT stream, but peek at events to log partials and gate early prompting.
        async def _tap_stream():
            async for ev in Agent.default.stt_node(self, audio, model_settings):
                if isinstance(ev, SpeechEvent):
                    if ev.type == SpeechEventType.INTERIM_TRANSCRIPT and ev.alternatives:
                        alt = ev.alternatives[0]
                        conf = getattr(alt, "confidence", None)
                        text = alt.text or ""
                        logger.debug(
                            f"[STT partial] conf={conf if conf is not None else 'NA'} text={text!r}"
                        )

                        # Early-prompt gate (only if we are not also using built-in preemptive_generation)
                        preemptive_generation = os.getenv("PREEMPTIVE_GENERATION", "true").strip().lower() in {"1","true","t","yes","y"}
                        if (
                            self._early_mode == "threshold"
                            and not self._early_sent
                            and conf is not None and conf >= self._early_conf_threshold
                            and len(text) >= self._early_min_chars
                            and not preemptive_generation
                        ):
                            print("--------------------------------")
                            print(conf, text, len(text), self._early_min_chars, preemptive_generation, self._early_mode, self._early_sent)
                            print(ev)
                            print("--------------------------------")

                            self._early_sent = True
                            self._early_text = text
                            self._early_t0 = time.perf_counter()
                            logger.info(f"[EARLY] triggering LLM at conf={conf:.2f} text={text!r}")
                            # Fire-and-forget so we don't block the STT loop
                            self.session.generate_reply(user_input=text)

                    elif ev.type == SpeechEventType.FINAL_TRANSCRIPT and ev.alternatives:
                        alt = ev.alternatives[0]
                        final_text = alt.text or ""
                        conf = getattr(alt, "confidence", None)
                        self._last_final_text = final_text
                        logger.debug(f"[STT final] conf={conf if conf is not None else 'NA'} text={final_text!r}")

                        if self._early_sent and self._early_text:
                            wer = word_error_rate(final_text, self._early_text)
                            overlap = lcs_ratio(final_text, self._early_text)
                            hallucination_risk = overlap < 0.5 or wer > 0.35
                            logger.info(
                                f"[COMPARE] WER(early→final)={wer:.3f} overlap={overlap:.2f} hallucination_risk={hallucination_risk}"
                            )
                            # reset early gate (keep _early_t0 until we log TTS ttfb)
                            self._early_sent = False
                            self._early_text = ""

                yield ev

        return _tap_stream()
    # ------------------------------------------------------------------------------------

def prewarm(proc: JobProcess):
    """Preload VAD model for better performance"""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the voice assistant"""
    trace_provider = setup_langfuse(
        metadata={
            "langfuse.session.id": ctx.room.name,
        }
    )
    
    usage_collector = metrics.UsageCollector()

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    async def flush_trace():
        trace_provider.force_flush()

    ctx.add_shutdown_callback(log_usage)
    ctx.add_shutdown_callback(flush_trace)

    min_endpointing_delay = float(os.getenv("MIN_ENDPOINTING_DELAY", "0.2"))
    max_endpointing_delay = float(os.getenv("MAX_ENDPOINTING_DELAY", "6.0"))
    allow_interruptions = os.getenv("ALLOW_INTERRUPTIONS", "true").strip().lower() in {"1", "true", "t", "yes", "y"}
    preemptive_generation = os.getenv("PREEMPTIVE_GENERATION", "true").strip().lower() in {"1", "true", "t", "yes", "y"}

    # Create agent session
    session = AgentSession(
        llm=openai.LLM(model="gpt-4o-mini"),
        # CHANGED: enable streaming partials explicitly; keep your model choice
        stt=deepgram.STT(model="nova-2", language="en", interim_results=True),  # NEW flag
        tts=openai.TTS(
            model="gpt-4o-mini-tts",
            voice="ash",
        ),
        turn_detection=EnglishModel(),
        vad=ctx.proc.userdata["vad"],
        min_endpointing_delay=min_endpointing_delay,
        max_endpointing_delay=max_endpointing_delay,
        allow_interruptions=allow_interruptions,
        preemptive_generation=preemptive_generation
    )
    
    logger.info(
        f"Configuration: min_endpointing_delay={min_endpointing_delay}, "
        f"max_endpointing_delay={max_endpointing_delay}, "
        f"allow_interruptions={allow_interruptions}, "
        f"preemptive_generation={preemptive_generation}"
    )

    # Start the session
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # NEW: also log the text-only transcription stream the high-level way
    from livekit.agents import UserInputTranscribedEvent
    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev: UserInputTranscribedEvent):
        logger.debug(f"[TRANSCRIBE] final={ev.is_final} text={ev.transcript!r}")

    # Keep your existing metrics logging, and add latency calc when we see first TTS metric after early prompt
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

        # EARLY latency: from the time we triggered early LLM to the first audio byte reported by TTS
        try:
            # The active agent instance is available on the session
            agent = getattr(session, "agent", None)
            if agent and getattr(agent, "_early_t0", None) is not None:
                if isinstance(ev.metrics, TTSMetrics):
                    perceived_latency = time.perf_counter() - agent._early_t0
                    logger.info(f"[LATENCY] early_prompt_to_ttfb={perceived_latency:.3f}s")
                    agent._early_t0 = None  # only compute once per early generation
        except Exception as e:
            logger.debug(f"latency calc skipped: {e!r}")
    
    # Connect to the room
    await ctx.connect()
    session.say("Hello, I am your voice assistant")
    logger.info(f"Voice assistant started for session")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint, 
        prewarm_fnc=prewarm,
        initialize_process_timeout=300.0,
        multiprocessing_context="spawn",
    ))
