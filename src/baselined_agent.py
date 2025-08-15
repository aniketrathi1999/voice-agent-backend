"""Baseline voice agent with per-turn Langfuse tracing and clear configuration.

This module defines a minimal voice assistant that:
- Uses Deepgram STT, OpenAI LLM, and OpenAI TTS
- Measures key turn metrics and logs them to Langfuse (one trace per turn)
- Reads runtime configuration from environment variables
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langfuse import Langfuse, get_client
from livekit.agents import (
    Agent,
    AgentSession,
    ConversationItemAddedEvent,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    SpeechCreatedEvent,
    UserInputTranscribedEvent,
    WorkerOptions,
    cli,
)
from livekit.plugins import deepgram, noise_cancellation, openai, silero


# Load environment
load_dotenv(".env.local")

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("baseline")

# Ensure no global OTEL export for this baseline (we manage our own per-turn spans)
for key in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_HEADERS"):
    os.environ.pop(key, None)
os.environ["OTEL_TRACES_EXPORTER"] = "none"


# ---- Configuration (env-driven) ----
MIN_ENDPOINT_DELAY = float(os.getenv("MIN_ENDPOINTING_DELAY", "0.2"))
MAX_ENDPOINT_DELAY = float(os.getenv("MAX_ENDPOINTING_DELAY", "6.0"))
ALLOW_INTERRUPTIONS = os.getenv("ALLOW_INTERRUPTIONS", "true").lower() in {"1", "true", "t", "yes", "y"}
PREEMPTIVE_GENERATION = os.getenv("PREEMPTIVE_GENERATION", "false").lower() in {"1", "true", "t", "yes", "y"}

DEEPGRAM_MODEL = os.getenv("DG_MODEL", "nova-2")
DEEPGRAM_LANGUAGE = os.getenv("DG_LANG", "en")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "ash")


class BaselineAgent(Agent):
    """Agent with configured STT/LLM/TTS backends."""

    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice assistant. Keep answers short and clear.",
            # Non-streaming STT: only react on final transcripts.
            stt=deepgram.STT(
                model=DEEPGRAM_MODEL,
                language=DEEPGRAM_LANGUAGE,
                interim_results=False,
                punctuate=True,
            ),
            llm=openai.LLM(model=LLM_MODEL),
            tts=openai.TTS(model=TTS_MODEL, voice=TTS_VOICE),
        )


def prewarm(proc: JobProcess):
    """Load VAD once per worker for faster cold-starts."""
    proc.userdata["vad"] = silero.VAD.load()
    log.info("VAD prewarmed")


def now_ms() -> float:
    """Current time in milliseconds."""
    return time.time() * 1000.0


def as_float(value) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    try:
        return float(value)
    except Exception:
        return None


def safe_get(obj: Any, key: str, default=None):
    """Safe attribute/dict access (works for dicts and Pydantic-like models)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class TurnState:
    """Tracks per-turn data for metrics and Langfuse logging."""

    def __init__(self) -> None:
        self.index: int = 0
        self.reset()
        self.span_cm = None
        self.span_task: Optional[asyncio.Task] = None
        self.done_event: Optional[asyncio.Event] = None

    def reset(self) -> None:
        self.user_text: str = ""
        self.final_received_ms: Optional[float] = None
        self.agent_first_audio_ms: Optional[float] = None
        self.agent_text: str = ""

        # Pipeline metrics (seconds)
        self.eou_delay_s: Optional[float] = None
        self.llm_ttft_s: Optional[float] = None
        self.tts_ttfb_s: Optional[float] = None
        self.llm_cancelled: bool = False
        self.tts_cancelled: bool = False

        self.response_latency_ms: Optional[float] = None

        # Detached span reference (not bound to OTel global context)
        self.langfuse_client: Optional[Langfuse] = None
        self.root_span: Any = None

    def clear_for_next(self) -> None:
        self.user_text = ""
        self.final_received_ms = None
        self.agent_first_audio_ms = None
        self.agent_text = ""
        self.eou_delay_s = None
        self.llm_ttft_s = None
        self.tts_ttfb_s = None
        self.llm_cancelled = False
        self.tts_cancelled = False
        self.response_latency_ms = None
        self.root_span = None
        self.langfuse_client = None
        self.span_cm = None
        self.span_task = None
        self.done_event = None

    def compute_latency(self) -> Optional[float]:
        """Compute response latency in ms from pipeline or wall-clock."""
        if (
            self.eou_delay_s is not None
            and self.llm_ttft_s is not None
            and self.tts_ttfb_s is not None
        ):
            return (self.eou_delay_s + self.llm_ttft_s + self.tts_ttfb_s) * 1000.0

        if self.final_received_ms and self.agent_first_audio_ms:
            return self.agent_first_audio_ms - self.final_received_ms
        return None


async def entrypoint(ctx: JobContext):
    """Entrypoint for the baseline voice assistant."""

    langfuse_client = get_client()
    turn = TurnState()

    async def flush_langfuse():
        try:
            langfuse_client.flush()
        except Exception:
            pass

    ctx.add_shutdown_callback(flush_langfuse)

    # Disable global session tracing (we only push per-turn spans manually)
    from opentelemetry.sdk.trace import TracerProvider
    from livekit.agents.telemetry import set_tracer_provider

    def disable_session_tracing() -> None:
        for key in (
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "OTEL_EXPORTER_OTLP_HEADERS",
            "OTEL_TRACES_EXPORTER",
        ):
            os.environ.pop(key, None)
        tracer_provider = TracerProvider()
        set_tracer_provider(tracer_provider, metadata={"otel_export": "disabled"})

    disable_session_tracing()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        turn_detection="stt",
        min_endpointing_delay=MIN_ENDPOINT_DELAY,
        max_endpointing_delay=MAX_ENDPOINT_DELAY,
        allow_interruptions=ALLOW_INTERRUPTIONS,
        preemptive_generation=PREEMPTIVE_GENERATION,
        
    )

    log.info(
        "Config: min=%.2f max=%.2f allow_interruptions=%s preemptive_generation=%s",
        MIN_ENDPOINT_DELAY,
        MAX_ENDPOINT_DELAY,
        ALLOW_INTERRUPTIONS,
        PREEMPTIVE_GENERATION,
    )

    # -------- Langfuse helpers (one trace/span per turn) --------
    async def start_turn_trace():
        # Increment index, then reset but preserve index
        turn.index += 1
        idx = turn.index
        turn.reset()
        turn.index = idx

        trace_id = langfuse_client.create_trace_id()
        turn.done_event = asyncio.Event()

        async def span_runner():
            # Keep enter/exit in same task to avoid context detach errors
            cm = langfuse_client.start_as_current_span(
                name="conversation_turn",
                trace_context={"trace_id": trace_id},
            )
            span = cm.__enter__()
            turn.span_cm = cm
            turn.root_span = span

            span.update_trace(
                session_id=ctx.room.name,
                user_id=ctx.room.name,
                tags=["baseline", "voice", "per_turn"],
                metadata={
                    "turn_index": turn.index,
                    "min_endpointing_delay": MIN_ENDPOINT_DELAY,
                    "max_endpointing_delay": MAX_ENDPOINT_DELAY,
                    "allow_interruptions": ALLOW_INTERRUPTIONS,
                    "preemptive_generation": PREEMPTIVE_GENERATION,
                    "stt": f"deepgram:{DEEPGRAM_MODEL}",
                    "llm": LLM_MODEL,
                    "tts": f"{TTS_MODEL}:{TTS_VOICE}",
                },
            )

            await turn.done_event.wait()
            cm.__exit__(None, None, None)

        turn.span_task = asyncio.create_task(span_runner())
        log.info("Started per-turn trace: turn=%d trace_id=%s", turn.index, trace_id)

    def update_turn_span(metadata: Dict[str, Any]):
        if turn.root_span is None:
            return
        try:
            turn.root_span.update_trace(metadata=metadata)
        except Exception as exc:
            log.warning("Langfuse span update failed: %s", exc)

    def complete_turn():
        if turn.root_span is None or turn.done_event is None:
            return

        turn.response_latency_ms = turn.compute_latency()
        try:
            turn.root_span.update_trace(
                input=turn.user_text or None,
                output=turn.agent_text or None,
                metadata={
                    "turn_index": turn.index,
                    "response_latency_ms": turn.response_latency_ms,
                    "eou_delay_ms": turn.eou_delay_s * 1000.0 if turn.eou_delay_s is not None else None,
                    "llm_ttft_ms": turn.llm_ttft_s * 1000.0 if turn.llm_ttft_s is not None else None,
                    "tts_ttfb_ms": turn.tts_ttfb_s * 1000.0 if turn.tts_ttfb_s is not None else None,
                    # "cancelled_generation": bool(turn.llm_cancelled or turn.tts_cancelled),
                    "latency_ms": turn.response_latency_ms,
                },
            )
        finally:
            turn.done_event.set()

            async def after_close():
                try:
                    if turn.span_task is not None:
                        await turn.span_task
                    langfuse_client.flush()
                except Exception:
                    pass
                log.info(
                    "Turn %d complete | latency=%sms",
                    turn.index,
                    f"{turn.response_latency_ms:.2f}" if turn.response_latency_ms is not None else "NA",
                )
                turn.clear_for_next()

            asyncio.create_task(after_close())

    @session.on("user_input_transcribed")
    def on_transcribed(ev: UserInputTranscribedEvent):
        if turn.root_span is None:
            asyncio.create_task(start_turn_trace())

        turn.user_text = (ev.transcript or "").strip()
        if ev.is_final:
            turn.final_received_ms = now_ms()
            session.generate_reply(user_input=turn.user_text)

    @session.on("speech_created")
    def on_speech(_: SpeechCreatedEvent):
        if turn.agent_first_audio_ms is None:
            turn.agent_first_audio_ms = now_ms()

    @session.on("conversation_item_added")
    def on_item(ev: ConversationItemAddedEvent):
        if ev.item.role != "assistant":
            return
        turn.agent_text = ev.item.text_content or ""
        # Add this:
        setattr(turn, "assistant_interrupted", bool(getattr(ev.item, "interrupted", False)))
        # Also write it to LF span:
        update_turn_span(metadata={"assistant_interrupted": getattr(turn, "assistant_interrupted", False)})
        complete_turn()

    @session.on("metrics_collected")
    def on_metrics(ev: MetricsCollectedEvent):
        metrics = ev.metrics
        metric_type = safe_get(metrics, "type")

        if metric_type == "eou_metrics":
            eou_delay = as_float(safe_get(metrics, "end_of_utterance_delay"))
            if eou_delay is not None:
                turn.eou_delay_s = eou_delay
                update_turn_span({"eou_delay_s": eou_delay})

        elif metric_type == "llm_metrics":
            if bool(safe_get(metrics, "cancelled")):
                turn.llm_cancelled = True
            ttft = as_float(safe_get(metrics, "ttft"))
            if ttft is not None:
                turn.llm_ttft_s = ttft
                update_turn_span({"llm_ttft_s": ttft})

        elif metric_type == "tts_metrics":
            if bool(safe_get(metrics, "cancelled")):
                turn.tts_cancelled = True
            ttfb = as_float(safe_get(metrics, "ttfb"))
            if ttfb is not None and ttfb >= 0:
                turn.tts_ttfb_s = ttfb
                update_turn_span({"tts_ttfb_s": ttfb})

    # Start session
    await session.start(
        agent=BaselineAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )
    await ctx.connect()
    await session.say("Hello, I am Jarvis, How may I help you today")
    log.info("Baseline voice agent started (per-turn Langfuse traces)")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
