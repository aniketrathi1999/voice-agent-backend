"""Baseline voice agent with per-turn Langfuse tracing and clear configuration.

- Deepgram STT, OpenAI LLM, Cartesia TTS
- Per-turn Langfuse tracing (updates via a single task to avoid OTEL context issues)
- Optional Cartesia-friendly SSML tweaks
- Session-close feedback capture (best-effort voice prompt at shutdown)
"""

import asyncio
import logging
import os
import time
import re
from typing import Any, Dict, Optional
from typing import AsyncIterable
from livekit import rtc
from livekit.agents.voice.agent import ModelSettings
from dotenv import load_dotenv
from langfuse import get_client
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
from livekit.plugins import deepgram, noise_cancellation, openai, silero, cartesia

# Load environment
load_dotenv(".env.local")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("baseline")

# Disable global OTEL export (we manage spans manually)
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
CARTESIA_MODEL = os.getenv("CARTESIA_MODEL", "gpt-4o-mini")
CARTESIA_VOICE = os.getenv("CARTESIA_VOICE", "ash")
SSML_ENABLE = os.getenv("SSML_ENABLE", "true").lower() in {"1", "true", "t", "yes", "y"}

# How long to wait at shutdown for a spoken rating (seconds)
FINAL_FEEDBACK_WAIT = float(os.getenv("FINAL_FEEDBACK_WAIT", "7"))

# --- MOS parsing helpers ---
WORD_TO_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "for": 4, "five": 5}
RATING_DIGIT_RE = re.compile(r"\b([1-5])\b")
RATING_OUTOF_RE = re.compile(r"\b([1-5])\s*(?:/|out of|over)\s*5\b", re.I)
RATING_STARS_RE = re.compile(r"\b([1-5])\s*stars?\b", re.I)
RATING_DECIMAL_RE = re.compile(r"\b([1-5])(?:[.,]\s*([0-9]))\b")  # 4.5 → round

_UPPER_TOKEN = re.compile(r"\b[A-Z]{2,6}\b")


def parse_mos_rating(text: str) -> Optional[int]:
    """Parse a 1–5 MOS rating from freeform speech text."""
    if not text:
        return None
    t = text.strip().lower()

    m = RATING_OUTOF_RE.search(t)
    if m:
        return int(m.group(1))

    m = RATING_STARS_RE.search(t)
    if m:
        return int(m.group(1))

    m = RATING_DIGIT_RE.search(t)
    if m:
        return int(m.group(1))

    m = RATING_DECIMAL_RE.search(t)
    if m:
        base = int(m.group(1))
        frac = int(m.group(2))
        val = base + (1 if frac >= 5 else 0)
        return max(1, min(5, val))

    for word, val in WORD_TO_NUM.items():
        if re.fullmatch(rf"{word}(?:\s*stars?)?", t):
            return val
        if re.search(rf"\b{word}\b", t) and re.search(r"\bstars?\b", t):
            return val

    if t in WORD_TO_NUM:
        return WORD_TO_NUM[t]

    return None


def _to_cartesia_markup(text: str) -> str:
    """Convert general SSML-ish input into Cartesia-friendly markup."""
    t = text or ""
    t = re.sub(r"</?\s*speak[^>]*>", "", t, flags=re.I)
    t = re.sub(r"</?\s*prosody[^>]*>", "", t, flags=re.I)
    t = re.sub(r"</?\s*emphasis[^>]*>", "", t, flags=re.I)
    # SSML say-as spell-out -> Cartesia <spell>
    t = re.sub(
        r'<\s*say-as[^>]*interpret-as="spell-out"[^>]*>(.*?)</\s*say-as\s*>',
        r"<spell>\1</spell>",
        t,
        flags=re.I | re.S,
    )
    return t


def _inject_cartesia_enhancements(text: str) -> str:
    """Lightweight, safe enhancements that Cartesia parses."""
    if "<" in (text or ""):
        return _to_cartesia_markup(text)
    def spell(m): return f"<spell>{m.group(0)}</spell>"
    out = _UPPER_TOKEN.sub(spell, text or "")
    out = re.sub(r"\bslow and clear\b", 'slow and clear<break time="250ms"/>', out, flags=re.I)
    return out


class BaselineAgent(Agent):
    """Agent with configured STT/LLM/TTS backends."""

    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice assistant. Keep answers short and clear.
SYSTEM: Cartesia-friendly SSML only. Use only <break time="..."/> and <spell>…</spell>.
No other tags, no wrappers, no Markdown, no JSON, no emojis.
Keep sentences short and conversational.""",
            stt=deepgram.STT(
                model=DEEPGRAM_MODEL,
                language=DEEPGRAM_LANGUAGE,
                interim_results=False,
                punctuate=True,
            ),
            llm=openai.LLM(model=LLM_MODEL),
            tts=cartesia.TTS(
                model=CARTESIA_MODEL,
                voice=CARTESIA_VOICE,
            ),
        )

    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ) -> AsyncIterable[rtc.AudioFrame]:
        """Intercept outgoing text and make it Cartesia-friendly when needed."""
        async def transformed(stream: AsyncIterable[str]):
            async for chunk in stream:
                if SSML_ENABLE and isinstance(self.tts, cartesia.TTS):
                    yield _inject_cartesia_enhancements(chunk)
                else:
                    yield chunk

        async for frame in super().tts_node(transformed(text), model_settings):
            yield frame


def prewarm(proc: JobProcess):
    """Load VAD once per worker for faster cold-starts."""
    proc.userdata["vad"] = silero.VAD.load()
    log.info("VAD prewarmed")


def now_ms() -> float:
    return time.time() * 1000.0


def as_float(value) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def safe_get(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class TurnState:
    """Tracks per-turn data and funnels span updates through a single task."""

    def __init__(self) -> None:
        self.index: int = 0
        self.reset()
        # Span lifecycle & queue
        self.span_cm = None
        self.root_span: Any = None
        self.span_task: Optional[asyncio.Task] = None
        self.done_event: Optional[asyncio.Event] = None
        self.span_update_queue: Optional[asyncio.Queue] = None

        # Dedupe
        self.seen_assistant_item_ids: set[str] = set()

        # Session counters
        self.session_turns_total: int = 0

    def reset(self) -> None:
        # Content & timing
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

        # Flow guards
        self.reply_requested: bool = False

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
        self.span_cm = None
        self.span_task = None
        self.done_event = None
        self.span_update_queue = None
        self.reply_requested = False
        self.seen_assistant_item_ids.clear()

    def compute_latency(self) -> Optional[float]:
        if (
            self.eou_delay_s is not None
            and self.llm_ttft_s is not None
            and self.tts_ttfb_s is not None
        ):
            return (self.eou_delay_s + self.llm_ttft_s + self.tts_ttfb_s) * 1000.0
        if self.final_received_ms and self.agent_first_audio_ms:
            return self.agent_first_audio_ms - self.final_received_ms
        return None


class SessionState:
    """Session-level feedback (logged once at shutdown)."""
    def __init__(self) -> None:
        self.feedback_score: Optional[int] = None
        self.feedback_raw: Optional[str] = None
        self.turns_total: int = 0
        self.collecting_final_feedback: bool = False
        self.final_feedback_event: Optional[asyncio.Event] = None


async def entrypoint(ctx: JobContext):
    """Entrypoint for the baseline voice assistant."""

    langfuse_client = get_client()
    turn = TurnState()
    session_state = SessionState()

    # Disable global session tracing (OTEL)
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

    # Session
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

    # ---------- Per-turn Langfuse span (single-task context) ----------
    async def start_turn_trace():
        # Increment counters
        turn.index += 1
        turn.session_turns_total += 1
        session_state.turns_total = turn.session_turns_total

        # Reset per-turn state but preserve index
        idx = turn.index
        turn.reset()
        turn.index = idx

        trace_id = langfuse_client.create_trace_id()
        turn.done_event = asyncio.Event()
        turn.span_update_queue = asyncio.Queue()

        async def span_runner():
            # Enter in this task; all updates go through this queue.
            cm = langfuse_client.start_as_current_span(
                name="conversation_turn",
                trace_context={"trace_id": trace_id},
            )
            span = cm.__enter__()
            turn.span_cm = cm
            turn.root_span = span

            # Initial metadata
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

            try:
                # Process queued updates until done_event is set and queue is empty
                while True:
                    if turn.done_event.is_set() and turn.span_update_queue.empty():
                        break
                    try:
                        md = await asyncio.wait_for(turn.span_update_queue.get(), timeout=0.2)
                        span.update_trace(metadata=md)
                    except asyncio.TimeoutError:
                        pass
            finally:
                cm.__exit__(None, None, None)

        turn.span_task = asyncio.create_task(span_runner())
        log.info("Started per-turn trace: turn=%d", turn.index)

    def update_turn_span(metadata: Dict[str, Any]):
        # Enqueue metadata for the span_runner task
        try:
            if turn.span_update_queue is not None:
                turn.span_update_queue.put_nowait(metadata)
        except Exception as exc:
            log.warning("Failed to enqueue span update: %s", exc)

    def complete_turn():
        """Close the current per-turn span."""
        if turn.root_span is None or turn.done_event is None:
            return

        turn.response_latency_ms = turn.compute_latency()
        md = {
            "turn_index": turn.index,
            "response_latency_ms": turn.response_latency_ms,
            "eou_delay_ms": turn.eou_delay_s * 1000.0 if turn.eou_delay_s is not None else None,
            "llm_ttft_ms": turn.llm_ttft_s * 1000.0 if turn.llm_ttft_s is not None else None,
            "tts_ttfb_ms": turn.tts_ttfb_s * 1000.0 if turn.tts_ttfb_s is not None else None,
            "latency_ms": turn.response_latency_ms,
            "input": turn.user_text or None,
            "output": turn.agent_text or None,
        }
        update_turn_span(md)

        # Signal span_runner to exit after draining queue
        turn.done_event.set()

        async def after_close():
            try:
                if turn.span_task is not None:
                    await turn.span_task
                get_client().flush()
            except Exception:
                pass
            log.info(
                "Turn %d complete | latency=%sms",
                turn.index,
                f"{turn.response_latency_ms:.2f}" if turn.response_latency_ms is not None else "NA",
            )
            turn.clear_for_next()

        asyncio.create_task(after_close())

    # ---------- Session shutdown feedback ----------
    def on_shutdown():
        try:
            session_state.collecting_final_feedback = True
            session_state.final_feedback_event = asyncio.Event()
            try:
                #  take input from user from console
                rating = input("Please rate the audio quality from 1 to 5: ")
                session_state.feedback_score = int(rating)
                session_state.feedback_raw = rating

            except Exception as e:
                log.debug("Could not speak final feedback prompt (session closing?): %s", e)

            # Wait a bit for a spoken rating
            try:
                asyncio.wait_for(session_state.final_feedback_event.wait(), timeout=FINAL_FEEDBACK_WAIT)
            except asyncio.TimeoutError:
                log.info("No final rating received during shutdown window.")

            # Write a dedicated session-level trace with final feedback
            trace_id = get_client().create_trace_id()
            cm = get_client().start_as_current_span(
                name="session_feedback",
                trace_context={"trace_id": trace_id},
            )
            span = cm.__enter__()
            try:
                span.update_trace(
                    session_id=ctx.room.name,
                    user_id=ctx.room.name,
                    tags=["baseline", "voice", "session_feedback"],
                    input=None,
                    output=None,
                    metadata={
                        "turns_total": session_state.turns_total,
                        "feedback_score": session_state.feedback_score,
                        "feedback_raw": session_state.feedback_raw,
                        "prompted_at_shutdown": True,
                        "wait_seconds": FINAL_FEEDBACK_WAIT,
                    },
                )
            finally:
                cm.__exit__(None, None, None)
        except Exception as e:
            log.warning("Failed to capture/write session feedback: %s", e)
        finally:
            session_state.collecting_final_feedback = False
            try:
                get_client().flush()
            except Exception:
                pass

    # Register shutdown callback
    # ctx.add_shutdown_callback(on_shutdown)
    
    @session.on("close")
    def on_close():
        print("ON close calleddddddddddddddddddddddddddddddd")
        on_shutdown()

    # ---------- Event handlers ----------
    @session.on("user_input_transcribed")
    def on_transcribed(ev: UserInputTranscribedEvent):
        # If we are in shutdown feedback mode, only listen for a rating
        if session_state.collecting_final_feedback:
            if ev.is_final:
                text = (ev.transcript or "").strip()
                mos = parse_mos_rating(text)
                if mos is not None:
                    session_state.feedback_score = mos
                    session_state.feedback_raw = text
                    if session_state.final_feedback_event and not session_state.final_feedback_event.is_set():
                        session_state.final_feedback_event.set()
                    log.info("Captured final MOS rating at shutdown: %s (raw: %s)", mos, text)
            return

        # Normal per-turn flow
        if turn.root_span is None:
            asyncio.create_task(start_turn_trace())

        turn.user_text = (ev.transcript or "").strip()
        if ev.is_final and not turn.reply_requested:
            turn.reply_requested = True
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
        # Dedupe assistant items by id
        item_id = getattr(ev.item, "id", None)
        if item_id is not None:
            if item_id in turn.seen_assistant_item_ids:
                return
            turn.seen_assistant_item_ids.add(item_id)

        if not turn.agent_text:
            turn.agent_text = ev.item.text_content or ""

        setattr(turn, "assistant_interrupted", bool(getattr(ev.item, "interrupted", False)))
        update_turn_span({"assistant_interrupted": getattr(turn, "assistant_interrupted", False)})
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
    await session.say("Hello")
    log.info("Baseline voice agent started (per-turn Langfuse traces)")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
