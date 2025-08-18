# optimised_agent.py
"""
Optimised Voice Agent (English-only)

Features:
- OpenAI LLM + Deepgram STT (with interim results for early response)
- Cartesia TTS (sonic-2-2025-03-07 by default) with Cartesia-safe markup
- Guarded SSML passthrough: supports <break time="..."/> and <spell>...</spell>
- Early response on confident partial transcripts
- Session-level Langfuse via OTEL (no per-turn tracing)
- Silero VAD prewarm and LiveKit noise cancellation

ENV (.env.local):
  # LiveKit agent controls
  MIN_ENDPOINTING_DELAY=0.2
  MAX_ENDPOINTING_DELAY=6.0
  ALLOW_INTERRUPITIONS=true
  PREEMPTIVE_GENERATION=false

  # Langfuse (session-level tracing via OTEL)
  LANGFUSE_PUBLIC_KEY=...
  LANGFUSE_SECRET_KEY=...
  LANGFUSE_HOST=https://cloud.langfuse.com

  # Deepgram
  DG_MODEL=nova-2
  DG_LANG=en

  # OpenAI LLM
  LLM_MODEL=gpt-4o-mini

  # Cartesia
  CARTESIA_MODEL=sonic-2-2025-03-07
  CARTESIA_VOICE=<your-voice-id>

  # Early response tuning (optional)
  EARLY_CONF_THRESHOLD=0.70
  EARLY_MIN_CHARS=12
"""

from __future__ import annotations

import os
import re
import base64
import logging
from typing import AsyncIterable, Optional

from dotenv import load_dotenv

from livekit import rtc
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
)
from livekit.agents.voice.agent import ModelSettings
from livekit.agents.stt import SpeechEvent, SpeechEventType
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins import deepgram, noise_cancellation, openai, silero, cartesia

# Langfuse via OTEL (session-level, not per-turn)
from livekit.agents.telemetry import set_tracer_provider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.util.types import AttributeValue

# --------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------
log = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
load_dotenv(".env.local")


def setup_langfuse(
    metadata: dict[str, AttributeValue] | None = None,
    *,
    host: str | None = None,
    public_key: str | None = None,
    secret_key: str | None = None,
) -> TracerProvider:
    """
    Session-level Langfuse via OTEL (like exp-3).
    """
    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    host = host or os.getenv("LANGFUSE_HOST")

    if not public_key or not secret_key:
        raise ValueError("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set")
    if not host:
        raise ValueError("LANGFUSE_HOST must be set, e.g. https://cloud.langfuse.com")

    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host.rstrip('/')}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth}"

    tp = TracerProvider()
    tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    set_tracer_provider(tp, metadata=metadata)
    log.info("Langfuse tracing initialised (session-level)")
    return tp


def prewarm(proc: JobProcess) -> None:
    """
    Load Silero VAD once per worker for faster cold-starts.
    """
    proc.userdata["vad"] = silero.VAD.load()
    log.info("VAD prewarmed")


# --------------------------------------------------------------------
# Cartesia-safe markup helpers (inspired by exp-5)
# --------------------------------------------------------------------
_UPPER_TOKEN = re.compile(r"\b[A-Z]{2,6}\b")  # acronyms to <spell>...</spell>

def _to_cartesia_markup(text: str) -> str:
    """
    Convert generic SSML to Cartesia-friendly subset:
    - strip <speak>, <prosody>, <emphasis>
    - map SSML <say-as interpret-as="spell-out">..</say-as> -> <spell>..</spell>
    """
    t = text or ""
    t = re.sub(r"</?\s*speak[^>]*>", "", t, flags=re.I)
    t = re.sub(r"</?\s*prosody[^>]*>", "", t, flags=re.I)
    t = re.sub(r"</?\s*emphasis[^>]*>", "", t, flags=re.I)
    t = re.sub(
        r'<\s*say-as[^>]*interpret-as="spell-out"[^>]*>(.*?)</\s*say-as\s*>',
        r"<spell>\1</spell>",
        t,
        flags=re.I | re.S,
    )
    return t


def _inject_cartesia_enhancements(text: str) -> str:
    """
    If text already has tags, normalise to Cartesia subset.
    Otherwise, lightly enhance:
    - spell ALLCAPS tokens (GPT, API)
    - add a short break after 'slow and clear'
    """
    if "<" in (text or ""):
        return _to_cartesia_markup(text)
    def spell(m): return f"<spell>{m.group(0)}</spell>"
    out = _UPPER_TOKEN.sub(spell, text or "")
    out = re.sub(r"\bslow and clear\b", 'slow and clear<break time="250ms"/>', out, flags=re.I)
    return out


# --------------------------------------------------------------------
# Assistant
# --------------------------------------------------------------------
class Assistant(Agent):
    """
    English-only assistant with:
    - concise, speakable answers
    - Cartesia-safe SSML guidance
    - early response on confident partials
    """

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant. Keep answers short, friendly, and clear.\n"
                "SYSTEM: Output text that is Cartesia-compatible. Allowed tags only:\n"
                "  <break time=\"...\"/> and <spell>...</spell>\n"
                "Avoid any other markup. No JSON, no Markdown, no emojis.\n"
                "Prefer short sentences and conversational style."
            )
        )
        # Early-response tuning
        self._early_conf_threshold = float(os.getenv("EARLY_CONF_THRESHOLD", "0.70"))
        self._early_min_chars = int(os.getenv("EARLY_MIN_CHARS", "12"))
        self._early_sent = False
        self._early_text = ""

    # --- Early response on interim transcripts (based on exp-4) ---
    async def stt_node(
        self,
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings: ModelSettings,
    ) -> Optional[AsyncIterable[SpeechEvent]]:
        """Process audio stream with speech-to-text and handle early response logic.
        
        This wraps the default STT node to implement early response capabilities
        when the system has high confidence in the partial transcript.
        
        Args:
            audio: Stream of audio frames to process
            model_settings: Configuration for the speech recognition model
            
        Returns:
            AsyncIterable of SpeechEvent objects containing transcription results
        """
        async def stream():
            async for event in Agent.default.stt_node(self, audio, model_settings):
                if isinstance(event, SpeechEvent):
                    if event.type == SpeechEventType.INTERIM_TRANSCRIPT and event.alternatives:
                        alt = event.alternatives[0]
                        txt = (alt.text or "").strip()
                        conf = getattr(alt, "confidence", None)
                        if (
                            not self._early_sent
                            and conf is not None and conf >= self._early_conf_threshold
                            and len(txt) >= self._early_min_chars
                            and os.getenv("PREEMPTIVE_GENERATION", "false").lower() not in {"1", "true", "yes", "y"}
                        ):
                            self._early_sent = True
                            self._early_text = txt
                            log.info(f"Early response triggered (conf={conf:.2f}): {txt!r}")
                            self.session.generate_reply(user_input=txt)

                    elif event.type == SpeechEventType.FINAL_TRANSCRIPT and event.alternatives:
                        self._early_sent = False
                        self._early_text = ""
                yield event
        return stream()

    # --- Cartesia text normalisation before synthesis (based on exp-5) ---
    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ) -> AsyncIterable[rtc.AudioFrame]:
        async def transformed(stream: AsyncIterable[str]):
            async for chunk in stream:
                if isinstance(self.tts, cartesia.TTS):
                    yield _inject_cartesia_enhancements(chunk)
                else:
                    yield chunk
        async for frame in super().tts_node(transformed(text), model_settings):
            yield frame


# --------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------
async def entrypoint(ctx: JobContext):
    # Session-level Langfuse (like exp-3)
    tp = setup_langfuse(metadata={"langfuse.session.id": ctx.room.name})

    usage = metrics.UsageCollector()

    async def log_usage():
        log.info("Usage: %s", usage.get_summary())

    async def flush_trace():
        try:
            tp.force_flush()
        except Exception:
            pass

    ctx.add_shutdown_callback(log_usage)
    ctx.add_shutdown_callback(flush_trace)

    # Runtime config
    min_endpoint = float(os.getenv("MIN_ENDPOINTING_DELAY", "0.2"))
    max_endpoint = float(os.getenv("MAX_ENDPOINTING_DELAY", "6.0"))
    allow_interruptions = os.getenv("ALLOW_INTERRUPTIONS", "true").strip().lower() in {"1", "true", "t", "yes", "y"}
    preemptive = os.getenv("PREEMPTIVE_GENERATION", "false").strip().lower() in {"1", "true", "t", "yes", "y"}

    # Backends
    dg_model = os.getenv("DG_MODEL", "nova-2")
    dg_lang = os.getenv("DG_LANG", "en")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    cart_model = os.getenv("CARTESIA_MODEL", "sonic-2-2025-03-07")  # supports speed/emotion
    cart_voice = os.getenv("CARTESIA_VOICE", "")

    # TTS init (prewarm; set speed/emotion only for supported model)
    tts_kwargs = {"model": cart_model, "voice": cart_voice} if cart_voice else {"model": cart_model}
    tts = cartesia.TTS(**tts_kwargs)
    try:
        # Optional: set defaults when supported
        if cart_model == "sonic-2-2025-03-07":
            try:
                tts.update_options(speed=1.0, emotion=["friendly"])
            except Exception as e:
                log.debug(f"tts.update_options skipped: {e}")
        # Warm Cartesia to reduce first-utterance latency
        try:
            tts.prewarm()
        except Exception as e:
            log.debug(f"TTS prewarm skipped: {e}")
    except Exception as e:
        log.warning(f"Cartesia options init failed: {e}")

    session = AgentSession(
        llm=openai.LLM(model=llm_model),
        stt=deepgram.STT(model=dg_model, language=dg_lang, interim_results=True, punctuate=True),
        tts=tts,
        turn_detection=EnglishModel(),
        vad=ctx.proc.userdata["vad"],
        min_endpointing_delay=min_endpoint,
        max_endpointing_delay=max_endpoint,
        allow_interruptions=allow_interruptions,
        preemptive_generation=preemptive,
        use_tts_aligned_transcript=True,
    )

    log.info(
        "Config: min_endpoint=%.2f max_endpoint=%.2f allow_interruptions=%s preemptive=%s",
        min_endpoint, max_endpoint, allow_interruptions, preemptive
    )

    # Metrics passthrough (keep it simple like exp-3)
    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    # Start
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()

    # First greeting (Cartesia-safe)
    try:
        await session.say(_inject_cartesia_enhancements(
            "Hello! I’m ready to help. If anything is important, I’ll make it slow and clear."
        ))
    except Exception as e:
        log.exception("Greeting TTS failed: %s", e)

    log.info("Optimised voice assistant started")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
        initialize_process_timeout=300.0,
        multiprocessing_context="spawn",
    ))
