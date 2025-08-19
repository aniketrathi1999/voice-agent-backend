import logging
import os
import base64
import asyncio
import time
from typing import AsyncIterable, Optional

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
    ModelSettings,
)
from livekit.agents.metrics import TTSMetrics
from livekit.agents.stt import SpeechEventType, SpeechEvent
from livekit import rtc
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins import deepgram, noise_cancellation, openai, silero
from livekit.agents.telemetry import set_tracer_provider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.util.types import AttributeValue

# Initialize logging
logger = logging.getLogger("agent")
load_dotenv(".env.local")

def _levenshtein_distance(source_tokens: list[str], target_tokens: list[str]) -> int:
    """Calculate Levenshtein distance between two token lists.
    
    Args:
        source_tokens: List of source tokens
        target_tokens: List of target tokens
        
    Returns:
        int: Minimum number of single-token edits (insertions, deletions, substitutions)
             needed to change source into target.
    """
    source_len = len(source_tokens)
    target_len = len(target_tokens)
    
    # Initialize dynamic programming table
    dp = list(range(target_len + 1))
    
    for i in range(1, source_len + 1):
        prev_diag = dp[0]
        dp[0] = i
        
        for j in range(1, target_len + 1):
            temp = dp[j]
            if source_tokens[i - 1] == target_tokens[j - 1]:
                dp[j] = prev_diag
            else:
                dp[j] = min(prev_diag, dp[j], dp[j - 1]) + 1
            prev_diag = temp
            
    return dp[target_len]

def word_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate between reference and hypothesis strings.
    
    Args:
        reference: Reference text
        hypothesis: Hypothesis text to evaluate
        
    Returns:
        float: Word Error Rate (0.0 to 1.0), where 0.0 is perfect match
    """
    ref_tokens = reference.strip().split()
    hyp_tokens = hypothesis.strip().split()
    
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
        
    return _levenshtein_distance(ref_tokens, hyp_tokens) / max(1, len(ref_tokens))

def longest_common_subsequence_ratio(str_a: str, str_b: str) -> float:
    """Calculate ratio of longest common subsequence to max string length.
    
    Args:
        str_a: First string for comparison
        str_b: Second string for comparison
        
    Returns:
        float: Ratio of LCS length to maximum input length (0.0 to 1.0)
    """
    tokens_a = str_a.split()
    tokens_b = str_b.split()
    
    len_a = len(tokens_a)
    len_b = len(tokens_b)
    
    if len_a == 0 or len_b == 0:
        return 0.0
        
    # Initialize DP table
    dp = [0] * (len_b + 1)
    
    for i in range(1, len_a + 1):
        prev = 0
        for j in range(1, len_b + 1):
            temp = dp[j]
            if tokens_a[i - 1] == tokens_b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
            
    lcs_length = dp[len_b]
    return lcs_length / max(len_a, len_b)

def setup_langfuse(
    metadata: dict[str, AttributeValue] | None = None,
    *,
    host: str | None = None,
    public_key: str | None = None,
    secret_key: str | None = None,
) -> TracerProvider:
    """Initialize and configure Langfuse tracing with proper authentication.
    
    Args:
        metadata: Optional metadata to associate with traces
        host: Langfuse host URL (defaults to LANGFUSE_HOST env var)
        public_key: Langfuse public key (defaults to LANGFUSE_PUBLIC_KEY env var)
        secret_key: Langfuse secret key (defaults to LANGFUSE_SECRET_KEY env var)
        
    Returns:
        TracerProvider: Configured OpenTelemetry TracerProvider
        
    Raises:
        ValueError: If required authentication keys are not provided
    """
    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    host = host or os.getenv("LANGFUSE_HOST")

    if not public_key or not secret_key:
        raise ValueError(
            "Missing Langfuse credentials. "
            "Please set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables."
        )

    if not host:
        raise ValueError("Langfuse host not specified. Set LANGFUSE_HOST environment variable.")

    # Configure authentication headers for Langfuse
    auth_token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host.rstrip('/')}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth_token}"
    
    # Initialize OpenTelemetry tracing
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    set_tracer_provider(trace_provider, metadata=metadata)
    
    logger.info("Langfuse tracing initialized")
    return trace_provider

class Assistant(Agent):
    """Voice assistant that handles speech-to-text, processing, and text-to-speech.
    
    This agent includes early response capabilities that can begin processing
    user speech before they finish talking, based on confidence thresholds.
    """
    
    def __init__(self) -> None:
        """Initialize the voice assistant with configuration from environment."""
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )
        # Early response configuration
        self._early_confidence_threshold = float(os.getenv("EARLY_CONF_THRESHOLD", "0.70"))
        self._min_chars_for_early_response = int(os.getenv("EARLY_MIN_CHARS", "12"))
        self._early_response_mode = os.getenv("EARLY_MODE", "threshold")  # 'off' or 'threshold'
        
        # Response tracking state
        self._is_early_response_sent = False
        self._early_response_text = ""
        self._early_response_start_time: Optional[float] = None
        self._last_final_transcript = ""

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
        async def process_audio_stream():
            async for event in Agent.default.stt_node(self, audio, model_settings):
                if not isinstance(event, SpeechEvent):
                    yield event
                    continue

                if event.type == SpeechEventType.INTERIM_TRANSCRIPT and event.alternatives:
                    await self._handle_interim_transcript(event)
                elif event.type == SpeechEventType.FINAL_TRANSCRIPT and event.alternatives:
                    await self._handle_final_transcript(event)
                
                yield event

        return process_audio_stream()
    
    async def _handle_interim_transcript(self, event: SpeechEvent) -> None:
        """Process interim (partial) transcript for potential early response.
        
        Args:
            event: SpeechEvent containing the interim transcript
        """
        if not event.alternatives:
            return
            
        transcript = event.alternatives[0]
        confidence = getattr(transcript, "confidence", None)
        text = transcript.text or ""
        
        logger.debug(
            f"Partial transcript - confidence: {confidence:.2f}, text: {text!r}"
        )

        # Check if we should trigger an early response
        is_preemptive_enabled = os.getenv("PREEMPTIVE_GENERATION", "true").strip().lower() in {
            "1", "true", "t", "yes", "y"
        }
        
        should_respond_early = (
            self._early_response_mode == "threshold"
            and not self._is_early_response_sent
            and confidence is not None 
            and confidence >= self._early_confidence_threshold
            and len(text) >= self._min_chars_for_early_response
            and not is_preemptive_enabled
        )
        
        if should_respond_early:
            self._is_early_response_sent = True
            self._early_response_text = text
            self._early_response_start_time = time.perf_counter()
            
            logger.info(
                f"Initiating early response at confidence {confidence:.2f}: {text!r}"
            )
            
            # Start response generation without blocking
            self.session.generate_reply(user_input=text)
    
    async def _handle_final_transcript(self, event: SpeechEvent) -> None:
        """Process final transcript and evaluate early response accuracy.
        
        Args:
            event: SpeechEvent containing the final transcript
        """
        if not event.alternatives:
            return
            
        transcript = event.alternatives[0]
        final_text = transcript.text or ""
        confidence = getattr(transcript, "confidence", None)
        
        self._last_final_transcript = final_text
        logger.debug(
            f"Final transcript - confidence: {confidence:.2f}, text: {final_text!r}"
        )
        
        # Evaluate early response accuracy if we sent one
        if self._is_early_response_sent and self._early_response_text:
            error_rate = word_error_rate(final_text, self._early_response_text)
            overlap = longest_common_subsequence_ratio(final_text, self._early_response_text)
            is_potential_hallucination = overlap < 0.5 or error_rate > 0.35
            
            logger.info(
                f"Early response evaluation - "
                f"WER: {error_rate:.3f}, "
                f"overlap: {overlap:.2f}, "
                f"hallucination_risk: {is_potential_hallucination}"
            )
            
            # Reset early response state (keep start time for latency measurement)
            self._is_early_response_sent = False
            self._early_response_text = ""

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
        stt=deepgram.STT(model="nova-2", language="en", interim_results=True),
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
