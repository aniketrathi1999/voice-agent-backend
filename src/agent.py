import logging
import os
import base64
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
)
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.agents.telemetry import set_tracer_provider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.util.types import AttributeValue

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# Thresholds for metrics calculation
MISSED_ENDPOINT_GRACE_S = 2.0
MISSED_ENDPOINT_IDLE_S = 6.0


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


def prewarm(proc: JobProcess):
    """Preload VAD model for better performance"""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the voice assistant"""
    # Get room SID for session identification
    trace_provider = setup_langfuse(
        # metadata will be set as attributes on all spans created by the tracer
        metadata={
            "langfuse.session.id": ctx.room.name,
        }
    )
    async def flush_trace():
        trace_provider.force_flush()

    ctx.add_shutdown_callback(flush_trace)

    # Create agent session
    session = AgentSession(
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(model="nova-2", language="en"),  # Simplified for reliability
        tts=cartesia.TTS(voice="6f84f4b8-58a2-430c-8c79-688dad597532"),
        turn_detection="vad",
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        min_endpointing_delay=0.5,
        max_endpointing_delay=6.0,
        allow_interruptions=True
    )
    
    # Start the session
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        logger.info(f"Metrics collected for session")
        metrics.log_metrics(ev.metrics)
    
    # Connect to the room
    await ctx.connect()
    
    logger.info(f"Voice assistant started for session")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))