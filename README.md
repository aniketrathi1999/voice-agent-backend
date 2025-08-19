<a href="https://livekit.io/">
  <img src="./.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# Voice AI Assistant with LiveKit

An optimized voice assistant built with LiveKit Agents for Python, featuring real-time speech recognition, natural language understanding, and high-quality text-to-speech synthesis.

## 🚀 Key Features

- **Optimized Voice Pipeline**
  - OpenAI GPT-4 for natural language understanding
  - Deepgram Nova-2 for high-accuracy speech recognition
  - Cartesia TTS for natural-sounding speech synthesis
  - LiveKit Turn Detector for smooth conversation flow

- **Enhanced Performance**
  - Early response on confident partial transcripts
  - Optimized endpointing for natural turn-taking
  - Pre-warmed VAD (Voice Activity Detection) for faster response times
  - Session-level tracing with Langfuse integration

- **Developer Friendly**
  - Clean, modular codebase
  - Comprehensive logging and metrics
  - Environment-based configuration
  - Easy integration with custom frontends

## 🛠️ Technical Stack

- **Backend**: Python 3.9+
- **Real-time Communication**: LiveKit
- **Speech Recognition**: Deepgram Nova-2
- **Language Model**: OpenAI GPT-4
- **Text-to-Speech**: Cartesia TTS
- **Voice Activity Detection**: Silero VAD
- **Observability**: Langfuse + OpenTelemetry

## 🛠️ Configuration

The agent can be configured using environment variables in `.env.local`:

```env
# LiveKit Configuration
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret

# AI Services
OPENAI_API_KEY=your-openai-key
DEEPGRAM_API_KEY=your-deepgram-key
CARTESIA_API_KEY=your-cartesia-key

# Optional Tuning
MIN_ENDPOINTING_DELAY=0.2  # Minimum delay before considering speech complete (seconds)
MAX_ENDPOINTING_DELAY=6.0   # Maximum delay before forcing speech to complete
ALLOW_INTERRUPTIONS=true    # Whether to allow barge-in
PREEMPTIVE_GENERATION=false # Generate responses before user stops speaking

# Early Response Tuning
EARLY_CONF_THRESHOLD=0.70   # Confidence threshold for early responses
EARLY_MIN_CHARS=12          # Minimum characters before considering early response

# Langfuse Tracing (Optional)
LANGFUSE_PUBLIC_KEY=your-key
LANGFUSE_SECRET_KEY=your-secret
LANGFUSE_HOST=https://cloud.langfuse.com
```

## 🚀 Optimizations

### Early Response System
- **Confidence-based triggering**: Responds when confidence exceeds threshold
- **Minimum character check**: Ensures enough context before responding
- **Configurable thresholds**: Adjust sensitivity based on your needs

### Performance Optimizations
- **Pre-warmed VAD**: Reduces cold start time
- **Efficient audio processing**: Optimized pipeline for low-latency responses
- **Session management**: Clean resource handling and error recovery

## 📦 Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/voice-ai-assistant.git
   cd agent-starter-python
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env.local
   # Edit .env.local with your API keys
   ```

4. Download required models:
   ```bash
   uv run python src/agent.py download-files
   ```

## 🚀 Running the Agent

### Development Mode
```bash
uv run python src/agent.py dev
```

### Production Mode
```bash
uv run python src/agent.py start
```

### Console Mode (for testing)
```bash
uv run python src/agent.py console
```

## 🔧 Development

### Testing
Run the test suite:
```bash
uv run pytest
```

### Debugging
Set `LOG_LEVEL=DEBUG` in your environment for detailed logging:
```bash
export LOG_LEVEL=DEBUG
uv run python src/agent.py dev
```

## 📊 Monitoring

The agent includes built-in monitoring:
- **Langfuse Integration**: Session-level tracing and analytics
- **LiveKit Metrics**: Real-time performance metrics
- **Structured Logging**: Easy to parse and analyze

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- [LiveKit](https://livekit.io/) for the real-time communication framework
- [OpenAI](https://openai.com/) for the language model
- [Deepgram](https://deepgram.com/) for speech recognition
- [Cartesia](https://cartesia.ai/) for text-to-speech