# LiveKit Voice AI Agent

A production-quality minimal Voice AI Agent built with LiveKit Agents Python SDK, featuring:
- **Mock STT/TTS** for development without API keys
- **Groq LLM** for genuine tool-calling
- **`get_order_status` tool** demonstrating function-calling flow

## Architecture

```
User Input → MockSTT (stdin) → Groq Llama 3.3 → MockTTS (stdout)
                                    │
                              get_order_status()
```

## Setup

1. Create `.env` from `.env.example` and fill in your keys.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run:

```bash
python main.py console
```

Type your input in the terminal and press Enter. The agent will process it through the LLM and respond via MockTTS (printed to stdout).

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LIVEKIT_URL` | `ws://localhost:7880` | LiveKit server URL |
| `GROQ_API_KEY` | — | Required for LLM |
| `USE_MOCK_STT` | `True` | Use stdin-based mock STT |
| `USE_MOCK_TTS` | `True` | Use print-based mock TTS |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Tests

```bash
pytest tests/ -v
```

## Tools

- **`get_order_status(order_id)`** — Returns mocked delivery status for known IDs (`ORD-001`, `ORD-002`, `ORD-003`, `12345`).

## Project Structure

```
livekit-voice-agent/
├── agent/           # Business logic: persona, tools, session
├── config/          # Settings (pydantic-settings)
├── mocks/           # Mock STT (stdin) and TTS (stdout)
├── plugins/         # Plugin selectors (real or mock)
├── tests/           # Unit and integration tests
├── logs/            # Runtime log files
├── main.py          # Entry point
└── requirements.txt
```

# Run the frontend
cd frontend
python server.py
# url : http://localhost:3000

# how to run:
# conda create -n LiveKit_Agents python=3.12
# .\livekit-server.exe --config .\livekit.yaml
# python main.py start
# python main_api.py

