# 🚀 How to Run — Section 4: Model Deployment

> **Stack:** Python 3.11 · FastAPI · Ollama · Qwen2.5:0.5B · Docker

---

## Table of Contents

1. [Option A — Docker (Recommended)](#option-a--docker-recommended)
2. [Option B — Local (Without Docker)](#option-b--local-without-docker)
3. [Testing the Endpoints](#testing-the-endpoints)
4. [Running the Benchmark](#running-the-benchmark)
5. [Running the Tests](#running-the-tests)
6. [Common Issues](#common-issues)

---

## Option A — Docker (Recommended)

> No Ollama or Python installation needed. Everything runs inside the container.

### Step 1 — Build the image

```bash
docker build -t section4 .
```

> ⏳ First build takes a few minutes — it installs Ollama and all Python dependencies inside the image.

---

### Step 2 — Run the container

```bash
docker run -p 8000:8000 section4
```

**What happens inside the container:**
1. Ollama server starts in the background
2. `qwen2.5:0.5b` is automatically pulled (only on first run)
3. FastAPI/Uvicorn starts on port `8000`

> ⏳ Allow up to **2 minutes** on first run for the model to be pulled.

---

### Step 2 (Alternative) — Run with docker-compose

```bash
docker-compose up --build
```

This variant mounts a persistent volume for the model weights, so the model is **not re-downloaded** on every restart.

---

### Step 3 — Verify it's running

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "ollama": "reachable",
  "model": "qwen2.5:0.5b",
  "version": "1.0.0"
}
```

---

## Option B — Local (Without Docker)

### Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.11+ | [python.org](https://www.python.org/downloads/) |
| Ollama | Latest | [ollama.com](https://ollama.com/download) |

---

### Step 1 — Install Ollama and pull the model

```bash
# Download and install Ollama from https://ollama.com/download
# Then pull the model:
ollama pull qwen2.5:0.5b
```

---

### Step 2 — Clone / navigate to the project

```bash
cd "Section 4 — Model Deployment"
```

---

### Step 3 — Create a virtual environment

```bash
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (Linux / macOS)
source .venv/bin/activate
```

---

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 5 — Configure environment

```bash
# Copy the example env file
cp .env.example .env

# Edit if needed (defaults work out of the box)
# notepad .env   (Windows)
# nano .env      (Linux/macOS)
```

---

### Step 6 — Start Ollama

```bash
ollama serve
```

> Keep this terminal open. Ollama will run on `http://localhost:11434`.

---

### Step 7 — Start the FastAPI server

Open a **new terminal** in the project directory:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API is now live at **http://localhost:8000**

---

## Testing the Endpoints

### 1. Health Check

```bash
curl http://localhost:8000/health
```

---

### 2. Blocking Generation

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?", "max_tokens": 200}'
```

Expected response shape:
```json
{
  "request_id": "550e8400-...",
  "text": "Machine learning is...",
  "model": "qwen2.5:0.5b",
  "metrics": {
    "completion_tokens": 45,
    "total_duration_ms": 1823.4,
    "eval_rate_tps": 24.7
  }
}
```

---

### 3. Streaming Generation (Token-by-Token)

```bash
curl -N -X POST http://localhost:8000/api/v1/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain neural networks in 3 sentences."}'
```

You will see tokens arriving one by one:

```
data: {"token": "Neural", "index": 0, "done": false}
data: {"token": " networks", "index": 1, "done": false}
data: {"token": " are", "index": 2, "done": false}
...
data: {"token": "", "index": 42, "done": true, "metrics": {...}}
```

---

### 4. Interactive API Docs (Swagger UI)

Open in browser:

```
http://localhost:8000/docs
```

---

## Running the Benchmark

The benchmark measures **Time-To-First-Token (TTFT)** and **Total Latency** under **10 concurrent requests**.

### Prerequisites

Make sure the API server is running (via Docker or locally).

### Install benchmark dependencies (local only)

```bash
pip install httpx
```

### Run the benchmark

```bash
python -m benchmark.runner
```

### Expected output

```
════════════════════════════════════════════════════════════════
           Benchmark Report — Section 4 (Task 4.1)
           Model: qwen2.5:0.5b  |  Concurrent: 10
════════════════════════════════════════════════════════════════

┌────────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐
│ Metric         │ Min (ms)   │ Max (ms)   │ Mean (ms)  │ P95 (ms)   │ StdDev     │
├────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤
│ TTFT           │  ...       │  ...       │  ...       │  ...       │  ...       │
│ Total Latency  │  ...       │  ...       │  ...       │  ...       │  ...       │
└────────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘

Results saved to: benchmark/results/report_YYYYMMDD_HHMMSS.json
```

---

## Running the Tests

### Install dev dependencies

```bash
pip install -r requirements-dev.txt
```

### Run all tests

```bash
pytest tests/ -v
```

### Run only unit tests

```bash
pytest tests/unit/ -v
```

---

## Common Issues

| Problem | Cause | Fix |
|---|---|---|
| `curl: (7) Failed to connect` | Server not started yet | Wait for model pull to finish (check docker logs) |
| `503 Service Unavailable` on `/health` | Ollama not reachable | Run `ollama serve` or check container logs |
| `404` on generate | Model not pulled | Run `ollama pull qwen2.5:0.5b` |
| Slow first response | Model cold-start (loading into RAM) | Send 1 warm-up request; subsequent ones are faster |
| `ModuleNotFoundError: app` | Wrong working directory | Run all commands from project root (`Section 4 — Model Deployment/`) |

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `MODEL_NAME` | `qwen2.5:0.5b` | Model to load |
| `API_PORT` | `8000` | FastAPI port |
| `LOG_LEVEL` | `info` | Logging level (`debug`, `info`, `warning`) |
| `WORKERS` | `1` | Number of Uvicorn workers |
| `REQUEST_TIMEOUT` | `120.0` | Max seconds to wait for Ollama (blocking) |
| `STREAM_TIMEOUT` | `180.0` | Max seconds for a streaming response |
