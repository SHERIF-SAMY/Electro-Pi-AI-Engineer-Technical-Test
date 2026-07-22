#!/bin/bash
set -e

# Start Ollama in the background
echo "Starting Ollama server..."
ollama serve &

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
while ! curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 1
done

echo "Ollama is ready. Pulling model $MODEL_NAME..."
ollama pull $MODEL_NAME

echo "Model pulled successfully. Starting FastAPI..."
exec uvicorn app.main:app --host $API_HOST --port $API_PORT --workers $WORKERS
