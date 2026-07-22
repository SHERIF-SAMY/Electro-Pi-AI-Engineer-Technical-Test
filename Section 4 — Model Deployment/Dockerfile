FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/
COPY docker/ docker/

# Make scripts executable
RUN chmod +x docker/entrypoint.sh docker/healthcheck.sh

# Environment variables
ENV OLLAMA_HOST=http://127.0.0.1:11434
ENV MODEL_NAME=qwen2.5:0.5b
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV LOG_LEVEL=info
ENV WORKERS=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD /app/docker/healthcheck.sh

ENTRYPOINT ["/app/docker/entrypoint.sh"]
