FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY service ./service
COPY configs ./configs
COPY prompts ./prompts
COPY mcp_server ./mcp_server
COPY eval ./eval
COPY scripts ./scripts
COPY tests ./tests
COPY assets ./assets
COPY docs ./docs
COPY Makefile LICENSE ./

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ".[dev]" \
    && chmod +x scripts/download_adapters.sh

ENV EEDI_LIGHT=1 \
    PYTHONPATH=/app \
    HF_HOME=/data/hf_cache \
    EEDI_DATA=/data/eedi-data

EXPOSE 6006

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:6006/health || exit 1

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "6006"]
