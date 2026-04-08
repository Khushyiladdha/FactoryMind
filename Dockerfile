FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY pyproject.toml ./

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir openenv-core || echo "[INFO] openenv-core skipped"

COPY factory_mind/ ./factory_mind/
COPY server/ ./server/
COPY tests/ ./tests/
COPY openenv.yaml ./
COPY inference.py ./
COPY test_local.py ./
COPY README.md ./

RUN openenv validate . || echo "[INFO] openenv validate skipped"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]