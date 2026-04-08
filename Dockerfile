FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (layer caching)
COPY requirements.txt ./
COPY pyproject.toml ./

# Install pinned dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install openenv-core if available (best-effort)
RUN pip install --no-cache-dir openenv-core || echo "[INFO] openenv-core not yet published; skipping"

# Copy all project files
COPY factory_mind/ ./factory_mind/
COPY server/ ./server/
COPY tests/ ./tests/
COPY openenv.yaml ./
COPY inference.py ./
COPY test_local.py ./
COPY README.md ./

# Validate OpenEnv spec (best-effort; don't fail build if CLI unavailable)
RUN openenv validate . || echo "[INFO] openenv validate skipped (CLI not available at build time)"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
