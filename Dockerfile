# ============================================================
# Student Task Manager Environment – Dockerfile
# ============================================================
# Multi-stage, production-ready Dockerfile
# Exposes REST endpoints on port 8000
# ============================================================

FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# -------------------------------------------------------
# Dependency installation layer (cached separately)
# -------------------------------------------------------
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------------
# Application layer
# -------------------------------------------------------
FROM deps AS app

# Copy source
COPY --chown=appuser:appuser env/ ./env/
COPY --chown=appuser:appuser main.py .
COPY --chown=appuser:appuser server/ ./server/
COPY --chown=appuser:appuser inference.py .
COPY --chown=appuser:appuser openenv.yaml .

# Create __init__.py if not present
RUN touch env/__init__.py

# Switch to non-root user
USER appuser

# Environment defaults (override via docker run -e)
ENV ENV_SCENARIO=medium \
    ENV_SEED=42 \
    HOST=0.0.0.0 \
    PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

