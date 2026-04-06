# ── Builder stage ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PATH="/opt/poetry/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    build-essential git curl \
    libpq-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/local/bin/python && \
    pip install --no-cache-dir "poetry==$POETRY_VERSION"

# ── Dependencies ────────────────────────────────────────────────────────────
WORKDIR /app
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root --no-dev --no-interaction --no-ansi

# ── Application ─────────────────────────────────────────────────────────────
COPY . .
RUN poetry install --only-root --no-interaction --no-ansi

# Create necessary directories
RUN mkdir -p data/logs data/raw data/processed

EXPOSE 8000 5055

CMD ["uvicorn", "pharmaai.api.app:app", "--host", "0.0.0.0", "--port", "8000"]