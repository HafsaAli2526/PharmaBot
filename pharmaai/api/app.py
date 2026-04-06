"""
pharmaai/api/app.py
FastAPI application factory with:
  – CORS, rate limiting, logging middleware
  – Prometheus instrumentation
  – Lifespan startup (load models, init DB, load index)
  – Routes: /v1/ask, /v1/search, /v1/classify
"""
from __future__ import annotations

import logging
import logging.config
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from pharmaai.api.middleware import RateLimitMiddleware
from pharmaai.api.routes import ask, classify, search
from pharmaai.core.config import get_settings
from pharmaai.core.database import create_tables

logger = logging.getLogger("pharmaai.api")


def _setup_logging() -> None:
    log_cfg_path = Path("configs/logging.yaml")
    if log_cfg_path.exists():
        with open(log_cfg_path) as f:
            cfg = yaml.safe_load(f)
        Path("./data/logs").mkdir(parents=True, exist_ok=True)
        logging.config.dictConfig(cfg)
    else:
        logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup & shutdown lifecycle."""
    _setup_logging()
    settings = get_settings()
    logger.info("PharmaAI starting up…")

    # Init database tables
    try:
        await create_tables()
        logger.info("Database ready.")
    except Exception as exc:
        logger.error("DB init failed: %s", exc)

    # Load vector index
    try:
        from pharmaai.embeddings.index import faiss_index
        faiss_index.load_or_build()
        logger.info("FAISS index ready (size=%d).", faiss_index.size)
    except Exception as exc:
        logger.error("Index init failed: %s", exc)

    # Pre-load embedding models (non-blocking but we start the registry)
    try:
        from pharmaai.embeddings.models import registry
        import threading
        t = threading.Thread(target=registry.load_all, daemon=True)
        t.start()
        logger.info("Embedding model loading started in background.")
    except Exception as exc:
        logger.error("Model loading failed: %s", exc)

    yield

    # Shutdown
    logger.info("PharmaAI shutting down…")
    try:
        from pharmaai.embeddings.index import faiss_index
        faiss_index.save()
    except Exception:
        pass


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="PharmaAI Assistant",
        description="Pharmaceutical intelligence platform – adverse events, R&D, regulatory & formula analysis.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Rate limiting ─────────────────────────────────────────────────────────
    rate_cfg = settings.api.rate_limit
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=rate_cfg.get("requests_per_minute", 60),
        burst=rate_cfg.get("burst", 20),
    )

    # ── Prometheus ────────────────────────────────────────────────────────────
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    # ── Request logging middleware ────────────────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "%s %s → %d (%.1fms)",
            request.method, request.url.path,
            response.status_code, elapsed,
        )
        return response

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exc_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health():
        from pharmaai.embeddings.index import faiss_index
        from pharmaai.embeddings.models import registry
        return {
            "status": "ok",
            "index_size": faiss_index.size,
            "models_loaded": registry._loaded,
        }

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(ask.router, prefix="/v1", tags=["QA"])
    app.include_router(search.router, prefix="/v1", tags=["Search"])
    app.include_router(classify.router, prefix="/v1", tags=["Classify"])

    return app


app = create_app()