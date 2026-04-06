"""
pharmaai/core/metrics.py
Custom Prometheus metrics for PharmaAI.
Import and instrument throughout the codebase.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Summary

# ── Ingestion ─────────────────────────────────────────────────────────────────
documents_ingested = Counter(
    "pharmaai_documents_ingested_total",
    "Total documents ingested",
    ["source"],
)

ingestion_errors = Counter(
    "pharmaai_ingestion_errors_total",
    "Total ingestion errors",
    ["source", "error_type"],
)

# ── Embedding ─────────────────────────────────────────────────────────────────
embedding_latency = Histogram(
    "pharmaai_embedding_latency_seconds",
    "Time to embed a batch of texts",
    ["model"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# ── Index ─────────────────────────────────────────────────────────────────────
faiss_index_size = Gauge(
    "pharmaai_faiss_index_size",
    "Number of vectors in the FAISS index",
)

search_latency = Histogram(
    "pharmaai_search_latency_seconds",
    "Hybrid search latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# ── RAG ───────────────────────────────────────────────────────────────────────
rag_latency = Histogram(
    "pharmaai_rag_latency_seconds",
    "End-to-end RAG pipeline latency",
    ["domain"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

rag_requests = Counter(
    "pharmaai_rag_requests_total",
    "Total RAG pipeline requests",
    ["domain", "cached"],
)

# ── Notifications ─────────────────────────────────────────────────────────────
notifications_sent = Counter(
    "pharmaai_notifications_sent_total",
    "Total notifications sent",
    ["channel", "severity"],
)

notification_errors = Counter(
    "pharmaai_notification_errors_total",
    "Total notification errors",
    ["channel"],
)

# ── Classification ────────────────────────────────────────────────────────────
classification_requests = Counter(
    "pharmaai_classification_requests_total",
    "Total domain classification requests",
    ["domain"],
)