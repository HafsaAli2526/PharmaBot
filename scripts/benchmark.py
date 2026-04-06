"""
scripts/benchmark.py
Benchmark embedding generation, FAISS search latency, and API throughput.
"""
from __future__ import annotations

import asyncio
import statistics
import time
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pharmaai.benchmark")


def bench_embedding(n_texts: int = 100, n_runs: int = 3) -> dict:
    from pharmaai.embeddings.models import embed_texts, registry
    registry.load_all()

    texts = [f"Adverse event report for drug compound {i}." for i in range(n_texts)]
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        embs = embed_texts(texts, "biobert", batch_size=32)
        latencies.append((time.perf_counter() - t0) * 1000)
        assert embs.shape == (n_texts, 768)

    return {
        "model": "biobert",
        "n_texts": n_texts,
        "mean_ms": round(statistics.mean(latencies), 1),
        "min_ms": round(min(latencies), 1),
        "throughput_per_sec": round(n_texts / (statistics.mean(latencies) / 1000), 1),
    }


def bench_faiss(n_index: int = 10_000, n_queries: int = 100) -> dict:
    from pharmaai.embeddings.index import FaissIndex
    from pharmaai.core.config import get_settings
    settings = get_settings()

    idx = FaissIndex()
    idx.build()

    # Populate with random vectors
    vecs = np.random.randn(n_index, settings.index.dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    doc_ids = [f"doc_{i}" for i in range(n_index)]
    idx.add(vecs, doc_ids)

    # Benchmark search
    queries = np.random.randn(n_queries, settings.index.dim).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        idx.search(q, top_k=10)
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "index_size": n_index,
        "n_queries": n_queries,
        "mean_ms": round(statistics.mean(latencies), 2),
        "p95_ms": round(sorted(latencies)[int(n_queries * 0.95)], 2),
        "p99_ms": round(sorted(latencies)[int(n_queries * 0.99)], 2),
    }


def main():
    print("\n=== PharmaAI Benchmark ===\n")

    print("── Embedding Benchmark ──")
    try:
        emb_results = bench_embedding(n_texts=64, n_runs=3)
        for k, v in emb_results.items():
            print(f"  {k}: {v}")
    except Exception as exc:
        print(f"  Skipped (models not loaded): {exc}")

    print("\n── FAISS Search Benchmark ──")
    faiss_results = bench_faiss(n_index=50_000, n_queries=200)
    for k, v in faiss_results.items():
        print(f"  {k}: {v}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()