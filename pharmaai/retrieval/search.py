"""
pharmaai/retrieval/search.py
Hybrid retrieval: dense (FAISS) + sparse (BM25) with optional metadata filters.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from pharmaai.core.database import document_store
from pharmaai.core.schemas import (
    ContentType, Document, Domain, SearchResult,
)
from pharmaai.embeddings.index import faiss_index
from pharmaai.embeddings.models import embedding_service

logger = logging.getLogger("pharmaai.retrieval.search")


@dataclass
class SearchFilters:
    domain: Domain | None = None
    content_type: ContentType | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    source: str | None = None
    metadata_filter: dict[str, Any] = field(default_factory=dict)


class HybridSearch:
    """
    Two-stage retrieval:
      1. Dense search via FAISS (fast ANN)
      2. Optional BM25 re-ranking on the dense candidates
    """

    def __init__(self, dense_candidates_multiplier: int = 3):
        self._multiplier = dense_candidates_multiplier

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: SearchFilters | None = None,
        use_bm25_rerank: bool = True,
    ) -> list[SearchResult]:
        t0 = time.perf_counter()
        filters = filters or SearchFilters()

        # ── 1. Dense retrieval ──────────────────────────────────────────────
        domain = filters.domain
        query_emb = embedding_service.embed(
            [query], domain=domain, combine=True
        )[0]

        n_candidates = top_k * self._multiplier
        raw = faiss_index.search(query_emb, top_k=n_candidates)
        if not raw:
            return []

        faiss_indices = [idx for idx, _ in raw]
        score_map = {idx: score for idx, score in raw}

        # ── 2. Fetch documents from DB ──────────────────────────────────────
        docs_by_idx = await document_store.get_by_faiss_indices(faiss_indices)
        candidates: list[tuple[Document, float]] = []
        for idx in faiss_indices:
            doc = docs_by_idx.get(idx)
            if doc is None:
                continue
            # Apply metadata filters
            if filters.domain and doc.domain != filters.domain:
                continue
            if filters.content_type and doc.content_type != filters.content_type:
                continue
            if filters.date_from and doc.timestamp < filters.date_from:
                continue
            if filters.date_to and doc.timestamp > filters.date_to:
                continue
            if filters.source and doc.source != filters.source:
                continue
            candidates.append((doc, score_map[idx]))

        if not candidates:
            return []

        # ── 3. BM25 re-ranking ──────────────────────────────────────────────
        if use_bm25_rerank and len(candidates) > top_k:
            tokenized_corpus = [d.content.lower().split() for d, _ in candidates]
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = bm25.get_scores(query.lower().split())

            # Combine: 0.7 × dense + 0.3 × normalized BM25
            if bm25_scores.max() > 0:
                bm25_norm = bm25_scores / bm25_scores.max()
            else:
                bm25_norm = bm25_scores

            combined = [
                (doc, 0.7 * dense + 0.3 * float(bm25_norm[i]))
                for i, (doc, dense) in enumerate(candidates)
            ]
            combined.sort(key=lambda x: x[1], reverse=True)
            candidates = combined

        # ── 4. Build results ────────────────────────────────────────────────
        results = [
            SearchResult(document=doc, score=round(score, 4), rank=rank)
            for rank, (doc, score) in enumerate(candidates[:top_k], 1)
        ]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "HybridSearch: query=%r top_k=%d results=%d latency=%.1fms",
            query[:60], top_k, len(results), elapsed_ms,
        )
        return results


hybrid_search = HybridSearch()