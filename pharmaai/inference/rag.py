"""
pharmaai/inference/rag.py
Full Retrieval-Augmented Generation pipeline.
  1. Classify query domain
  2. Generate sub-queries (query expansion)
  3. Multi-query retrieval + RRF fusion
  4. Build prompt with retrieved context
  5. Generate answer with fine-tuned SLM
  6. Cache response + record Prometheus metrics
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from pharmaai.core.cache import cache
from pharmaai.core.metrics import rag_latency, rag_requests
from pharmaai.core.schemas import (
    AskRequest, AskResponse, Domain, SearchResult
)
from pharmaai.inference.query_generator import query_generator
from pharmaai.inference.small_lm import slm
from pharmaai.processing.domain_classifier import classifier
from pharmaai.retrieval.fusion import multi_query_search
from pharmaai.retrieval.search import SearchFilters, hybrid_search

logger = logging.getLogger("pharmaai.inference.rag")


SYSTEM_PROMPT = """You are PharmaAI, a pharmaceutical intelligence assistant.
You answer questions accurately based on the provided context.
If the context does not contain sufficient information, say so clearly.
Do NOT fabricate drug names, dosages, or clinical data.
Always cite the source title when referencing specific information.
"""

ANSWER_PROMPT_TEMPLATE = """{system}

## Context Documents
{context}

## Question
{question}

## Instructions
Provide a clear, concise answer based on the context above.
Cite sources by their title where relevant.
If answering about adverse events or drug interactions, emphasize safety.

## Answer
"""


def _build_context(results: list[SearchResult], max_chars: int = 6000) -> str:
    parts = []
    chars = 0
    for r in results:
        doc = r.document
        snippet = (
            f"[{r.rank}] **{doc.title or doc.source}** "
            f"(Source: {doc.source}, Domain: {doc.domain.value})\n"
            f"{doc.content[:800]}\n"
        )
        if chars + len(snippet) > max_chars:
            break
        parts.append(snippet)
        chars += len(snippet)
    return "\n---\n".join(parts)


class RAGPipeline:
    def __init__(self, use_query_expansion: bool = True):
        self._expand = use_query_expansion

    async def answer(self, request: AskRequest) -> AskResponse:
        t0 = time.perf_counter()

        # 0. Cache lookup
        cache_key = cache.make_ask_key(
            request.question,
            request.domain.value if request.domain else None,
        )
        if request.use_cache:
            cached = await cache.get(cache_key)
            if cached:
                resp = AskResponse(**cached)
                resp.cached = True
                rag_requests.labels(domain=resp.domain.value, cached="true").inc()
                return resp

        # 1. Classify domain
        domain = request.domain
        if domain is None:
            result = classifier.classify(request.question)
            domain = result.domain

        # 2. Query expansion
        queries = [request.question]
        if self._expand:
            try:
                extra = query_generator.generate(request.question, domain)
                queries.extend(extra)
            except Exception as exc:
                logger.warning("Query expansion failed: %s", exc)

        # 3. Retrieval
        filters = SearchFilters(domain=domain if domain != Domain.UNKNOWN else None)
        if len(queries) > 1:
            sources = await multi_query_search(
                queries, top_k=request.top_k, filters=filters
            )
        else:
            sources = await hybrid_search.search(
                queries[0], top_k=request.top_k, filters=filters
            )

        # 4. Build prompt
        context = _build_context(sources)
        prompt = ANSWER_PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            context=context if context else "No relevant documents found.",
            question=request.question,
        )

        # 5. Generate answer
        answer_text = slm.generate(prompt)

        latency = (time.perf_counter() - t0) * 1000
        logger.info(
            "RAG complete: domain=%s queries=%d sources=%d latency=%.1fms",
            domain.value, len(queries), len(sources), latency,
        )

        response = AskResponse(
            answer=answer_text,
            sources=sources,
            domain=domain,
            latency_ms=round(latency, 1),
        )

        # 6. Metrics + cache
        rag_latency.labels(domain=domain.value).observe(latency / 1000)
        rag_requests.labels(domain=domain.value, cached="false").inc()
        await cache.set(cache_key, response.model_dump(mode="json"))

        return response


rag_pipeline = RAGPipeline()