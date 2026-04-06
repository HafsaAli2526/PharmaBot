"""
pharmaai/retrieval/fusion.py
Reciprocal Rank Fusion (RRF) to merge result lists from multiple queries
or multiple retrieval strategies.
"""
from __future__ import annotations

from pharmaai.core.schemas import SearchResult


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60,
) -> list[SearchResult]:
    """
    Merge multiple ranked lists via RRF.

    score(d) = Σ_i  1 / (k + rank_i(d))

    Documents not appearing in a list are simply not scored for that list.
    Returns a single merged list sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    docs: dict[str, SearchResult] = {}

    for results in result_lists:
        for rank, res in enumerate(results, 1):
            doc_id = res.document.id
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in docs:
                docs[doc_id] = res

    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)
    merged = []
    for rank, doc_id in enumerate(sorted_ids, 1):
        result = docs[doc_id]
        merged.append(SearchResult(
            document=result.document,
            score=round(scores[doc_id], 6),
            rank=rank,
        ))
    return merged


async def multi_query_search(
    queries: list[str],
    top_k: int = 10,
    filters=None,
) -> list[SearchResult]:
    """Run the same set of queries independently and fuse results."""
    import asyncio
    from pharmaai.retrieval.search import hybrid_search

    tasks = [
        hybrid_search.search(q, top_k=top_k * 2, filters=filters)
        for q in queries
    ]
    results_per_query = await asyncio.gather(*tasks)
    return reciprocal_rank_fusion(list(results_per_query), k=60)[:top_k]