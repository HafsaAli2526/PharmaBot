"""pharmaai/api/routes/search.py"""
from __future__ import annotations
import time
from fastapi import APIRouter
from pharmaai.core.schemas import SearchRequest, SearchResponse
from pharmaai.retrieval.search import hybrid_search, SearchFilters

router = APIRouter()

@router.post("/search", response_model=SearchResponse, summary="Search documents")
async def search(request: SearchRequest) -> SearchResponse:
    """
    Semantic + keyword hybrid search across all indexed pharmaceutical documents.
    """
    t0 = time.perf_counter()
    filters = SearchFilters(
        domain=request.domain,
        content_type=request.content_type,
        date_from=request.date_from,
        date_to=request.date_to,
    )
    results = await hybrid_search.search(
        request.query, top_k=request.top_k, filters=filters
    )
    return SearchResponse(
        results=results,
        total=len(results),
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )