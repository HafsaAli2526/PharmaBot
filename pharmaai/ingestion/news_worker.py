"""
pharmaai/ingestion/news_worker.py
News & preprint ingestion:
  – NewsAPI
  – BioRxiv / MedRxiv RSS/API feeds
  – EventRegistry (optional)
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import AsyncIterator

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from pharmaai.core.config import get_settings
from pharmaai.core.schemas import Document, ContentType, Domain
from pharmaai.processing.formatter import formatter
from pharmaai.ingestion.progress_tracker import progress_tracker

logger = logging.getLogger("pharmaai.ingestion.news_worker")

_RETRY = retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20), reraise=True)

PHARMA_QUERY = (
    "pharmaceutical OR drug approval OR FDA OR EMA OR "
    "adverse drug OR clinical trial OR biotech"
)


# ─────────────────────────────────────────────────────────────────────────────
# NewsAPI
# ─────────────────────────────────────────────────────────────────────────────

class NewsAPIWorker:
    SOURCE = "newsapi"

    def __init__(self):
        settings = get_settings()
        self._key = settings.apis.newsapi.api_key
        self._base = settings.apis.newsapi.base_url
        self._client = httpx.AsyncClient(timeout=20)

    @_RETRY
    async def _get(self, params: dict) -> dict:
        r = await self._client.get(
            f"{self._base}/everything",
            params={**params, "apiKey": self._key},
        )
        r.raise_for_status()
        return r.json()

    async def ingest(
        self,
        query: str = PHARMA_QUERY,
        days_back: int = 7,
        page_size: int = 100,
    ) -> AsyncIterator[list[Document]]:
        from_dt = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        page = 1
        total = 0
        while True:
            data = await self._get({
                "q": query,
                "from": from_dt,
                "pageSize": page_size,
                "page": page,
                "language": "en",
                "sortBy": "publishedAt",
            })
            articles = data.get("articles", [])
            if not articles:
                break
            docs = []
            for a in articles:
                pub_at = a.get("publishedAt", "")
                try:
                    ts = datetime.fromisoformat(pub_at.replace("Z", "+00:00"))
                except Exception:
                    ts = datetime.utcnow()
                content = formatter.build_content(
                    a.get("title", ""),
                    a.get("description", "") + " " + (a.get("content") or ""),
                )
                docs.append(Document(
                    content=formatter.clean(content),
                    content_type=ContentType.NEWS_ARTICLE,
                    domain=Domain.PHARMACOVIGILANCE,
                    source=self.SOURCE,
                    source_id=a.get("url", ""),
                    title=a.get("title", ""),
                    url=a.get("url", ""),
                    timestamp=ts,
                    metadata={"source_name": a.get("source", {}).get("name", "")},
                ))
            yield docs
            total += len(docs)
            progress_tracker.update(self.SOURCE, increment=len(docs))
            if len(articles) < page_size or total >= data.get("totalResults", 0):
                break
            page += 1
            await asyncio.sleep(1)


# ─────────────────────────────────────────────────────────────────────────────
# BioRxiv / MedRxiv
# ─────────────────────────────────────────────────────────────────────────────

class BioRxivWorker:
    """Fetches recent preprints from BioRxiv and MedRxiv via their REST API."""
    SOURCE = "biorxiv"
    BASE = "https://api.biorxiv.org"

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=30)

    @_RETRY
    async def _get(self, server: str, interval: str, cursor: int) -> dict:
        url = f"{self.BASE}/details/{server}/{interval}/{cursor}/100/json"
        r = await self._client.get(url)
        r.raise_for_status()
        return r.json()

    async def ingest(
        self,
        server: str = "medrxiv",
        days_back: int = 7,
    ) -> AsyncIterator[list[Document]]:
        end = datetime.utcnow()
        start = end - timedelta(days=days_back)
        interval = f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        cursor = 0
        while True:
            data = await self._get(server, interval, cursor)
            collection = data.get("collection", [])
            if not collection:
                break
            docs = []
            for item in collection:
                ts_str = item.get("date", "2020-01-01")
                try:
                    ts = datetime.fromisoformat(ts_str)
                except Exception:
                    ts = datetime.utcnow()
                content = formatter.build_content(
                    item.get("title", ""),
                    item.get("abstract", ""),
                    {"authors": item.get("authors", ""), "doi": item.get("doi", "")},
                )
                docs.append(Document(
                    content=formatter.clean(content),
                    content_type=ContentType.RND_ARTICLE,
                    domain=Domain.RND,
                    source=f"{server}",
                    source_id=item.get("doi", ""),
                    title=item.get("title", ""),
                    url=f"https://doi.org/{item.get('doi', '')}",
                    timestamp=ts,
                    metadata={"doi": item.get("doi", ""), "server": server},
                ))
            yield docs
            progress_tracker.update(self.SOURCE, increment=len(docs))
            cursor += 100
            total = data.get("messages", [{}])[0].get("total", 0)
            if cursor >= total:
                break
            await asyncio.sleep(0.5)