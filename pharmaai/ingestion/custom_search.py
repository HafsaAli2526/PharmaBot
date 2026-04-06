"""
pharmaai/ingestion/custom_search.py
Google Custom Search Engine (CSE) client.
Returns web pages as Documents for supplementary context.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import AsyncIterator

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from pharmaai.core.config import get_settings
from pharmaai.core.schemas import Document, ContentType, Domain
from pharmaai.processing.formatter import formatter

logger = logging.getLogger("pharmaai.ingestion.custom_search")

_RETRY = retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))

CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"


class GoogleCSEWorker:
    SOURCE = "google_cse"

    def __init__(self):
        settings = get_settings()
        self._key = settings.apis.google_cse.api_key
        self._cx = settings.apis.google_cse.cx
        self._client = httpx.AsyncClient(timeout=20)

    @_RETRY
    async def search(
        self,
        query: str,
        num: int = 10,
        start: int = 1,
    ) -> list[Document]:
        params = {
            "key": self._key,
            "cx": self._cx,
            "q": query,
            "num": min(num, 10),
            "start": start,
        }
        r = await self._client.get(CSE_ENDPOINT, params=params)
        r.raise_for_status()
        data = r.json()
        docs = []
        for item in data.get("items", []):
            snippet = item.get("snippet", "")
            title = item.get("title", "")
            url = item.get("link", "")
            content = formatter.build_content(title, snippet)
            docs.append(Document(
                content=formatter.clean(content),
                content_type=ContentType.WEB_PAGE,
                domain=Domain.UNKNOWN,
                source=self.SOURCE,
                source_id=url,
                title=title,
                url=url,
                timestamp=datetime.utcnow(),
                metadata={"query": query},
            ))
        return docs

    async def search_all_pages(
        self, query: str, max_results: int = 30
    ) -> AsyncIterator[list[Document]]:
        """Yield pages of results (10 per page, up to max_results)."""
        fetched = 0
        start = 1
        while fetched < max_results:
            docs = await self.search(query, num=10, start=start)
            if not docs:
                break
            yield docs
            fetched += len(docs)
            start += 10
            await asyncio.sleep(1)  # respect rate limit