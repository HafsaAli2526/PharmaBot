"""
pharmaai/ingestion/api_workers.py
Async ingestion workers for external APIs:
  – PubMed (Entrez eUtils)
  – OpenFDA (drug labels + adverse events)
  – ClinicalTrials.gov v2
Each worker normalises records into the canonical Document schema and
publishes them via the queue publisher (or yields them for direct use).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import AsyncIterator
from xml.etree import ElementTree as ET

import httpx
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type
)

from pharmaai.core.config import get_settings
from pharmaai.core.schemas import Document, ContentType, Domain
from pharmaai.processing.formatter import formatter
from pharmaai.processing.domain_classifier import classifier
from pharmaai.ingestion.progress_tracker import progress_tracker

logger = logging.getLogger("pharmaai.ingestion.api_workers")

_RETRY = retry(
    retry=retry_if_exception_type(httpx.HTTPError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)


class RateLimiter:
    """Simple token-bucket rate limiter."""
    def __init__(self, rate: float):
        self._rate = rate
        self._last = 0.0

    async def acquire(self):
        now = asyncio.get_event_loop().time()
        sleep = max(0, (1 / self._rate) - (now - self._last))
        if sleep:
            await asyncio.sleep(sleep)
        self._last = asyncio.get_event_loop().time()


# ─────────────────────────────────────────────────────────────────────────────
# PubMed
# ─────────────────────────────────────────────────────────────────────────────

class PubMedWorker:
    SOURCE = "pubmed"

    def __init__(self):
        settings = get_settings()
        cfg = settings.apis.pubmed
        self._base = cfg.base_url
        self._key = cfg.api_key
        self._limiter = RateLimiter(cfg.rate_limit_per_second)
        self._client = httpx.AsyncClient(timeout=30)

    @_RETRY
    async def _get(self, url: str, params: dict) -> dict | ET.Element:
        await self._limiter.acquire()
        r = await self._client.get(url, params=params)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "json" in ct:
            return r.json()
        return ET.fromstring(r.text)

    async def search_ids(
        self, query: str, retmax: int = 100, date_from: str = ""
    ) -> list[str]:
        params = dict(
            db="pubmed", term=query, retmax=retmax,
            retmode="json", sort="pub_date",
        )
        if self._key:
            params["api_key"] = self._key
        if date_from:
            params["mindate"] = date_from
            params["datetype"] = "pdat"
        data = await self._get(f"{self._base}/esearch.fcgi", params)
        return data.get("esearchresult", {}).get("idlist", [])

    async def fetch_abstracts(self, pmids: list[str]) -> list[Document]:
        if not pmids:
            return []
        params = dict(
            db="pubmed", id=",".join(pmids), retmode="xml",
        )
        if self._key:
            params["api_key"] = self._key
        root = await self._get(f"{self._base}/efetch.fcgi", params)
        docs = []
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else ""
            title_el = article.find(".//ArticleTitle")
            title = title_el.text or "" if title_el is not None else ""
            abstract_els = article.findall(".//AbstractText")
            abstract = " ".join(
                (el.text or "") for el in abstract_els if el.text
            )
            pub_date_el = article.find(".//PubDate/Year")
            year = pub_date_el.text if pub_date_el is not None else "2000"
            try:
                ts = datetime(int(year), 1, 1)
            except ValueError:
                ts = datetime.utcnow()

            content = formatter.build_content(title, abstract)
            classification = classifier.classify(content)

            docs.append(Document(
                content=formatter.clean(content),
                content_type=ContentType.RND_ARTICLE,
                domain=classification.domain if classification.domain != Domain.UNKNOWN else Domain.RND,
                source=self.SOURCE,
                source_id=pmid,
                title=title,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                timestamp=ts,
                metadata={"pmid": pmid},
            ))
        return docs

    async def ingest(
        self,
        query: str = "pharmaceutical adverse event",
        batch_size: int = 200,
        max_total: int = 10_000,
    ) -> AsyncIterator[list[Document]]:
        """Yield batches of Documents for the given query."""
        progress = progress_tracker.get(self.SOURCE)
        date_from = progress.get("last_timestamp") or "2010/01/01"
        if isinstance(date_from, datetime):
            date_from = date_from.strftime("%Y/%m/%d")

        retstart = 0
        total_done = 0
        while total_done < max_total:
            ids = await self.search_ids(
                query, retmax=min(batch_size, max_total - total_done),
                date_from=str(date_from),
            )
            if not ids:
                break
            docs = await self.fetch_abstracts(ids)
            yield docs
            progress_tracker.update(
                self.SOURCE,
                last_id=ids[-1],
                increment=len(docs),
            )
            total_done += len(docs)
            if len(ids) < batch_size:
                break
            await asyncio.sleep(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# OpenFDA
# ─────────────────────────────────────────────────────────────────────────────

class OpenFDAWorker:
    SOURCE = "openfda"

    def __init__(self):
        settings = get_settings()
        cfg = settings.apis.openfda
        self._base = cfg.base_url
        self._key = cfg.api_key
        self._limiter = RateLimiter(cfg.rate_limit_per_second)
        self._client = httpx.AsyncClient(timeout=30)

    @_RETRY
    async def _get(self, endpoint: str, params: dict) -> dict:
        await self._limiter.acquire()
        if self._key:
            params["api_key"] = self._key
        r = await self._client.get(f"{self._base}{endpoint}", params=params)
        r.raise_for_status()
        return r.json()

    async def fetch_adverse_events(
        self, skip: int = 0, limit: int = 100
    ) -> list[Document]:
        data = await self._get(
            "/drug/event.json",
            {"limit": limit, "skip": skip, "sort": "receivedate:desc"},
        )
        docs = []
        for result in data.get("results", []):
            reactions = ", ".join(
                r.get("reactionmeddrapt", "")
                for r in result.get("patient", {}).get("reaction", [])
            )
            drugs = ", ".join(
                d.get("medicinalproduct", "")
                for d in result.get("patient", {}).get("drug", [])
            )
            serious = result.get("serious", 0) == 1
            report_id = result.get("safetyreportid", "")
            receive_date = result.get("receivedate", "20000101")
            try:
                ts = datetime.strptime(str(receive_date)[:8], "%Y%m%d")
            except ValueError:
                ts = datetime.utcnow()

            content = (
                f"Drug(s): {drugs}. "
                f"Adverse reactions: {reactions}. "
                f"Serious: {'Yes' if serious else 'No'}."
            )

            docs.append(Document(
                content=formatter.clean(content),
                content_type=ContentType.ADVERSE_EVENT,
                domain=Domain.PHARMACOVIGILANCE,
                source=self.SOURCE,
                source_id=report_id,
                title=f"FAERS Report {report_id}",
                timestamp=ts,
                metadata={
                    "drugs": drugs,
                    "reactions": reactions,
                    "serious": serious,
                    "raw": result,
                },
            ))
        return docs

    async def ingest(
        self, total: int = 10_000, batch_size: int = 100
    ) -> AsyncIterator[list[Document]]:
        progress = progress_tracker.get(self.SOURCE)
        skip = int(progress.get("total_ingested", 0))
        while skip < total:
            docs = await self.fetch_adverse_events(skip=skip, limit=batch_size)
            if not docs:
                break
            yield docs
            progress_tracker.update(
                self.SOURCE,
                last_id=docs[-1].source_id,
                increment=len(docs),
            )
            skip += len(docs)
            await asyncio.sleep(0.3)


# ─────────────────────────────────────────────────────────────────────────────
# ClinicalTrials.gov v2
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalTrialsWorker:
    SOURCE = "clinical_trials"
    BASE = "https://clinicaltrials.gov/api/v2"

    def __init__(self):
        settings = get_settings()
        self._limiter = RateLimiter(settings.apis.clinical_trials.rate_limit_per_second)
        self._client = httpx.AsyncClient(timeout=30)

    @_RETRY
    async def _get(self, params: dict) -> dict:
        await self._limiter.acquire()
        r = await self._client.get(f"{self.BASE}/studies", params=params)
        r.raise_for_status()
        return r.json()

    def _parse_study(self, study: dict) -> Document:
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status = proto.get("statusModule", {})
        desc = proto.get("descriptionModule", {})
        design = proto.get("designModule", {})

        nct_id = ident.get("nctId", "")
        title = ident.get("briefTitle", "")
        summary = desc.get("briefSummary", "")
        phase = str(design.get("phases", ["N/A"]))
        status_str = status.get("overallStatus", "")
        start_date = status.get("startDateStruct", {}).get("date", "2000-01-01")
        try:
            ts = datetime.fromisoformat(start_date[:10])
        except ValueError:
            ts = datetime.utcnow()

        content = formatter.build_content(
            title,
            summary,
            {"phase": phase, "status": status_str, "nct_id": nct_id},
        )

        return Document(
            content=formatter.clean(content),
            content_type=ContentType.CLINICAL_TRIAL,
            domain=Domain.RND,
            source=self.SOURCE,
            source_id=nct_id,
            title=title,
            url=f"https://clinicaltrials.gov/study/{nct_id}",
            timestamp=ts,
            metadata={"nct_id": nct_id, "phase": phase, "status": status_str},
        )

    async def ingest(
        self,
        query: str = "pharmaceutical",
        max_results: int = 5000,
        page_size: int = 50,
    ) -> AsyncIterator[list[Document]]:
        progress = progress_tracker.get(self.SOURCE)
        page_token = progress.get("last_id") or None
        total_fetched = int(progress.get("total_ingested", 0))

        while total_fetched < max_results:
            params: dict = {
                "query.term": query,
                "pageSize": page_size,
                "format": "json",
            }
            if page_token:
                params["pageToken"] = page_token
            data = await self._get(params)
            studies = data.get("studies", [])
            if not studies:
                break
            docs = [self._parse_study(s) for s in studies]
            yield docs
            next_token = data.get("nextPageToken")
            progress_tracker.update(
                self.SOURCE,
                last_id=next_token or "",
                increment=len(docs),
            )
            total_fetched += len(docs)
            if not next_token:
                break
            page_token = next_token
            await asyncio.sleep(0.2)