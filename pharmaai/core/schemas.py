"""
pharmaai/core/schemas.py
Canonical data models shared across ingestion, processing, retrieval, and API.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class Domain(str, Enum):
    PHARMACOVIGILANCE = "pharmacovigilance"
    INTERNAL = "internal"
    RND = "r&d"
    REGULATION = "regulation"
    FORMULAS = "formulas"
    UNKNOWN = "unknown"


class ContentType(str, Enum):
    ADVERSE_EVENT = "adverse_event"
    CLINICAL_TRIAL = "clinical_trial"
    REGULATORY_DOC = "regulatory_doc"
    INTERNAL_PROCESS = "internal_process"
    RND_ARTICLE = "r&d_article"
    FORMULA = "formula"
    NEWS_ARTICLE = "news_article"
    WEB_PAGE = "web_page"
    DRUG_LABEL = "drug_label"
    RESEARCH_ARTICLE = "research_article"
    UNKNOWN = "unknown"


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationChannel(str, Enum):
    TWILIO = "twilio"
    SLACK = "slack"
    EMAIL = "email"
    FCM = "fcm"
    SNS = "sns"


# ─────────────────────────────────────────────────────────────────────────────
# Core document model
# ─────────────────────────────────────────────────────────────────────────────

class Document(BaseModel):
    """Canonical normalised record inserted into the vector index."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    content_type: ContentType = ContentType.UNKNOWN
    domain: Domain = Domain.UNKNOWN
    source: str                         # e.g. "openfda", "pubmed", "internal"
    source_id: str = ""                 # original ID in the source system
    title: str = ""
    url: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None   # populated after embedding step

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class SearchResult(BaseModel):
    """A retrieved document with its similarity score."""
    document: Document
    score: float
    rank: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# API request / response models
# ─────────────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    domain: Domain | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    use_cache: bool = True


class AskResponse(BaseModel):
    answer: str
    sources: list[SearchResult]
    domain: Domain
    latency_ms: float
    cached: bool = False


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    domain: Domain | None = None
    content_type: ContentType | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    date_from: datetime | None = None
    date_to: datetime | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int
    latency_ms: float


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class ClassifyResponse(BaseModel):
    domain: Domain
    content_type: ContentType
    confidence: float


# ─────────────────────────────────────────────────────────────────────────────
# Notification models
# ─────────────────────────────────────────────────────────────────────────────

class NotificationPayload(BaseModel):
    title: str
    body: str
    severity: AlertSeverity = AlertSeverity.MEDIUM
    channels: list[NotificationChannel] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    recipient: str = ""   # phone / email / slack channel / FCM token


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion progress
# ─────────────────────────────────────────────────────────────────────────────

class IngestionProgress(BaseModel):
    source: str
    last_id: str = ""
    last_timestamp: datetime | None = None
    total_ingested: int = 0
    last_updated: datetime = Field(default_factory=datetime.utcnow)