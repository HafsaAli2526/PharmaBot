"""
pharmaai/core/database.py
Async PostgreSQL connection pool (SQLAlchemy 2.x + asyncpg).
Also provides a simple document store helper.
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Any

from sqlalchemy import (
    Column, DateTime, Float, Index, Integer,
    String, Text, text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from pharmaai.core.config import get_settings
from pharmaai.core.schemas import Document, ContentType, Domain

logger = logging.getLogger("pharmaai.database")


# ─────────────────────────────────────────────────────────────────────────────
# ORM Base
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=False), primary_key=True)
    content = Column(Text, nullable=False)
    content_type = Column(String(64), nullable=False, index=True)
    domain = Column(String(64), nullable=False, index=True)
    source = Column(String(128), nullable=False, index=True)
    source_id = Column(String(256), index=True)
    title = Column(Text, default="")
    url = Column(Text, default="")
    timestamp = Column(DateTime(timezone=True), index=True)
    meta = Column(JSONB, default={})
    faiss_idx = Column(Integer, index=True)  # position in FAISS index

    __table_args__ = (
        Index("ix_domain_content_type", "domain", "content_type"),
        Index("ix_source_source_id", "source", "source_id", unique=True),
    )


class IngestionProgressRecord(Base):
    __tablename__ = "ingestion_progress"

    source = Column(String(128), primary_key=True)
    last_id = Column(String(256), default="")
    last_timestamp = Column(DateTime(timezone=True), nullable=True)
    total_ingested = Column(Integer, default=0)
    last_updated = Column(DateTime(timezone=True))


# ─────────────────────────────────────────────────────────────────────────────
# Engine + session factory
# ─────────────────────────────────────────────────────────────────────────────

_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.postgres.dsn,
            pool_size=settings.postgres.pool_size,
            max_overflow=settings.postgres.max_overflow,
            echo=False,
        )
    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ─────────────────────────────────────────────────────────────────────────────
# Schema helpers
# ─────────────────────────────────────────────────────────────────────────────

async def create_tables() -> None:
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created / verified.")


# ─────────────────────────────────────────────────────────────────────────────
# Document store
# ─────────────────────────────────────────────────────────────────────────────

class DocumentStore:
    """Thin async wrapper over PostgreSQL for document CRUD."""

    async def upsert(self, doc: Document, faiss_idx: int | None = None) -> None:
        async with get_session() as session:
            existing = await session.get(DocumentRecord, doc.id)
            if existing:
                existing.content = doc.content
                existing.meta = doc.metadata
                if faiss_idx is not None:
                    existing.faiss_idx = faiss_idx
            else:
                record = DocumentRecord(
                    id=doc.id,
                    content=doc.content,
                    content_type=doc.content_type.value,
                    domain=doc.domain.value,
                    source=doc.source,
                    source_id=doc.source_id,
                    title=doc.title,
                    url=doc.url,
                    timestamp=doc.timestamp,
                    meta=doc.metadata,
                    faiss_idx=faiss_idx,
                )
                session.add(record)

    async def get_by_faiss_indices(
        self, indices: list[int]
    ) -> dict[int, Document]:
        async with get_session() as session:
            result = await session.execute(
                text("SELECT * FROM documents WHERE faiss_idx = ANY(:idxs)"),
                {"idxs": indices},
            )
            rows = result.mappings().all()
        out: dict[int, Document] = {}
        for row in rows:
            doc = Document(
                id=row["id"],
                content=row["content"],
                content_type=ContentType(row["content_type"]),
                domain=Domain(row["domain"]),
                source=row["source"],
                source_id=row["source_id"] or "",
                title=row["title"] or "",
                url=row["url"] or "",
                timestamp=row["timestamp"],
                metadata=row["meta"] or {},
            )
            out[row["faiss_idx"]] = doc
        return out

    async def get_by_source_id(
        self, source: str, source_id: str
    ) -> Document | None:
        async with get_session() as session:
            result = await session.execute(
                text(
                    "SELECT * FROM documents WHERE source=:s AND source_id=:sid"
                ),
                {"s": source, "sid": source_id},
            )
            row = result.mappings().first()
        if not row:
            return None
        return Document(
            id=row["id"],
            content=row["content"],
            content_type=ContentType(row["content_type"]),
            domain=Domain(row["domain"]),
            source=row["source"],
            source_id=row["source_id"] or "",
            title=row["title"] or "",
            url=row["url"] or "",
            timestamp=row["timestamp"],
            metadata=row["meta"] or {},
        )

    async def save_progress(self, source: str, last_id: str,
                             last_ts: datetime | None, total: int) -> None:
        async with get_session() as session:
            existing = await session.get(IngestionProgressRecord, source)
            if existing:
                existing.last_id = last_id
                existing.last_timestamp = last_ts
                existing.total_ingested = total
                existing.last_updated = datetime.utcnow()
            else:
                session.add(IngestionProgressRecord(
                    source=source,
                    last_id=last_id,
                    last_timestamp=last_ts,
                    total_ingested=total,
                    last_updated=datetime.utcnow(),
                ))

    async def get_progress(self, source: str) -> dict[str, Any]:
        async with get_session() as session:
            row = await session.get(IngestionProgressRecord, source)
        if not row:
            return {"last_id": "", "last_timestamp": None, "total_ingested": 0}
        return {
            "last_id": row.last_id,
            "last_timestamp": row.last_timestamp,
            "total_ingested": row.total_ingested,
        }


document_store = DocumentStore()