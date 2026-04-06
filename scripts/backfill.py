"""
scripts/backfill.py
Re-generate embeddings for all documents already in the database
and rebuild the FAISS index from scratch.
Useful after:
  – Changing the embedding model
  – Upgrading FAISS index configuration
  – Recovering from index corruption

Usage:
  python scripts/backfill.py [--batch-size 64] [--dry-run]
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import numpy as np
import typer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pharmaai.backfill")
app = typer.Typer()


async def _fetch_all_documents(session, batch_size: int = 1000):
    """Generator: yield batches of DocumentRecord rows."""
    from sqlalchemy import text
    offset = 0
    while True:
        result = await session.execute(
            text("SELECT * FROM documents ORDER BY id LIMIT :lim OFFSET :off"),
            {"lim": batch_size, "off": offset},
        )
        rows = result.mappings().all()
        if not rows:
            break
        yield rows
        offset += len(rows)
        if len(rows) < batch_size:
            break


async def run_backfill(batch_size: int = 64, dry_run: bool = False):
    from pharmaai.core.database import get_session
    from pharmaai.core.schemas import Document, ContentType, Domain
    from pharmaai.embeddings.models import embedding_service, registry
    from pharmaai.embeddings.index import faiss_index

    # Load models
    logger.info("Loading embedding models…")
    registry.load_all()

    # Rebuild fresh index
    logger.info("Building fresh FAISS index…")
    faiss_index.build()

    total_processed = 0
    total_vectors: list[np.ndarray] = []
    total_ids: list[str] = []

    async with get_session() as session:
        async for rows in _fetch_all_documents(session, batch_size=batch_size * 4):
            # Build Document objects
            docs = []
            for row in rows:
                try:
                    docs.append(Document(
                        id=row["id"],
                        content=row["content"],
                        content_type=ContentType(row["content_type"]),
                        domain=Domain(row["domain"]),
                        source=row["source"],
                        source_id=row["source_id"] or "",
                        metadata=row["meta"] or {},
                    ))
                except Exception as exc:
                    logger.warning("Skipping malformed row %s: %s", row["id"], exc)

            if not docs:
                continue

            # Embed in sub-batches
            for i in range(0, len(docs), batch_size):
                sub = docs[i : i + batch_size]
                texts = [d.content for d in sub]
                try:
                    embs = embedding_service.embed(texts, combine=True)
                    total_vectors.append(embs)
                    total_ids.extend([d.id for d in sub])
                    total_processed += len(sub)
                except Exception as exc:
                    logger.error("Embedding failed for batch: %s", exc)

            logger.info("Embedded %d documents so far…", total_processed)

    if not total_vectors:
        logger.warning("No documents found in database. Nothing to backfill.")
        return

    all_vecs = np.vstack(total_vectors)
    logger.info("Total embeddings computed: %d (dim=%d)", len(all_vecs), all_vecs.shape[1])

    if dry_run:
        logger.info("Dry run – skipping index write.")
        return

    # Train and populate index
    faiss_index._index.train(all_vecs)
    faiss_indices = faiss_index.add(all_vecs, total_ids)
    logger.info("FAISS index populated with %d vectors.", faiss_index.size)

    # Persist
    faiss_index.save()
    logger.info("Index saved.")

    # Update faiss_idx in DB
    logger.info("Updating faiss_idx in database…")
    id_to_faiss = dict(zip(total_ids, faiss_indices))
    async with get_session() as session:
        from sqlalchemy import text
        for doc_id, fidx in id_to_faiss.items():
            await session.execute(
                text("UPDATE documents SET faiss_idx=:fidx WHERE id=:doc_id"),
                {"fidx": fidx, "doc_id": doc_id},
            )
    logger.info("✓ Backfill complete. %d documents re-indexed.", total_processed)


@app.command()
def backfill(
    batch_size: int = typer.Option(64, help="Embedding batch size"),
    dry_run: bool = typer.Option(False, help="Compute embeddings but don't write"),
):
    t0 = time.perf_counter()
    asyncio.run(run_backfill(batch_size=batch_size, dry_run=dry_run))
    elapsed = time.perf_counter() - t0
    logger.info("Backfill completed in %.1f seconds.", elapsed)


if __name__ == "__main__":
    app()