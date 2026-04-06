"""
scripts/init_db.py
Initialise PostgreSQL tables and build a fresh FAISS index.
Run once before starting the application.
"""
import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pharmaai.init")


async def main():
    logger.info("─── PharmaAI Initialisation ───")

    # Create directories
    for p in ["data/logs", "data/raw", "data/processed", "models"]:
        Path(p).mkdir(parents=True, exist_ok=True)

    # Init database
    logger.info("Creating database tables…")
    from pharmaai.core.database import create_tables
    await create_tables()
    logger.info("✓ Database tables ready.")

    # Build empty FAISS index
    logger.info("Building FAISS index…")
    from pharmaai.embeddings.index import faiss_index
    faiss_index.build()
    faiss_index.save()
    logger.info("✓ FAISS index initialised at %s", faiss_index._index_path)

    logger.info("─── Initialisation complete ───")


if __name__ == "__main__":
    asyncio.run(main())