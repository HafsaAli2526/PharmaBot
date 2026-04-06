"""
scripts/run_ingestion.py
Orchestrates all ingestion workers.
Can be run as a one-shot batch job or continuous loop.
"""
import asyncio
import logging
import signal
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pharmaai.ingestion_runner")

RUNNING = True


def _handle_shutdown(signum, frame):
    global RUNNING
    logger.info("Shutdown signal received.")
    RUNNING = False


signal.signal(signal.SIGINT, _handle_shutdown)
signal.signal(signal.SIGTERM, _handle_shutdown)


async def run_worker(worker, name: str, processor) -> int:
    total = 0
    try:
        async for batch in worker.ingest():
            if not RUNNING:
                break
            n = await processor.process_batch(batch)
            total += n
            logger.info("[%s] Processed %d docs (total=%d)", name, n, total)
    except Exception as exc:
        logger.error("[%s] Worker error: %s", name, exc)
    return total


async def main():
    from pharmaai.ingestion.api_workers import PubMedWorker, OpenFDAWorker, ClinicalTrialsWorker
    from pharmaai.ingestion.news_worker import NewsAPIWorker, BioRxivWorker
    from pharmaai.ingestion.queue_publisher import DirectProcessor

    processor = DirectProcessor()

    workers = [
        (OpenFDAWorker(), "OpenFDA"),
        (PubMedWorker(), "PubMed"),
        (ClinicalTrialsWorker(), "ClinicalTrials"),
        (BioRxivWorker(), "BioRxiv"),
    ]

    # Run NewsAPI only if key is configured
    from pharmaai.core.config import get_settings
    if get_settings().apis.newsapi.api_key:
        workers.append((NewsAPIWorker(), "NewsAPI"))

    tasks = [run_worker(w, name, processor) for w, name in workers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    total = sum(r for r in results if isinstance(r, int))
    logger.info("Ingestion complete. Total documents processed: %d", total)


if __name__ == "__main__":
    asyncio.run(main())