"""
pharmaai/ingestion/progress_tracker.py
Persist and retrieve per-source ingestion progress so workers can resume
after interruption.  Uses both an in-memory dict and a JSON file on disk
(PostgreSQL upsert is also available via DocumentStore).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock

from pharmaai.core.config import get_settings

logger = logging.getLogger("pharmaai.ingestion.progress")


class ProgressTracker:
    """Thread-safe progress tracker backed by a JSON file."""

    def __init__(self, path: str | None = None):
        settings = get_settings()
        self._path = Path(path or settings.ingestion.progress_file)
        self._lock = Lock()
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    self._data = json.load(f)
                logger.info("Progress loaded from %s", self._path)
            except Exception as exc:
                logger.warning("Could not load progress file: %s", exc)
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    def get(self, source: str) -> dict:
        with self._lock:
            return self._data.get(source, {
                "last_id": "",
                "last_timestamp": None,
                "total_ingested": 0,
            })

    def update(
        self,
        source: str,
        last_id: str = "",
        last_timestamp: datetime | str | None = None,
        increment: int = 0,
    ) -> None:
        with self._lock:
            current = self._data.get(source, {
                "last_id": "", "last_timestamp": None, "total_ingested": 0
            })
            current["last_id"] = last_id or current["last_id"]
            if last_timestamp:
                current["last_timestamp"] = (
                    last_timestamp.isoformat()
                    if isinstance(last_timestamp, datetime)
                    else last_timestamp
                )
            current["total_ingested"] = current.get("total_ingested", 0) + increment
            current["last_updated"] = datetime.utcnow().isoformat()
            self._data[source] = current
            self._save()
        logger.debug(
            "Progress updated: source=%s last_id=%s total=%d",
            source, last_id, current["total_ingested"],
        )

    def reset(self, source: str) -> None:
        with self._lock:
            self._data.pop(source, None)
            self._save()

    def all_sources(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())


progress_tracker = ProgressTracker()