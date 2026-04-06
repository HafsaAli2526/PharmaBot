"""
pharmaai/embeddings/index.py
FAISS vector index management:
  – Build (IVF-Flat with cosine via inner-product on L2-normalised vecs)
  – Persist to disk / load from disk
  – Incremental add
  – Search with metadata filter support
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from pharmaai.core.config import get_settings

logger = logging.getLogger("pharmaai.embeddings.index")


class FaissIndex:
    """
    Thread-safe FAISS index wrapper.

    We use an IVFFlat index with inner-product (IP) distance.
    Vectors MUST be L2-normalised before insertion (cosine similarity ≡ IP).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._index: faiss.Index | None = None
        self._id_map: dict[int, str] = {}   # faiss_idx → document uuid
        self._next_idx: int = 0
        settings = get_settings()
        self._dim: int = settings.index.dim
        self._nlist: int = settings.index.nlist
        self._nprobe: int = settings.index.nprobe
        self._index_path = Path(settings.index.path)
        self._id_map_path = Path(settings.index.id_map_path)

    # ── Build / init ──────────────────────────────────────────────────────────

    def build(self, vectors: np.ndarray | None = None) -> None:
        """
        (Re)build a fresh IVFFlat index.
        If vectors are provided, train and populate immediately.
        """
        with self._lock:
            quantizer = faiss.IndexFlatIP(self._dim)
            self._index = faiss.IndexIVFFlat(
                quantizer, self._dim, self._nlist, faiss.METRIC_INNER_PRODUCT
            )
            self._index.nprobe = self._nprobe
            self._id_map = {}
            self._next_idx = 0

            if vectors is not None and len(vectors) > 0:
                self._train_and_add(vectors, list(range(len(vectors))))
            logger.info("FAISS index built (dim=%d, nlist=%d).", self._dim, self._nlist)

    def _train_and_add(
        self, vectors: np.ndarray, doc_ids: list[str]
    ) -> None:
        """Internal helper – assumes lock is held."""
        if not self._index.is_trained:
            logger.info("Training FAISS index on %d vectors…", len(vectors))
            self._index.train(vectors.astype(np.float32))
        start = self._next_idx
        self._index.add(vectors.astype(np.float32))
        for i, doc_id in enumerate(doc_ids):
            self._id_map[start + i] = doc_id
        self._next_idx += len(vectors)

    # ── Add ───────────────────────────────────────────────────────────────────

    def add(self, vectors: np.ndarray, doc_ids: list[str]) -> list[int]:
        """
        Add vectors to the index.  Returns the assigned FAISS indices.
        """
        if self._index is None:
            self.build()
        if not self._index.is_trained:
            logger.info("Index not yet trained – training now on %d vecs.", len(vectors))
            self._index.train(vectors.astype(np.float32))

        with self._lock:
            start = self._next_idx
            self._index.add(vectors.astype(np.float32))
            assigned = list(range(start, start + len(vectors)))
            for i, doc_id in enumerate(doc_ids):
                self._id_map[start + i] = doc_id
            self._next_idx += len(vectors)
        return assigned

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self, query_vector: np.ndarray, top_k: int = 10
    ) -> list[tuple[int, float]]:
        """
        Search the index.
        Returns list of (faiss_idx, score) sorted by descending score.
        Skips indices not in id_map (deleted / out-of-range).
        """
        if self._index is None or self._next_idx == 0:
            return []
        with self._lock:
            q = query_vector.reshape(1, -1).astype(np.float32)
            scores, indices = self._index.search(q, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx in self._id_map:
                results.append((int(idx), float(score)))
        return results

    def faiss_idx_to_doc_id(self, faiss_idx: int) -> str | None:
        return self._id_map.get(faiss_idx)

    # ── Persist / load ────────────────────────────────────────────────────────

    def save(self) -> None:
        if self._index is None:
            return
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            faiss.write_index(self._index, str(self._index_path))
            with open(self._id_map_path, "w") as f:
                # keys must be strings in JSON
                json.dump(
                    {str(k): v for k, v in self._id_map.items()}, f
                )
        logger.info(
            "FAISS index saved (%d vectors) → %s",
            self._next_idx, self._index_path,
        )

    def load(self) -> bool:
        if not self._index_path.exists():
            logger.warning("No saved FAISS index found at %s.", self._index_path)
            return False
        with self._lock:
            self._index = faiss.read_index(str(self._index_path))
            self._index.nprobe = self._nprobe
            with open(self._id_map_path) as f:
                raw = json.load(f)
            self._id_map = {int(k): v for k, v in raw.items()}
            self._next_idx = max(self._id_map.keys(), default=-1) + 1
        logger.info(
            "FAISS index loaded (%d vectors) from %s",
            self._next_idx, self._index_path,
        )
        return True

    def load_or_build(self) -> None:
        if not self.load():
            self.build()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return self._next_idx

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.is_trained


faiss_index = FaissIndex()