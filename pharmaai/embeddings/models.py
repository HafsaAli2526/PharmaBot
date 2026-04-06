"""
pharmaai/embeddings/models.py
Load and manage local transformer models on the H200 GPU.
Provides a unified EmbeddingService with per-domain model routing.
"""
from __future__ import annotations

import logging
import threading
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from pharmaai.core.config import get_settings
from pharmaai.core.schemas import Domain, ContentType

logger = logging.getLogger("pharmaai.embeddings.models")

ModelName = Literal["biobert", "clinicalbert", "chemberta", "slm", "slm_finetuned"]

# Mapping: domain → preferred encoder
DOMAIN_MODEL_MAP: dict[Domain, list[ModelName]] = {
    Domain.PHARMACOVIGILANCE: ["biobert", "clinicalbert"],
    Domain.INTERNAL: ["biobert"],
    Domain.RND: ["clinicalbert", "biobert"],
    Domain.REGULATION: ["biobert"],
    Domain.FORMULAS: ["chemberta", "biobert"],
    Domain.UNKNOWN: ["biobert", "clinicalbert", "chemberta"],
}

CONTENT_TYPE_MODEL_MAP: dict[ContentType, list[ModelName]] = {
    ContentType.ADVERSE_EVENT: ["biobert", "clinicalbert"],
    ContentType.CLINICAL_TRIAL: ["clinicalbert"],
    ContentType.FORMULA: ["chemberta"],
    ContentType.DRUG_LABEL: ["biobert", "chemberta"],
    ContentType.RND_ARTICLE: ["clinicalbert", "biobert"],
    ContentType.REGULATORY_DOC: ["biobert"],
}


class ModelRegistry:
    """Thread-safe registry for loaded transformer models."""

    _instance: ModelRegistry | None = None
    _lock = threading.Lock()

    def __new__(cls) -> ModelRegistry:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._models: dict[str, AutoModel] = {}
                cls._instance._tokenizers: dict[str, AutoTokenizer] = {}
                cls._instance._loaded = False
        return cls._instance

    def load_all(self, device: str | None = None) -> None:
        if self._loaded:
            return
        settings = get_settings()
        self._device = device or settings.gpu.device
        if not torch.cuda.is_available():
            logger.warning("CUDA unavailable – falling back to CPU.")
            self._device = "cpu"

        model_cfgs = {
            "biobert": settings.models.biobert,
            "clinicalbert": settings.models.clinicalbert,
            "chemberta": settings.models.chemberta,
        }
        for name, cfg in model_cfgs.items():
            logger.info("Loading %s from %s …", name, cfg.path)
            tok = AutoTokenizer.from_pretrained(cfg.path)
            mdl = AutoModel.from_pretrained(cfg.path)
            if settings.gpu.mixed_precision and self._device != "cpu":
                mdl = mdl.to(torch.bfloat16)
            mdl = mdl.to(self._device).eval()
            self._tokenizers[name] = tok
            self._models[name] = mdl
            logger.info("  ✓ %s loaded on %s", name, self._device)

        self._loaded = True

    def get_model(self, name: ModelName) -> tuple[AutoTokenizer, AutoModel]:
        if not self._loaded:
            self.load_all()
        return self._tokenizers[name], self._models[name]

    @property
    def device(self) -> str:
        return getattr(self, "_device", "cpu")


registry = ModelRegistry()


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling over token embeddings, respecting padding."""
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


@torch.inference_mode()
def embed_texts(
    texts: list[str],
    model_name: ModelName,
    max_length: int = 512,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Tokenise and encode a list of texts using the specified model.
    Returns an (N, dim) float32 numpy array of L2-normalised embeddings.
    """
    tok, mdl = registry.get_model(model_name)
    all_embeddings: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(registry.device)

        outputs = mdl(**encoded)
        embeddings = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        all_embeddings.append(embeddings.cpu().float().numpy())

    return np.vstack(all_embeddings)


def combine_embeddings(*arrays: np.ndarray) -> np.ndarray:
    """
    Concatenate multiple embedding arrays along the feature dimension.
    All arrays must have the same number of rows.
    Result is L2-normalised after concatenation.
    """
    combined = np.concatenate(arrays, axis=-1)
    # L2 normalise each row
    norms = np.linalg.norm(combined, axis=1, keepdims=True).clip(min=1e-9)
    return combined / norms


class EmbeddingService:
    """
    High-level service: select models based on domain/content_type,
    generate and optionally combine embeddings.
    """

    def __init__(self):
        self._settings = get_settings()

    def get_model_names(
        self,
        domain: Domain | None = None,
        content_type: ContentType | None = None,
        combine: bool = True,
    ) -> list[ModelName]:
        if content_type and content_type in CONTENT_TYPE_MODEL_MAP:
            return CONTENT_TYPE_MODEL_MAP[content_type]
        if domain and domain in DOMAIN_MODEL_MAP:
            models = DOMAIN_MODEL_MAP[domain]
            if not combine:
                return [models[0]]
            return models
        return ["biobert", "clinicalbert", "chemberta"]

    def embed(
        self,
        texts: list[str],
        domain: Domain | None = None,
        content_type: ContentType | None = None,
        combine: bool = True,
        max_length: int = 512,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Produce embeddings for a list of texts.
        If combine=True and multiple models are selected, concatenate their
        outputs to produce a richer representation.
        """
        model_names = self.get_model_names(domain, content_type, combine)

        if not combine or len(model_names) == 1:
            return embed_texts(texts, model_names[0], max_length, batch_size)

        all_embs = [
            embed_texts(texts, name, max_length, batch_size)
            for name in model_names
        ]
        # Pad to 3-model concatenation (2304-dim) if needed for FAISS compatibility
        while len(all_embs) < 3:
            all_embs.append(all_embs[-1])  # duplicate last embedding
        return combine_embeddings(*all_embs[:3])


embedding_service = EmbeddingService()