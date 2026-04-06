"""
pharmaai/processing/domain_classifier.py
Two-stage domain & content-type classifier:
  1. Rule-based keyword matching (fast, deterministic)
  2. Optional ML fallback using a small BERT-based classifier (if trained)
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import NamedTuple

from pharmaai.core.schemas import Domain, ContentType

logger = logging.getLogger("pharmaai.processing.classifier")


# ─────────────────────────────────────────────────────────────────────────────
# Rule tables
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_RULES: dict[Domain, list[str]] = {
    Domain.PHARMACOVIGILANCE: [
        "adverse event", "side effect", "adverse reaction", "pharmacovigilance",
        "faers", "adr", "drug reaction", "toxicity report", "signal detection",
        "safety signal", "post-market surveillance",
    ],
    Domain.INTERNAL: [
        "inventory", "sop", "standard operating procedure", "workflow",
        "approval", "hr policy", "employee", "procurement", "internal memo",
        "requisition", "onboarding",
    ],
    Domain.RND: [
        "clinical trial", "phase i", "phase ii", "phase iii", "phase iv",
        "study design", "randomised", "randomized", "placebo", "endpoint",
        "biomarker", "efficacy", "clinical study", "investigational",
        "nct number", "cohort", "double-blind",
    ],
    Domain.REGULATION: [
        "iso standard", "fda guidance", "ema guideline", "gmp", "gcp", "glp",
        "regulatory requirement", "cfr", "21 cfr", "directive", "ich guideline",
        "compliance", "audit", "inspection", "dossier", "marketing authorisation",
    ],
    Domain.FORMULAS: [
        "compound", "dosage", "formula", "chemical structure", "smiles",
        "molecular weight", "pharmacokinetics", "pharmacodynamics",
        "drug interaction", "solubility", "bioavailability", "excipient",
        "active pharmaceutical ingredient", "api ",
    ],
}

CONTENT_TYPE_RULES: dict[ContentType, list[str]] = {
    ContentType.ADVERSE_EVENT: [
        "adverse event", "side effect", "adr", "faers", "adverse reaction",
        "safety report", "drug reaction",
    ],
    ContentType.CLINICAL_TRIAL: [
        "clinical trial", "nct", "phase i", "phase ii", "phase iii",
        "randomised controlled", "study protocol",
    ],
    ContentType.REGULATORY_DOC: [
        "fda guidance", "ema guideline", "iso ", "gmp", "regulatory dossier",
        "marketing authorisation", "ich ", "21 cfr",
    ],
    ContentType.INTERNAL_PROCESS: [
        "sop", "standard operating procedure", "inventory", "hr policy",
        "internal memo", "workflow approval",
    ],
    ContentType.RND_ARTICLE: [
        "research article", "pubmed", "abstract", "doi:", "keywords:",
        "methods:", "results:", "conclusion:",
    ],
    ContentType.FORMULA: [
        "smiles", "inchi", "molecular weight", "chemical formula",
        "compound id", "pubchem", "chembl",
    ],
    ContentType.NEWS_ARTICLE: [
        "reported today", "breaking news", "press release", "announced that",
        "according to reuters", "according to bloomberg", "newsapi",
    ],
    ContentType.DRUG_LABEL: [
        "indications", "contraindications", "dosage and administration",
        "warnings and precautions", "drug label", "prescribing information",
    ],
}


class ClassificationResult(NamedTuple):
    domain: Domain
    content_type: ContentType
    confidence: float        # 0.0–1.0 (1.0 = definite rule match)
    matched_keywords: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

class DomainClassifier:
    """
    Classify text into Domain + ContentType.
    Uses weighted keyword scoring per category, returns best match.
    Falls back to Domain.UNKNOWN / ContentType.UNKNOWN if score too low.
    """

    def __init__(self, min_score: float = 0.1):
        self.min_score = min_score

    def _score(self, text_lower: str, rules: dict) -> dict:
        scores: dict = {}
        matched: dict = {}
        for key, keywords in rules.items():
            hits = [kw for kw in keywords if kw in text_lower]
            if hits:
                # score = fraction of keywords matched, bonus for multiple hits
                scores[key] = len(hits) / len(keywords) + 0.05 * (len(hits) - 1)
                matched[key] = hits
        return {"scores": scores, "matched": matched}

    def classify(self, text: str) -> ClassificationResult:
        lower = text.lower()

        domain_result = self._score(lower, DOMAIN_RULES)
        content_result = self._score(lower, CONTENT_TYPE_RULES)

        # Pick best domain
        if domain_result["scores"]:
            best_domain = max(domain_result["scores"], key=domain_result["scores"].get)
            domain_score = domain_result["scores"][best_domain]
            matched = domain_result["matched"].get(best_domain, [])
        else:
            best_domain = Domain.UNKNOWN
            domain_score = 0.0
            matched = []

        # Pick best content type
        if content_result["scores"]:
            best_ct = max(content_result["scores"], key=content_result["scores"].get)
            ct_score = content_result["scores"][best_ct]
        else:
            best_ct = ContentType.UNKNOWN
            ct_score = 0.0

        confidence = max(domain_score, ct_score)

        if confidence < self.min_score:
            best_domain = Domain.UNKNOWN
            best_ct = ContentType.UNKNOWN

        return ClassificationResult(
            domain=best_domain,
            content_type=best_ct,
            confidence=min(1.0, confidence),
            matched_keywords=matched,
        )

    def classify_batch(self, texts: list[str]) -> list[ClassificationResult]:
        return [self.classify(t) for t in texts]


# Singleton
classifier = DomainClassifier()