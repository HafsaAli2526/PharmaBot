"""
pharmaai/inference/query_generator.py
Generate diverse sub-queries from a user question to improve recall.
Uses template-based expansion + optional SLM generation.
"""
from __future__ import annotations

import logging
import re
from pharmaai.core.schemas import Domain

logger = logging.getLogger("pharmaai.inference.query_generator")

EXPANSION_TEMPLATES: dict[Domain, list[str]] = {
    Domain.PHARMACOVIGILANCE: [
        "{query} adverse reactions",
        "{query} side effects FAERS",
        "{query} safety signal pharmacovigilance",
    ],
    Domain.RND: [
        "{query} clinical trial results",
        "{query} PubMed research",
        "{query} efficacy safety study",
    ],
    Domain.REGULATION: [
        "{query} FDA guideline",
        "{query} EMA regulatory requirement",
        "{query} compliance standard",
    ],
    Domain.FORMULAS: [
        "{query} chemical structure interaction",
        "{query} dosage pharmacokinetics",
        "{query} drug compound analysis",
    ],
    Domain.INTERNAL: [
        "{query} SOP procedure",
        "{query} inventory workflow",
    ],
}

SLM_EXPANSION_PROMPT = """Given the pharmaceutical question below, generate 3 alternative
search queries that would help retrieve relevant information. Output one query per line,
no numbering, no extra text.

Question: {question}

Queries:"""


class QueryGenerator:
    def __init__(self, use_slm: bool = False, max_queries: int = 3):
        self._use_slm = use_slm
        self._max = max_queries

    def generate(self, question: str, domain: Domain | None = None) -> list[str]:
        """Return a list of alternative search queries."""
        queries: list[str] = []

        # Template-based
        templates = EXPANSION_TEMPLATES.get(domain or Domain.UNKNOWN, [])
        if not templates:
            templates = [
                "{query} pharmaceutical",
                "{query} drug",
            ]

        for tpl in templates[: self._max]:
            q = tpl.format(query=question)
            if q != question:
                queries.append(q)

        # Optionally use SLM for richer expansion
        if self._use_slm and len(queries) < self._max:
            try:
                from pharmaai.inference.small_lm import slm
                prompt = SLM_EXPANSION_PROMPT.format(question=question)
                raw = slm.generate(prompt, max_new_tokens=120, temperature=0.5)
                for line in raw.strip().split("\n"):
                    line = line.strip().lstrip("-•123456789. ").strip()
                    if line and len(line) > 5:
                        queries.append(line)
                        if len(queries) >= self._max:
                            break
            except Exception as exc:
                logger.warning("SLM query expansion failed: %s", exc)

        return queries[: self._max]


query_generator = QueryGenerator(use_slm=False)