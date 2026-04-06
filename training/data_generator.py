"""
training/data_generator.py
Generates fine-tuning data (question, context, answer) triples from:
  – OpenFDA adverse event records
  – PubMed abstracts
  – ClinicalTrials summaries
  – SIDER drug-side-effect pairs
Outputs JSONL to data/processed/train.jsonl
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from pharmaai.core.schemas import Document

logger = logging.getLogger("pharmaai.training.data_gen")

TEMPLATES: dict[str, list[dict]] = {
    "adverse_event": [
        {
            "q_tpl": "What are the adverse reactions reported for {drug}?",
            "a_tpl": "Based on FAERS data, the following adverse reactions have been reported for {drug}: {reactions}.",
        },
        {
            "q_tpl": "Is {drug} associated with serious adverse events?",
            "a_tpl": "According to pharmacovigilance reports, {drug} has been associated with the following reactions: {reactions}. Serious: {serious}.",
        },
    ],
    "clinical_trial": [
        {
            "q_tpl": "What is the status of the clinical trial {nct_id}?",
            "a_tpl": "Clinical trial {nct_id} ({title}) is currently {status}. Phase: {phase}.",
        },
        {
            "q_tpl": "Summarise the clinical trial titled '{title}'.",
            "a_tpl": "Trial {nct_id}: {content}",
        },
    ],
    "r_and_d": [
        {
            "q_tpl": "Summarise the research article: '{title}'.",
            "a_tpl": "The article '{title}' reports: {content}",
        },
    ],
}


def doc_to_qa_pairs(doc: Document) -> list[dict]:
    pairs = []
    meta = doc.metadata
    ct = doc.content_type.value

    if ct == "adverse_event":
        for tpl in TEMPLATES.get("adverse_event", []):
            q = tpl["q_tpl"].format(
                drug=meta.get("drugs", "the drug"),
            )
            a = tpl["a_tpl"].format(
                drug=meta.get("drugs", "the drug"),
                reactions=meta.get("reactions", "reactions listed below"),
                serious=str(meta.get("serious", False)),
            )
            pairs.append({
                "question": q,
                "context": doc.content[:600],
                "answer": a,
                "domain": doc.domain.value,
                "source": doc.source,
            })

    elif ct == "clinical_trial":
        for tpl in TEMPLATES.get("clinical_trial", []):
            q = tpl["q_tpl"].format(
                nct_id=meta.get("nct_id", "N/A"),
                title=doc.title[:80],
            )
            a = tpl["a_tpl"].format(
                nct_id=meta.get("nct_id", "N/A"),
                title=doc.title[:80],
                status=meta.get("status", "unknown"),
                phase=meta.get("phase", "N/A"),
                content=doc.content[:400],
            )
            pairs.append({
                "question": q,
                "context": doc.content[:600],
                "answer": a,
                "domain": doc.domain.value,
                "source": doc.source,
            })

    elif ct in ("r&d_article", "research_article"):
        for tpl in TEMPLATES.get("r_and_d", []):
            q = tpl["q_tpl"].format(title=doc.title[:80])
            a = tpl["a_tpl"].format(
                title=doc.title[:80], content=doc.content[:400]
            )
            pairs.append({
                "question": q,
                "context": doc.content[:600],
                "answer": a,
                "domain": doc.domain.value,
                "source": doc.source,
            })

    return pairs


def generate_from_documents(
    documents: list[Document],
    output_path: Path,
    max_pairs: int = 50_000,
    shuffle: bool = True,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_pairs = []
    for doc in documents:
        pairs = doc_to_qa_pairs(doc)
        all_pairs.extend(pairs)
        if len(all_pairs) >= max_pairs:
            break

    if shuffle:
        random.shuffle(all_pairs)

    with open(output_path, "w") as f:
        for pair in all_pairs[:max_pairs]:
            f.write(json.dumps(pair) + "\n")

    logger.info("Generated %d QA pairs → %s", len(all_pairs), output_path)
    return len(all_pairs)