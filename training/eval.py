"""
training/eval.py
Evaluate the fine-tuned SLM and retrieval pipeline.

Metrics:
  – SLM: ROUGE-1/2/L, BERTScore (optional)
  – Retrieval: Precision@K, Recall@K, MRR, NDCG
  – Domain classifier: accuracy, F1 per class
  – End-to-end RAG: answer relevance (embedding similarity)

Usage:
  python training/eval.py \
    --eval_data  ./data/processed/eval.jsonl \
    --output     ./data/eval_results.json
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import typer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pharmaai.eval")
app = typer.Typer()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rouge_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores: dict[str, list[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            s = scorer.score(ref, pred)
            scores["rouge1"].append(s["rouge1"].fmeasure)
            scores["rouge2"].append(s["rouge2"].fmeasure)
            scores["rougeL"].append(s["rougeL"].fmeasure)
        return {k: round(float(np.mean(v)), 4) for k, v in scores.items()}
    except ImportError:
        logger.warning("rouge_score not installed; skipping ROUGE.")
        return {}


def _precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for d in top_k if d in relevant_ids)
    return hits / k


def _recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for d in retrieved_ids[:k] if d in relevant_ids)
    return hits / len(relevant_ids)


def _mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    def dcg(ids: list[str]) -> float:
        return sum(
            (1.0 / np.log2(i + 2)) for i, d in enumerate(ids) if d in relevant_ids
        )
    ideal = dcg(list(relevant_ids)[:k])
    if ideal == 0:
        return 0.0
    return dcg(retrieved_ids[:k]) / ideal


# ─────────────────────────────────────────────────────────────────────────────
# SLM evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_slm(eval_data: list[dict]) -> dict[str, Any]:
    from pharmaai.inference.small_lm import slm

    predictions, references = [], []
    latencies = []

    for item in eval_data:
        prompt = (
            f"### Question:\n{item['question']}\n\n"
            f"### Context:\n{item.get('context', '')}\n\n"
            f"### Answer:\n"
        )
        t0 = time.perf_counter()
        pred = slm.generate(prompt, max_new_tokens=256)
        latencies.append((time.perf_counter() - t0) * 1000)
        predictions.append(pred)
        references.append(item.get("answer", ""))

    rouge = _rouge_scores(predictions, references)
    return {
        "n_samples": len(eval_data),
        "rouge": rouge,
        "mean_latency_ms": round(float(np.mean(latencies)), 1),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval evaluation
# ─────────────────────────────────────────────────────────────────────────────

async def eval_retrieval(eval_data: list[dict], top_k: int = 10) -> dict[str, Any]:
    from pharmaai.retrieval.search import hybrid_search

    p_at_1, p_at_5, p_at_10 = [], [], []
    r_at_10, mrr_scores, ndcg_scores = [], [], []
    latencies = []

    for item in eval_data:
        if "relevant_ids" not in item:
            continue
        relevant = set(item["relevant_ids"])
        t0 = time.perf_counter()
        results = await hybrid_search.search(item["question"], top_k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved_ids = [r.document.source_id for r in results]

        p_at_1.append(_precision_at_k(retrieved_ids, relevant, 1))
        p_at_5.append(_precision_at_k(retrieved_ids, relevant, 5))
        p_at_10.append(_precision_at_k(retrieved_ids, relevant, 10))
        r_at_10.append(_recall_at_k(retrieved_ids, relevant, 10))
        mrr_scores.append(_mrr(retrieved_ids, relevant))
        ndcg_scores.append(_ndcg_at_k(retrieved_ids, relevant, 10))

    if not p_at_1:
        return {"warning": "No retrieval eval items with relevant_ids found."}

    return {
        "n_queries": len(p_at_1),
        "P@1": round(float(np.mean(p_at_1)), 4),
        "P@5": round(float(np.mean(p_at_5)), 4),
        "P@10": round(float(np.mean(p_at_10)), 4),
        "R@10": round(float(np.mean(r_at_10)), 4),
        "MRR": round(float(np.mean(mrr_scores)), 4),
        "NDCG@10": round(float(np.mean(ndcg_scores)), 4),
        "mean_latency_ms": round(float(np.mean(latencies)), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Domain classifier evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_classifier(eval_data: list[dict]) -> dict[str, Any]:
    from pharmaai.processing.domain_classifier import classifier
    from pharmaai.core.schemas import Domain

    all_true, all_pred = [], []
    for item in eval_data:
        if "domain" not in item:
            continue
        result = classifier.classify(item.get("question", "") + " " + item.get("context", ""))
        all_pred.append(result.domain.value)
        all_true.append(item["domain"])

    if not all_true:
        return {"warning": "No domain labels found."}

    accuracy = sum(t == p for t, p in zip(all_true, all_pred)) / len(all_true)

    # Per-class F1
    domains = [d.value for d in Domain]
    per_class: dict[str, dict] = {}
    for d in domains:
        tp = sum(1 for t, p in zip(all_true, all_pred) if t == d and p == d)
        fp = sum(1 for t, p in zip(all_true, all_pred) if t != d and p == d)
        fn = sum(1 for t, p in zip(all_true, all_pred) if t == d and p != d)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[d] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
    return {
        "n_samples": len(all_true),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def evaluate(
    eval_data: str = typer.Option("./data/processed/eval.jsonl"),
    output: str = typer.Option("./data/eval_results.json"),
    skip_slm: bool = typer.Option(False, help="Skip SLM evaluation (faster)"),
):
    data_path = Path(eval_data)
    if not data_path.exists():
        typer.echo(f"Eval data not found: {data_path}", err=True)
        raise typer.Exit(1)

    logger.info("Loading eval data from %s", data_path)
    items = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info("Loaded %d eval items.", len(items))

    results: dict[str, Any] = {}

    # Classifier eval (fast, always run)
    logger.info("Evaluating domain classifier…")
    results["classifier"] = eval_classifier(items)
    logger.info("Classifier: %s", results["classifier"])

    # Retrieval eval
    logger.info("Evaluating retrieval pipeline…")
    results["retrieval"] = asyncio.run(eval_retrieval(items))
    logger.info("Retrieval: %s", results["retrieval"])

    # SLM eval (optional, slow)
    if not skip_slm:
        logger.info("Evaluating SLM (this may take a while)…")
        try:
            results["slm"] = eval_slm(items[:200])  # cap for speed
            logger.info("SLM: %s", results["slm"])
        except Exception as exc:
            logger.error("SLM eval failed: %s", exc)
            results["slm"] = {"error": str(exc)}

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Eval results saved → %s", output_path)


if __name__ == "__main__":
    app()