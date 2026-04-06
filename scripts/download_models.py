"""
scripts/download_models.py
Download all required models from HuggingFace Hub to ./models/
Run this on a machine with internet access, then rsync to the H200 server.

Usage:
  python scripts/download_models.py [--models all] [--token HF_TOKEN]
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pharmaai.download_models")
app = typer.Typer()

MODELS = {
    "biobert": "dmis-lab/biobert-v1.1",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "chemberta": "seyonec/ChemBERTa-zinc-base-v1",
    "slm": "microsoft/Phi-3-mini-4k-instruct",
}


@app.command()
def download(
    models: str = typer.Option("all", help="Comma-separated model keys or 'all'"),
    output_dir: str = typer.Option("./models", help="Directory to save models"),
    token: Optional[str] = typer.Option(None, envvar="HF_TOKEN", help="HuggingFace token"),
):
    keys = list(MODELS.keys()) if models == "all" else [m.strip() for m in models.split(",")]
    out = Path(output_dir)

    for key in keys:
        if key not in MODELS:
            logger.warning("Unknown model key: %s. Available: %s", key, list(MODELS))
            continue
        repo_id = MODELS[key]
        dest = out / key
        if dest.exists() and any(dest.iterdir()):
            logger.info("✓ %s already exists at %s – skipping.", key, dest)
            continue
        logger.info("Downloading %s (%s) → %s …", key, repo_id, dest)
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(dest),
                token=token,
                ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
            )
            logger.info("  ✓ %s downloaded.", key)
        except Exception as exc:
            logger.error("  ✗ Failed to download %s: %s", key, exc)

    logger.info("All downloads complete. Copy ./models/ to your H200 server.")
    logger.info("  rsync -av --progress ./models/ user@h200-server:/path/to/pharma_ai/models/")


if __name__ == "__main__":
    app()