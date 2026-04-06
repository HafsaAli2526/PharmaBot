"""
training/train_classifier.py
Train an optional ML-based domain classifier on top of the rule-based one.
Uses BioBERT embeddings + a lightweight classification head.
Fine-tunes only the classification head (frozen encoder) for speed.

Usage:
  python training/train_classifier.py \
    --data ./data/processed/train.jsonl \
    --output ./models/domain_classifier \
    --epochs 5
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import typer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pharmaai.training.classifier")
app = typer.Typer()

DOMAIN_LABELS = [
    "pharmacovigilance", "internal", "r&d", "regulation", "formulas", "unknown"
]
LABEL2ID = {l: i for i, l in enumerate(DOMAIN_LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


class DomainDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = [
            i for i in items
            if i.get("domain") in LABEL2ID
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        text = item.get("question", "") + " " + item.get("context", "")
        label = LABEL2ID[item["domain"]]
        return text, label


class DomainClassifierHead(nn.Module):
    """Classification head on top of frozen BioBERT CLS embeddings."""

    def __init__(self, input_dim: int = 768, num_classes: int = 6, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def get_embeddings(texts: list[str], batch_size: int = 64) -> np.ndarray:
    from pharmaai.embeddings.models import embed_texts, registry
    registry.load_all()
    return embed_texts(texts, "biobert", batch_size=batch_size)


@app.command()
def train(
    data: str = typer.Option("./data/processed/train.jsonl"),
    output: str = typer.Option("./models/domain_classifier"),
    epochs: int = typer.Option(10),
    batch_size: int = typer.Option(128),
    lr: float = typer.Option(1e-3),
    eval_split: float = typer.Option(0.1),
):
    logger.info("Loading training data…")
    items = []
    with open(data) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    dataset = DomainDataset(items)
    logger.info("Dataset size: %d", len(dataset))

    # Split
    val_size = int(len(dataset) * eval_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Pre-compute embeddings (batch all texts)
    logger.info("Computing BioBERT embeddings (this may take a while)…")
    all_texts = [dataset[i][0] for i in range(len(dataset))]
    all_labels = [dataset[i][1] for i in range(len(dataset))]
    all_embs = get_embeddings(all_texts, batch_size=64)

    train_indices = train_ds.indices
    val_indices = val_ds.indices

    X_train = torch.tensor(all_embs[train_indices], dtype=torch.float32)
    y_train = torch.tensor([all_labels[i] for i in train_indices], dtype=torch.long)
    X_val = torch.tensor(all_embs[val_indices], dtype=torch.float32)
    y_val = torch.tensor([all_labels[i] for i in val_indices], dtype=torch.long)

    from pharmaai.core.config import get_settings
    device = torch.device(get_settings().gpu.device if torch.cuda.is_available() else "cpu")

    model = DomainClassifierHead(input_dim=768, num_classes=len(DOMAIN_LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(X_train) // batch_size + 1, epochs=epochs
    )

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]

        total_loss, correct = 0.0, 0
        for i in range(0, len(X_train), batch_size):
            xb = X_train[i : i + batch_size].to(device)
            yb = y_train[i : i + batch_size].to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(xb)
            correct += (logits.argmax(-1) == yb).sum().item()

        train_acc = correct / len(X_train)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device))
            val_loss = F.cross_entropy(val_logits, y_val.to(device)).item()
            val_acc = (val_logits.argmax(-1) == y_val.to(device)).float().mean().item()

        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
            epoch, epochs, total_loss / len(X_train), train_acc, val_loss, val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            out = Path(output)
            out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out / "classifier_head.pt")
            json.dump(
                {"label2id": LABEL2ID, "id2label": ID2LABEL, "input_dim": 768},
                open(out / "config.json", "w"),
                indent=2,
            )
            logger.info("  ✓ Best model saved (val_acc=%.4f)", best_val_acc)

    logger.info("Training complete. Best val_acc=%.4f", best_val_acc)


if __name__ == "__main__":
    app()