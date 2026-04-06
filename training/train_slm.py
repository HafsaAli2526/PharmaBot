"""
training/train_slm.py
Fine-tune the SLM (Phi-3-mini or similar) on pharmaceutical QA data
using QLoRA (4-bit quantisation + LoRA adapters).

Usage:
  python training/train_slm.py \
    --model_path ./models/slm \
    --data_path ./data/processed/train.jsonl \
    --output_dir ./models/slm_finetuned \
    --epochs 3 \
    --batch_size 4
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import typer
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pharmaai.training.slm")

app = typer.Typer()

PROMPT_TEMPLATE = (
    "### Question:\n{question}\n\n"
    "### Context:\n{context}\n\n"
    "### Answer:\n{answer}"
)


def load_dataset(data_path: Path) -> Dataset:
    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def tokenize(example: dict, tokenizer, max_length: int = 1024) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        question=example.get("question", ""),
        context=example.get("context", ""),
        answer=example.get("answer", ""),
    )
    tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


@app.command()
def train(
    model_path: str = typer.Option("./models/slm", help="Base model path"),
    data_path: str = typer.Option("./data/processed/train.jsonl"),
    output_dir: str = typer.Option("./models/slm_finetuned"),
    epochs: int = typer.Option(3),
    batch_size: int = typer.Option(4),
    gradient_accumulation: int = typer.Option(4),
    learning_rate: float = typer.Option(2e-4),
    lora_r: int = typer.Option(16),
    lora_alpha: int = typer.Option(32),
    max_seq_length: int = typer.Option(1024),
    eval_split: float = typer.Option(0.05),
):
    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 4-bit quantisation ────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    logger.info("Loading base model…")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA config ───────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    raw_ds = load_dataset(Path(data_path))
    split = raw_ds.train_test_split(test_size=eval_split, seed=42)
    train_ds = split["train"].map(
        lambda ex: tokenize(ex, tokenizer, max_seq_length),
        remove_columns=raw_ds.column_names,
    )
    eval_ds = split["test"].map(
        lambda ex: tokenize(ex, tokenizer, max_seq_length),
        remove_columns=raw_ds.column_names,
    )

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        fp16=False,
        bf16=True,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        optim="paged_adamw_8bit",
        dataloader_num_workers=4,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
    )

    logger.info("Starting training…")
    trainer.train()

    logger.info("Saving fine-tuned model to %s", output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training complete.")


if __name__ == "__main__":
    app()