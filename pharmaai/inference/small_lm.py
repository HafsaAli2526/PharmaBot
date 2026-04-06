"""
pharmaai/inference/small_lm.py
Load the fine-tuned SLM (Phi-3-mini or similar) for RAG answer generation.
Uses 4-bit quantisation via bitsandbytes to fit on the H200 alongside
the three BERT encoders.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    TextStreamer,
)

from pharmaai.core.config import get_settings

logger = logging.getLogger("pharmaai.inference.small_lm")


class SLMInference:
    """Thread-safe singleton for SLM inference."""

    _instance: SLMInference | None = None
    _lock = threading.Lock()

    def __new__(cls) -> SLMInference:
        with cls._lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._model = None
                obj._tokenizer = None
                obj._loaded = False
                cls._instance = obj
        return cls._instance

    def load(self) -> None:
        if self._loaded:
            return
        settings = get_settings()
        cfg = settings.models.slm_finetuned or settings.models.slm
        model_path = cfg.path

        if not Path(model_path).exists():
            # Fallback to base SLM
            model_path = settings.models.slm.path
            logger.warning("Fine-tuned SLM not found; using base SLM at %s", model_path)

        logger.info("Loading SLM from %s …", model_path)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self._model.eval()
        self._loaded = True
        logger.info("SLM loaded on %s", next(self._model.parameters()).device)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        if not self._loaded:
            self.load()

        settings = get_settings()
        cfg = settings.models.slm
        _max = max_new_tokens or cfg.max_new_tokens
        _temp = temperature or cfg.temperature
        _top_p = top_p or cfg.top_p

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=3500
        ).to(next(self._model.parameters()).device)

        gen_config = GenerationConfig(
            max_new_tokens=_max,
            temperature=_temp,
            top_p=_top_p,
            do_sample=_temp > 0,
            repetition_penalty=1.15,
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        output_ids = self._model.generate(**inputs, generation_config=gen_config)
        # Strip prompt tokens from output
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()


slm = SLMInference()