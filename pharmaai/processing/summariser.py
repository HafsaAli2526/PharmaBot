"""
pharmaai/processing/summariser.py
Two-tier summarisation:
  1. TextRank extractive summary (fast, no GPU, good for long docs)
  2. SLM abstractive summary (H200 GPU, richer but slower)
"""
from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Literal

logger = logging.getLogger("pharmaai.processing.summariser")


class SummaryMode(str, Enum):
    EXTRACTIVE = "extractive"   # TextRank – fast, no model needed
    ABSTRACTIVE = "abstractive" # SLM     – better quality, uses GPU
    AUTO = "auto"               # TextRank for short texts, SLM for long


# ─────────────────────────────────────────────────────────────────────────────
# Extractive (TextRank via sumy)
# ─────────────────────────────────────────────────────────────────────────────

def _extractive_summary(text: str, sentence_count: int = 4) -> str:
    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.text_rank import TextRankSummarizer
        from sumy.nlp.stemmers import Stemmer
        from sumy.utils import get_stop_words

        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        stemmer = Stemmer("english")
        summarizer = TextRankSummarizer(stemmer)
        summarizer.stop_words = get_stop_words("english")
        sentences = summarizer(parser.document, sentence_count)
        return " ".join(str(s) for s in sentences)
    except Exception as exc:
        logger.warning("TextRank failed (%s); falling back to truncation.", exc)
        # Naive fallback: first N sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sentences[:sentence_count])


# ─────────────────────────────────────────────────────────────────────────────
# Abstractive (SLM)
# ─────────────────────────────────────────────────────────────────────────────

ABSTRACTIVE_PROMPT = (
    "Summarise the following pharmaceutical text in 3-5 concise sentences. "
    "Preserve key drug names, dosages, and clinical findings.\n\n"
    "Text:\n{text}\n\nSummary:"
)


def _abstractive_summary(text: str, max_new_tokens: int = 200) -> str:
    from pharmaai.inference.small_lm import slm
    prompt = ABSTRACTIVE_PROMPT.format(text=text[:3000])
    return slm.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class Summariser:
    ABSTRACTIVE_THRESHOLD = 800  # chars; use SLM only for longer texts in AUTO mode

    def summarise(
        self,
        text: str,
        mode: SummaryMode = SummaryMode.AUTO,
        sentence_count: int = 4,
        max_new_tokens: int = 200,
    ) -> str:
        if not text or not text.strip():
            return ""

        if mode == SummaryMode.EXTRACTIVE:
            return _extractive_summary(text, sentence_count)

        if mode == SummaryMode.ABSTRACTIVE:
            return _abstractive_summary(text, max_new_tokens)

        # AUTO: use SLM for longer, richer texts
        if len(text) > self.ABSTRACTIVE_THRESHOLD:
            try:
                return _abstractive_summary(text, max_new_tokens)
            except Exception as exc:
                logger.warning("Abstractive summary failed: %s; falling back.", exc)
        return _extractive_summary(text, sentence_count)

    def summarise_batch(
        self,
        texts: list[str],
        mode: SummaryMode = SummaryMode.EXTRACTIVE,
        sentence_count: int = 3,
    ) -> list[str]:
        return [self.summarise(t, mode=mode, sentence_count=sentence_count) for t in texts]


summariser = Summariser()