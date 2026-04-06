"""
pharmaai/processing/formatter.py
Text cleaning, normalisation, and truncation utilities.
"""
from __future__ import annotations

import html
import re
import unicodedata
from functools import lru_cache

# Common medical abbreviations NOT to split on
_ABBREV = {"Dr", "Mr", "Mrs", "Ms", "vs", "et al", "i.e", "e.g", "Fig", "Eq"}


class TextFormatter:
    """
    Cleans and normalises raw text before embedding or display.
    """

    def __init__(self, max_chars: int = 4096):
        self.max_chars = max_chars
        self._whitespace_re = re.compile(r"\s+")
        self._url_re = re.compile(r"https?://\S+")
        self._xml_re = re.compile(r"<[^>]+>")
        self._special_re = re.compile(r"[^\w\s.,;:()\-/\\%°μα-ωΑ-Ω]")

    def clean(self, text: str, keep_urls: bool = False) -> str:
        if not text:
            return ""
        # HTML entities
        text = html.unescape(text)
        # XML/HTML tags
        text = self._xml_re.sub(" ", text)
        # Optionally remove URLs
        if not keep_urls:
            text = self._url_re.sub(" ", text)
        # Unicode normalise to NFC
        text = unicodedata.normalize("NFC", text)
        # Collapse whitespace
        text = self._whitespace_re.sub(" ", text).strip()
        # Truncate
        if len(text) > self.max_chars:
            text = text[: self.max_chars]
        return text

    def clean_batch(self, texts: list[str], keep_urls: bool = False) -> list[str]:
        return [self.clean(t, keep_urls) for t in texts]

    @staticmethod
    def truncate_for_embedding(text: str, max_tokens: int = 512) -> str:
        """
        Rough token budget: 1 token ≈ 4 chars for English biomedical text.
        Preserve whole sentences where possible.
        """
        char_limit = max_tokens * 4
        if len(text) <= char_limit:
            return text
        # Try to cut at a sentence boundary
        truncated = text[:char_limit]
        last_period = truncated.rfind(".")
        if last_period > char_limit * 0.7:
            return truncated[: last_period + 1]
        return truncated

    @staticmethod
    def build_content(title: str, body: str, metadata: dict | None = None) -> str:
        """Compose a single string from title + body (+ optional metadata fields)."""
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if body:
            parts.append(body)
        if metadata:
            useful = {
                k: v for k, v in metadata.items()
                if k in ("drug_name", "compound", "nct_id", "source", "authors")
                and v
            }
            if useful:
                parts.append(
                    " | ".join(f"{k}: {v}" for k, v in useful.items())
                )
        return "\n".join(parts)


formatter = TextFormatter()