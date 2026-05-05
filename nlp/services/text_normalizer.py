"""
Text normalization for Vietnamese medical NLP pipeline.
Handles text preprocessing including segmentation, accent removal, and cleanup.
"""

import builtins
import os
import re
import threading
import unicodedata
from typing import Any

import py_vncorenlp

from chatbot.models import ChatbotConfig
from nlp.services.constants import (
    BASIC_PUNCTUATION_PATTERN,
    MEDICAL_PUNCTUATION_PATTERN,
    VNCORENLP_CACHE_KEY,
    VNCORENLP_PATH,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class TextNormalizer:
    """
    Text preprocessing for NER and Negation Detection.
    Processing order: normalize -> segment -> clean punctuation
    """

    # Class-level singleton to avoid re-initializing JVM / segmenter
    _SEGMENTER: py_vncorenlp.VnCoreNLP | None = None
    _SEGMENTER_INIT_TRIED: bool = False
    _SEGMENTER_LOCK = threading.Lock()
    _GLOBAL_CACHE_KEY = VNCORENLP_CACHE_KEY

    def __init__(self, vncorenlp_path: str | None = None) -> None:
        """
        Initialize TextNormalizer with VnCoreNLP segmenter.

        Args:
            vncorenlp_path: Path to VnCoreNLP jar file. If None, uses default path.
        """
        self.vncorenlp_segmenter: py_vncorenlp.VnCoreNLP | None = None
        self.vncorenlp_path = vncorenlp_path or self._get_default_vncorenlp_path()
        self.enabled = self._get_vncorenlp_enabled()

        if not self.enabled:
            logger.debug("VnCoreNLP disabled by configuration")
            return

        # Use process-global cache to avoid JVM restart
        cached = getattr(builtins, TextNormalizer._GLOBAL_CACHE_KEY, None)
        if cached is not None:
            self.vncorenlp_segmenter = cached
            TextNormalizer._SEGMENTER = cached
            TextNormalizer._SEGMENTER_INIT_TRIED = True
        elif TextNormalizer._SEGMENTER is not None:
            self.vncorenlp_segmenter = TextNormalizer._SEGMENTER
        else:
            if not TextNormalizer._SEGMENTER_INIT_TRIED:
                with TextNormalizer._SEGMENTER_LOCK:
                    if (
                        TextNormalizer._SEGMENTER is None
                        and not TextNormalizer._SEGMENTER_INIT_TRIED
                    ):
                        self._initialize_vncorenlp()
                        TextNormalizer._SEGMENTER = self.vncorenlp_segmenter
                        TextNormalizer._SEGMENTER_INIT_TRIED = True
                        try:
                            setattr(
                                builtins,
                                TextNormalizer._GLOBAL_CACHE_KEY,
                                self.vncorenlp_segmenter,
                            )
                        except Exception:
                            pass
                    else:
                        self.vncorenlp_segmenter = TextNormalizer._SEGMENTER
            else:
                self.vncorenlp_segmenter = TextNormalizer._SEGMENTER

    def _get_vncorenlp_enabled(self) -> bool:
        """Get VnCoreNLP enabled setting from config."""
        try:
            return bool(ChatbotConfig.get_config("VNCORENLP_ENABLED", default=True))
        except Exception:
            return True

    def _get_default_vncorenlp_path(self) -> str:
        """Get default VnCoreNLP path."""
        return VNCORENLP_PATH

    def _initialize_vncorenlp(self) -> None:
        """Initialize VnCoreNLP for word segmentation."""
        try:
            old_cwd = os.getcwd()
            try:
                self.vncorenlp_segmenter = py_vncorenlp.VnCoreNLP(
                    annotators=["wseg"],
                    save_dir=self.vncorenlp_path,
                )
            finally:
                try:
                    os.chdir(old_cwd)
                except Exception:
                    pass
            logger.info("VnCoreNLP initialized at: %s", self.vncorenlp_path)
        except Exception as e:
            logger.warning("Cannot initialize VnCoreNLP: %s. Using fallback.", e)
            self.vncorenlp_segmenter = None

    def remove_accents(self, text: str) -> str:
        """Remove Vietnamese accents from text."""
        normalized = unicodedata.normalize("NFD", text)
        without_accents = "".join(
            c for c in normalized if unicodedata.category(c) != "Mn"
        )
        return without_accents

    def normalize_text(self, text: str) -> str:
        """Basic text normalization: lowercase and whitespace cleanup."""
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def segment_text(self, text: str) -> str:
        """Segment text using VnCoreNLP word segmentation."""
        if self.vncorenlp_segmenter is None:
            return text.replace(" ", "_")

        try:
            old_cwd = os.getcwd()
            try:
                segmented_sentences = self.vncorenlp_segmenter.word_segment(text)
            finally:
                try:
                    os.chdir(old_cwd)
                except Exception:
                    pass

            if segmented_sentences:
                return " ".join(segmented_sentences)
            else:
                return text.replace(" ", "_")
        except Exception as e:
            logger.warning("VnCoreNLP segmentation error: %s", e)
            return text

    def clean_punctuation(
        self,
        text: str,
        remove_semicolon: bool = True,
        keep_medical_punctuation: bool = True,
    ) -> str:
        """Clean punctuation from text."""
        if remove_semicolon:
            text = text.replace(";", "")

        if keep_medical_punctuation:
            text = re.sub(MEDICAL_PUNCTUATION_PATTERN, "", text)
        else:
            text = re.sub(BASIC_PUNCTUATION_PATTERN, "", text)

        text = re.sub(r"\s+", " ", text).strip()
        return text

    def process_text(self, text: str, include_variants: bool = False) -> dict[str, str]:
        """
        Complete text preprocessing pipeline.

        Args:
            text: Input text
            include_variants: Whether to include all text variants in output

        Returns:
            Dictionary with processed text variants
        """
        if not text or not text.strip():
            return {"processed": ""}

        text_normalized = self.normalize_text(text)
        text_no_accents = self.remove_accents(text_normalized)
        text_clean = self.clean_punctuation(
            text_normalized, remove_semicolon=True, keep_medical_punctuation=False
        )
        text_segmented = self.segment_text(text_clean)
        processed_text = text_clean

        result: dict[str, str] = {"processed": processed_text}

        if include_variants:
            result.update(
                {
                    "original": text,
                    "normalized": text_normalized,
                    "no_accents": text_no_accents,
                    "clean": text_clean,
                    "segmented": text_segmented,
                }
            )

        return result

    def batch_process(
        self, texts: list[str], include_variants: bool = False
    ) -> list[dict[str, str]]:
        """Process multiple texts in batch."""
        return [self.process_text(text, include_variants) for text in texts]

    def get_processing_info(self) -> dict[str, Any]:
        """Get information about the normalizer configuration."""
        return {
            "vncorenlp_enabled": getattr(self, "enabled", True),
            "vncorenlp_available": self.vncorenlp_segmenter is not None,
            "vncorenlp_path": self.vncorenlp_path,
            "vncorenlp_init_tried": TextNormalizer._SEGMENTER_INIT_TRIED,
            "processing_steps": [
                "1. Basic normalization (lowercase, whitespace cleanup)",
                "2. Remove Vietnamese accents",
                "3. Clean punctuation (remove semicolons)",
                "4. Word segmentation with VnCoreNLP",
            ],
        }

    def is_ready(self) -> bool:
        """Check if normalizer is ready for processing."""
        return self.vncorenlp_segmenter is not None

    @classmethod
    def warmup(cls, vncorenlp_path: str | None = None) -> bool:
        """Pre-initialize JVM on startup to avoid first-request latency."""
        inst = cls(vncorenlp_path=vncorenlp_path)
        return inst.vncorenlp_segmenter is not None


def quick_normalize(text: str) -> str:
    """Quick text normalization without creating TextNormalizer instance."""
    normalizer = TextNormalizer()
    return normalizer.process_text(text)["processed"]
