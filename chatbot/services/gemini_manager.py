"""
Gemini API Manager
Service for managing Google Gemini LLM with API key rotation and retry logic.
"""

import logging
import random
import time
import unicodedata
from typing import Any, ClassVar, cast, override

from django.conf import settings

import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from chatbot.models import ChatbotConfig

logger = logging.getLogger(__name__)


def vn_norm(s: str) -> str:
    """Normalize Vietnamese text using NFC (Canonical Composition) and strip whitespace."""
    return unicodedata.normalize("NFC", str(s)).strip()


class NormalizedGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    """Google Generative AI Embeddings with L2 normalization and Vietnamese text normalization."""

    EPS: ClassVar[float] = 1e-12

    @override
    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Embed documents with L2 normalization."""
        normalized_texts = [vn_norm(text) for text in texts]
        vecs = super().embed_documents(normalized_texts, **kwargs)
        arr = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, self.EPS)
        normed = arr / norms
        return normed.tolist()

    @override
    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Embed query with L2 normalization."""
        normalized_text = vn_norm(text)
        v = np.asarray(super().embed_query(normalized_text, **kwargs), dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n < self.EPS:
            return v.tolist()
        return (v / n).tolist()


class GeminiAPIManager:
    """Manages Gemini API with key rotation and robust error handling."""

    def __init__(self) -> None:
        """Initialize with API keys from Django settings."""
        self.api_keys = self._load_api_keys()
        if not self.api_keys:
            raise ValueError("No Gemini API keys configured")
        self.current_key_index = 0
        self._setup_client()

    def _load_api_keys(self) -> list[str]:
        """Load API keys from settings based on paid/free flag."""
        is_paid = bool(
            ChatbotConfig.get_config("IS_USE_PAID_GOOGLE_GEMINI_API_KEY", False)
        )
        if is_paid:
            keys = getattr(settings, "GOOGLE_GEMINI_PAID_API_KEYS", [])
        else:
            keys = getattr(settings, "GOOGLE_GEMINI_API_KEYS", [])

        # Fallback to single key
        if not keys:
            single_key = getattr(settings, "GOOGLE_API_KEY", None)
            if single_key:
                keys = [single_key]
        return cast(list[str], keys)

    def _setup_client(self) -> None:
        """Log current API key configuration (LangChain handles API key via constructor)."""
        current_key = self.get_current_key()
        logger.info(f"Gemini API key configured: {current_key[:10]}...")

    def get_current_key(self) -> str:
        """Get the current API key."""
        return self.api_keys[self.current_key_index]

    def rotate_key(self) -> None:
        """Rotate to the next API key."""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self._setup_client()
        logger.info(f"Rotated to API key index: {self.current_key_index}")

    def create_llm(
        self,
        model: str | None = None,
        temperature: float | None = None,
    ) -> ChatGoogleGenerativeAI:
        """Create a LangChain ChatGoogleGenerativeAI instance."""
        llm_model = str(
            model or ChatbotConfig.get_config("LLM_MODEL", "gemini-2.5-flash")
        )
        config_temp_raw = ChatbotConfig.get_config("TEMPERATURE", 0.3)
        config_temp = (
            float(cast(float, config_temp_raw)) if config_temp_raw is not None else 0.3
        )
        llm_temp = temperature if temperature is not None else config_temp

        return ChatGoogleGenerativeAI(
            model=llm_model,
            temperature=llm_temp,
            google_api_key=self.get_current_key(),
            streaming=True,
        )

    def create_embeddings(
        self,
        model: str | None = None,
        task_type: str | None = None,
        normalized: bool = False,
    ) -> GoogleGenerativeAIEmbeddings | NormalizedGoogleGenerativeAIEmbeddings:
        """Create an embeddings instance."""
        emb_model = str(
            model or ChatbotConfig.get_config("EMBEDDING_MODEL", "models/embedding-001")
        )
        kwargs: dict[str, Any] = {
            "model": emb_model,
            "google_api_key": self.get_current_key(),
        }
        if task_type:
            kwargs["task_type"] = task_type

        if normalized:
            return NormalizedGoogleGenerativeAIEmbeddings(**kwargs)  # type: ignore[arg-type]
        return GoogleGenerativeAIEmbeddings(**kwargs)  # type: ignore[arg-type]

    def should_rotate(self, err: Exception) -> bool:
        """Check if error indicates rate limiting."""
        msg = str(err).lower()
        indicators = ["429", "rate limit", "quota", "exceed", "resource exhausted"]
        return any(s in msg for s in indicators)

    def backoff_sleep(self, attempt: int) -> None:
        """Exponential backoff with jitter."""
        base = min(8, 2 ** max(0, attempt))
        jitter = random.uniform(0, 0.3)
        time.sleep(base + jitter)


# Singleton instance
_gemini_manager: GeminiAPIManager | None = None


def get_gemini_manager() -> GeminiAPIManager:
    """Get or create the global GeminiAPIManager instance."""
    global _gemini_manager
    if _gemini_manager is None:
        _gemini_manager = GeminiAPIManager()
    return _gemini_manager
