"""
Gemini API Manager
Service for managing Google Gemini LLM with API key rotation and retry logic.
"""

import logging
import os
import random
import time
import unicodedata
from typing import Any, ClassVar, cast, override

from django.conf import settings

import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from chatbot.models import ChatbotConfig
from vector_store.services.constants import (
    DEFAULT_OPENROUTER_BASE_URL,
    OPENROUTER_API_KEY_ENV_NAME,
    OPENROUTER_BASE_URL_ENV_NAME,
)

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
        """Initialize manager without touching provider-specific secrets yet."""
        self.api_keys: list[str] = []
        self.current_key_index = 0

    def _to_str_list(self, value: object) -> list[str]:
        """Normalize DB/settings values into a list of non-empty strings."""
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, tuple):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return []

    def _get_db_or_setting(self, key: str, setting_fallback: object = None) -> object:
        """Read config from ChatbotConfig first, then Django settings fallback."""
        db_value = ChatbotConfig.get_config(key, None)
        if db_value not in (None, ""):
            return db_value
        if setting_fallback is not None:
            return setting_fallback
        return getattr(settings, key, None)

    def _load_api_keys(self) -> list[str]:
        """Load Gemini API keys from DB config first, then Django settings."""
        is_paid = bool(
            ChatbotConfig.get_config("IS_USE_PAID_GOOGLE_GEMINI_API_KEY", False)
        )
        key_name = (
            "GOOGLE_GEMINI_PAID_API_KEYS" if is_paid else "GOOGLE_GEMINI_API_KEYS"
        )

        keys = self._to_str_list(self._get_db_or_setting(key_name))
        if keys:
            return keys

        if is_paid:
            setting_keys = getattr(settings, "GOOGLE_GEMINI_PAID_API_KEYS", [])
        else:
            setting_keys = getattr(settings, "GOOGLE_GEMINI_API_KEYS", [])
        keys = self._to_str_list(setting_keys)
        if keys:
            return keys

        # Fallback to single key
        single_key_value = self._get_db_or_setting("GOOGLE_API_KEY")
        single_keys = self._to_str_list(single_key_value)
        if single_keys:
            return [single_keys[0]]

        setting_single_key = getattr(settings, "GOOGLE_API_KEY", None)
        setting_single_keys = self._to_str_list(setting_single_key)
        if setting_single_keys:
            return [setting_single_keys[0]]
        return []

    def _setup_client(self) -> None:
        """Record that Gemini credentials are available without exposing secrets."""
        logger.info("Gemini provider configured with %s API key(s)", len(self.api_keys))

    def _ensure_gemini_api_keys(self) -> None:
        """Ensure Gemini API keys are available before using Gemini provider."""
        if self.api_keys:
            return
        self.api_keys = self._load_api_keys()
        if self.api_keys:
            self.current_key_index = 0
            self._setup_client()
            return
        raise ValueError(
            "No Gemini API keys configured. Set GOOGLE_API_KEY/GOOGLE_GEMINI_API_KEYS "
            "in ChatbotConfig or environment."
        )

    def _resolve_llm_provider(self) -> str:
        """Resolve LLM provider from DB config."""
        provider = ChatbotConfig.get_config(
            "LLM_PROVIDER",
            getattr(settings, "LLM_PROVIDER", "gemini"),
        )
        if isinstance(provider, str) and provider.strip():
            return provider.strip().lower()
        return "gemini"

    def _resolve_openrouter_base_url(self) -> str:
        """Resolve OpenRouter base URL from DB config or environment."""
        db_base_url = ChatbotConfig.get_config("OPENROUTER_BASE_URL", None)
        if isinstance(db_base_url, str) and db_base_url.strip():
            return db_base_url.strip()
        env_base_url = os.getenv(OPENROUTER_BASE_URL_ENV_NAME)
        if env_base_url:
            return env_base_url
        return DEFAULT_OPENROUTER_BASE_URL

    def _resolve_openrouter_api_key(self) -> str | None:
        """Resolve OpenRouter API key from DB config or environment."""
        db_api_key = ChatbotConfig.get_config(OPENROUTER_API_KEY_ENV_NAME, None)
        if isinstance(db_api_key, str) and db_api_key.strip():
            return db_api_key.strip()
        env_api_key = os.getenv(OPENROUTER_API_KEY_ENV_NAME)
        if env_api_key and env_api_key.strip():
            return env_api_key.strip()
        return None

    def get_current_key(self) -> str:
        """Get the current API key."""
        self._ensure_gemini_api_keys()
        return self.api_keys[self.current_key_index]

    def rotate_key(self) -> None:
        """Rotate to the next API key."""
        self._ensure_gemini_api_keys()
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        logger.info("Rotated Gemini API key index to %s", self.current_key_index)

    def create_llm(
        self,
        model: str | None = None,
        temperature: float | None = None,
    ) -> ChatGoogleGenerativeAI | ChatOpenAI:
        """Create a LangChain chat model based on DB-configured provider."""
        llm_provider = self._resolve_llm_provider()
        config_temp_raw = ChatbotConfig.get_config("TEMPERATURE", 0.3)
        config_temp = (
            float(cast(float, config_temp_raw)) if config_temp_raw is not None else 0.3
        )
        llm_temp = temperature if temperature is not None else config_temp

        if llm_provider == "openrouter":
            default_openrouter_model = "openai/gpt-4.1-mini"
            llm_model = str(
                model or ChatbotConfig.get_config("LLM_MODEL", default_openrouter_model)
            )
            api_key = self._resolve_openrouter_api_key()
            if not api_key:
                raise ValueError(
                    f"{OPENROUTER_API_KEY_ENV_NAME} is required for LLM_PROVIDER=openrouter"
                )
            logger.info(
                "LLM provider selected: %s (model=%s)",
                llm_provider,
                llm_model,
            )
            return ChatOpenAI(
                model=llm_model,
                api_key=SecretStr(api_key),
                base_url=self._resolve_openrouter_base_url(),
                temperature=llm_temp,
                streaming=True,
            )

        llm_model = str(
            model or ChatbotConfig.get_config("LLM_MODEL", "gemini-2.5-flash")
        )
        self._ensure_gemini_api_keys()
        logger.info("LLM provider selected: gemini (model=%s)", llm_model)
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


def reset_gemini_manager() -> None:
    """Reset the shared manager for tests."""
    global _gemini_manager
    _gemini_manager = None
