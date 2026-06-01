"""
OpenRouter Embedding Provider
Implementation using OpenAI-compatible embeddings endpoint via OpenRouter.
"""

import logging
import os

import numpy as np
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pydantic import SecretStr
from typing_extensions import override

from chatbot.models import ChatbotConfig
from vector_store.services.constants import (
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_MODEL,
    OPENROUTER_API_KEY_ENV_NAME,
    OPENROUTER_BASE_URL_ENV_NAME,
)
from vector_store.services.embedding_providers.embedding_interface import (
    EmbeddingProvider,
)

logger = logging.getLogger(__name__)


class OpenRouterEmbeddingProvider(EmbeddingProvider):
    """OpenRouter embedding provider via OpenAI-compatible API."""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        """
        Initialize OpenRouter embedding provider.

        Args:
            model: OpenRouter embedding model name
            base_url: OpenRouter API base URL
        """
        if not model:
            model_name = ChatbotConfig.get_config("EMBEDDING_MODEL")
            if isinstance(model_name, str) and model_name.strip():
                model = model_name.strip()
            else:
                logger.warning(
                    "EMBEDDING_MODEL not found in config, using provider default."
                )
                model = DEFAULT_OPENROUTER_MODEL

        api_key_raw = ChatbotConfig.get_config(OPENROUTER_API_KEY_ENV_NAME, None)
        api_key = (
            api_key_raw.strip()
            if isinstance(api_key_raw, str) and api_key_raw.strip()
            else os.getenv(OPENROUTER_API_KEY_ENV_NAME)
        )
        if not api_key:
            raise ValueError(
                f"{OPENROUTER_API_KEY_ENV_NAME} environment variable is required"
            )

        db_base_url = ChatbotConfig.get_config("OPENROUTER_BASE_URL", None)
        resolved_base_url = (
            base_url
            or (
                db_base_url.strip()
                if isinstance(db_base_url, str) and db_base_url.strip()
                else None
            )
            or os.getenv(OPENROUTER_BASE_URL_ENV_NAME)
            or DEFAULT_OPENROUTER_BASE_URL
        )

        self.model = model
        try:
            self.embeddings = OpenAIEmbeddings(
                model=model,
                api_key=SecretStr(api_key),
                base_url=resolved_base_url,
            )
            self.client = OpenAI(api_key=api_key, base_url=resolved_base_url)
        except TypeError:
            logger.error(
                "Failed to initialize OpenRouterEmbeddings. Check if the model and base_url are correct."
            )
            raise
        logger.info(
            "Initialized OpenRouterEmbeddingProvider with model: %s, base_url: %s",
            model,
            resolved_base_url,
        )

    def _normalize_768(self, embedding: list[float]) -> list[float]:
        """Normalize vector with 768-d truncation for pgvector compatibility."""
        truncated = embedding[:768]
        arr = np.array(truncated, dtype=float)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    @override
    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding vector for the given text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            raw_embedding = self.embeddings.embed_query(text)
            return self._normalize_768(raw_embedding)
        except Exception as e:
            logger.error(f"Error embedding text with OpenRouter: {e}")
            raise

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts with one OpenRouter request."""
        try:
            response = self.client.embeddings.create(model=self.model, input=texts)
            data = response.data
            if len(data) != len(texts):
                raise ValueError(
                    f"OpenRouter returned {len(data)} embeddings for {len(texts)} texts"
                )
            ordered = sorted(data, key=lambda item: item.index)
            return [self._normalize_768(item.embedding) for item in ordered]
        except Exception as e:
            logger.error(f"Error embedding text batch with OpenRouter: {e}")
            raise

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embedding vectors for multiple documents.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            raw_embeddings = self.embeddings.embed_documents(texts)
            return [self._normalize_768(emb) for emb in raw_embeddings]
        except Exception as e:
            logger.error(f"Error embedding documents with OpenRouter: {e}")
            raise
