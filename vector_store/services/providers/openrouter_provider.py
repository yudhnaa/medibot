"""
OpenRouter Embedding Provider
Implementation using OpenAI-compatible embeddings endpoint via OpenRouter.
"""

import logging
import os

import numpy as np
from pydantic import SecretStr

from langchain_openai import OpenAIEmbeddings
from typing_extensions import override

from chatbot.models import ChatbotConfig
from vector_store.services.constants import (
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_EMBEDDING_MODEL,
    OPENROUTER_API_KEY_ENV_NAME,
    OPENROUTER_BASE_URL_ENV_NAME,
)
from vector_store.services.providers.embedding_interface import EmbeddingProvider

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
            model_name = ChatbotConfig.get_config("OPENROUTER_EMBEDDING_MODEL")
            if isinstance(model_name, str):
                model = model_name
            else:
                logger.warning(
                    "OPENROUTER_EMBEDDING_MODEL not found in config, using default."
                )
                model = DEFAULT_OPENROUTER_EMBEDDING_MODEL

        api_key = os.getenv(OPENROUTER_API_KEY_ENV_NAME)
        if not api_key:
            raise ValueError(
                f"{OPENROUTER_API_KEY_ENV_NAME} environment variable is required"
            )

        resolved_base_url = (
            base_url
            or os.getenv(OPENROUTER_BASE_URL_ENV_NAME)
            or DEFAULT_OPENROUTER_BASE_URL
        )

        try:
            self.embeddings = OpenAIEmbeddings(
                model=model,
                api_key=SecretStr(api_key),
                base_url=resolved_base_url,
            )
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
