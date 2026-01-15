"""
Embedding Service
Main service for generating text embeddings using different providers.
"""

import logging

from embeddings.services.constants import (
    EMBEDDING_PROVIDER_GEMINI,
    EMBEDDING_PROVIDER_TEI,
    EMBEDDING_PROVIDER_TRANSFORMERS,
)
from embeddings.services.providers.embedding_interface import EmbeddingProvider
from embeddings.services.providers.gemini_provider import GeminiEmbeddingProvider
from embeddings.services.providers.tei_provider import TEIEmbeddingProvider
from embeddings.services.providers.transformers_provider import (
    TransformersEmbeddingProvider,
)

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings.
    Supports multiple providers: gemini, tei, transformers.
    """

    PROVIDERS = {
        EMBEDDING_PROVIDER_GEMINI: GeminiEmbeddingProvider,
        EMBEDDING_PROVIDER_TEI: TEIEmbeddingProvider,
        EMBEDDING_PROVIDER_TRANSFORMERS: TransformersEmbeddingProvider,
    }

    def __init__(self, provider: str = EMBEDDING_PROVIDER_TRANSFORMERS, **kwargs):  # pyright: ignore[reportUnknownParameterType]
        """
        Initialize embedding service with specified provider.

        Args:
            provider: Provider name ('gemini' or 'tei')
            **kwargs: Additional arguments passed to provider initialization
        """
        if provider not in self.PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider}. Available: {list(self.PROVIDERS.keys())}"
            )

        provider_class = self.PROVIDERS[provider]
        self.provider: EmbeddingProvider = provider_class(**kwargs)
        self.provider_name = provider
        logger.info(f"EmbeddingService initialized with provider: {provider}")

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding vector for the given text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        return self.provider.embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embedding vectors for multiple documents.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        return self.provider.embed_documents(texts)

    def get_provider_name(self) -> str:
        """Get the current provider name."""
        return self.provider_name
