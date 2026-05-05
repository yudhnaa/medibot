"""
Embedding Service
Main service for generating text embeddings using different providers.
"""

import logging

from chatbot.models import ChatbotConfig
from vector_store.services.constants import (
    DEFAULT_EMBEDDING_PROVIDER,
    EMBEDDING_PROVIDER_GEMINI,
    EMBEDDING_PROVIDER_OPENROUTER,
    EMBEDDING_PROVIDER_TEI,
    EMBEDDING_PROVIDER_TRANSFORMERS,
)
from vector_store.services.embedding_providers.embedding_interface import (
    EmbeddingProvider,
)
from vector_store.services.embedding_providers.gemini_provider import (
    GeminiEmbeddingProvider,
)
from vector_store.services.embedding_providers.openrouter_provider import (
    OpenRouterEmbeddingProvider,
)
from vector_store.services.embedding_providers.tei_provider import TEIEmbeddingProvider
from vector_store.services.embedding_providers.transformers_provider import (
    TransformersEmbeddingProvider,
)

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings.
    Supports multiple providers: gemini, openrouter, tei, transformers.
    """

    PROVIDERS = {
        EMBEDDING_PROVIDER_GEMINI: GeminiEmbeddingProvider,
        EMBEDDING_PROVIDER_OPENROUTER: OpenRouterEmbeddingProvider,
        EMBEDDING_PROVIDER_TEI: TEIEmbeddingProvider,
        EMBEDDING_PROVIDER_TRANSFORMERS: TransformersEmbeddingProvider,
    }

    @classmethod
    def resolve_provider(cls, provider: str | None = None) -> str:
        """Resolve embedding provider from explicit input or ChatbotConfig."""
        if provider:
            resolved = provider
        else:
            config_provider = ChatbotConfig.get_config(
                "EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER
            )
            resolved = (
                config_provider
                if isinstance(config_provider, str)
                else DEFAULT_EMBEDDING_PROVIDER
            )

        if resolved not in cls.PROVIDERS:
            logger.warning(
                "Invalid EMBEDDING_PROVIDER '%s', falling back to '%s'",
                resolved,
                DEFAULT_EMBEDDING_PROVIDER,
            )
            return DEFAULT_EMBEDDING_PROVIDER
        return resolved

    def __init__(
        self, provider: str | None = None, **kwargs
    ):  # pyright: ignore[reportUnknownParameterType]
        """
        Initialize embedding service with specified provider.

        Args:
            provider: Provider name override (None to use ChatbotConfig.EMBEDDING_PROVIDER)
            **kwargs: Additional arguments passed to provider initialization
        """
        provider = self.resolve_provider(provider)
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
