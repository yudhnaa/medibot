"""
Embedding Interface
Abstract base class for embedding providers.
"""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding vector for the given text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        pass

    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # pyright: ignore[reportUnusedParameter]
        """
        Generate embedding vectors for multiple documents.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors

        Raises:
            NotImplementedError: If the provider does not support batch embedding.
        """
        raise NotImplementedError("Batch embedding is not supported by this provider.")
