"""
TEI Embedding Provider
"""

import logging
import os
from typing_extensions import override

import requests

from vector_store.services.constants import (
    TEI_EMBEDDING_ENDPOINT_SUFFIX,
    TEI_ENDPOINT_ENV_NAME,
)
from vector_store.services.providers.embedding_interface import EmbeddingProvider

logger = logging.getLogger(__name__)


class TEIEmbeddingProvider(EmbeddingProvider):
    """TEI embedding provider using direct HTTP requests."""

    def __init__(self, endpoint_url: str | None = None):
        """
        Initialize TEI embedding provider.

        Args:
            endpoint_url: TEI server endpoint URL. If None, reads from TEI_ENDPOINT_URL env var.
        """
        endpoint = endpoint_url or os.getenv(TEI_ENDPOINT_ENV_NAME)
        if not endpoint:
            raise ValueError(
                f"{TEI_ENDPOINT_ENV_NAME} must be provided or set in environment"
            )

        # Ensure endpoint has the correct path
        if not endpoint.endswith(TEI_EMBEDDING_ENDPOINT_SUFFIX):
            endpoint = f"{endpoint.rstrip('/')}{TEI_EMBEDDING_ENDPOINT_SUFFIX}"

        self.endpoint_url: str = endpoint
        logger.info(
            f"Initialized TEIEmbeddingProvider with endpoint: {self.endpoint_url}"
        )

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
            response = requests.post(
                self.endpoint_url,
                json={"input": text},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error embedding text with TEI: {e}")
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
            response = requests.post(
                self.endpoint_url,
                json={"input": texts},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            logger.error(f"Error embedding documents with TEI: {e}")
            raise
