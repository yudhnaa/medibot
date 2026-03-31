"""
Gemini Embedding Provider
Implementation using Google Gemini via langchain-google-genai.
"""

import logging
import os
from typing import override

import numpy as np

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

from chatbot.models import ChatbotConfig
from vector_store.services.constants import (
    DEFAULT_EMBEDDING_MODEL,
    GOOGLE_API_KEY_ENV_NAME,
)
from vector_store.services.providers.embedding_interface import EmbeddingProvider

logger = logging.getLogger(__name__)


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini embedding provider using langchain-google-genai."""

    def __init__(self, model: str | None = None):
        """
        Initialize Gemini embedding provider.

        Args:
            model: Gemini embedding model name
        """
        if not model:
            # Load from database config
            model_name = ChatbotConfig.get_config("GEMINI_EMBEDDING_MODEL")
            if isinstance(model_name, str):
                model = model_name
            else:
                logger.warning(
                    "GEMINI_EMBEDDING_MODEL not found in config, using default."
                )
                model = DEFAULT_EMBEDDING_MODEL

        api_key = os.getenv(GOOGLE_API_KEY_ENV_NAME)
        if not api_key:
            raise ValueError(
                f"{GOOGLE_API_KEY_ENV_NAME} environment variable is required"
            )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model,
            api_key=SecretStr(api_key),
        )
        logger.info(f"Initialized GeminiEmbeddingProvider with model: {model}")

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
            # Matryoshka Representation Learning (MRL) truncation to 768d and L2 normalization
            truncated = raw_embedding[:768]
            arr = np.array(truncated, dtype=float)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            return arr.tolist()
        except Exception as e:
            logger.error(f"Error embedding text with Gemini: {e}")
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
            # Matryoshka Representation Learning (MRL) truncation to 768d and L2 normalization
            normalized_embeddings = []
            for emb in raw_embeddings:
                truncated = emb[:768]
                arr = np.array(truncated, dtype=float)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                normalized_embeddings.append(arr.tolist())
            return normalized_embeddings
        except Exception as e:
            logger.error(f"Error embedding documents with Gemini: {e}")
            raise
