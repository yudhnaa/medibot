"""
Transformers Embedding Provider
Implementation using Hugging Face Transformers.
"""

import logging
from typing import override

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from chatbot.models import ChatbotConfig

from embeddings.services.constants import (
    DEFAULT_TRANSFORMERS_INSTRUCTION,
    DEFAULT_TRANSFORMERS_MODEL,
)
from embeddings.services.providers.embedding_interface import EmbeddingProvider

logger = logging.getLogger(__name__)


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pooling strategy for Qwen3-Embedding."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


class TransformersEmbeddingProvider(EmbeddingProvider):
    """Generic embedding provider using Transformers."""

    def __init__(
        self,
        model_name: str | None = None,
        instruction: str = DEFAULT_TRANSFORMERS_INSTRUCTION,
    ):
        """
        Initialize Transformers embedding provider.

        Args:
            model_name: HuggingFace model hub name or path.
            instruction: Instruction to prepend to queries (embed_text).
        """
        if not model_name:
            # Load from database config
            db_model = ChatbotConfig.get_config("TRANSFORMERS_EMBEDDING_MODEL")
            if isinstance(db_model, str):
                model_name = db_model
            else:
                logger.warning(
                    "TRANSFORMERS_EMBEDDING_MODEL not found in config, using default."
                )
                model_name = DEFAULT_TRANSFORMERS_MODEL

        self.instruction = instruction

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            f"Initializing TransformersEmbeddingProvider {model_name} on {self.device}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set to evaluation mode

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Internal method to embed a batch of texts."""
        max_length = 8192
        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_dict = batch_dict.to(self.device)

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = last_token_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.tolist()

    @override
    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding vector for a single query text.
        Prepends the configured instruction.
        """
        # "Each query must come with a one-sentence instruction"
        input_text = f"Instruct: {self.instruction}\nQuery: {text}"
        embeddings = self._embed_batch([input_text])
        return embeddings[0]

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embedding vectors for multiple documents.
        Does NOT prepend instruction (assumed to be retrieval documents).
        """
        # "No need to add instruction for retrieval documents"
        return self._embed_batch(texts)
