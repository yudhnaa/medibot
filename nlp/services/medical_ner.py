"""
Medical Named Entity Recognition using PhoBERT-based model.
Identifies medical entities (diseases, drugs, tests, etc.) in Vietnamese text.
"""

import threading
from typing import Any

from django.conf import settings

import torch
from transformers import RobertaForTokenClassification

from nlp.services.constants import (
    MEDICAL_NER_MODEL_PATH,
    MEDICAL_NER_TOKENIZER_PATH,
    NER_LABEL_MAPPING,
)
from nlp.services.libs.VietMed_NER.tokenizer.tokenization_phobert_fast import (
    PhobertTokenizerFast,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class MedicalNER:
    """Medical Named Entity Recognition model wrapper."""

    _TOKENIZER_CACHE: PhobertTokenizerFast | None = None
    _MODEL_CACHE: RobertaForTokenClassification | None = None
    _LABEL_MAP_CACHE: dict[int, str] = {}
    _CACHE_SIGNATURE: tuple[str, str] | None = None
    _CACHE_LOCK = threading.Lock()

    def __init__(
        self,
        model_path: str | None = None,
        tokenizer_path: str | None = None,
    ) -> None:
        """
        Initialize the medical NER model.

        Args:
            model_path: Path to the NER model directory. If None, uses default path.
            tokenizer_path: Path to the PhoBERT tokenizer assets. If None, uses default.
        """
        if model_path is None:
            model_path = getattr(
                settings,
                "MEDICAL_NER_MODEL_PATH",
                MEDICAL_NER_MODEL_PATH,
            )
        if tokenizer_path is None:
            tokenizer_path = getattr(
                settings,
                "MEDICAL_NER_TOKENIZER_PATH",
                MEDICAL_NER_TOKENIZER_PATH,
            )

        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.tokenizer: PhobertTokenizerFast | None = None
        self.model: RobertaForTokenClassification | None = None
        self.label_map: dict[int, str] = {}
        self._load_model()

    @classmethod
    def clear_cache(cls) -> None:
        """Clear process-wide model/tokenizer cache for tests or reloads."""
        cls._TOKENIZER_CACHE = None
        cls._MODEL_CACHE = None
        cls._LABEL_MAP_CACHE = {}
        cls._CACHE_SIGNATURE = None

    def _load_model(self) -> None:
        """Load model and tokenizer."""
        cache_signature = (self.model_path, self.tokenizer_path)
        if self.__class__._CACHE_SIGNATURE == cache_signature:
            self._apply_cached_components()
            return

        with self.__class__._CACHE_LOCK:
            if self.__class__._CACHE_SIGNATURE == cache_signature:
                self._apply_cached_components()
                return

            try:
                tokenizer = PhobertTokenizerFast.from_pretrained(
                    self.tokenizer_path,
                    local_files_only=True,
                )
                model = RobertaForTokenClassification.from_pretrained(self.model_path)
                model.eval()

                config = model.config
                label_map: dict[int, str] = {}
                if config.id2label:
                    label_map = {int(k): v for k, v in config.id2label.items()}

                self.__class__._TOKENIZER_CACHE = tokenizer
                self.__class__._MODEL_CACHE = model
                self.__class__._LABEL_MAP_CACHE = label_map
                self.__class__._CACHE_SIGNATURE = cache_signature
                self._apply_cached_components()

                logger.info(
                    "Medical NER runtime loaded from model=%s tokenizer=%s",
                    self.model_path,
                    self.tokenizer_path,
                )
                logger.debug("Using local PhobertTokenizerFast with offset mapping")

            except Exception as e:
                logger.error("Failed to load NER model: %s", e)
                raise

    def _apply_cached_components(self) -> None:
        """Attach cached tokenizer/model instances to the wrapper."""
        self.tokenizer = self.__class__._TOKENIZER_CACHE
        self.model = self.__class__._MODEL_CACHE
        self.label_map = dict(self.__class__._LABEL_MAP_CACHE)

    def _create_begin_entity(
        self,
        text: str,
        offset: Any,
        norm_label: str,
        score: Any,
    ) -> dict[str, Any]:
        """Create a new entity starting at B- tag."""
        return {
            "span": (offset[0].item(), offset[1].item()),
            "text": text[offset[0] : offset[1]],
            "label": norm_label,
            "score": score.item(),
        }

    def _extend_current_entity(
        self,
        text: str,
        current_entity: dict[str, Any],
        offset: Any,
        score: Any,
    ) -> None:
        """Extend current entity with I- tag token."""
        current_entity["span"] = (
            current_entity["span"][0],
            offset[1].item(),
        )
        current_entity["text"] = text[
            current_entity["span"][0] : current_entity["span"][1]
        ]
        current_entity["score"] = (current_entity["score"] + score.item()) / 2

    def _should_extend_entity(
        self,
        label: str,
        norm_label: str,
        current_entity: dict[str, Any] | None,
    ) -> bool:
        """Check if the current I- tag should extend the existing entity."""
        return (
            label.startswith("I-")
            and current_entity is not None
            and norm_label == current_entity["label"]
        )

    def _process_token(
        self,
        text: str,
        label: str,
        norm_label: str,
        offset: Any,
        score: Any,
        current_entity: dict[str, Any] | None,
        entities: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Process a single token and update entity state."""
        if label not in ("O", "0"):
            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                return self._create_begin_entity(text, offset, norm_label, score)
            elif self._should_extend_entity(label, norm_label, current_entity):
                self._extend_current_entity(text, current_entity, offset, score)  # type: ignore[arg-type]
                return current_entity
            else:
                if current_entity:
                    entities.append(current_entity)
                return None
        else:
            if current_entity:
                entities.append(current_entity)
            return None

    def predict(self, text: str) -> list[dict[str, Any]]:
        """
        Predict medical entities in text.

        Args:
            text: Input text to analyze.

        Returns:
            List of entities with format:
            [
                {
                    'span': (start, end),
                    'text': 'entity_text',
                    'label': 'DISEASE',
                    'score': 0.95
                }
            ]
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_offsets_mapping=True,
        )

        offsets = inputs.pop("offset_mapping")[0]

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_labels = torch.argmax(predictions, dim=-1)[0]
            confidence_scores = torch.max(predictions, dim=-1)[0][0]

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])  # pyright: ignore[reportIndexIssue]
        special_tokens = set(self.tokenizer.all_special_tokens)
        entities: list[dict[str, Any]] = []
        current_entity: dict[str, Any] | None = None

        for token, label_id, score, offset in zip(
            tokens, predicted_labels, confidence_scores, offsets
        ):
            if token in special_tokens or offset[0] == offset[1]:
                continue

            label = self.label_map.get(int(label_id.item()), "O")
            norm_label = (
                self.normalize_label(label[2:])
                if label.startswith("B-") or label.startswith("I-")
                else label
            )

            current_entity = self._process_token(
                text, label, norm_label, offset, score, current_entity, entities
            )

        if current_entity:
            entities.append(current_entity)

        return entities

    def normalize_label(self, label: str) -> str:
        """Normalize label from model to standard format."""
        return NER_LABEL_MAPPING.get(label.upper(), label)
