"""
NER and Negation detection integrator for medical text processing.
Combines NER, negation detection, and text normalization into a unified pipeline.
"""

import json
from typing import Any

from nlp.services.medical_ner import MedicalNER
from nlp.services.negation_detector import VietnameseNegationDetector
from nlp.services.text_normalizer import TextNormalizer
from utils.logger import get_logger

logger = get_logger(__name__)


class NERNegationIntegrator:
    """
    Integrates NER and negation detection for medical text processing.
    Processes text through: normalization -> NER -> negation detection -> entity classification.
    """

    def __init__(
        self,
        ner_model_path: str | None = None,
        enable_text_normalization: bool = True,
    ) -> None:
        """
        Initialize the integrator.

        Args:
            ner_model_path: Path to NER model. If None, uses default.
            enable_text_normalization: Whether to enable text normalization.
        """
        self.ner = MedicalNER(ner_model_path)
        self.negation_detector = VietnameseNegationDetector()
        self.enable_text_normalization = enable_text_normalization
        self.text_normalizer: TextNormalizer | None = None

        if self.enable_text_normalization:
            try:
                self.text_normalizer = TextNormalizer()
                logger.info("Text normalizer initialized successfully")
            except Exception as e:
                logger.warning("Failed to initialize text normalizer: %s", e)
                self.text_normalizer = None
                self.enable_text_normalization = False

    def process_text(self, text: str, include_debug: bool = False) -> dict[str, Any]:
        """
        Process text through the complete NER + negation pipeline.

        Args:
            text: Input text to process
            include_debug: Whether to include debug information

        Returns:
            Dictionary with entities, negation info, and processing metadata
        """
        original_text = text
        normalization_info: dict[str, Any]

        # Text normalization (if enabled)
        if self.enable_text_normalization and self.text_normalizer:
            try:
                norm_result = self.text_normalizer.process_text(
                    text, include_variants=False
                )
                processed_text = norm_result["processed"]
                normalization_info = {
                    "enabled": True,
                    "original": original_text,
                    "variants": norm_result,
                }
                text = processed_text
                logger.debug(
                    "Text normalized: '%s' -> '%s'", original_text, processed_text
                )
            except Exception as e:
                logger.warning("Text normalization failed: %s", e)
                normalization_info = {
                    "enabled": True,
                    "error": str(e),
                    "original": original_text,
                }
        else:
            normalization_info = {"enabled": False}

        # Run NER
        entities = self.ner.predict(text)

        # Detect negation
        negation_info = self.negation_detector.detect_negation(text)

        # Classify entities by negation status
        all_entities: list[dict[str, Any]] = []
        negated_entities: list[dict[str, Any]] = []
        non_negated_entities: list[dict[str, Any]] = []

        negation_scopes = negation_info["negation_scopes"]

        for entity in entities:
            entity_span = entity["span"]
            label = self.negation_detector.is_entity_negated(
                entity_span, negation_scopes
            )

            entity_with_negation = entity.copy()
            entity_with_negation["is_negated"] = label
            all_entities.append(entity_with_negation)

            if label == "negated":
                negated_entities.append(entity_with_negation)
            else:
                non_negated_entities.append(entity_with_negation)

        result: dict[str, Any] = {
            "input_text": original_text,
            "processed_text": text,
            "entities": {
                "all": all_entities,
                "negated": negated_entities,
                "non_negated": non_negated_entities,
                "total_count": len(all_entities),
                "negated_count": len(negated_entities),
                "non_negated_count": len(non_negated_entities),
            },
            "negation_info": {
                "has_negation": len(negation_info["negation_scopes"]) > 0,
                "negation_cues": negation_info["negation_cues"],
                "negation_scopes": negation_info["negation_scopes"],
            },
            "normalization_info": normalization_info,
        }

        # Update is_negated for all entities
        for entity in result["entities"]["all"]:
            entity_span = entity["span"]
            entity["is_negated"] = self.negation_detector.is_entity_negated(
                entity_span, negation_scopes
            )

        if include_debug:
            result["debug"] = {
                "ner_raw_output": entities,
                "negation_raw_output": negation_info,
                "processing_steps": [
                    "0. Text normalization and preprocessing",
                    "1. Extracted entities using Medical NER",
                    "2. Detected negation cues and scopes",
                    "3. Mapped entities to negation scopes",
                    "4. Classified entities as negated/non-negated",
                ],
            }

        return result

    def get_entities_by_label(
        self,
        result: dict[str, Any],
        label: str,
        include_negated: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get entities filtered by label.

        Args:
            result: Processing result from process_text
            label: Entity label to filter by
            include_negated: Whether to include negated entities

        Returns:
            List of matching entities
        """
        entities = result["entities"]["all"]
        filtered = [e for e in entities if e["label"] == label]

        if not include_negated:
            filtered = [e for e in filtered if e["is_negated"] != "negated"]

        return filtered

    def get_summary(self, result: dict[str, Any]) -> dict[str, Any]:
        """
        Get a summary of the processing result.

        Args:
            result: Processing result from process_text

        Returns:
            Summary dictionary with statistics
        """
        entities = result["entities"]["all"]

        label_stats: dict[str, dict[str, int]] = {}
        for entity in entities:
            lab = entity["label"]
            if lab not in label_stats:
                label_stats[lab] = {
                    "total": 0,
                    "negated": 0,
                    "non_negated": 0,
                    "uncertain": 0,
                }

            label_stats[lab]["total"] += 1
            tag = entity["is_negated"]
            if tag == "negated":
                label_stats[lab]["negated"] += 1
            elif tag == "uncertain":
                label_stats[lab]["uncertain"] += 1
                label_stats[lab]["non_negated"] += 1
            else:
                label_stats[lab]["non_negated"] += 1

        return {
            "text_length": len(result["input_text"]),
            "processed_text_length": len(result.get("processed_text", "")),
            "text_normalization_enabled": result["normalization_info"]["enabled"],
            "has_medical_entities": len(entities) > 0,
            "has_negation": result["negation_info"]["has_negation"],
            "total_entities": len(entities),
            "negated_entities": len(result["entities"]["negated"]),
            "non_negated_entities": len(result["entities"]["non_negated"]),
            "negation_cues_count": len(result["negation_info"]["negation_cues"]),
            "label_statistics": label_stats,
            "detected_labels": list(label_stats.keys()),
        }


def format_output_json(result: dict[str, Any], pretty: bool = True) -> str:
    """Format processing result as JSON string."""
    if pretty:
        return json.dumps(result, ensure_ascii=False, indent=2)
    else:
        return json.dumps(result, ensure_ascii=False)
