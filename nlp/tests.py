"""
Tests for NLP services pipeline.
Tests NER, negation detection, text normalization, and the integrator.
"""

from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings

from typing_extensions import override

from nlp.services.medical_ner import MedicalNER
from nlp.services.negation_detector import (
    CueType,
    NegationCue,
    VietnameseNegationDetector,
)
from nlp.services.runtime import get_shared_integrator, reset_nlp_runtime_cache
from nlp.services.text_normalizer import TextNormalizer


class VietnameseNegationDetectorTests(TestCase):
    """Tests for Vietnamese negation detection."""

    detector: VietnameseNegationDetector | None = None

    @override
    def setUp(self) -> None:
        self.detector = VietnameseNegationDetector()

    def _get_detector(self) -> VietnameseNegationDetector:
        """Helper to get detector with type assertion."""
        assert self.detector is not None
        return self.detector

    def test_detect_pre_negation(self) -> None:
        """Test detection of pre-negation cues."""
        text = "Bệnh nhân không đau đầu"
        result = self._get_detector().detect_negation(text)

        self.assertIn("negation_cues", result)
        self.assertIn("negation_scopes", result)
        self.assertTrue(len(result["negation_cues"]) > 0)

    def test_detect_no_negation(self) -> None:
        """Test text without negation."""
        text = "Bệnh nhân bị đau đầu"
        result = self._get_detector().detect_negation(text)

        self.assertEqual(len(result["negation_scopes"]), 0)

    def test_detect_pseudo_negation(self) -> None:
        """Test pseudo-negation detection (không chỉ, không những)."""
        text = "Bệnh nhân không chỉ đau đầu mà còn sốt"
        result = self._get_detector().detect_negation(text)

        # Pseudo negation should be detected as cue but not create scope
        cue_types = [c["cue_type"] for c in result["negation_cues"]]
        self.assertIn("PSEUDO", cue_types)

    def test_detect_uncertain_cues(self) -> None:
        """Test uncertain cue detection."""
        text = "Có thể bệnh nhân bị viêm phổi"
        result = self._get_detector().detect_negation(text)

        cue_types = [c["cue_type"] for c in result["negation_cues"]]
        self.assertIn("UNCERTAIN", cue_types)

    def test_is_entity_negated_affirmed(self) -> None:
        """Test entity not in negation scope is affirmed."""
        entity_span = (20, 30)
        scopes: list[tuple[int, int, CueType]] = [(0, 10, "NEG")]

        label = self._get_detector().is_entity_negated(entity_span, scopes)
        self.assertEqual(label, "affirmed")

    def test_is_entity_negated_negated(self) -> None:
        """Test entity in negation scope is negated."""
        entity_span = (5, 10)
        scopes: list[tuple[int, int, CueType]] = [(0, 15, "NEG")]

        label = self._get_detector().is_entity_negated(entity_span, scopes)
        self.assertEqual(label, "negated")

    def test_is_entity_negated_uncertain(self) -> None:
        """Test entity in uncertain scope."""
        entity_span = (5, 10)
        scopes: list[tuple[int, int, CueType]] = [(0, 15, "UNCERTAIN")]

        label = self._get_detector().is_entity_negated(entity_span, scopes)
        self.assertEqual(label, "uncertain")

    def test_find_negation_cues(self) -> None:
        """Test finding negation cues."""
        text = "không có triệu chứng"
        cues = self._get_detector().find_negation_cues(text)

        self.assertTrue(len(cues) > 0)
        self.assertIsInstance(cues[0], NegationCue)
        self.assertEqual(cues[0].direction, "PRE")

    def test_scope_terminator(self) -> None:
        """Test that scope terminators limit negation scope."""
        text = "Bệnh nhân không đau đầu nhưng sốt cao"
        result = self._get_detector().detect_negation(text)

        # Check that negation scope doesn't extend past "nhung"
        if result["negation_scopes"]:
            scope_end = result["negation_scopes"][0][1]
            nhung_pos = text.find("nhưng")
            self.assertLessEqual(scope_end, nhung_pos + len("nhưng"))


class TextNormalizerTests(TestCase):
    """Tests for text normalization."""

    normalizer: TextNormalizer | None = None

    @override
    def setUp(self) -> None:
        # Mock VnCoreNLP to avoid JVM initialization
        with patch("nlp.services.text_normalizer.py_vncorenlp"):
            self.normalizer = TextNormalizer()
            self.normalizer.vncorenlp_segmenter = None  # Force fallback

    def _get_normalizer(self) -> TextNormalizer:
        """Helper to get normalizer with type assertion."""
        assert self.normalizer is not None
        return self.normalizer

    def test_normalize_text_lowercase(self) -> None:
        """Test text is converted to lowercase."""
        text = "Bệnh Nhân BỊ ĐAU"
        result = self._get_normalizer().normalize_text(text)

        self.assertEqual(result, "bệnh nhân bị đau")

    def test_normalize_text_whitespace(self) -> None:
        """Test extra whitespace is removed."""
        text = "Bệnh   nhân    bị   đau"
        result = self._get_normalizer().normalize_text(text)

        self.assertEqual(result, "bệnh nhân bị đau")

    def test_remove_accents(self) -> None:
        """Test Vietnamese accent removal."""
        text = "bệnh nhân"
        result = self._get_normalizer().remove_accents(text)

        self.assertEqual(result, "benh nhan")

    def test_clean_punctuation(self) -> None:
        """Test punctuation cleaning."""
        text = "triệu chứng; đau đầu!"
        result = self._get_normalizer().clean_punctuation(text, remove_semicolon=True)

        self.assertNotIn(";", result)

    def test_process_text_empty(self) -> None:
        """Test processing empty text."""
        result = self._get_normalizer().process_text("")

        self.assertEqual(result["processed"], "")

    def test_process_text_with_variants(self) -> None:
        """Test processing with variants included."""
        text = "Bệnh nhân bị đau"
        result = self._get_normalizer().process_text(text, include_variants=True)

        self.assertIn("processed", result)
        self.assertIn("original", result)
        self.assertIn("normalized", result)
        self.assertIn("no_accents", result)

    def test_batch_process(self) -> None:
        """Test batch processing multiple texts."""
        texts = ["Đau đầu", "Sốt cao"]
        results = self._get_normalizer().batch_process(texts)

        self.assertEqual(len(results), 2)
        self.assertIn("processed", results[0])
        self.assertIn("processed", results[1])

    def test_get_processing_info(self) -> None:
        """Test getting processing info."""
        info = self._get_normalizer().get_processing_info()

        self.assertIn("vncorenlp_available", info)
        self.assertIn("processing_steps", info)

    def test_segment_text_fallback(self) -> None:
        """Test segmentation fallback when VnCoreNLP unavailable."""
        text = "đau đầu"
        result = self._get_normalizer().segment_text(text)

        # Fallback replaces spaces with underscores
        self.assertEqual(result, "đau_đầu")


class NERNegationIntegratorTests(TestCase):
    """Tests for the NER-Negation integrator."""

    @patch("nlp.services.integrator.MedicalNER")
    @patch("nlp.services.integrator.TextNormalizer")
    def test_init_with_normalization(
        self, mock_normalizer: MagicMock, _mock_ner: MagicMock
    ) -> None:
        """Test integrator initialization with normalization enabled."""
        from nlp.services.integrator import NERNegationIntegrator

        integrator = NERNegationIntegrator(enable_text_normalization=True)

        self.assertTrue(integrator.enable_text_normalization)
        mock_normalizer.assert_called_once()

    @patch("nlp.services.integrator.MedicalNER")
    @patch("nlp.services.integrator.TextNormalizer")
    def test_init_without_normalization(
        self, mock_normalizer: MagicMock, _mock_ner: MagicMock
    ) -> None:
        """Test integrator initialization with normalization disabled."""
        from nlp.services.integrator import NERNegationIntegrator

        integrator = NERNegationIntegrator(enable_text_normalization=False)

        self.assertFalse(integrator.enable_text_normalization)
        mock_normalizer.assert_not_called()

    @patch("nlp.services.integrator.MedicalNER")
    @patch("nlp.services.integrator.TextNormalizer")
    def test_process_text_structure(
        self, mock_normalizer: MagicMock, mock_ner: MagicMock
    ) -> None:
        """Test process_text returns expected structure."""
        from nlp.services.integrator import NERNegationIntegrator

        mock_ner_instance = MagicMock()
        mock_ner_instance.predict.return_value = [
            {"span": (10, 20), "text": "đau đầu", "label": "DISEASE", "score": 0.9}
        ]
        mock_ner.return_value = mock_ner_instance

        mock_norm_instance = MagicMock()
        mock_norm_instance.process_text.return_value = {"processed": "test text"}
        mock_normalizer.return_value = mock_norm_instance

        integrator = NERNegationIntegrator(enable_text_normalization=True)
        result = integrator.process_text("test text")

        self.assertIn("input_text", result)
        self.assertIn("processed_text", result)
        self.assertIn("entities", result)
        self.assertIn("negation_info", result)
        self.assertIn("normalization_info", result)

    @patch("nlp.services.integrator.MedicalNER")
    @patch("nlp.services.integrator.TextNormalizer")
    def test_process_text_with_debug(
        self, mock_normalizer: MagicMock, mock_ner: MagicMock
    ) -> None:
        """Test process_text with debug info included."""
        from nlp.services.integrator import NERNegationIntegrator

        mock_ner_instance = MagicMock()
        mock_ner_instance.predict.return_value = []
        mock_ner.return_value = mock_ner_instance

        mock_norm_instance = MagicMock()
        mock_norm_instance.process_text.return_value = {"processed": "text"}
        mock_normalizer.return_value = mock_norm_instance

        integrator = NERNegationIntegrator()
        result = integrator.process_text("text", include_debug=True)

        self.assertIn("debug", result)
        self.assertIn("processing_steps", result["debug"])

    @patch("nlp.services.integrator.MedicalNER")
    @patch("nlp.services.integrator.TextNormalizer")
    def test_get_entities_by_label(
        self, mock_normalizer: MagicMock, mock_ner: MagicMock
    ) -> None:
        """Test filtering entities by label."""
        from nlp.services.integrator import NERNegationIntegrator

        mock_ner.return_value = MagicMock()
        mock_normalizer.return_value = MagicMock()

        integrator = NERNegationIntegrator(enable_text_normalization=False)

        result = {
            "entities": {
                "all": [
                    {"label": "DISEASE", "is_negated": "affirmed"},
                    {"label": "DRUG", "is_negated": "affirmed"},
                    {"label": "DISEASE", "is_negated": "negated"},
                ]
            }
        }

        diseases = integrator.get_entities_by_label(result, "DISEASE")
        self.assertEqual(len(diseases), 2)

        diseases_non_negated = integrator.get_entities_by_label(
            result, "DISEASE", include_negated=False
        )
        self.assertEqual(len(diseases_non_negated), 1)

    @patch("nlp.services.integrator.MedicalNER")
    @patch("nlp.services.integrator.TextNormalizer")
    def test_get_summary(self, mock_normalizer: MagicMock, mock_ner: MagicMock) -> None:
        """Test getting summary of processing result."""
        from nlp.services.integrator import NERNegationIntegrator

        mock_ner.return_value = MagicMock()
        mock_normalizer.return_value = MagicMock()

        integrator = NERNegationIntegrator(enable_text_normalization=False)

        result = {
            "input_text": "test",
            "processed_text": "test",
            "entities": {
                "all": [
                    {"label": "DISEASE", "is_negated": "affirmed"},
                    {"label": "DISEASE", "is_negated": "negated"},
                ],
                "negated": [{"label": "DISEASE"}],
                "non_negated": [{"label": "DISEASE"}],
            },
            "negation_info": {
                "has_negation": True,
                "negation_cues": [{}],
            },
            "normalization_info": {"enabled": False},
        }

        summary = integrator.get_summary(result)

        self.assertEqual(summary["total_entities"], 2)
        self.assertEqual(summary["negated_entities"], 1)
        self.assertEqual(summary["non_negated_entities"], 1)
        self.assertIn("DISEASE", summary["detected_labels"])


class FormatOutputJsonTests(TestCase):
    """Tests for JSON output formatting."""

    def test_format_output_json_pretty(self) -> None:
        """Test pretty JSON formatting."""
        from nlp.services.integrator import format_output_json

        result = {"key": "value", "nested": {"a": 1}}
        output = format_output_json(result, pretty=True)

        self.assertIn("\n", output)
        self.assertIn("  ", output)

    def test_format_output_json_compact(self) -> None:
        """Test compact JSON formatting."""
        from nlp.services.integrator import format_output_json

        result = {"key": "value"}
        output = format_output_json(result, pretty=False)

        self.assertNotIn("\n", output)

    def test_format_output_json_unicode(self) -> None:
        """Test JSON with Vietnamese unicode."""
        from nlp.services.integrator import format_output_json

        result = {"text": "Bệnh nhân đau đầu"}
        output = format_output_json(result)

        self.assertIn("Bệnh nhân đau đầu", output)


class MedicalNERRuntimeTests(TestCase):
    """Tests for local tokenizer loading and NLP runtime caching."""

    def tearDown(self) -> None:
        MedicalNER.clear_cache()
        reset_nlp_runtime_cache()

    @override_settings(MEDICAL_NER_TOKENIZER_PATH="/tmp/local-tokenizer")
    @patch("nlp.services.medical_ner.RobertaForTokenClassification.from_pretrained")
    @patch("nlp.services.medical_ner.PhobertTokenizerFast.from_pretrained")
    def test_medical_ner_loads_local_tokenizer(
        self,
        mock_tokenizer_loader: MagicMock,
        mock_model_loader: MagicMock,
    ) -> None:
        """Tokenizer must be loaded from local vendor files only."""
        mock_tokenizer_loader.return_value = MagicMock(all_special_tokens=[])
        mock_model = MagicMock()
        mock_model.config.id2label = {0: "O"}
        mock_model_loader.return_value = mock_model

        MedicalNER(model_path="/tmp/local-model")

        mock_tokenizer_loader.assert_called_once_with(
            "/tmp/local-tokenizer",
            local_files_only=True,
        )

    @override_settings(MEDICAL_NER_TOKENIZER_PATH="/tmp/local-tokenizer")
    @patch("nlp.services.medical_ner.RobertaForTokenClassification.from_pretrained")
    @patch("nlp.services.medical_ner.PhobertTokenizerFast.from_pretrained")
    def test_medical_ner_reuses_cached_runtime(
        self,
        mock_tokenizer_loader: MagicMock,
        mock_model_loader: MagicMock,
    ) -> None:
        """Repeated wrapper creation should reuse the same model/tokenizer."""
        mock_tokenizer_loader.return_value = MagicMock(all_special_tokens=[])
        mock_model = MagicMock()
        mock_model.config.id2label = {0: "O"}
        mock_model_loader.return_value = mock_model

        first = MedicalNER(model_path="/tmp/local-model")
        second = MedicalNER(model_path="/tmp/local-model")

        self.assertIs(first.model, second.model)
        self.assertIs(first.tokenizer, second.tokenizer)
        mock_tokenizer_loader.assert_called_once()
        mock_model_loader.assert_called_once()

    @patch("nlp.services.runtime.NERNegationIntegrator")
    def test_get_shared_integrator_reuses_singleton(
        self,
        mock_integrator_cls: MagicMock,
    ) -> None:
        """Shared NLP integrator should initialize once per process."""
        reset_nlp_runtime_cache()
        mock_integrator = MagicMock()
        mock_integrator_cls.return_value = mock_integrator

        first = get_shared_integrator()
        second = get_shared_integrator()

        self.assertIs(first, second)
        mock_integrator_cls.assert_called_once_with(enable_text_normalization=True)
