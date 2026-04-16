from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from django.test import SimpleTestCase

from chatbot.services.chatbot_service import ChatbotService
from chatbot.services.constants import XRAY_RESPIRATORY_DOMAIN_CONTEXT_VI
from chatbot.services.gemini_manager import GeminiAPIManager, reset_gemini_manager


class GeminiProviderSelectionTests(SimpleTestCase):
    """Tests for lazy provider selection and OpenRouter path."""

    def tearDown(self) -> None:
        reset_gemini_manager()

    @patch.object(
        GeminiAPIManager,
        "_load_api_keys",
        side_effect=AssertionError("Gemini keys must not load in constructor"),
    )
    def test_manager_init_is_lazy(self, _mock_load_keys: MagicMock) -> None:
        """Constructing the manager should not touch Gemini secrets."""
        manager = GeminiAPIManager()
        self.assertEqual(manager.api_keys, [])

    @patch("chatbot.services.gemini_manager.ChatOpenAI")
    @patch("chatbot.services.gemini_manager.ChatbotConfig.get_config")
    def test_openrouter_llm_does_not_touch_gemini_path(
        self,
        mock_get_config: MagicMock,
        mock_chat_openai: MagicMock,
    ) -> None:
        """OpenRouter selection must not trigger Gemini key resolution."""
        manager = GeminiAPIManager()
        mock_chat_openai.return_value = MagicMock(name="openrouter-llm")
        config = {
            "TEMPERATURE": 0.2,
            "LLM_MODEL": "openai/gpt-4.1-mini",
        }
        mock_get_config.side_effect = lambda key, default=None: config.get(key, default)

        with patch.object(
            manager,
            "_resolve_llm_provider",
            return_value="openrouter",
        ), patch.object(
            manager,
            "_resolve_openrouter_api_key",
            return_value="test-key",
        ), patch.object(
            manager,
            "_resolve_openrouter_base_url",
            return_value="https://openrouter.ai/api/v1",
        ), patch.object(
            manager,
            "_ensure_gemini_api_keys",
            side_effect=AssertionError("Gemini path must not be used"),
        ):
            llm = manager.create_llm()

        self.assertIs(llm, mock_chat_openai.return_value)
        mock_chat_openai.assert_called_once()


class XRayContextTests(SimpleTestCase):
    """Tests for X-ray prompt context integration."""

    def test_build_xray_context_contains_domain_signal(self) -> None:
        """Built X-ray context should include domain bias and findings."""
        service = object.__new__(ChatbotService)
        analysis = SimpleNamespace(
            pred_label="COVID",
            class_probs={"COVID": 0.91, "Normal": 0.05, "Viral Pneumonia": 0.04},
            findings=["bilateral involvement", "diffuse involvement"],
        )

        context = ChatbotService._build_xray_context(service, analysis)

        self.assertIn(XRAY_RESPIRATORY_DOMAIN_CONTEXT_VI.strip(), context)
        self.assertIn("COVID", context)
        self.assertIn("bilateral involvement", context)

    async def test_aquery_passes_xray_context_into_chain(self) -> None:
        """Async query flow should pass the resolved X-ray context into the chain."""
        service = object.__new__(ChatbotService)
        service._chain = MagicMock()
        service._chain.ainvoke = AsyncMock(return_value="ok")
        service._save_message = MagicMock()
        service._apply_analysis_to_intake = MagicMock()
        service._analyze_query = MagicMock(
            return_value={
                "q_cleaned": "tôi bị sốt và ho",
                "q_symptom": "sốt, ho",
                "positives": {"SYMPTOM": ["sốt", "ho"], "ETIOLOGY": [], "RISK": []},
                "negatives": {"SYMPTOM": []},
                "disease_mentions": [],
                "patient_state_extract": {},
            }
        )
        service._get_chat_history = MagicMock(return_value=[])
        service._get_xray_analysis_payload = MagicMock(
            return_value={
                "context": "XRAY-CONTEXT",
                "serialized": {"id": 7},
                "user_metadata": {"attachment": "/media/xray.png"},
            }
        )
        service._last_audit = {"audit_id": "audit-1", "mode": "multi-disease"}

        response = await ChatbotService.aquery(
            service,
            "Tôi bị sốt và ho",
            xray_analysis_id=7,
        )

        self.assertEqual(response, "ok")
        self.assertEqual(
            service._chain.ainvoke.await_args.args[0]["xray_context"],
            "XRAY-CONTEXT",
        )
        self.assertIn("analysis", service._chain.ainvoke.await_args.args[0])
        service._analyze_query.assert_called_once_with("Tôi bị sốt và ho")
        service._apply_analysis_to_intake.assert_called_once()


class RouterAndRetrievalTests(SimpleTestCase):
    """Tests for C/A/B routing and multi-stage retrieval behavior."""

    def test_get_last_source_urls_dedupes_and_extracts_nested_metadata(self) -> None:
        service = object.__new__(ChatbotService)
        service._last_docs_cache = [
            SimpleNamespace(
                metadata={"url": "https://example.org/covid-19"},
                page_content="doc 1",
            ),
            SimpleNamespace(
                metadata={"metadata": {"url": "https://example.org/covid-19"}},
                page_content="doc 2",
            ),
            SimpleNamespace(
                metadata={"source_url": "https://example.org/prevention"},
                page_content="doc 3",
            ),
        ]

        urls = ChatbotService.get_last_source_urls(service)

        self.assertEqual(urls, ["https://example.org/covid-19", "https://example.org/prevention"])

    def test_filter_generic_symptom_terms_removes_meta_terms(self) -> None:
        service = object.__new__(ChatbotService)
        filtered = ChatbotService._filter_generic_symptom_terms(
            service,
            ["triệu chứng", "dấu hiệu", "sốt", "triệu chứng: ho", "covid 19"],
            disease_mentions={"covid 19"},
        )
        self.assertEqual(filtered, ["sốt", "ho"])

    @patch("chatbot.services.chatbot_service.ChatbotConfig.get_config")
    def test_gate_with_index_c_uses_q_cleaned_threshold(
        self, mock_get_config: MagicMock
    ) -> None:
        config = {"RAG_THRESH_C": 0.8}
        mock_get_config.side_effect = lambda key, default=None: config.get(key, default)

        service = object.__new__(ChatbotService)
        service.vector_manager = MagicMock()
        service._user_intake_db = SimpleNamespace(symptoms=[], age=None, sex="unknown")
        service.vector_manager.embedding_service = None
        service.vector_manager.search_similar.return_value = [
            SimpleNamespace(
                distance=0.3,  # score = 0.85
                title="covid-19",
                content="covid-19",
                metadata={"canonical_title": "covid-19"},
            ),
        ]

        gate = ChatbotService._gate_with_index_c(service, "triệu chứng covid-19")

        self.assertTrue(gate["go_single"])
        self.assertEqual(gate["reason"], "above_0.8")
        self.assertEqual(gate["title"], "covid-19")
        self.assertGreaterEqual(gate["top_score"], 0.8)

    @patch("chatbot.services.chatbot_service.ChatbotConfig.get_config")
    def test_multi_disease_retrieval_builds_candidates_and_summary(
        self, mock_get_config: MagicMock
    ) -> None:
        config = {
            "RAG_B_TOPK": 5,
            "RAG_MERGED_LIMIT": 10,
            "RAG_TITLE_TOP_M": 3,
            "RAG_FINAL_TITLES": 2,
            "RAG_MERGE_WEIGHT_ENTITIES": 0.5,
            "RAG_MERGE_WEIGHT_QUERY": 0.5,
            "RAG_NEG_SYM_SIM_THRESH": 0.75,
            "RAG_PENALTY_ALPHA": 0.5,
        }
        mock_get_config.side_effect = lambda key, default=None: config.get(key, default)

        service = object.__new__(ChatbotService)
        service.vector_manager = MagicMock()
        service._user_intake_db = SimpleNamespace(
            symptoms=["sốt"],
            age=25,
            sex="female",
        )
        service.vector_manager.embedding_service = None

        b_docs_query_a = [
            SimpleNamespace(
                distance=0.2,
                title="bệnh sởi",
                content="sốt cao, phát ban, viêm kết mạc",
                section_type="symptom",
                source="admin_url",
                metadata={"canonical_title": "bệnh sởi", "section": "symptom"},
            ),
        ]
        b_docs_query_b = [
            SimpleNamespace(
                distance=0.25,
                title="bệnh sởi",
                content="virus sởi lây qua đường hô hấp",
                section_type="aetiologies",
                source="admin_url",
                metadata={"canonical_title": "bệnh sởi", "section": "aetiologies"},
            ),
        ]
        a_summary_docs = [
            SimpleNamespace(
                distance=0.1,
                title="bệnh sởi",
                content="bệnh sởi là bệnh truyền nhiễm cấp tính do virus sởi.",
                metadata={"canonical_title": "bệnh sởi"},
            )
        ]
        service.vector_manager.search_similar.side_effect = [
            b_docs_query_a,  # Stage 3 query 2a
            b_docs_query_b,  # Stage 3 query 2b
            a_summary_docs,  # Stage 6 summary fetch
        ]

        result = ChatbotService._multi_disease_retrieval(
            service,
            analysis={
                "q_cleaned": "tôi bị sốt phát ban",
                "q_symptom": "sốt, phát ban",
                "positives": {"SYMPTOM": ["sốt", "phát ban"], "ETIOLOGY": [], "RISK": []},
                "negatives": {"SYMPTOM": ["đau bụng"]},
            },
        )

        self.assertGreaterEqual(len(result["candidates"]), 1)
        first = result["candidates"][0]
        self.assertEqual(first["title"], "bệnh sởi")
        self.assertIn("summary", first)
        self.assertTrue(first["final_score"] > 0)


class BenchmarkIntakeTests(SimpleTestCase):
    """Tests for benchmark-only intake replacement semantics."""

    def test_apply_benchmark_intake_payload_replaces_persistent_state(self) -> None:
        service = object.__new__(ChatbotService)
        intake = SimpleNamespace(
            disease_name="bệnh cũ",
            age=72,
            sex="female",
            symptoms=["sốt cũ"],
            symptoms_negated=["không đau đầu"],
            onset_days=9,
            pregnancy_status="yes",
            location_country="VN",
            chronic_conditions=["hen suyễn"],
            allergies=["penicillin"],
            meds=["thuốc cũ"],
            save=MagicMock(),
        )
        service._user_intake_db = intake

        ChatbotService._apply_benchmark_intake_payload(
            service,
            {
                "age": 25,
                "sex": "male",
                "symptoms": ["sốt", "sốt", " "],
                "meds": ["paracetamol"],
            },
        )

        self.assertIsNone(intake.disease_name)
        self.assertEqual(intake.age, 25)
        self.assertEqual(intake.sex, "male")
        self.assertEqual(intake.symptoms, ["sốt"])
        self.assertEqual(intake.symptoms_negated, [])
        self.assertIsNone(intake.onset_days)
        self.assertIsNone(intake.pregnancy_status)
        self.assertIsNone(intake.location_country)
        self.assertEqual(intake.chronic_conditions, [])
        self.assertEqual(intake.allergies, [])
        self.assertEqual(intake.meds, ["paracetamol"])
        intake.save.assert_called_once_with()

    def test_run_benchmark_case_applies_empty_payload_reset(self) -> None:
        service = object.__new__(ChatbotService)
        service._apply_benchmark_intake_payload = MagicMock()
        service._analyze_query = MagicMock(side_effect=RuntimeError("boom"))
        service._last_audit = {}

        result = ChatbotService.run_benchmark_case(
            service,
            question="test benchmark",
            intake_payload={},
        )

        service._apply_benchmark_intake_payload.assert_called_once_with({})
        self.assertEqual(result["generation_output"]["error"], "boom")
