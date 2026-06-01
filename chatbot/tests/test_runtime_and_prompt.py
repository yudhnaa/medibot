from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from django.test import SimpleTestCase

from langchain_core.documents import Document

from chatbot.models import UserIntake
from chatbot.services.chatbot_benchmark_service import ChatbotBenchmarkService
from chatbot.services.chatbot_service import ChatbotService
from chatbot.services.constants import XRAY_RESPIRATORY_DOMAIN_CONTEXT_VI
from chatbot.services.gemini_manager import GeminiAPIManager, reset_gemini_manager
from vision.models.xray_analysis import XRayAnalysis


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

        with (
            patch.object(
                manager,
                "_resolve_llm_provider",
                return_value="openrouter",
            ),
            patch.object(
                manager,
                "_resolve_openrouter_api_key",
                return_value="test-key",
            ),
            patch.object(
                manager,
                "_resolve_openrouter_base_url",
                return_value="https://openrouter.ai/api/v1",
            ),
            patch.object(
                manager,
                "_ensure_gemini_api_keys",
                side_effect=AssertionError("Gemini path must not be used"),
            ),
        ):
            llm = manager.create_llm()

        self.assertIs(llm, mock_chat_openai.return_value)
        mock_chat_openai.assert_called_once()


class XRayContextTests(SimpleTestCase):
    """Tests for X-ray prompt context integration."""

    def test_build_xray_context_contains_domain_signal(self) -> None:
        """Built X-ray context should include domain bias and findings."""
        service = object.__new__(ChatbotService)
        analysis = cast(
            XRayAnalysis,
            SimpleNamespace(
                pred_label="COVID",
                class_probs={"COVID": 0.91, "Normal": 0.05, "Viral Pneumonia": 0.04},
                findings=["bilateral involvement", "diffuse involvement"],
            ),
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
    """Tests for collection routing and multi-stage retrieval behavior."""

    def test_get_last_source_urls_dedupes_and_extracts_nested_metadata(self) -> None:
        service = object.__new__(ChatbotService)
        service._last_docs_cache = [
            Document(
                page_content="doc 1",
                metadata={"url": "https://example.org/covid-19"},
            ),
            Document(
                page_content="doc 2",
                metadata={"metadata": {"url": "https://example.org/covid-19"}},
            ),
            Document(
                page_content="doc 3",
                metadata={"source_url": "https://example.org/prevention"},
            ),
        ]

        urls = ChatbotService.get_last_source_urls(service)

        self.assertEqual(
            urls, ["https://example.org/covid-19", "https://example.org/prevention"]
        )

    def test_filter_generic_symptom_terms_removes_meta_terms(self) -> None:
        service = object.__new__(ChatbotService)
        filtered = ChatbotService._filter_generic_symptom_terms(
            service,
            ["triệu chứng", "dấu hiệu", "sốt", "triệu chứng: ho", "covid 19"],
            disease_mentions={"covid 19"},
        )
        self.assertEqual(filtered, ["sốt", "ho"])

    @patch("chatbot.services.chatbot_service.ChatbotConfig.get_config")
    def test_gate_with_titles_collection_uses_q_cleaned_threshold(
        self, mock_get_config: MagicMock
    ) -> None:
        config = {"RAG_TITLES_COLLECTION_THRESHOLD": 0.8}
        mock_get_config.side_effect = lambda key, default=None: config.get(key, default)

        service = object.__new__(ChatbotService)
        service.vector_manager = MagicMock()
        service._user_intake_db = cast(
            UserIntake,
            SimpleNamespace(symptoms=[], age=None, sex="unknown"),
        )
        service.vector_manager.embedding_service = None
        service.vector_manager.search_collection.return_value = [
            SimpleNamespace(
                distance=0.3,  # score = 0.85
                title="covid-19",
                content="covid-19",
                metadata={"canonical_title": "covid-19"},
            ),
        ]

        gate = ChatbotService._gate_with_titles_collection(
            service, "triệu chứng covid-19"
        )

        self.assertTrue(gate["go_single"])
        self.assertEqual(gate["reason"], "above_0.8")
        self.assertEqual(gate["title"], "covid-19")
        self.assertGreaterEqual(gate["top_score"], 0.8)

    @patch("chatbot.services.chatbot_service.ChatbotConfig.get_config")
    def test_gate_with_disease_mention_uses_mentions_and_soft_threshold(
        self, mock_get_config: MagicMock
    ) -> None:
        config = {"RAG_TITLES_COLLECTION_THRESHOLD": 0.65}
        mock_get_config.side_effect = lambda key, default=None: config.get(key, default)

        service = object.__new__(ChatbotService)
        service.vector_manager = MagicMock()
        service.vector_manager.search_collection.return_value = [
            SimpleNamespace(
                distance=0.76,  # score = 0.62
                title="pleural effusion",
                content="pleural effusion | tràn dịch màng phổi",
                metadata={"canonical_title": "pleural effusion"},
            ),
        ]

        gate = ChatbotService._gate_with_titles_collection(
            service,
            "bạn có biết bệnh tràn dịch màng phổi không?",
            disease_mentions=["tràn dịch màng phổi"],
        )

        service.vector_manager.search_collection.assert_called_once()
        self.assertEqual(
            service.vector_manager.search_collection.call_args.kwargs["query"],
            "tràn dịch màng phổi",
        )
        self.assertTrue(gate["go_single"])
        self.assertEqual(gate["reason"], "disease_mention_above_0.6")
        self.assertEqual(gate["title"], "pleural effusion")

    @patch("chatbot.services.chatbot_service.ChatbotConfig.get_config")
    def test_multi_disease_retrieval_builds_candidates_and_summary(
        self, mock_get_config: MagicMock
    ) -> None:
        config = {
            "RAG_CHUNKS_COLLECTION_TOPK": 5,
            "RAG_MERGED_LIMIT": 10,
            "RAG_TITLE_TOP_M": 3,
            "RAG_FINAL_TITLES": 2,
            "RAG_MERGE_WEIGHT_ENTITIES": 0.5,
            "RAG_MERGE_WEIGHT_QUERY": 0.5,
            "RAG_NEG_SYM_SIM_THRESH": 0.75,
            "RAG_NEG_EMBED_BATCH_SIZE": 32,
            "RAG_PENALTY_ALPHA": 0.5,
        }
        mock_get_config.side_effect = lambda key, default=None: config.get(key, default)

        service = object.__new__(ChatbotService)
        service.vector_manager = MagicMock()
        service._user_intake_db = cast(
            UserIntake,
            SimpleNamespace(
                symptoms=["sốt"],
                age=25,
                sex="female",
            ),
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
        service.vector_manager.search_collection.side_effect = [
            b_docs_query_a,  # Stage 3 query 2a
            b_docs_query_b,  # Stage 3 query 2b
            a_summary_docs,  # Stage 6 summary fetch
        ]

        service._similarity_embedding_cache = {}

        result = ChatbotService._multi_disease_retrieval(
            service,
            analysis={
                "q_cleaned": "tôi bị sốt phát ban",
                "q_symptom": "sốt, phát ban",
                "positives": {
                    "SYMPTOM": ["sốt", "phát ban"],
                    "ETIOLOGY": [],
                    "RISK": [],
                },
                "negatives": {"SYMPTOM": ["đau bụng"]},
            },
        )

        self.assertGreaterEqual(len(result["candidates"]), 1)
        first = result["candidates"][0]
        self.assertEqual(first["title"], "bệnh sởi")
        self.assertIn("summary", first)
        self.assertTrue(first["final_score"] > 0)

    @patch("chatbot.services.chatbot_service.ChatbotConfig.get_config")
    def test_multi_disease_retrieval_orders_evidence_by_final_candidates(
        self,
        mock_get_config: MagicMock,
    ) -> None:
        config = {
            "RAG_CHUNKS_COLLECTION_TOPK": 4,
            "RAG_MERGED_LIMIT": 4,
            "RAG_TITLE_TOP_M": 1,
            "RAG_FINAL_TITLES": 2,
            "RAG_MERGE_WEIGHT_ENTITIES": 1.0,
            "RAG_MERGE_WEIGHT_QUERY": 0.0,
            "RAG_PENALTY_ALPHA": 0.0,
            "RAG_NEG_SYM_SIM_THRESH": 0.7,
            "RAG_NEG_EMBED_BATCH_SIZE": 32,
        }
        mock_get_config.side_effect = lambda key, default=None: config.get(key, default)

        service = object.__new__(ChatbotService)
        service.vector_manager = MagicMock()
        service._user_intake_db = cast(
            UserIntake,
            SimpleNamespace(symptoms=[], age=None, sex="unknown"),
        )
        service.vector_manager.embedding_service = None
        service._similarity_embedding_cache = {}

        low_rank_doc = SimpleNamespace(
            distance=0.8,
            title="covid-19",
            content="đau nhức cơ thể",
            section_type="symptom",
            source="admin_url",
            metadata={
                "canonical_title": "covid-19",
                "section": "symptom",
                "url": "https://example.test/covid",
            },
        )
        top_rank_doc = SimpleNamespace(
            distance=0.1,
            title="vẹo cột sống",
            content="đau lưng âm ỉ",
            section_type="symptom",
            source="admin_url",
            metadata={
                "canonical_title": "vẹo cột sống",
                "section": "symptom",
                "url": "https://example.test/scoliosis",
            },
        )
        service.vector_manager.search_collection.side_effect = [
            [low_rank_doc, top_rank_doc],
            [],
            [],
            [],
        ]

        result = ChatbotService._multi_disease_retrieval(
            service,
            analysis={
                "q_cleaned": "tôi bị đau lưng",
                "q_symptom": "đau lưng",
                "positives": {"SYMPTOM": ["đau lưng"], "ETIOLOGY": [], "RISK": []},
                "negatives": {"SYMPTOM": []},
            },
        )

        self.assertEqual(
            [candidate["title"] for candidate in result["candidates"]],
            ["vẹo cột sống", "covid-19"],
        )
        self.assertEqual(
            [doc.metadata["title"] for doc in result["evidence_docs"]],
            ["vẹo cột sống", "covid-19"],
        )
        self.assertEqual(
            [candidate["source_urls"] for candidate in result["candidates"]],
            [["https://example.test/scoliosis"], ["https://example.test/covid"]],
        )
        self.assertEqual(
            result["source_urls"],
            ["https://example.test/scoliosis", "https://example.test/covid"],
        )

    def test_negation_similarity_primes_unique_texts_in_chunks(self) -> None:
        service = object.__new__(ChatbotService)
        service.vector_manager = MagicMock()
        service._similarity_embedding_cache = {}
        service.vector_manager.embedding_service.embed_texts.side_effect = [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0]],
        ]

        neg_frac = ChatbotService._multi_disease_neg_frac(
            service,
            neg_symptoms=["không sốt", "không sốt"],
            symptom_texts=["không sốt", "ho khan"],
            neg_thresh=0.8,
            batch_size=2,
        )

        self.assertEqual(neg_frac, 1.0)
        self.assertEqual(
            service.vector_manager.embedding_service.embed_texts.call_count, 1
        )
        self.assertEqual(
            service.vector_manager.embedding_service.embed_texts.call_args_list[0].args[
                0
            ],
            ["không sốt", "ho khan"],
        )

    def test_negation_similarity_falls_back_to_lexical_on_embedding_failure(
        self,
    ) -> None:
        service = object.__new__(ChatbotService)
        service.vector_manager = MagicMock()
        service._similarity_embedding_cache = {}
        service.vector_manager.embedding_service.embed_texts.side_effect = RuntimeError(
            "boom"
        )
        service.vector_manager.embedding_service.embed_text.side_effect = RuntimeError(
            "boom"
        )

        neg_frac = ChatbotService._multi_disease_neg_frac(
            service,
            neg_symptoms=["đau bụng"],
            symptom_texts=["đau bụng nhiều"],
            neg_thresh=0.5,
            batch_size=2,
        )

        self.assertEqual(neg_frac, 1.0)


class IntakeRoutingTests(SimpleTestCase):
    def _service_with_intake(self) -> ChatbotService:
        service = object.__new__(ChatbotService)
        service._user_intake_db = cast(
            UserIntake,
            SimpleNamespace(
                disease_name="cúm mùa",
                symptoms=["sốt", "ho"],
                symptoms_negated=["khó thở"],
                age=30,
                sex="female",
                pregnancy_status="no",
                location_country="VN",
                chronic_conditions=["hen suyễn"],
                allergies=["penicillin"],
                onset_days=3,
                meds=["paracetamol"],
                refresh_from_db=MagicMock(),
            ),
        )
        setattr(
            service, "vector_manager", SimpleNamespace(search_collection=MagicMock())
        )
        service._last_docs_cache = []
        service._last_audit = {}
        return service

    def test_greeting_routes_without_retrieval_or_covid_context(self) -> None:
        service = self._service_with_intake()

        analysis = ChatbotService._fallback_query_analysis_heuristic(
            service, "xin chào"
        )
        response = ChatbotService._get_non_retrieval_response(service, analysis)

        self.assertEqual(analysis["intent"], "greeting")
        self.assertFalse(analysis["should_retrieve"])
        self.assertNotIn("COVID", response or "")
        self.assertEqual(ChatbotService.get_last_source_urls(service), [])
        vector_manager = cast(Any, service.vector_manager)
        vector_manager.search_collection.assert_not_called()

    def test_intake_query_uses_saved_user_intake_fields(self) -> None:
        service = self._service_with_intake()

        analysis = ChatbotService._fallback_query_analysis_heuristic(
            service,
            "Thông tin intake của tôi là gì?",
        )
        response = ChatbotService._get_non_retrieval_response(service, analysis)

        self.assertEqual(analysis["intent"], "intake_query")
        self.assertFalse(analysis["should_retrieve"])
        assert response is not None
        self.assertIn("## Thông tin intake hiện có của bạn", response)
        self.assertIn("### Thông tin bệnh nhân", response)
        self.assertIn("- Bệnh: cúm mùa", response)
        self.assertIn("- Triệu chứng (+): sốt, ho", response)
        self.assertIn("Bệnh nền: hen suyễn", response)
        self.assertIn("Dị ứng: penicillin", response)
        self.assertIn("Số ngày khởi phát: 3", response)
        self.assertNotIn("không có quyền truy cập", response.lower())
        vector_manager = cast(Any, service.vector_manager)
        vector_manager.search_collection.assert_not_called()

    def test_analyzer_intake_route_overrides_saved_symptom_medical_signal(self) -> None:
        service = self._service_with_intake()

        analysis = ChatbotService._add_response_route(
            service,
            {
                "intent": "intake_query",
                "response_mode": "intake",
                "should_retrieve": False,
                "disease_mentions": [],
                "positives": {"SYMPTOM": [], "ETIOLOGY": [], "RISK": []},
                "negatives": {"SYMPTOM": []},
            },
            "Bạn có nắm các thông tin cơ bản về tôi không?",
        )
        response = ChatbotService._get_non_retrieval_response(service, analysis)

        self.assertEqual(analysis["intent"], "intake_query")
        self.assertFalse(analysis["should_retrieve"])
        self.assertIn("Tuổi: 30", response or "")
        self.assertEqual(ChatbotService.get_last_source_urls(service), [])

    def test_vietnamese_basic_info_questions_route_to_intake_in_fallback(self) -> None:
        service = self._service_with_intake()

        for question in (
            "Tôi bao nhiêu tuổi?",
            "Bạn có nắm các thông tin cơ bản về tôi không?",
            "Bạn biết tôi bao nhiều tuổi không?",
            "Bạn biết tôi bao nhiêu tuổi không?",
        ):
            analysis = ChatbotService._fallback_query_analysis_heuristic(
                service,
                question,
            )
            response = ChatbotService._get_non_retrieval_response(service, analysis)

            self.assertEqual(analysis["intent"], "intake_query")
            self.assertFalse(analysis["should_retrieve"])
            self.assertIn("Tuổi: 30", response or "")
            self.assertEqual(ChatbotService.get_last_source_urls(service), [])

    def test_medical_query_still_allows_retrieval(self) -> None:
        service = self._service_with_intake()

        analysis = ChatbotService._add_response_route(
            service,
            {
                "disease_mentions": [],
                "positives": {"SYMPTOM": ["sốt"], "ETIOLOGY": [], "RISK": []},
                "negatives": {"SYMPTOM": []},
            },
            "Tôi bị sốt và ho",
        )

        self.assertEqual(analysis["intent"], "medical_query")
        self.assertTrue(analysis["should_retrieve"])
        self.assertIsNone(ChatbotService._get_non_retrieval_response(service, analysis))


class BenchmarkIntakeTests(SimpleTestCase):
    """Tests for benchmark-only intake replacement semantics."""

    def test_apply_benchmark_intake_payload_replaces_persistent_state(self) -> None:
        service = object.__new__(ChatbotBenchmarkService)
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
        service._user_intake_db = cast(UserIntake, intake)

        ChatbotBenchmarkService._apply_benchmark_intake_payload(
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
        service = object.__new__(ChatbotBenchmarkService)
        service._apply_benchmark_intake_payload = MagicMock()
        service._analyze_query = MagicMock(side_effect=RuntimeError("boom"))
        service._last_audit = {}

        result = ChatbotBenchmarkService.run_benchmark_case(
            service,
            question="test benchmark",
            intake_payload={},
        )

        service._apply_benchmark_intake_payload.assert_called_once_with({})
        self.assertEqual(result["generation_output"]["error"], "boom")


class BenchmarkLanguagePolicyTests(SimpleTestCase):
    @patch("chatbot.services.chatbot_service.ChatbotConfig.get_config")
    def test_get_benchmark_output_language_normalizes_alias(
        self,
        mock_get_config: MagicMock,
    ) -> None:
        mock_get_config.return_value = "english"
        service = object.__new__(ChatbotBenchmarkService)

        output_language = ChatbotBenchmarkService._get_benchmark_output_language(
            service,
            "What causes COVID-19?",
        )

        self.assertEqual(output_language, "en")

    def test_build_benchmark_answer_policy_includes_english_rule(self) -> None:
        service = object.__new__(ChatbotBenchmarkService)

        policy = ChatbotBenchmarkService._build_benchmark_answer_policy(
            service,
            question="Is cancer a risk factor for COVID-19?",
            retrieval_output={"rerank": {"insufficient_evidence": False}},
            output_language="en",
        )

        self.assertIn("Output language: English only.", policy)
        self.assertIn(
            "For single-aspect questions, use 1-2 concise evidence-based sentences.",
            policy,
        )
        self.assertIn("start with `Yes.` or `No.`", policy)

    def test_build_benchmark_answer_policy_adds_what_is_definition_rule(self) -> None:
        service = object.__new__(ChatbotBenchmarkService)

        policy = ChatbotBenchmarkService._build_benchmark_answer_policy(
            service,
            question="What is a cough a symptom of?",
            retrieval_output={
                "rerank": {
                    "insufficient_evidence": False,
                    "intent": {"target_sections": ["symptom", "general"]},
                }
            },
            output_language="en",
        )

        self.assertIn(
            "For 'what is' questions, answer in direct definitional form", policy
        )
        self.assertIn("Prefer sentence shape", policy)

    def test_build_benchmark_answer_policy_adds_role_explanation_rule(self) -> None:
        service = object.__new__(ChatbotBenchmarkService)

        policy = ChatbotBenchmarkService._build_benchmark_answer_policy(
            service,
            question="What role does fever play as a symptom of COVID-19?",
            retrieval_output={
                "rerank": {
                    "insufficient_evidence": False,
                    "intent": {"target_sections": ["symptom"]},
                }
            },
            output_language="en",
        )

        self.assertIn("For role/explanation questions", policy)

    def test_build_benchmark_answer_policy_adds_what_is_risk_phrase_rule(self) -> None:
        service = object.__new__(ChatbotBenchmarkService)

        policy = ChatbotBenchmarkService._build_benchmark_answer_policy(
            service,
            question="What is the risk of COVID-19 for older people?",
            retrieval_output={
                "rerank": {
                    "insufficient_evidence": False,
                    "intent": {"target_sections": ["risk", "general"]},
                }
            },
            output_language="en",
        )

        self.assertIn("The risk for older people is higher", policy)

    def test_build_benchmark_answer_policy_adds_multi_aspect_coverage_rule(
        self,
    ) -> None:
        service = object.__new__(ChatbotBenchmarkService)

        policy = ChatbotBenchmarkService._build_benchmark_answer_policy(
            service,
            question=(
                "What are the key characteristics of COVID-19, including its cause, "
                "symptoms, and prevention?"
            ),
            retrieval_output={
                "rerank": {
                    "insufficient_evidence": False,
                    "intent": {
                        "target_sections": [
                            "general",
                            "aetiologies",
                            "symptom",
                            "living_and_preventive",
                        ]
                    },
                }
            },
            output_language="en",
        )

        self.assertIn(
            "This is a multi-aspect question: cover each asked aspect", policy
        )
        self.assertIn("Use 2-4 sentences", policy)
        self.assertIn("Focus only on evidence relevant to sections", policy)
        self.assertIn(
            "When cause is asked, explicitly mention SARS-CoV-2 as the cause.", policy
        )
        self.assertIn(
            "For prevention-focused questions, include vaccination, masking, and distancing",
            policy,
        )

    def test_detect_benchmark_intent_includes_risk_for_older_people(self) -> None:
        service = object.__new__(ChatbotBenchmarkService)

        intent = ChatbotBenchmarkService._detect_benchmark_intent(
            service,
            "What is the risk of COVID-19 for older people?",
        )

        self.assertIn("risk", intent["target_sections"])
        self.assertIn("general", intent["target_sections"])
        self.assertTrue(
            any(term in intent["matched_terms"] for term in ["risk", "older people"])
        )

    def test_build_benchmark_answer_policy_includes_risk_specific_rule(self) -> None:
        service = object.__new__(ChatbotBenchmarkService)

        policy = ChatbotBenchmarkService._build_benchmark_answer_policy(
            service,
            question="What is the risk of COVID-19 for older people?",
            retrieval_output={
                "rerank": {
                    "insufficient_evidence": False,
                    "intent": {"target_sections": ["risk", "general"]},
                }
            },
            output_language="en",
        )

        self.assertTrue(
            (
                "When describing risk, explicitly state that older people are at higher risk of becoming seriously ill."
                in policy
            )
            or ("The risk for older people is higher" in policy)
        )

    def test_shape_benchmark_answer_for_what_is_symptom_question(self) -> None:
        service = object.__new__(ChatbotBenchmarkService)

        shaped = ChatbotBenchmarkService._shape_benchmark_answer(
            service,
            question="What is a cough a symptom of?",
            answer="Cough is a symptom of COVID-19.",
            retrieval_output={
                "rerank": {"intent": {"target_sections": ["symptom", "general"]}}
            },
            output_language="en",
        )

        self.assertEqual(shaped, "COVID-19 is what cough is a symptom of.")

    def test_shape_benchmark_answer_for_what_is_risk_question(self) -> None:
        service = object.__new__(ChatbotBenchmarkService)

        shaped = ChatbotBenchmarkService._shape_benchmark_answer(
            service,
            question="What is the risk of COVID-19 for older people?",
            answer="Older people are at higher risk of becoming seriously ill from COVID-19.",
            retrieval_output={
                "rerank": {"intent": {"target_sections": ["risk", "general"]}}
            },
            output_language="en",
        )

        self.assertIn("The risk for older people is higher", shaped)

    def test_shape_benchmark_answer_for_role_question(self) -> None:
        service = object.__new__(ChatbotBenchmarkService)

        shaped = ChatbotBenchmarkService._shape_benchmark_answer(
            service,
            question="What role does fever play as a symptom of COVID-19?",
            answer="Fever is listed as one of the top three most common symptoms of COVID-19.",
            retrieval_output={"rerank": {"intent": {"target_sections": ["symptom"]}}},
            output_language="en",
        )

        self.assertIn("Fever plays the role", shaped)

    def test_shape_benchmark_answer_for_main_cause_question(self) -> None:
        service = object.__new__(ChatbotBenchmarkService)

        shaped = ChatbotBenchmarkService._shape_benchmark_answer(
            service,
            question=(
                "What is the main cause of the disease that is called covid-19, and "
                "what is the name of the virus that causes it, the sars-cov-2?"
            ),
            answer="COVID-19 is caused by the SARS-CoV-2 virus.",
            retrieval_output={
                "rerank": {"intent": {"target_sections": ["aetiologies"]}}
            },
            output_language="en",
        )

        self.assertEqual(shaped, "The main cause of COVID-19 is the SARS-CoV-2 virus.")

    def test_enforce_benchmark_output_language_rewrites_vietnamese_to_english(
        self,
    ) -> None:
        service = object.__new__(ChatbotBenchmarkService)
        service.llm = MagicMock(
            invoke=MagicMock(
                return_value=SimpleNamespace(
                    content="COVID-19 is caused by the SARS-CoV-2 virus."
                )
            )
        )

        rewritten = ChatbotBenchmarkService._enforce_benchmark_output_language(
            service,
            question="What causes COVID-19?",
            answer="COVID-19 là bệnh do virus SARS-CoV-2 gây ra.",
            output_language="en",
        )

        self.assertEqual(rewritten, "COVID-19 is caused by the SARS-CoV-2 virus.")
        service.llm.invoke.assert_called_once()
