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
