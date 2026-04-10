from unittest.mock import MagicMock, patch

from django.contrib.admin.sites import AdminSite
from django.contrib.auth import get_user_model
from django.contrib.messages.storage.fallback import FallbackStorage
from django.contrib.sessions.middleware import SessionMiddleware
from django.test import RequestFactory, TestCase

from chatbot.admin import ExtendedMedicalDocumentAdmin
from chatbot.forms import CovidQAEmbedForm
from chatbot.models import EmbeddingJob, MedicalDocument


class CovidQAEmbedFormTests(TestCase):
    def test_valid_num_articles(self):
        form = CovidQAEmbedForm(data={"num_articles": 147, "start_article": 1})
        self.assertTrue(form.is_valid())

    def test_invalid_num_articles_above_limit(self):
        form = CovidQAEmbedForm(data={"num_articles": 148, "start_article": 1})
        self.assertFalse(form.is_valid())
        self.assertIn("num_articles", form.errors)

    def test_invalid_num_articles_below_limit(self):
        form = CovidQAEmbedForm(data={"num_articles": 0, "start_article": 1})
        self.assertFalse(form.is_valid())
        self.assertIn("num_articles", form.errors)

    def test_invalid_range_when_start_plus_count_exceeds_limit(self):
        form = CovidQAEmbedForm(data={"num_articles": 10, "start_article": 140})
        self.assertFalse(form.is_valid())
        self.assertIn("__all__", form.errors)


class ExtendedMedicalDocumentAdminTests(TestCase):
    def setUp(self):
        self.site = AdminSite()
        self.admin = ExtendedMedicalDocumentAdmin(MedicalDocument, self.site)
        self.factory = RequestFactory()
        User = get_user_model()
        self.superuser = User.objects.create_superuser(
            username="admin_embed",
            email="admin_embed@example.com",
            password="test-password-123",
        )

    def _attach_messages(self, request):
        session_middleware = SessionMiddleware(lambda req: None)
        session_middleware.process_request(request)
        request.session.save()
        setattr(request, "_messages", FallbackStorage(request))

    @patch("chatbot.admin.process_covid_qa_embed.delay")
    @patch("chatbot.admin.EmbeddingService.resolve_provider")
    @patch("chatbot.admin.ChatbotConfig.get_config")
    def test_embed_covid_qa_view_dispatches_task(
        self,
        mock_get_config,
        mock_resolve_provider,
        mock_delay,
    ):
        mock_get_config.side_effect = lambda key, default=None: {
            "LLM_PROVIDER": "gemini",
            "LLM_MODEL": "gemini-2.5-flash",
        }.get(key, default)
        mock_resolve_provider.return_value = "transformers"
        mock_delay.return_value = MagicMock(id="celery-task-id")

        request = self.factory.post(
            "/admin/chatbot/medicaldocument/embed-covid-qa/",
            data={"num_articles": "25", "start_article": "5"},
        )
        request.user = self.superuser
        self._attach_messages(request)

        response = self.admin.embed_covid_qa_view(request)
        self.assertEqual(response.status_code, 302)

        self.assertEqual(EmbeddingJob.objects.count(), 1)
        job = EmbeddingJob.objects.first()
        self.assertIsNotNone(job)
        if job is not None:
            self.assertEqual(job.provider, "transformers")
            self.assertEqual(job.celery_task_id, "celery-task-id")

        mock_delay.assert_called_once()
        kwargs = mock_delay.call_args.kwargs
        self.assertEqual(kwargs["num_articles"], 25)
        self.assertEqual(kwargs["start_article"], 5)
        self.assertEqual(kwargs["embedding_provider"], "transformers")
        self.assertEqual(kwargs["llm_provider"], "gemini")

    @patch("chatbot.admin.process_covid_qa_embed.delay")
    @patch("chatbot.admin.EmbeddingService.resolve_provider")
    @patch("chatbot.admin.ChatbotConfig.get_config")
    def test_embed_covid_qa_view_uses_openrouter_llm_model_key(
        self,
        mock_get_config,
        mock_resolve_provider,
        mock_delay,
    ):
        mock_get_config.side_effect = lambda key, default=None: {
            "LLM_PROVIDER": "openrouter",
            "LLM_MODEL": "openai/gpt-4.1-mini",
        }.get(key, default)
        mock_resolve_provider.return_value = "openrouter"
        mock_delay.return_value = MagicMock(id="celery-task-id-openrouter")

        request = self.factory.post(
            "/admin/chatbot/medicaldocument/embed-covid-qa/",
            data={"num_articles": "10", "start_article": "1"},
        )
        request.user = self.superuser
        self._attach_messages(request)

        response = self.admin.embed_covid_qa_view(request)
        self.assertEqual(response.status_code, 302)

        mock_delay.assert_called_once()
        kwargs = mock_delay.call_args.kwargs
        self.assertEqual(kwargs["llm_provider"], "openrouter")
        self.assertEqual(kwargs["llm_model"], "openai/gpt-4.1-mini")
