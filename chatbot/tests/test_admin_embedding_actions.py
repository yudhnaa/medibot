from unittest.mock import MagicMock, patch

from django.contrib.admin.sites import AdminSite
from django.contrib.auth import get_user_model
from django.contrib.messages.storage.fallback import FallbackStorage
from django.contrib.sessions.middleware import SessionMiddleware
from django.test import RequestFactory, TestCase

from chatbot.admin import ExtendedMedicalDocumentAdmin
from chatbot.forms import ArticleUrlEmbedForm
from chatbot.models import EmbeddingJob, MedicalDocument


class ArticleUrlEmbedFormTests(TestCase):
    def test_valid_url(self):
        form = ArticleUrlEmbedForm(data={"url": "https://example.com/article"})
        self.assertTrue(form.is_valid())

    def test_invalid_url_scheme(self):
        form = ArticleUrlEmbedForm(data={"url": "ftp://example.com/article"})
        self.assertFalse(form.is_valid())
        self.assertIn("url", form.errors)


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

    @patch("chatbot.admin.process_article_url_embed.delay")
    @patch("chatbot.admin.EmbeddingService.resolve_provider")
    def test_embed_url_view_dispatches_task(
        self,
        mock_resolve_provider,
        mock_delay,
    ):
        mock_resolve_provider.return_value = "transformers"
        mock_delay.return_value = MagicMock(id="celery-task-id")

        request = self.factory.post(
            "/admin/chatbot/medicaldocument/embed-url/",
            data={"url": "https://example.com/medical-article"},
        )
        request.user = self.superuser
        self._attach_messages(request)

        response = self.admin.embed_url_view(request)
        self.assertEqual(response.status_code, 302)

        self.assertEqual(EmbeddingJob.objects.count(), 1)
        job = EmbeddingJob.objects.first()
        self.assertIsNotNone(job)
        if job is not None:
            self.assertEqual(job.provider, "transformers")
            self.assertEqual(job.celery_task_id, "celery-task-id")

        mock_delay.assert_called_once()
        kwargs = mock_delay.call_args.kwargs
        self.assertEqual(kwargs["url"], "https://example.com/medical-article")
        self.assertEqual(kwargs["embedding_provider"], "transformers")
