from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.contrib.admin.sites import AdminSite
from django.contrib.auth import get_user_model
from django.contrib.messages.storage.fallback import FallbackStorage
from django.contrib.sessions.middleware import SessionMiddleware
from django.test import RequestFactory, TestCase

from chatbot.admin import ExtendedMedicalDocumentAdmin
from chatbot.forms import ArticleUrlEmbedForm, ReembeddingForm
from chatbot.models import EmbeddingJob, MedicalDocument, SectionType
from chatbot.tasks import process_reembedding_job


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

    @patch("chatbot.tasks.process_article_url_embed.delay")
    @patch("vector_store.services.embedding_service.EmbeddingService.resolve_provider")
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

    @patch("chatbot.admin.document_admin.render")
    @patch("vector_store.services.vector_store_manager.VectorStoreManager")
    def test_vector_search_view_deduplicates_and_filters_results(
        self,
        mock_manager_cls,
        mock_render,
    ):
        manager = mock_manager_cls.return_value
        duplicate_doc = SimpleNamespace(
            pk=1,
            title="Sởi",
            content="Sốt phát ban",
            section_type="symptom",
            index_type="B",
            embedding_provider="mock",
            distance=0.1,
        )
        manager.search_similar.return_value = [
            duplicate_doc,
            duplicate_doc,
            SimpleNamespace(
                pk=2,
                title="Cảm lạnh",
                content="Ho",
                section_type="symptom",
                index_type="B",
                embedding_provider="mock",
                distance=0.8,
            ),
        ]
        mock_render.return_value = MagicMock(status_code=200)

        request = self.factory.post(
            "/admin/chatbot/medicaldocument/vector-search/",
            data={
                "query_text": "sốt phát ban",
                "k": "5",
                "section_types": [],
                "min_similarity": "0.5",
            },
        )
        request.user = self.superuser
        self._attach_messages(request)

        response = self.admin.vector_search_view(request)

        self.assertEqual(response.status_code, 200)
        context = mock_render.call_args.args[2]
        self.assertEqual(context["count"], 1)
        self.assertEqual(context["results"][0]["id"], 1)

    def test_reembedding_form_excludes_selected_option(self):
        form = ReembeddingForm()

        self.assertNotIn("selected", dict(form.fields["reembed_type"].choices))

    def test_reembedding_form_requires_section_type_for_section(self):
        form = ReembeddingForm(
            data={
                "reembed_type": "section",
                "batch_size": "100",
                "confirm": "on",
            }
        )

        self.assertFalse(form.is_valid())
        self.assertIn("section_type", form.errors)

    @patch(
        "vector_store.services.reembed_service.ReembeddingService.reembed_by_section"
    )
    @patch("vector_store.services.embedding_service.EmbeddingService.resolve_provider")
    def test_reembed_manager_sync_section_runs_service(
        self,
        mock_resolve_provider,
        mock_reembed_by_section,
    ):
        mock_resolve_provider.return_value = "transformers"

        request = self._post_reembed_request(
            {
                "reembed_type": "section",
                "section_type": SectionType.SYMPTOM,
                "batch_size": "100",
                "confirm": "on",
            }
        )

        response = self.admin.reembed_manager_view(request)

        self.assertEqual(response.status_code, 302)
        job = EmbeddingJob.objects.get()
        self.assertEqual(job.job_type, "reembed_section")
        self.assertEqual(job.provider, "transformers")
        self.assertEqual(job.section_type, SectionType.SYMPTOM)
        mock_reembed_by_section.assert_called_once_with(
            section_type=SectionType.SYMPTOM,
            provider="transformers",
            job=job,
        )

    @patch("vector_store.services.reembed_service.ReembeddingService.reembed_missing")
    @patch("vector_store.services.embedding_service.EmbeddingService.resolve_provider")
    def test_reembed_manager_sync_missing_runs_service(
        self,
        mock_resolve_provider,
        mock_reembed_missing,
    ):
        mock_resolve_provider.return_value = "transformers"

        request = self._post_reembed_request(
            {
                "reembed_type": "missing",
                "batch_size": "100",
                "confirm": "on",
            }
        )

        response = self.admin.reembed_manager_view(request)

        self.assertEqual(response.status_code, 302)
        job = EmbeddingJob.objects.get()
        self.assertEqual(job.job_type, "reembed_missing")
        self.assertIsNone(job.section_type)
        mock_reembed_missing.assert_called_once_with(
            provider="transformers",
            job=job,
        )

    @patch("chatbot.tasks.process_reembedding_job.delay")
    @patch("vector_store.services.embedding_service.EmbeddingService.resolve_provider")
    def test_reembed_manager_async_queues_real_task(
        self,
        mock_resolve_provider,
        mock_delay,
    ):
        mock_resolve_provider.return_value = "transformers"
        mock_delay.return_value = MagicMock(id="celery-reembed-id")

        request = self._post_reembed_request(
            {
                "reembed_type": "missing",
                "batch_size": "100",
                "run_async": "on",
                "confirm": "on",
            }
        )

        response = self.admin.reembed_manager_view(request)

        self.assertEqual(response.status_code, 302)
        job = EmbeddingJob.objects.get()
        self.assertEqual(job.celery_task_id, "celery-reembed-id")
        mock_delay.assert_called_once_with(job.pk)

    def test_process_reembedding_job_handles_missing_job(self):
        result = process_reembedding_job.run(999999)

        self.assertEqual(result["status"], "error")
        self.assertIn("not found", result["message"])

    def _post_reembed_request(self, data):
        request = self.factory.post(
            "/admin/chatbot/medicaldocument/reembed-manager/",
            data=data,
        )
        request.user = self.superuser
        self._attach_messages(request)
        return request
