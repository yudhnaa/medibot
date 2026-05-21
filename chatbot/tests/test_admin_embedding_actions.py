from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

from django.contrib.admin.sites import AdminSite
from django.contrib.auth import get_user_model
from django.contrib.messages.storage.fallback import FallbackStorage
from django.contrib.sessions.middleware import SessionMiddleware
from django.forms import ChoiceField
from django.http import HttpResponse
from django.test import RequestFactory, TestCase

from chatbot.admin import ExtendedMedicalDocumentAdmin, MedicalDocumentTitleAdmin
from chatbot.forms import ArticlePasteEmbedForm, ArticleUrlEmbedForm, ReembeddingForm
from chatbot.models import (
    EmbeddingJob,
    MedicalDiseaseDocument,
    MedicalDocumentChunk,
    MedicalDocumentTitle,
    SectionType,
)
from chatbot.tasks import process_reembedding_job
from vector_store.services.reembed_service import ReembeddingService


class ArticleUrlEmbedFormTests(TestCase):
    def test_valid_url(self):
        form = ArticleUrlEmbedForm(data={"url": "https://example.com/article"})
        self.assertTrue(form.is_valid())

    def test_invalid_url_scheme(self):
        form = ArticleUrlEmbedForm(data={"url": "ftp://example.com/article"})
        self.assertFalse(form.is_valid())
        self.assertIn("url", form.errors)


class ArticlePasteEmbedFormTests(TestCase):
    def test_valid_pasted_content(self):
        form = ArticlePasteEmbedForm(
            data={
                "title": "Scoliosis",
                "source_url": "https://example.com/scoliosis",
                "content": "Scoliosis is a sideways curvature of the spine.",
            }
        )
        self.assertTrue(form.is_valid())

    def test_empty_content(self):
        form = ArticlePasteEmbedForm(
            data={"title": "Scoliosis", "source_url": "", "content": "  "}
        )
        self.assertFalse(form.is_valid())
        self.assertIn("content", form.errors)

    def test_invalid_source_url_scheme(self):
        form = ArticlePasteEmbedForm(
            data={
                "title": "Scoliosis",
                "source_url": "ftp://example.com/scoliosis",
                "content": "Scoliosis is a sideways curvature of the spine.",
            }
        )
        self.assertFalse(form.is_valid())
        self.assertIn("source_url", form.errors)


class MedicalDocumentTitleAdminTests(TestCase):
    def setUp(self):
        self.site = AdminSite()
        self.admin = MedicalDocumentTitleAdmin(MedicalDocumentTitle, self.site)
        self.factory = RequestFactory()
        User = cast(Any, get_user_model())
        self.superuser = User.objects.create_superuser(
            username="title_admin",
            email="title_admin@example.com",
            password="test-password-123",
        )

    def _attach_messages(self, request):
        session_middleware = SessionMiddleware(lambda req: HttpResponse())
        session_middleware.process_request(request)
        request.session.save()
        setattr(request, "_messages", FallbackStorage(request))

    def test_delete_all_about_this_disease_deletes_three_collections(self):
        metadata = {"canonical_title": "pleural effusion"}
        selected_title = MedicalDocumentTitle.objects.create(
            title="Pleural Effusion",
            content="pleural effusion | tràn dịch màng phổi",
            metadata=metadata,
        )
        MedicalDiseaseDocument.objects.create(
            title="pleural effusion",
            content="summary",
            metadata=metadata,
        )
        MedicalDocumentChunk.objects.create(
            title="pleural effusion",
            content="chunk",
            metadata=metadata,
        )
        MedicalDocumentTitle.objects.create(
            title="unrelated disease",
            content="unrelated",
            metadata={"canonical_title": "unrelated disease"},
        )

        request = self.factory.post("/admin/chatbot/medicaldocumenttitle/")
        request.user = self.superuser
        self._attach_messages(request)

        self.admin.delete_all_about_this_disease(
            request,
            MedicalDocumentTitle.objects.filter(pk=selected_title.pk),
        )

        self.assertFalse(
            MedicalDiseaseDocument.objects.filter(
                metadata__canonical_title="pleural effusion"
            ).exists()
        )
        self.assertFalse(
            MedicalDocumentChunk.objects.filter(
                metadata__canonical_title="pleural effusion"
            ).exists()
        )
        self.assertFalse(
            MedicalDocumentTitle.objects.filter(
                metadata__canonical_title="pleural effusion"
            ).exists()
        )
        self.assertTrue(
            MedicalDocumentTitle.objects.filter(
                metadata__canonical_title="unrelated disease"
            ).exists()
        )

    @patch("chatbot.tasks.process_reembedding_job.delay")
    @patch("vector_store.services.embedding_service.EmbeddingService.resolve_provider")
    def test_mark_for_reembedding_queues_all_three_disease_collections(
        self,
        mock_resolve_provider,
        mock_delay,
    ):
        mock_resolve_provider.return_value = "transformers"
        mock_delay.return_value = MagicMock(id="title-reembed-task")
        metadata = {"canonical_title": "pleural effusion"}
        selected_title = MedicalDocumentTitle.objects.create(
            title="Pleural Effusion",
            content="pleural effusion | tràn dịch màng phổi",
            metadata=metadata,
        )
        disease_doc = MedicalDiseaseDocument.objects.create(
            title="pleural effusion",
            content="summary",
            metadata=metadata,
        )
        chunk_doc = MedicalDocumentChunk.objects.create(
            title="pleural effusion",
            content="chunk",
            metadata=metadata,
        )
        unrelated = MedicalDocumentChunk.objects.create(
            title="unrelated disease",
            content="unrelated",
            metadata={"canonical_title": "unrelated disease"},
        )

        request = self.factory.post("/admin/chatbot/medicaldocumenttitle/")
        request.user = self.superuser
        self._attach_messages(request)

        self.admin.mark_for_reembedding(
            request,
            MedicalDocumentTitle.objects.filter(pk=selected_title.pk),
        )

        job = EmbeddingJob.objects.get()
        self.assertEqual(job.job_type, "reembed_selected")
        self.assertEqual(job.provider, "transformers")
        self.assertEqual(job.celery_task_id, "title-reembed-task")
        self.assertCountEqual(
            job.document_ids,
            [
                f"medical_documents_disease:{disease_doc.pk}",
                f"medical_documents_chunks:{chunk_doc.pk}",
                f"medical_documents_titles:{selected_title.pk}",
            ],
        )
        self.assertNotIn(f"medical_documents_chunks:{unrelated.pk}", job.document_ids)
        mock_delay.assert_called_once_with(job.pk)


class ExtendedMedicalDocumentAdminTests(TestCase):
    def setUp(self):
        self.site = AdminSite()
        self.admin = ExtendedMedicalDocumentAdmin(MedicalDocumentChunk, self.site)
        self.factory = RequestFactory()
        User = cast(Any, get_user_model())
        self.superuser = User.objects.create_superuser(
            username="admin_embed",
            email="admin_embed@example.com",
            password="test-password-123",
        )

    def _attach_messages(self, request):
        session_middleware = SessionMiddleware(lambda req: HttpResponse())
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
            "/admin/chatbot/medicaldocumentchunk/embed-url/",
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

    @patch("chatbot.tasks.process_article_paste_embed.delay")
    @patch("vector_store.services.embedding_service.EmbeddingService.resolve_provider")
    def test_paste_content_view_dispatches_task(
        self,
        mock_resolve_provider,
        mock_delay,
    ):
        mock_resolve_provider.return_value = "transformers"
        mock_delay.return_value = MagicMock(id="paste-celery-task-id")

        request = self.factory.post(
            "/admin/chatbot/medicaldocumentchunk/paste-content/",
            data={
                "title": "Scoliosis",
                "source_url": "https://example.com/scoliosis",
                "content": "Scoliosis is a sideways curvature of the spine.",
            },
        )
        request.user = self.superuser
        self._attach_messages(request)

        response = self.admin.paste_content_view(request)
        self.assertEqual(response.status_code, 302)

        self.assertEqual(EmbeddingJob.objects.count(), 1)
        job = EmbeddingJob.objects.first()
        self.assertIsNotNone(job)
        if job is not None:
            self.assertEqual(job.job_type, "paste_ingest")
            self.assertEqual(job.provider, "transformers")
            self.assertEqual(job.celery_task_id, "paste-celery-task-id")

        mock_delay.assert_called_once()
        kwargs = mock_delay.call_args.kwargs
        self.assertEqual(kwargs["title"], "Scoliosis")
        self.assertEqual(
            kwargs["content"],
            "Scoliosis is a sideways curvature of the spine.",
        )
        self.assertEqual(kwargs["source_url"], "https://example.com/scoliosis")
        self.assertEqual(kwargs["embedding_provider"], "transformers")
        self.assertEqual(kwargs["source"], "admin_paste")

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
            collection_name="medical_documents_chunks",
            embedding_provider="mock",
            distance=0.1,
        )
        manager.search_collection.return_value = [
            duplicate_doc,
            duplicate_doc,
            SimpleNamespace(
                pk=2,
                title="Cảm lạnh",
                content="Ho",
                section_type="symptom",
                collection_name="medical_documents_chunks",
                embedding_provider="mock",
                distance=0.8,
            ),
        ]
        mock_render.return_value = MagicMock(status_code=200)

        request = self.factory.post(
            "/admin/chatbot/medicaldocumentchunk/vector-search/",
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
        reembed_type_field = cast(ChoiceField, form.fields["reembed_type"])
        choice_values = [
            value
            for value, _label in cast(list[tuple[str, str]], reembed_type_field.choices)
        ]

        self.assertNotIn("selected", choice_values)

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

    def test_reembed_documents_uses_collection_qualified_refs(self):
        disease_doc = MedicalDiseaseDocument.objects.create(
            title="disease",
            content="summary",
            embedding=[0.1] * 768,
        )
        chunk_doc = MedicalDocumentChunk.objects.create(
            title="chunk",
            content="detail",
            embedding=[0.2] * 768,
        )

        with patch(
            "vector_store.services.reembed_service.EmbeddingService"
        ) as mock_service_cls:
            mock_service_cls.resolve_provider.return_value = "transformers"
            mock_service = mock_service_cls.return_value
            mock_service.embed_text.return_value = [0.9] * 768

            result = ReembeddingService.reembed_documents(
                document_ids=[f"medical_documents_chunks:{chunk_doc.pk}"],
                provider="transformers",
            )

        self.assertEqual(result["successful"], 1)
        disease_doc.refresh_from_db()
        chunk_doc.refresh_from_db()
        self.assertEqual(list(disease_doc.embedding), [0.1] * 768)
        self.assertEqual(list(chunk_doc.embedding), [0.9] * 768)

    def test_process_reembedding_job_handles_missing_job(self):
        result = cast(Any, process_reembedding_job).run(999999)

        self.assertEqual(result["status"], "error")
        self.assertIn("not found", result["message"])

    def _post_reembed_request(self, data):
        request = self.factory.post(
            "/admin/chatbot/medicaldocumentchunk/reembed-manager/",
            data=data,
        )
        request.user = self.superuser
        self._attach_messages(request)
        return request
