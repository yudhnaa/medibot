import os
from typing import Any, cast

from django.contrib import admin, messages
from django.db import transaction
from django.db.models import Q, QuerySet
from django.http import HttpRequest
from django.shortcuts import redirect, render
from django.urls import path

from chatbot.admin.document_admin import (
    EmbeddingJobAdmin,
    MedicalVectorDocumentAdmin,
)
from chatbot.forms import ArticlePasteEmbedForm, ArticleUrlEmbedForm, CsvUploadForm
from chatbot.models import (
    ChatbotConfig,
    ChatMessage,
    ChatSession,
    EmbeddingJob,
    MedicalDiseaseDocument,
    MedicalDocumentChunk,
    MedicalDocumentTitle,
    UserIntake,
)


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ["id", "session_id", "customer", "title", "created_at", "is_active"]
    search_fields = ["session_id", "title", "customer__username"]
    list_filter = ["is_active", "created_at"]


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ["id", "session", "role", "created_at"]
    search_fields = ["content", "session__session_id"]
    list_filter = ["role", "created_at"]


# MedicalDocumentAdmin with CSV upload support
class ExtendedMedicalDocumentAdmin(MedicalVectorDocumentAdmin):
    """Medical document chunk admin extended with ingestion actions."""

    def get_urls(self):
        """Add custom URL for CSV upload."""
        urls = super().get_urls()
        custom_urls = [
            path(
                "upload-csv/",
                self.admin_site.admin_view(self.upload_csv_view),
                name="chatbot_medicaldocumentchunk_upload_csv",
            ),
            path(
                "embed-url/",
                self.admin_site.admin_view(self.embed_url_view),
                name="chatbot_medicaldocumentchunk_embed_url",
            ),
            path(
                "paste-content/",
                self.admin_site.admin_view(self.paste_content_view),
                name="chatbot_medicaldocumentchunk_paste_content",
            ),
        ]
        return custom_urls + urls

    def upload_csv_view(self, request: HttpRequest):
        """
        Handle CSV file upload for medical documents.
        Only accessible to superusers.
        """
        from chatbot.models import EmbeddingJobStatus

        user_obj = cast(Any, request.user)
        is_superuser = bool(getattr(user_obj, "is_superuser", False))
        user_pk = getattr(user_obj, "pk", None)

        # Check if user is superuser
        if not is_superuser:
            messages.error(
                request,
                (
                    "You do not have permission to upload CSV files. "
                    "Only superusers can perform this action."
                ),
            )
            return redirect("admin:chatbot_medicaldocumentchunk_changelist")

        if request.method == "POST":
            form = CsvUploadForm(request.POST, request.FILES)
            if form.is_valid():
                from vector_store.services.embedding_service import EmbeddingService

                # Get form data
                csv_file = form.cleaned_data["csv_file"]
                collections = form.cleaned_data["collections"]
                embedding_provider = EmbeddingService.resolve_provider()

                # Save file to temporary location using absolute path
                from django.conf import settings

                temp_dir = os.path.join(settings.BASE_DIR, "tmp", "csv_uploads")
                os.makedirs(temp_dir, exist_ok=True)
                file_path = os.path.join(temp_dir, csv_file.name)

                # Save the uploaded file
                with open(file_path, "wb+") as destination:
                    for chunk in csv_file.chunks():
                        destination.write(chunk)

                # Create EmbeddingJob record for CSV upload
                job = EmbeddingJob.objects.create(
                    job_type="csv_upload",
                    status=EmbeddingJobStatus.PENDING,
                    provider=embedding_provider,
                    created_by=user_obj,
                    notes=f"CSV file: {csv_file.name}",
                )

                # Trigger Celery task for background processing
                try:
                    from chatbot.tasks import process_csv_upload

                    task = process_csv_upload.delay(  # pyright: ignore[reportCallIssue]
                        file_path=file_path,
                        collections=list(collections),
                        source="admin_upload",
                        user_id=user_pk,
                        job_id=job.pk,
                    )

                    job.celery_task_id = task.id
                    job.save()

                    collections_str = ", ".join(collections)
                    messages.info(
                        request,
                        (
                            f"CSV upload started in background (Job #{job.pk}). "
                            f"Processing {csv_file.name} with {embedding_provider} provider "
                            f"for collections: {collections_str}. "
                            f"This may take several minutes. Check Embedding Jobs for progress."
                        ),
                    )
                except Exception as e:
                    job.status = EmbeddingJobStatus.FAILED
                    job.error_messages = [str(e)]
                    job.save()
                    messages.error(
                        request,
                        (
                            f"Failed to start background task: {str(e)}. "
                            f"Please ensure Celery worker is running."
                        ),
                    )

                return redirect("admin:chatbot_medicaldocumentchunk_changelist")
        else:
            form = CsvUploadForm()

        context = {
            **self.admin_site.each_context(request),
            "form": form,
            "title": "Upload CSV File",
            "opts": self.model._meta,
        }

        return render(
            request,
            "admin/chatbot/medicaldocument/upload_csv_form.html",
            context,
        )

    def embed_url_view(self, request: HttpRequest):
        """Trigger URL crawl + LLM extraction + embedding into collections."""
        from chatbot.models import EmbeddingJobStatus

        user_obj = cast(Any, request.user)
        is_superuser = bool(getattr(user_obj, "is_superuser", False))
        user_pk = getattr(user_obj, "pk", None)

        if not is_superuser:
            messages.error(
                request,
                (
                    "You do not have permission to run URL embedding. "
                    "Only superusers can perform this action."
                ),
            )
            return redirect("admin:chatbot_medicaldocumentchunk_changelist")

        if request.method == "POST":
            form = ArticleUrlEmbedForm(request.POST)
            if form.is_valid():
                from vector_store.services.embedding_service import EmbeddingService

                url = str(form.cleaned_data["url"]).strip()
                embedding_provider = EmbeddingService.resolve_provider()

                job = EmbeddingJob.objects.create(
                    job_type="url_ingest",
                    status=EmbeddingJobStatus.PENDING,
                    provider=embedding_provider,
                    created_by=user_obj,
                    notes=f"URL ingest request: {url}",
                )

                try:
                    from chatbot.tasks import process_article_url_embed

                    task = process_article_url_embed.delay(  # pyright: ignore[reportCallIssue]
                        url=url,
                        embedding_provider=embedding_provider,
                        source="admin_url",
                        user_id=user_pk,
                        job_id=job.pk,
                    )
                    job.celery_task_id = task.id
                    job.save(update_fields=["celery_task_id"])

                    messages.info(
                        request,
                        (
                            f"URL embedding started (Job #{job.pk}). "
                            f"url={url}, embedding_provider={embedding_provider}. "
                            "Check Embedding Jobs for progress."
                        ),
                    )
                except Exception as exc:
                    job.status = EmbeddingJobStatus.FAILED
                    job.error_messages = [str(exc)]
                    job.save(update_fields=["status", "error_messages"])
                    messages.error(
                        request,
                        (
                            f"Failed to start URL embedding: {exc}. "
                            "Please ensure Celery worker is running."
                        ),
                    )
                return redirect("admin:chatbot_medicaldocumentchunk_changelist")
        else:
            form = ArticleUrlEmbedForm()

        context = {
            **self.admin_site.each_context(request),
            "form": form,
            "title": "Crawl URL & Embed",
            "opts": self.model._meta,
        }
        return render(
            request,
            "admin/chatbot/medicaldocument/embed_url_form.html",
            context,
        )

    def paste_content_view(self, request: HttpRequest):
        """Trigger pasted content LLM extraction and embedding into medical document collections."""
        from chatbot.models import EmbeddingJobStatus

        user_obj = cast(Any, request.user)
        is_superuser = bool(getattr(user_obj, "is_superuser", False))
        user_pk = getattr(user_obj, "pk", None)

        if not is_superuser:
            messages.error(
                request,
                (
                    "You do not have permission to run pasted content embedding. "
                    "Only superusers can perform this action."
                ),
            )
            return redirect("admin:chatbot_medicaldocumentchunk_changelist")

        if request.method == "POST":
            form = ArticlePasteEmbedForm(request.POST)
            if form.is_valid():
                from vector_store.services.embedding_service import EmbeddingService

                title = str(form.cleaned_data["title"]).strip()
                content = str(form.cleaned_data["content"]).strip()
                source_url = str(form.cleaned_data.get("source_url", "")).strip()
                embedding_provider = EmbeddingService.resolve_provider()

                job = EmbeddingJob.objects.create(
                    job_type="paste_ingest",
                    status=EmbeddingJobStatus.PENDING,
                    provider=embedding_provider,
                    created_by=user_obj,
                    notes=f"Paste ingest request: {title}",
                )

                try:
                    from chatbot.tasks import process_article_paste_embed

                    task = process_article_paste_embed.delay(  # pyright: ignore[reportCallIssue]
                        title=title,
                        content=content,
                        source_url=source_url,
                        embedding_provider=embedding_provider,
                        source="admin_paste",
                        user_id=user_pk,
                        job_id=job.pk,
                    )
                    job.celery_task_id = task.id
                    job.save(update_fields=["celery_task_id"])

                    messages.info(
                        request,
                        (
                            f"Pasted content embedding started (Job #{job.pk}). "
                            f"title={title}, embedding_provider={embedding_provider}. "
                            "Check Embedding Jobs for progress."
                        ),
                    )
                except Exception as exc:
                    job.status = EmbeddingJobStatus.FAILED
                    job.error_messages = [str(exc)]
                    job.save(update_fields=["status", "error_messages"])
                    messages.error(
                        request,
                        (
                            f"Failed to start pasted content embedding: {exc}. "
                            "Please ensure Celery worker is running."
                        ),
                    )
                return redirect("admin:chatbot_medicaldocumentchunk_changelist")
        else:
            form = ArticlePasteEmbedForm()

        context = {
            **self.admin_site.each_context(request),
            "form": form,
            "title": "Paste Content & Embed",
            "opts": self.model._meta,
        }
        return render(
            request,
            "admin/chatbot/medicaldocument/paste_content_form.html",
            context,
        )


class CollectionOnlyMedicalDocumentAdmin(MedicalVectorDocumentAdmin):
    """Collection admin without chunk ingestion tools."""

    change_list_template = None

    def get_urls(self):
        return admin.ModelAdmin.get_urls(self)


class MedicalDocumentTitleAdmin(CollectionOnlyMedicalDocumentAdmin):
    actions = [
        "delete_all_about_this_disease",
        "mark_for_reembedding",
        "bulk_delete_documents",
        "bulk_export_json",
        "bulk_export_csv",
    ]

    def _selected_disease_titles(
        self, queryset: QuerySet[MedicalDocumentTitle]
    ) -> list[str]:
        return sorted(
            {
                str(doc.metadata.get("canonical_title") or doc.title).strip().lower()
                for doc in queryset
                if str(doc.metadata.get("canonical_title") or doc.title).strip()
            }
        )

    def _disease_title_query(self, titles: list[str]) -> Q:
        query = Q()
        for title in titles:
            query |= Q(metadata__canonical_title=title) | Q(title__iexact=title)
        return query

    @admin.action(description="Re-embed All About This Disease")
    def mark_for_reembedding(
        self,
        request: HttpRequest,
        queryset: QuerySet[MedicalDocumentTitle],
    ) -> None:
        titles = self._selected_disease_titles(queryset)
        if not titles:
            messages.warning(request, "No disease titles selected.")
            return

        from chatbot.tasks import process_reembedding_job
        from vector_store.services.embedding_service import EmbeddingService
        from vector_store.services.reembed_service import ReembeddingService

        disease_query = self._disease_title_query(titles)
        document_refs = []
        counts_by_collection: dict[str, int] = {}
        for model in (
            MedicalDiseaseDocument,
            MedicalDocumentChunk,
            MedicalDocumentTitle,
        ):
            collection_name = model._meta.db_table
            ids = list(model.objects.filter(disease_query).values_list("id", flat=True))
            counts_by_collection[collection_name] = len(ids)
            document_refs.extend(
                ReembeddingService.document_ref(collection_name, doc_id)
                for doc_id in ids
            )

        if not document_refs:
            messages.warning(request, "No documents found for selected disease titles.")
            return

        provider = EmbeddingService.resolve_provider()
        job = ReembeddingService.create_job(
            job_type="reembed_selected",
            provider=provider,
            document_ids=document_refs,
            created_by=request.user,
            notes="disease_titles=" + ",".join(titles),
        )
        task = process_reembedding_job.delay(job.pk)  # pyright: ignore[reportCallIssue]
        job.celery_task_id = task.id
        job.save(update_fields=["celery_task_id"])

        messages.success(
            request,
            (
                f"Re-embedding job {job.pk} started for {len(titles)} disease(s): "
                + ", ".join(
                    f"{collection}={count}"
                    for collection, count in counts_by_collection.items()
                )
            ),
        )

    @admin.action(description="Delete All About This Disease")
    def delete_all_about_this_disease(
        self,
        request: HttpRequest,
        queryset: QuerySet[MedicalDocumentTitle],
    ) -> None:
        titles = self._selected_disease_titles(queryset)
        if not titles:
            messages.warning(request, "No disease titles selected.")
            return

        disease_query = self._disease_title_query(titles)
        deleted_by_collection: dict[str, int] = {}
        with transaction.atomic():
            for model in (
                MedicalDiseaseDocument,
                MedicalDocumentChunk,
                MedicalDocumentTitle,
            ):
                deleted, _ = model.objects.filter(disease_query).delete()
                deleted_by_collection[model._meta.db_table] = deleted

        total_deleted = sum(deleted_by_collection.values())
        messages.success(
            request,
            (
                f"Deleted {total_deleted} documents for {len(titles)} disease(s): "
                + ", ".join(
                    f"{collection}={count}"
                    for collection, count in deleted_by_collection.items()
                )
            ),
        )


# Register physical medical document collections.
admin.site.register(MedicalDiseaseDocument, CollectionOnlyMedicalDocumentAdmin)
admin.site.register(MedicalDocumentTitle, MedicalDocumentTitleAdmin)
admin.site.register(MedicalDocumentChunk, ExtendedMedicalDocumentAdmin)
admin.site.register(EmbeddingJob, EmbeddingJobAdmin)


@admin.register(ChatbotConfig)
class ChatbotConfigAdmin(admin.ModelAdmin):
    list_display = ["key", "category", "is_active", "updated_at"]
    search_fields = ["key", "description"]
    list_filter = ["category", "is_active"]


@admin.register(UserIntake)
class UserIntakeAdmin(admin.ModelAdmin):
    list_display = ["customer_id_display", "customer_username_display", "updated_at"]
    fields = (
        "customer_id_display",
        "customer_username_display",
        "age",
        "sex",
        "pregnancy_status",
        "onset_days",
        "disease_name",
        "symptoms",
    )
    readonly_fields = (
        "customer_id_display",
        "customer_username_display",
        "age",
        "sex",
        "pregnancy_status",
        "onset_days",
        "disease_name",
        "symptoms",
    )

    @admin.display(description="Customer ID")
    def customer_id_display(self, obj):
        return obj.customer_id

    @admin.display(description="Customer Username")
    def customer_username_display(self, obj):
        return obj.customer.username
