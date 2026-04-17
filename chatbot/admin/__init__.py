"""
Django admin configuration for chatbot app.
Registers models and configures admin interfaces.
"""

from typing import Any, cast

from django.contrib import admin, messages
from django.http import HttpRequest
from django.shortcuts import redirect, render
from django.urls import path
import os

from chatbot.models import (
    ChatbotConfig,
    ChatMessage,
    ChatSession,
    MedicalDocument,
    UserPreference,
    EmbeddingJob,
    EmbeddingAuditLog,
)
from chatbot.forms import ArticleUrlEmbedForm, CsvUploadForm
from chatbot.admin.document_admin import (
    MedicalDocumentAdmin,
    EmbeddingJobAdmin,
    EmbeddingAuditLogAdmin,
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
class ExtendedMedicalDocumentAdmin(MedicalDocumentAdmin):
    """MedicalDocumentAdmin extended with CSV upload."""

    def get_urls(self):
        """Add custom URL for CSV upload."""
        urls = super().get_urls()
        custom_urls = [
            path(
                "upload-csv/",
                self.admin_site.admin_view(self.upload_csv_view),
                name="chatbot_medicaldocument_upload_csv",
            ),
            path(
                "embed-url/",
                self.admin_site.admin_view(self.embed_url_view),
                name="chatbot_medicaldocument_embed_url",
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
            return redirect("admin:chatbot_medicaldocument_changelist")

        if request.method == "POST":
            form = CsvUploadForm(request.POST, request.FILES)
            if form.is_valid():
                from vector_store.services.embedding_service import EmbeddingService

                # Get form data
                csv_file = form.cleaned_data["csv_file"]
                index_types = form.cleaned_data["index_types"]  # Now returns a list
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
                        index_types=list(index_types),
                        source="admin_upload",
                        user_id=user_pk,
                        job_id=job.pk,
                    )

                    job.celery_task_id = task.id
                    job.save()

                    index_types_str = ", ".join(index_types)
                    messages.info(
                        request,
                        (
                            f"CSV upload started in background (Job #{job.pk}). "
                            f"Processing {csv_file.name} with {embedding_provider} provider "
                            f"for index types: {index_types_str}. "
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

                return redirect("admin:chatbot_medicaldocument_changelist")
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
        """Trigger URL crawl + LLM extraction + embedding into C/A/B indexes."""
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
            return redirect("admin:chatbot_medicaldocument_changelist")

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
                return redirect("admin:chatbot_medicaldocument_changelist")
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


# Register MedicalDocument with extended admin
admin.site.register(MedicalDocument, ExtendedMedicalDocumentAdmin)
admin.site.register(EmbeddingJob, EmbeddingJobAdmin)
admin.site.register(EmbeddingAuditLog, EmbeddingAuditLogAdmin)


@admin.register(ChatbotConfig)
class ChatbotConfigAdmin(admin.ModelAdmin):
    list_display = ["key", "category", "is_active", "updated_at"]
    search_fields = ["key", "description"]
    list_filter = ["category", "is_active"]


@admin.register(UserPreference)
class UserPreferenceAdmin(admin.ModelAdmin):
    list_display = ["customer", "response_style", "language", "updated_at"]
    search_fields = ["customer__username"]
    list_filter = ["response_style", "language"]
