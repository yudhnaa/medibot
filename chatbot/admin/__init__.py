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
from chatbot.forms import CovidQAEmbedForm, CsvUploadForm
from chatbot.tasks import process_covid_qa_embed, process_csv_upload
from vector_store.services.embedding_service import EmbeddingService
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
                "embed-covid-qa/",
                self.admin_site.admin_view(self.embed_covid_qa_view),
                name="chatbot_medicaldocument_embed_covid_qa",
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

    def embed_covid_qa_view(self, request: HttpRequest):
        """Trigger covid_qa_deepset embedding with selectable article count."""
        from chatbot.models import EmbeddingJobStatus

        user_obj = cast(Any, request.user)
        is_superuser = bool(getattr(user_obj, "is_superuser", False))
        user_pk = getattr(user_obj, "pk", None)

        if not is_superuser:
            messages.error(
                request,
                (
                    "You do not have permission to run covid_qa_deepset embedding. "
                    "Only superusers can perform this action."
                ),
            )
            return redirect("admin:chatbot_medicaldocument_changelist")

        if request.method == "POST":
            form = CovidQAEmbedForm(request.POST)
            if form.is_valid():
                num_articles = int(form.cleaned_data["num_articles"])
                start_article = int(form.cleaned_data["start_article"])
                embedding_provider = EmbeddingService.resolve_provider()

                llm_provider_raw = ChatbotConfig.get_config("LLM_PROVIDER", "gemini")
                llm_provider = (
                    llm_provider_raw.lower().strip()
                    if isinstance(llm_provider_raw, str)
                    else "gemini"
                )
                llm_model_default = (
                    "openai/gpt-4.1-mini"
                    if llm_provider == "openrouter"
                    else "gemini-2.5-flash"
                )
                llm_model_raw = ChatbotConfig.get_config("LLM_MODEL", llm_model_default)
                llm_model = (
                    llm_model_raw if isinstance(llm_model_raw, str) else llm_model_default
                )

                job = EmbeddingJob.objects.create(
                    job_type="change_provider",
                    status=EmbeddingJobStatus.PENDING,
                    provider=embedding_provider,
                    created_by=user_obj,
                    notes=(
                        f"covid_qa_deepset embed request "
                        f"(start_article={start_article}, num_articles={num_articles}, "
                        f"llm_provider={llm_provider})"
                    ),
                )

                try:
                    task = process_covid_qa_embed.delay(  # pyright: ignore[reportCallIssue]
                        num_articles=num_articles,
                        start_article=start_article,
                        embedding_provider=embedding_provider,
                        llm_provider=llm_provider,
                        llm_model=llm_model,
                        user_id=user_pk,
                        job_id=job.pk,
                    )
                    job.celery_task_id = task.id
                    job.save(update_fields=["celery_task_id"])

                    messages.info(
                        request,
                        (
                            f"covid_qa_deepset embedding started (Job #{job.pk}). "
                            f"Range={start_article}-{start_article + num_articles - 1}, "
                            f"embedding_provider={embedding_provider}, "
                            f"llm_provider={llm_provider}. "
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
                            f"Failed to start covid_qa_deepset embedding: {exc}. "
                            "Please ensure Celery worker is running."
                        ),
                    )
                return redirect("admin:chatbot_medicaldocument_changelist")
        else:
            form = CovidQAEmbedForm()

        context = {
            **self.admin_site.each_context(request),
            "form": form,
            "title": "Embed covid_qa_deepset Articles",
            "opts": self.model._meta,
        }
        return render(
            request,
            "admin/chatbot/medicaldocument/embed_covid_qa_form.html",
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
