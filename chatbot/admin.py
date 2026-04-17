from __future__ import annotations

from typing import Any, cast, override

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
)
from chatbot.forms import CsvUploadForm
from chatbot.admin.document_admin import MedicalDocumentAdmin


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


# MedicalDocumentAdmin, EmbeddingJobAdmin, EmbeddingAuditLogAdmin imported from document_admin module
# BUT we need to extend MedicalDocumentAdmin here to add CSV upload functionality
class ExtendedMedicalDocumentAdmin(MedicalDocumentAdmin):
    """MedicalDocumentAdmin extended with CSV upload."""

    @override
    def get_urls(self):
        """Add custom URL for CSV upload."""
        urls = super().get_urls()
        custom_urls = [
            path(
                "upload-csv/",
                self.admin_site.admin_view(self.upload_csv_view),
                name="chatbot_medicaldocument_upload_csv",
            ),
        ]
        return custom_urls + urls

    def upload_csv_view(self, request: HttpRequest):
        """
        Handle CSV file upload for medical documents.
        Only accessible to superusers.
        """
        from chatbot.models import EmbeddingJob, EmbeddingJobStatus

        user = cast(Any, request.user)

        # Check if user is superuser
        if not user.is_superuser:
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
                    created_by=user,
                    notes=f"CSV file: {csv_file.name}",
                )

                job_obj = cast(Any, job)

                # Trigger Celery task for background processing
                try:
                    from chatbot.tasks import process_csv_upload

                    task = process_csv_upload.delay(  # pyright: ignore[reportCallIssue]
                        file_path=file_path,
                        index_types=list(index_types),
                        source="admin_upload",
                        user_id=user.pk,
                        job_id=job_obj.pk,
                    )

                    job.celery_task_id = task.id
                    job.save()

                    index_types_str = ", ".join(index_types)
                    messages.info(
                        request,
                        (
                            f"CSV upload started in background (Job #{job_obj.pk}). "
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


# Unregister the base MedicalDocumentAdmin and register extended version
admin.site.unregister(MedicalDocument)
admin.site.register(MedicalDocument, ExtendedMedicalDocumentAdmin)


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
