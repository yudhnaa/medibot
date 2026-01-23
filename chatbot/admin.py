from __future__ import annotations

from typing import override

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
from chatbot.tasks import process_csv_upload


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


@admin.register(MedicalDocument)
class MedicalDocumentAdmin(admin.ModelAdmin):
    list_display = ["title", "section_type", "index_type", "source", "created_at"]
    search_fields = ["title", "content"]
    list_filter = ["section_type", "index_type", "source"]
    change_list_template = "admin/chatbot/medicaldocument/change_list.html"

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
        # Check if user is superuser
        if not request.user.is_superuser:
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
                embedding_provider = form.cleaned_data["embedding_provider"]
                index_types = form.cleaned_data["index_types"]  # Now returns a list

                # Save file to temporary location using absolute path
                from django.conf import settings

                temp_dir = os.path.join(settings.BASE_DIR, "tmp", "csv_uploads")
                os.makedirs(temp_dir, exist_ok=True)
                file_path = os.path.join(temp_dir, csv_file.name)

                # Save the uploaded file
                with open(file_path, "wb+") as destination:
                    for chunk in csv_file.chunks():
                        destination.write(chunk)

                # Trigger Celery task for background processing
                try:
                    task = process_csv_upload.delay(  # pyright: ignore[reportCallIssue]
                        file_path=file_path,
                        embedding_provider=embedding_provider,
                        index_types=list(index_types),
                        source="admin_upload",
                        user_id=request.user.id,
                    )

                    index_types_str = ", ".join(index_types)
                    messages.info(
                        request,
                        (
                            f"CSV upload started in background (Task ID: {task.id}). "
                            f"Processing {csv_file.name} with {embedding_provider} provider "
                            f"for index types: {index_types_str}. "
                            f"This may take several minutes. Check logs for progress."
                        ),
                    )
                except Exception as e:
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
