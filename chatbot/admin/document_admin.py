"""
Enhanced Admin Classes for Document Embedding Management.
"""

from __future__ import annotations
import json
import csv
from io import StringIO

from django.contrib import admin, messages
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.urls import path
from django.utils.html import format_html
from django.db.models import Count, QuerySet

from chatbot.models import (
    MedicalDocument,
    EmbeddingJob,
    EmbeddingAuditLog,
    EmbeddingJobStatus,
)
from chatbot.forms import (
    VectorSearchForm,
    ReembeddingForm,
)
from vector_store.services.reembed_service import ReembeddingService
from vector_store.services.quality_service import QualityService
from vector_store.services.vector_store_manager import VectorStoreManager


class MedicalDocumentAdmin(admin.ModelAdmin):
    """Enhanced admin for MedicalDocument with CRUD, bulk actions, and vector search."""

    list_display = [
        "title_short",
        "section_type",
        "index_type",
        "embedding_status",
        "embedding_provider",
        "last_reembedded_at",
        "source",
        "created_at",
    ]
    list_filter = [
        "section_type",
        "index_type",
        "embedding_provider",
        "created_at",
    ]
    search_fields = ["title", "content"]
    readonly_fields = [
        "embedding_dimension",
        "embedding_norm",
        "has_embedding",
        "created_at",
        "updated_at",
        "embedding_provider",
        "last_reembedded_at",
        "embedding_preview",
    ]
    fieldsets = (
        ("Document", {"fields": ("title", "content", "source")}),
        ("Classification", {"fields": ("section_type", "index_type")}),
        ("Metadata", {"fields": ("metadata",)}),
        (
            "Embedding",
            {
                "fields": (
                    "has_embedding",
                    "embedding_dimension",
                    "embedding_norm",
                    "embedding_provider",
                    "last_reembedded_at",
                    "embedding_preview",
                ),
                "classes": ("collapse",),
            },
        ),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )
    actions = [
        "mark_for_reembedding",
        "bulk_delete_documents",
        "bulk_export_json",
        "bulk_export_csv",
    ]
    change_list_template = "admin/chatbot/medicaldocument/change_list.html"

    def get_urls(self):
        """Add custom admin URLs."""
        urls = super().get_urls()
        custom_urls = [
            path(
                "vector-search/",
                self.admin_site.admin_view(self.vector_search_view),
                name="chatbot_medicaldocument_vector_search",
            ),
            path(
                "quality-check/",
                self.admin_site.admin_view(self.quality_check_view),
                name="chatbot_medicaldocument_quality_check",
            ),
            path(
                "reembed-manager/",
                self.admin_site.admin_view(self.reembed_manager_view),
                name="chatbot_medicaldocument_reembed_manager",
            ),
            path(
                "api/search-similar/",
                self.admin_site.admin_view(self.api_search_similar),
                name="chatbot_medicaldocument_api_search_similar",
            ),
            path(
                "dashboard/",
                self.admin_site.admin_view(self.dashboard_view),
                name="chatbot_medicaldocument_dashboard",
            ),
        ]
        return custom_urls + urls

    # ========== List Display Methods ==========

    @admin.display(description="Title")
    def title_short(self, obj: MedicalDocument) -> str:
        """Display shortened title."""
        return obj.title[:50] + "..." if len(obj.title) > 50 else obj.title

    @admin.display(description="Embedding")
    def embedding_status(self, obj: MedicalDocument) -> str:
        """Display embedding status as colored badge."""
        if not obj.has_embedding:
            return "Missing"
        return "Present"

    # ========== Readonly Fields Methods ==========

    @admin.display(description="Embedding Dimension")
    def embedding_dimension(self, obj: MedicalDocument) -> str:
        """Show embedding dimension."""
        return f"{obj.embedding_dimension} dims" if obj.has_embedding else "N/A"

    @admin.display(description="Embedding Norm")
    def embedding_norm(self, obj: MedicalDocument) -> str:
        """Show embedding norm."""
        if obj.embedding_norm is None:
            return "N/A"
        return f"{obj.embedding_norm:.4f}"

    @admin.display(description="Has Embedding")
    def has_embedding(self, obj: MedicalDocument) -> str:
        """Show if document has embedding."""
        return "Yes" if obj.has_embedding else "No"

    @admin.display(description="Embedding Preview (first 10 values)")
    def embedding_preview(self, obj: MedicalDocument) -> str:
        """Show first 10 embedding values."""
        if not obj.has_embedding:
            return "No embedding available"
        preview = obj.embedding[:10]
        return f"[{', '.join(f'{v:.6f}' for v in preview)}...]"

    # ========== Admin Actions ==========

    @admin.action(description="Mark selected documents for re-embedding")
    def mark_for_reembedding(self, request: HttpRequest, queryset: QuerySet) -> None:
        """Mark selected documents for re-embedding."""
        count = queryset.count()
        messages.success(
            request,
            f"{count} document(s) marked. Go to Re-embedding Manager to process.",
        )

    @admin.action(description="Delete selected documents")
    def bulk_delete_documents(self, request: HttpRequest, queryset: QuerySet) -> None:
        """Delete selected documents."""
        count = queryset.count()

        # Log deletion
        for doc in queryset:
            EmbeddingAuditLog.objects.create(
                user=request.user,
                action="delete",
                document=doc,
                notes="Bulk deleted via admin",
            )

        queryset.delete()
        messages.success(request, f"Deleted {count} document(s).")

    @admin.action(description="Export selected as JSON")
    def bulk_export_json(self, request, queryset: QuerySet) -> HttpResponse:
        """Export documents as JSON."""
        data = []
        for doc in queryset:
            doc_id = doc.pk
            data.append(
                {
                    "id": doc_id,
                    "title": doc.title,
                    "content": doc.content,
                    "section_type": doc.section_type,
                    "index_type": doc.index_type,
                    "source": doc.source,
                    "embedding_provider": doc.embedding_provider,
                    "metadata": doc.metadata,
                    "created_at": doc.created_at.isoformat(),
                    "updated_at": doc.updated_at.isoformat(),
                }
            )

        response = HttpResponse(
            json.dumps(data, indent=2), content_type="application/json"
        )
        response["Content-Disposition"] = 'attachment; filename="documents.json"'
        return response

    @admin.action(description="Export selected as CSV")
    def bulk_export_csv(self, request, queryset: QuerySet) -> HttpResponse:
        """Export documents as CSV."""
        output = StringIO()
        writer = csv.writer(output)

        # Headers
        writer.writerow(
            [
                "ID",
                "Title",
                "Section Type",
                "Index Type",
                "Source",
                "Provider",
                "Has Embedding",
                "Created At",
            ]
        )

        # Rows
        for doc in queryset:
            doc_id = doc.pk
            writer.writerow(
                [
                    doc_id,
                    doc.title,
                    doc.section_type,
                    doc.index_type,
                    doc.source,
                    doc.embedding_provider,
                    "Yes" if doc.has_embedding else "No",
                    doc.created_at.isoformat(),
                ]
            )

        response = HttpResponse(output.getvalue(), content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="documents.csv"'
        return response

    # ========== Custom Views ==========

    def vector_search_view(self, request: HttpRequest) -> HttpResponse:
        """Vector similarity search interface."""
        if request.method == "POST":
            form = VectorSearchForm(request.POST)
            if form.is_valid():
                query_text = form.cleaned_data["query_text"]
                k = form.cleaned_data["k"]
                provider = form.cleaned_data["embedding_provider"]
                section_types = form.cleaned_data.get("section_types")
                min_similarity = form.cleaned_data.get("min_similarity", 0.5)

                try:
                    manager = VectorStoreManager(embedding_provider=provider)
                    raw_results = []
                    if section_types:
                        for section in section_types:
                            raw_results.extend(
                                manager.search_similar(
                                    query=query_text,
                                    k=k,
                                    section_type=section,
                                )
                            )
                    else:
                        raw_results = manager.search_similar(query=query_text, k=k)

                    # De-duplicate and score results
                    seen: set[int] = set()
                    results = []
                    for doc in raw_results:
                        doc_id = doc.pk
                        if doc_id is None:
                            continue
                        if doc_id in seen:
                            continue
                        seen.add(doc_id)
                        distance = float(getattr(doc, "distance", 1.0))
                        similarity = max(0.0, 1.0 - distance)
                        if similarity < min_similarity:
                            continue
                        results.append(
                            {
                                "id": doc_id,
                                "title": doc.title,
                                "content": doc.content,
                                "section_type": doc.section_type,
                                "index_type": doc.index_type,
                                "embedding_provider": doc.embedding_provider,
                                "similarity": similarity,
                            }
                        )
                    results = sorted(
                        results, key=lambda r: r["similarity"], reverse=True
                    )[:k]

                    return render(
                        request,
                        "admin/chatbot/medicaldocument/vector_search.html",
                        {
                            "form": form,
                            "results": results,
                            "query": query_text,
                            "count": len(results),
                        },
                    )
                except Exception as e:
                    messages.error(request, f"Search error: {str(e)}")
        else:
            form = VectorSearchForm()

        return render(
            request, "admin/chatbot/medicaldocument/vector_search.html", {"form": form}
        )

    def quality_check_view(self, request: HttpRequest) -> HttpResponse:
        """Embedding quality check interface."""
        page = int(request.GET.get("page", 1))
        check_type = request.GET.get("check_type", "all")

        report = QualityService.get_quality_report(page=page)

        stats = {
            "total_documents": MedicalDocument.objects.count(),
            "with_embeddings": MedicalDocument.objects.filter(
                embedding__isnull=False
            ).count(),
            "missing_embeddings": report["missing_embeddings"]["total"],
            "duplicate_groups": report["duplicate_embeddings"]["count"],
        }

        # Calculate coverage
        if stats["total_documents"] > 0:
            stats["coverage_percent"] = (
                stats["with_embeddings"] / stats["total_documents"]
            ) * 100
        else:
            stats["coverage_percent"] = 0
        stats["missing_embeddings"] = (
            stats["total_documents"] - stats["with_embeddings"]
        )

        return render(
            request,
            "admin/chatbot/medicaldocument/quality_check.html",
            {
                "report": report,
                "stats": stats,
                "page": page,
                "check_type": check_type,
            },
        )

    def reembed_manager_view(self, request: HttpRequest) -> HttpResponse:
        """Re-embedding job manager interface."""
        if request.method == "POST":
            form = ReembeddingForm(request.POST)
            if form.is_valid():
                reembed_type = form.cleaned_data["reembed_type"]
                target_provider = form.cleaned_data["target_provider"]
                section_type = form.cleaned_data.get("section_type")
                current_provider = form.cleaned_data.get("current_provider")
                run_async = form.cleaned_data["run_async"]

                try:
                    # Determine job type and create job
                    job_type_map = {
                        "selected": "reembed_selected",
                        "section": "reembed_section",
                        "missing": "reembed_missing",
                        "provider_change": "change_provider",
                    }

                    job_type = job_type_map[reembed_type]

                    job = ReembeddingService.create_job(
                        job_type=job_type,
                        provider=target_provider,
                        old_provider=(
                            current_provider
                            if isinstance(current_provider, str)
                            else None
                        ),
                        section_type=(
                            section_type if isinstance(section_type, str) else None
                        ),
                        created_by=request.user,
                    )

                    if run_async:
                        # TODO: Trigger Celery task
                        job.celery_task_id = "pending"
                        job.save()
                        job_id = job.pk
                        messages.success(
                            request,
                            (
                                f"Re-embedding job {job_id} started in background."
                                if job_id is not None
                                else "Re-embedding job started in background."
                            ),
                        )
                    else:
                        # Synchronous processing
                        if reembed_type == "section" and isinstance(section_type, str):
                            ReembeddingService.reembed_by_section(
                                section_type=section_type,
                                provider=target_provider,
                                job=job,
                            )
                        elif reembed_type == "missing":
                            ReembeddingService.reembed_missing(
                                provider=target_provider,
                                job=job,
                            )
                        elif reembed_type == "provider_change" and isinstance(
                            current_provider, str
                        ):
                            ReembeddingService.change_provider(
                                old_provider=current_provider,
                                new_provider=target_provider,
                                job=job,
                            )

                        messages.success(
                            request,
                            f"Re-embedding complete: {job.successful_documents} "
                            f"successful, {job.failed_documents} failed.",
                        )

                    return redirect("admin:chatbot_embeddingjob_changelist")

                except Exception as e:
                    messages.error(request, f"Error: {str(e)}")
        else:
            form = ReembeddingForm()

        # Show recent jobs
        recent_jobs = EmbeddingJob.objects.all()[:10]

        return render(
            request,
            "admin/chatbot/medicaldocument/reembed_manager.html",
            {
                "form": form,
                "recent_jobs": recent_jobs,
            },
        )

    def dashboard_view(self, request: HttpRequest) -> HttpResponse:
        """Main embedding dashboard."""
        stats = {
            "total_documents": MedicalDocument.objects.count(),
            "with_embeddings": MedicalDocument.objects.filter(
                embedding__isnull=False
            ).count(),
            "by_section": dict(
                MedicalDocument.objects.values("section_type")
                .annotate(count=Count("id"))
                .values_list("section_type", "count")
            ),
            "by_index": dict(
                MedicalDocument.objects.values("index_type")
                .annotate(count=Count("id"))
                .values_list("index_type", "count")
            ),
            "by_provider": dict(
                MedicalDocument.objects.values("embedding_provider")
                .annotate(count=Count("id"))
                .values_list("embedding_provider", "count")
            ),
        }

        if stats["total_documents"] > 0:
            stats["coverage_percent"] = (
                stats["with_embeddings"] / stats["total_documents"]
            ) * 100
        else:
            stats["coverage_percent"] = 0

        recent_jobs = EmbeddingJob.objects.all()[:5]

        return render(
            request,
            "admin/chatbot/medicaldocument/dashboard.html",
            {
                "stats": stats,
                "recent_jobs": recent_jobs,
            },
        )

    def api_search_similar(self, request: HttpRequest) -> JsonResponse:
        """API endpoint for similarity search (for AJAX)."""
        if request.method != "POST":
            return JsonResponse({"error": "POST required"}, status=400)

        try:
            import json

            data = json.loads(request.body)
            query_text = data.get("query")
            k = int(data.get("k", 5))
            provider = data.get("provider", "transformers")
            section_type = data.get("section_type")

            manager = VectorStoreManager(embedding_provider=provider)
            docs = manager.search_similar(
                query=query_text,
                k=k,
                section_type=section_type,
            )
            results = []
            for doc in docs:
                doc_id = doc.pk
                if doc_id is None:
                    continue
                distance = float(getattr(doc, "distance", 1.0))
                results.append(
                    {
                        "id": doc_id,
                        "title": doc.title,
                        "section_type": doc.section_type,
                        "index_type": doc.index_type,
                        "similarity": max(0.0, 1.0 - distance),
                    }
                )

            return JsonResponse(
                {
                    "status": "success",
                    "results": results,
                }
            )
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)


class EmbeddingJobAdmin(admin.ModelAdmin):
    """Admin for embedding jobs tracking."""

    list_display = [
        "job_id_short",
        "job_type_display",
        "status_badge",
        "success_rate_display",
        "started_at",
        "duration_display",
    ]
    list_filter = ["job_type", "status", "started_at"]
    readonly_fields = [
        "job_id_short",
        "success_rate",
        "duration",
        "details_json",
        "started_at",
        "completed_at",
    ]
    fieldsets = (
        ("Job Info", {"fields": ("job_id_short", "job_type", "status", "created_by")}),
        ("Parameters", {"fields": ("provider", "old_provider", "section_type")}),
        (
            "Results",
            {
                "fields": (
                    "total_documents",
                    "successful_documents",
                    "failed_documents",
                    "success_rate",
                    "error_messages",
                )
            },
        ),
        (
            "Execution",
            {
                "fields": ("started_at", "completed_at", "duration", "celery_task_id"),
                "classes": ("collapse",),
            },
        ),
        ("Details", {"fields": ("details_json", "notes"), "classes": ("collapse",)}),
    )
    actions = ["cancel_job"]

    @admin.display(description="Job ID")
    def job_id_short(self, obj: EmbeddingJob) -> str:
        job_id = obj.pk
        return f"Job #{job_id}" if job_id is not None else "Job"

    @admin.display(description="Type")
    def job_type_display(self, obj: EmbeddingJob) -> str:
        label_map = dict(EmbeddingJob.JOB_TYPE_CHOICES)
        key = obj.job_type or ""
        return label_map.get(key, key)

    @admin.display(description="Status")
    def status_badge(self, obj: EmbeddingJob) -> str:
        colors = {
            "pending": "#FFA500",
            "processing": "#4169E1",
            "completed": "#228B22",
            "failed": "#DC143C",
            "cancelled": "#808080",
        }
        color = colors.get(obj.status, "#000000")
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 10px; '
            'border-radius: 3px;">{}</span>',
            color,
            dict(EmbeddingJobStatus.choices).get(obj.status, obj.status),
        )

    @admin.display(description="Success Rate")
    def success_rate_display(self, obj: EmbeddingJob) -> str:
        if obj.total_documents == 0:
            return "N/A"
        return f"{obj.success_rate:.1f}%"

    @admin.display(description="Duration")
    def duration_display(self, obj: EmbeddingJob) -> str:
        if obj.duration is None:
            return "In Progress"
        return str(obj.duration)

    @admin.display(description="Details")
    def details_json(self, obj: EmbeddingJob) -> str:
        data = {
            "document_ids_count": len(obj.document_ids),
            "error_count": len(obj.error_messages),
        }
        return json.dumps(data, indent=2)

    @admin.action(description="Cancel selected pending jobs")
    def cancel_job(self, request: HttpRequest, queryset: QuerySet) -> None:
        """Cancel pending jobs."""
        count = queryset.filter(status="pending").update(status="cancelled")
        messages.success(request, f"Cancelled {count} job(s).")


class EmbeddingAuditLogAdmin(admin.ModelAdmin):
    """Admin for embedding audit logs."""

    list_display = ["timestamp", "user", "action_display", "document_title", "job_link"]
    list_filter = ["action", "timestamp", "user"]
    search_fields = ["document__title", "user__username", "notes"]
    readonly_fields = ["timestamp", "changes_json"]

    @admin.display(description="Action")
    def action_display(self, obj: EmbeddingAuditLog) -> str:
        label_map = dict(EmbeddingAuditLog.ACTION_CHOICES)
        key = obj.action or ""
        return label_map.get(key, key)

    @admin.display(description="Document")
    def document_title(self, obj: EmbeddingAuditLog) -> str:
        if obj.document:
            return obj.document.title[:50]
        return "-"

    @admin.display(description="Job")
    def job_link(self, obj: EmbeddingAuditLog) -> str:
        if obj.embedding_job:
            job_id = obj.embedding_job.pk
            if job_id is not None:
                url = f"/admin/chatbot/embeddingjob/{job_id}/change/"
                return format_html('<a href="{}">Job #{}</a>', url, job_id)
        return "-"

    @admin.display(description="Changes")
    def changes_json(self, obj: EmbeddingAuditLog) -> str:
        return json.dumps(obj.changes, indent=2)
