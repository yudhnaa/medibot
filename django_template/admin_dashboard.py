from django.contrib.auth import get_user_model
from django.db.models import Count, Q

from chatbot.models import (
    ChatMessage,
    ChatSession,
    EmbeddingAuditLog,
    EmbeddingJob,
    EmbeddingJobStatus,
    iter_collection_models,
)
from rag_benchmark.models import BenchmarkRun, BenchmarkRunStatus
from vision.models import XRayAnalysis


def _count_by_status(model, status_field: str = "status") -> dict[str, int]:
    return {
        str(status): count
        for status, count in model.objects.values_list(status_field)
        .annotate(count=Count("id"))
        .order_by()
    }


def _medical_document_stats() -> tuple[dict[str, int], int, int, float]:
    collection_counts: dict[str, int] = {}
    total_documents = 0
    documents_with_embeddings = 0

    for collection_name, model in iter_collection_models():
        stats = model.objects.aggregate(
            total=Count("id"),
            with_embeddings=Count("id", filter=Q(embedding__isnull=False)),
        )
        collection_total = int(stats["total"] or 0)
        collection_counts[collection_name] = collection_total
        total_documents += collection_total
        documents_with_embeddings += int(stats["with_embeddings"] or 0)

    embedding_coverage = (
        round((documents_with_embeddings / total_documents) * 100, 1)
        if total_documents
        else 0
    )
    return (
        collection_counts,
        total_documents,
        documents_with_embeddings,
        embedding_coverage,
    )


def get_admin_dashboard_context() -> dict:
    UserModel = get_user_model()
    (
        collection_counts,
        total_documents,
        documents_with_embeddings,
        embedding_coverage,
    ) = _medical_document_stats()
    user_stats = UserModel.objects.aggregate(
        total=Count("id"),
        staff=Count("id", filter=Q(is_staff=True)),
    )
    session_stats = ChatSession.objects.aggregate(
        total=Count("id"),
        active=Count("id", filter=Q(is_active=True)),
    )
    benchmark_stats = BenchmarkRun.objects.aggregate(
        total=Count("id"),
        completed=Count("id", filter=Q(status=BenchmarkRunStatus.COMPLETED)),
    )
    embedding_job_stats = EmbeddingJob.objects.aggregate(
        failed=Count("id", filter=Q(status=EmbeddingJobStatus.FAILED)),
    )

    return {
        "summary_cards": [
            {
                "label": "Customers",
                "value": user_stats["total"],
                "detail": f"{user_stats['staff']} staff users",
                "icon": "fas fa-users",
                "accent": "slate",
            },
            {
                "label": "Chat sessions",
                "value": session_stats["total"],
                "detail": f"{session_stats['active']} active sessions",
                "icon": "fas fa-comments",
                "accent": "sage",
            },
            {
                "label": "Chat messages",
                "value": ChatMessage.objects.count(),
                "detail": "Conversation records",
                "icon": "fas fa-comment-medical",
                "accent": "sky",
            },
            {
                "label": "Medical documents",
                "value": total_documents,
                "detail": f"{embedding_coverage}% embedding coverage",
                "icon": "fas fa-book-medical",
                "accent": "sand",
            },
            {
                "label": "Diseases",
                "value": collection_counts.get("medical_documents_titles", 0),
                "detail": "Titles collection records",
                "icon": "fas fa-notes-medical",
                "accent": "lavender",
            },
        ],
        "knowledge_report": {
            "collection_counts": collection_counts,
            "documents_with_embeddings": documents_with_embeddings,
            "embedding_coverage": embedding_coverage,
            "embedding_jobs_by_status": _count_by_status(EmbeddingJob),
            "recent_embedding_jobs": EmbeddingJob.objects.only(
                "id", "job_type", "status", "started_at", "total_documents"
            )[:5],
            "audit_events": EmbeddingAuditLog.objects.count(),
        },
        "operations_report": {
            "xray_analyses": XRayAnalysis.objects.count(),
            "benchmark_runs": benchmark_stats["total"],
            "completed_benchmark_runs": benchmark_stats["completed"],
            "benchmark_runs_by_status": _count_by_status(BenchmarkRun),
            "latest_benchmark_runs": BenchmarkRun.objects.select_related(
                "dataset"
            ).only("id", "dataset__name", "dataset__version", "status", "created_at")[
                :5
            ],
            "failed_embedding_jobs": embedding_job_stats["failed"],
        },
    }
