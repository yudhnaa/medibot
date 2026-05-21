"""
Re-embedding Service
Handles selective, bulk, and provider-switching re-embedding operations.
"""

import logging
from typing import Any

from django.db import transaction
from django.utils import timezone

from chatbot.models import (
    EmbeddingJob,
    EmbeddingJobStatus,
    get_collection_model,
    iter_collection_models,
)
from vector_store.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class ReembeddingService:
    """Service for managing document re-embedding operations."""

    @staticmethod
    def create_job(
        job_type: str,
        provider: str | None = None,
        old_provider: str | None = None,
        section_type: str | None = None,
        document_ids: list[Any] | None = None,
        created_by=None,
        notes: str = "",
    ) -> EmbeddingJob:
        """
        Create an embedding job record (before processing).

        Args:
            job_type: Type of job (reembed_selected, reembed_section, etc.)
            provider: Target embedding provider (None to use ChatbotConfig)
            old_provider: Previous provider (for provider changes)
            section_type: Section type filter (for section re-embedding)
            document_ids: List of document IDs to process
            created_by: User who created the job
            notes: Additional notes

        Returns:
            EmbeddingJob instance
        """
        resolved_provider = EmbeddingService.resolve_provider(provider)
        job = EmbeddingJob.objects.create(
            job_type=job_type,
            status=EmbeddingJobStatus.PENDING,
            provider=resolved_provider,
            old_provider=old_provider,
            section_type=section_type,
            document_ids=document_ids or [],
            created_by=created_by,
            notes=notes,
        )
        logger.info(f"Created embedding job {job.pk}: {job_type}")
        return job

    @staticmethod
    def document_ref(collection_name: str, document_id: int) -> str:
        return f"{collection_name}:{document_id}"

    @staticmethod
    def _resolve_document_refs(document_ids: list[Any]) -> list[Any]:
        grouped_ids: dict[str, list[int]] = {}
        for raw_ref in document_ids:
            if isinstance(raw_ref, dict):
                collection_name = str(raw_ref.get("collection_name", "")).strip()
                raw_id = raw_ref.get("id")
                if raw_id is None:
                    raise ValueError("Document ref id is required.")
                document_id = int(raw_id)
            elif isinstance(raw_ref, str) and ":" in raw_ref:
                collection_name, raw_id = raw_ref.split(":", 1)
                document_id = int(raw_id)
            else:
                raise ValueError(
                    "Document refs must include collection_name and id, e.g. medical_documents_chunks:123"
                )
            get_collection_model(collection_name)
            grouped_ids.setdefault(collection_name, []).append(document_id)

        docs = []
        for collection_name, ids in grouped_ids.items():
            model = get_collection_model(collection_name)
            docs.extend(model.objects.filter(id__in=ids))
        return docs

    @staticmethod
    @transaction.atomic
    def reembed_documents(
        document_ids: list[Any],
        provider: str | None = None,
        job: EmbeddingJob | None = None,
    ) -> dict[str, Any]:
        resolved_provider = EmbeddingService.resolve_provider(provider)
        if job:
            job.status = EmbeddingJobStatus.PROCESSING
            job.total_documents = len(document_ids)
            job.save()

        embedding_service = EmbeddingService(provider=resolved_provider)

        successful = 0
        failed = 0
        errors = []
        docs = ReembeddingService._resolve_document_refs(document_ids)

        for doc in docs:
            try:
                embedding = embedding_service.embed_text(doc.content)
                doc.embedding = embedding
                doc.embedding_provider = resolved_provider
                doc.last_reembedded_at = timezone.now()
                doc.embedding_job = job
                doc.save()

                successful += 1
                logger.info("Re-embedded document %s:%s", doc.collection_name, doc.pk)

            except Exception as e:
                failed += 1
                error_msg = f"Doc {doc.collection_name}:{doc.pk}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)

        result = {
            "total": len(document_ids),
            "successful": successful,
            "failed": failed,
            "errors": errors,
        }

        if job:
            job.status = EmbeddingJobStatus.COMPLETED
            job.successful_documents = successful
            job.failed_documents = failed
            job.error_messages = errors
            job.completed_at = timezone.now()
            job.save()

        return result

    @staticmethod
    @transaction.atomic
    def reembed_by_section(
        section_type: str,
        provider: str | None = None,
        job: EmbeddingJob | None = None,
    ) -> dict[str, Any]:
        """Re-embed all documents of a specific section type."""
        document_ids = []
        for collection_name, model in iter_collection_models():
            document_ids.extend(
                ReembeddingService.document_ref(collection_name, doc_id)
                for doc_id in model.objects.filter(
                    section_type=section_type
                ).values_list("id", flat=True)
            )

        return ReembeddingService.reembed_documents(
            document_ids=document_ids,
            provider=provider,
            job=job,
        )

    @staticmethod
    @transaction.atomic
    def reembed_missing(
        provider: str | None = None,
        job: EmbeddingJob | None = None,
    ) -> dict[str, Any]:
        """Re-embed all documents without embeddings."""
        document_ids = []
        for collection_name, model in iter_collection_models():
            document_ids.extend(
                ReembeddingService.document_ref(collection_name, doc_id)
                for doc_id in model.objects.filter(embedding__isnull=True).values_list(
                    "id", flat=True
                )
            )

        return ReembeddingService.reembed_documents(
            document_ids=document_ids,
            provider=provider,
            job=job,
        )

    @staticmethod
    @transaction.atomic
    def change_provider(
        old_provider: str,
        new_provider: str,
        document_ids: list[Any] | None = None,
        job: EmbeddingJob | None = None,
    ) -> dict[str, Any]:
        """
        Change embedding provider for documents.

        Args:
            old_provider: Current provider
            new_provider: New provider
            document_ids: Specific docs, or None for all with old provider
            job: EmbeddingJob to update

        Returns:
            Result dict
        """
        if document_ids is None:
            document_ids = []
            for collection_name, model in iter_collection_models():
                document_ids.extend(
                    ReembeddingService.document_ref(collection_name, doc_id)
                    for doc_id in model.objects.filter(
                        embedding_provider=old_provider
                    ).values_list("id", flat=True)
                )

        return ReembeddingService.reembed_documents(
            document_ids=document_ids,
            provider=new_provider,
            job=job,
        )
