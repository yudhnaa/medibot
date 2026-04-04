"""
Re-embedding Service
Handles selective, bulk, and provider-switching re-embedding operations.
"""

import logging
from typing import Any
from django.utils import timezone
from django.db import transaction

from chatbot.models import MedicalDocument, EmbeddingJob, EmbeddingJobStatus
from vector_store.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class ReembeddingService:
    """Service for managing document re-embedding operations."""

    @staticmethod
    def create_job(
        job_type: str,
        provider: str = "transformers",
        old_provider: str | None = None,
        section_type: str | None = None,
        document_ids: list[int] | None = None,
        created_by=None,
        notes: str = "",
    ) -> EmbeddingJob:
        """
        Create an embedding job record (before processing).

        Args:
            job_type: Type of job (reembed_selected, reembed_section, etc.)
            provider: Target embedding provider
            old_provider: Previous provider (for provider changes)
            section_type: Section type filter (for section re-embedding)
            document_ids: List of document IDs to process
            created_by: User who created the job
            notes: Additional notes

        Returns:
            EmbeddingJob instance
        """
        job = EmbeddingJob.objects.create(
            job_type=job_type,
            status=EmbeddingJobStatus.PENDING,
            provider=provider,
            old_provider=old_provider,
            section_type=section_type,
            document_ids=document_ids or [],
            created_by=created_by,
            notes=notes,
        )
        logger.info(f"Created embedding job {job.pk}: {job_type}")
        return job

    @staticmethod
    @transaction.atomic
    def reembed_documents(
        document_ids: list[int],
        provider: str = "transformers",
        job: EmbeddingJob | None = None,
    ) -> dict[str, Any]:
        """
        Re-embed specific documents.

        Args:
            document_ids: List of document IDs
            provider: Embedding provider
            job: EmbeddingJob to update (optional)

        Returns:
            Result dict with success/failure counts
        """
        if job:
            job.status = EmbeddingJobStatus.PROCESSING
            job.total_documents = len(document_ids)
            job.save()

        embedding_service = EmbeddingService(provider=provider)

        successful = 0
        failed = 0
        errors = []

        docs = MedicalDocument.objects.filter(id__in=document_ids)

        for doc in docs:
            try:
                # Generate new embedding
                embedding = embedding_service.embed_text(doc.content)

                # Update document
                doc.embedding = embedding
                doc.embedding_provider = provider
                doc.last_reembedded_at = timezone.now()
                doc.embedding_job = job
                doc.save()

                successful += 1
                logger.info(f"Re-embedded document {doc.pk}")

            except Exception as e:
                failed += 1
                error_msg = f"Doc {doc.pk}: {str(e)}"
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
        provider: str = "transformers",
        job: EmbeddingJob | None = None,
    ) -> dict[str, Any]:
        """Re-embed all documents of a specific section type."""
        docs = MedicalDocument.objects.filter(section_type=section_type)
        document_ids = list(docs.values_list("id", flat=True))

        return ReembeddingService.reembed_documents(
            document_ids=document_ids,
            provider=provider,
            job=job,
        )

    @staticmethod
    @transaction.atomic
    def reembed_missing(
        provider: str = "transformers",
        job: EmbeddingJob | None = None,
    ) -> dict[str, Any]:
        """Re-embed all documents without embeddings."""
        docs = MedicalDocument.objects.filter(embedding__isnull=True)
        document_ids = list(docs.values_list("id", flat=True))

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
        document_ids: list[int] | None = None,
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
            docs = MedicalDocument.objects.filter(embedding_provider=old_provider)
            document_ids = list(docs.values_list("id", flat=True))

        return ReembeddingService.reembed_documents(
            document_ids=document_ids,
            provider=new_provider,
            job=job,
        )
