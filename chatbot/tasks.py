"""Celery tasks for chatbot application."""

import logging
import os

from celery import shared_task

from vector_store.services.embedding_service import EmbeddingService
from chatbot.models import EmbeddingJobStatus
from vector_store.services.vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def process_csv_upload(
    self,
    file_path: str,
    index_types: list[str],  # Changed to list of index types
    embedding_provider: str | None = None,
    source: str = "admin_upload",
    user_id: int | None = None,
    job_id: int | None = None,
):
    """
    Process uploaded CSV file in background with multiple index types.

    Args:
        self: Celery task instance (for retries)
        file_path: Path to uploaded CSV file
        embedding_provider: Deprecated provider override (None to use ChatbotConfig)
        index_types: List of index types to create (e.g., ['A', 'B', 'C'])
        source: Source identifier for tracking
        user_id: User ID who initiated upload (for notifications)

    Returns:
        dict: Processing result with document count and status
    """
    job = None
    if job_id is not None:
        from chatbot.models import EmbeddingJob

        job = EmbeddingJob.objects.filter(id=job_id).first()
        if job:
            job.status = EmbeddingJobStatus.PROCESSING
            job.save(update_fields=["status"])

    try:
        resolved_provider = EmbeddingService.resolve_provider(embedding_provider)
        logger.info(
            f"Starting CSV processing: {file_path} with provider={resolved_provider}, index_types={index_types}"
        )

        # Initialize VectorStoreManager with resolved provider
        manager = VectorStoreManager(embedding_provider=resolved_provider)

        total_documents = []

        # Process CSV for each selected index type
        for index_type in index_types:
            logger.info(f"Processing index type: {index_type}")

            # Process CSV to document dictionaries
            documents = manager.process_csv_to_documents(
                csv_path=file_path,
                source=source,
                index_type=index_type,
            )

            if documents:
                # Add documents with embeddings to database
                created_docs = manager.add_documents(documents)
                total_documents.extend(created_docs)
                logger.info(
                    f"Created {len(created_docs)} documents for index type {index_type}"
                )
            else:
                logger.warning(f"No documents extracted for index type {index_type}")

        if not total_documents:
            logger.warning(f"No documents extracted from {file_path}")
            return {
                "status": "warning",
                "message": "No valid documents found in CSV file",
                "count": 0,
            }

        logger.info(
            f"Successfully processed {len(total_documents)} total documents from {file_path} with {len(index_types)} index type(s)"
        )

        if job:
            job.status = EmbeddingJobStatus.COMPLETED
            job.total_documents = len(total_documents)
            job.successful_documents = len(total_documents)
            job.failed_documents = 0
            job.save(
                update_fields=[
                    "status",
                    "total_documents",
                    "successful_documents",
                    "failed_documents",
                ]
            )

        # Clean up temporary file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup file {file_path}: {e}")

        return {
            "status": "success",
            "message": (
                f"Successfully processed {len(total_documents)} documents "
                f"across {len(index_types)} index type(s)"
            ),
            "count": len(total_documents),
            "index_types": index_types,
        }

    except Exception as exc:
        logger.error(f"Error processing CSV {file_path}: {exc}", exc_info=True)

        if job:
            job.status = EmbeddingJobStatus.FAILED
            job.error_messages = [str(exc)]
            job.save(update_fields=["status", "error_messages"])

        # Clean up file on error
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

        # Retry task if not exceeded max retries
        if self.request.retries < self.max_retries:
            logger.info(
                f"Retrying task (attempt {self.request.retries + 1}/{self.max_retries})"
            )
            raise self.retry(exc=exc)

        # Final failure - return error status
        return {
            "status": "error",
            "message": f"Failed to process CSV: {str(exc)}",
            "count": 0,
        }
