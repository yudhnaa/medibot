"""Celery tasks for chatbot application."""

import logging
import os
from datetime import datetime, timezone
from typing import Any

from celery import shared_task

from chatbot.models import EmbeddingJobStatus
from vector_store.services.embedding_docs_pipeline.embed_contexts import embed_contexts
from vector_store.services.embedding_service import EmbeddingService
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
    job = _start_embedding_job(job_id)

    try:
        resolved_provider = EmbeddingService.resolve_provider(embedding_provider)
        logger.info(
            f"Starting CSV processing: {file_path} with provider={resolved_provider}, index_types={index_types}"
        )

        # Initialize VectorStoreManager with resolved provider
        manager = VectorStoreManager(embedding_provider=resolved_provider)

        total_documents = []
        per_index_reports: dict[str, dict[str, Any]] = {}

        # Process CSV for each selected index type
        for index_type in index_types:
            created_docs, report = _process_csv_index_type(
                manager=manager,
                file_path=file_path,
                source=source,
                index_type=index_type,
                job_id=job_id,
            )
            per_index_reports[index_type] = report
            if created_docs:
                total_documents.extend(created_docs)

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
            _complete_embedding_job(
                job=job,
                total_documents=len(total_documents),
                per_index_reports=per_index_reports,
            )

        _cleanup_file(file_path)

        return {
            "status": "success",
            "message": (
                f"Successfully processed {len(total_documents)} documents "
                f"across {len(index_types)} index type(s)"
            ),
            "count": len(total_documents),
            "index_types": index_types,
            "reports": per_index_reports,
        }

    except Exception as exc:
        logger.error(f"Error processing CSV {file_path}: {exc}", exc_info=True)

        if job:
            _fail_embedding_job(job, exc)

        _cleanup_file(file_path, warn_on_error=False)

        return _retry_or_return_csv_error(self, exc)


def _start_embedding_job(job_id: int | None):
    if job_id is None:
        return None

    from chatbot.models import EmbeddingJob

    job = EmbeddingJob.objects.filter(id=job_id).first()
    if job:
        job.status = EmbeddingJobStatus.PROCESSING
        job.save(update_fields=["status"])
    return job


def _process_csv_index_type(
    *,
    manager: VectorStoreManager,
    file_path: str,
    source: str,
    index_type: str,
    job_id: int | None,
) -> tuple[list[Any], dict[str, Any]]:
    logger.info(f"Processing index type: {index_type}")
    result = manager.process_csv_to_documents_with_report(
        csv_path=file_path,
        source=source,
        index_type=index_type,
        ingestion_job_id=job_id,
    )
    documents = result["documents"]
    if not documents:
        logger.warning(f"No documents extracted for index type {index_type}")
        return [], result["report"]

    created_docs = manager.add_documents(documents)
    logger.info(f"Created {len(created_docs)} documents for index type {index_type}")
    return created_docs, result["report"]


def _complete_embedding_job(
    *,
    job: Any,
    total_documents: int,
    per_index_reports: dict[str, dict[str, Any]],
) -> None:
    failed_rows_total = sum(
        int(report.get("failed_rows", 0)) for report in per_index_reports.values()
    )
    report_chunks = [
        f"{index_type}: rows={report.get('total_rows', 0)}, "
        f"ok={report.get('success_rows', 0)}, "
        f"failed={report.get('failed_rows', 0)}"
        for index_type, report in per_index_reports.items()
    ]
    job.status = EmbeddingJobStatus.COMPLETED
    job.total_documents = total_documents
    job.successful_documents = total_documents
    job.failed_documents = failed_rows_total
    job.notes = " | ".join(report_chunks)
    job.save(
        update_fields=[
            "status",
            "total_documents",
            "successful_documents",
            "failed_documents",
            "notes",
        ]
    )


def _fail_embedding_job(job: Any, exc: Exception) -> None:
    job.status = EmbeddingJobStatus.FAILED
    job.error_messages = [str(exc)]
    job.save(update_fields=["status", "error_messages"])


@shared_task(bind=True, max_retries=2, default_retry_delay=60)
def process_reembedding_job(self, job_id: int):
    from chatbot.models import EmbeddingJob
    from vector_store.services.reembed_service import ReembeddingService

    job = EmbeddingJob.objects.filter(id=job_id).first()
    if job is None:
        message = f"Embedding job {job_id} not found."
        logger.error(message)
        return {"status": "error", "message": message}

    try:
        if job.job_type == "reembed_section":
            if not job.section_type:
                raise ValueError("Section type is required for section re-embedding.")
            return ReembeddingService.reembed_by_section(
                section_type=job.section_type,
                provider=job.provider,
                job=job,
            )
        if job.job_type == "reembed_missing":
            return ReembeddingService.reembed_missing(
                provider=job.provider,
                job=job,
            )
        raise ValueError(f"Unsupported re-embedding job type: {job.job_type}")
    except Exception as exc:
        logger.error(
            "Error processing re-embedding job %s: %s", job_id, exc, exc_info=True
        )
        _fail_embedding_job(job, exc)
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        return {"status": "error", "message": str(exc)}


def _cleanup_file(file_path: str, *, warn_on_error: bool = True) -> None:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as exc:
        if warn_on_error:
            logger.warning(f"Failed to cleanup file {file_path}: {exc}")


def _retry_or_return_csv_error(self: Any, exc: Exception) -> dict[str, Any]:
    if self.request.retries < self.max_retries:
        logger.info(
            f"Retrying task (attempt {self.request.retries + 1}/{self.max_retries})"
        )
        raise self.retry(exc=exc)

    return {
        "status": "error",
        "message": f"Failed to process CSV: {str(exc)}",
        "count": 0,
    }


@shared_task(
    bind=True,
    max_retries=2,
    default_retry_delay=60,
    soft_time_limit=20 * 60,
    time_limit=25 * 60,
)
def process_article_url_embed(
    self,
    url: str,
    embedding_provider: str | None = None,
    source: str = "admin_url",
    user_id: int | None = None,
    job_id: int | None = None,
):
    """Crawl URL, extract disease sections via LLM, and embed into A/B/C indexes."""
    job = None
    if job_id is not None:
        from chatbot.models import EmbeddingJob

        job = EmbeddingJob.objects.filter(id=job_id).first()
        if job:
            job.status = EmbeddingJobStatus.PROCESSING
            job.save(update_fields=["status"])

    try:
        from vector_store.services.article_ingestion_service import (
            ArticleIngestionService,
        )

        resolved_provider = EmbeddingService.resolve_provider(embedding_provider)
        service = ArticleIngestionService(embedding_provider=resolved_provider)
        result = service.ingest_from_url(
            url=url,
            source=source,
            ingestion_job_id=job_id,
        )

        if job:
            total = int(result.get("total", 0))
            job.status = EmbeddingJobStatus.COMPLETED
            job.total_documents = total
            job.successful_documents = total
            job.failed_documents = 0
            job.completed_at = datetime.now(timezone.utc)
            job.notes = (
                f"url={url}, index_c={result.get('index_c', 0)}, "
                f"index_a={result.get('index_a', 0)}, "
                f"index_b={result.get('index_b', 0)}"
            )
            job.save(
                update_fields=[
                    "status",
                    "total_documents",
                    "successful_documents",
                    "failed_documents",
                    "completed_at",
                    "notes",
                ]
            )

        return {
            "status": "success",
            "message": "URL crawl and embedding completed",
            **result,
        }
    except Exception as exc:
        logger.error(
            "Error processing URL embedding for %s: %s", url, exc, exc_info=True
        )
        if job:
            job.status = EmbeddingJobStatus.FAILED
            job.error_messages = [str(exc)]
            job.completed_at = datetime.now(timezone.utc)
            job.save(update_fields=["status", "error_messages", "completed_at"])

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)

        return {
            "status": "error",
            "message": f"Failed URL embedding: {str(exc)}",
            "total": 0,
        }


@shared_task(
    bind=True,
    max_retries=2,
    default_retry_delay=60,
    soft_time_limit=4 * 60 * 60 - 300,
    time_limit=4 * 60 * 60,
)
def process_covid_qa_embed(
    self,
    num_articles: int,
    start_article: int = 1,
    embedding_provider: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    user_id: int | None = None,
    job_id: int | None = None,
):
    """
    Run covid_qa_deepset embedding pipeline in background.

    Args:
        self: Celery task instance
        num_articles: Number of unique articles to process
        start_article: 1-based start index within deduplicated article list
        embedding_provider: Optional embedding provider override
        llm_provider: Optional llm provider override (gemini/openrouter)
        llm_model: Optional llm model override
        user_id: Initiator user id
        job_id: EmbeddingJob id for status tracking
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
            (
                "Starting covid_qa_deepset embedding: "
                "start_article=%s, num_articles=%s, provider=%s, llm_provider=%s"
            ),
            start_article,
            num_articles,
            resolved_provider,
            llm_provider,
        )

        result = embed_contexts(
            num_docs=num_articles,
            start_article=start_article,
            clear_existing=(start_article == 1),
            provider=resolved_provider,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )

        index_c = int(result.get("index_c", 0))
        index_a = int(result.get("index_a", 0))
        index_b = int(result.get("index_b", 0))
        total = int(result.get("total", index_c + index_a + index_b))

        if job:
            job.status = EmbeddingJobStatus.COMPLETED
            job.total_documents = total
            job.successful_documents = total
            job.failed_documents = 0
            job.completed_at = datetime.now(timezone.utc)
            job.notes = (
                f"covid_qa_deepset embedding completed. "
                f"start_article={start_article}, "
                f"articles={result.get('articles', num_articles)}, "
                f"index_c={index_c}, index_a={index_a}, index_b={index_b}, "
                f"dataset={result.get('extraction_dataset_path', '')}"
            )
            job.save(
                update_fields=[
                    "status",
                    "total_documents",
                    "successful_documents",
                    "failed_documents",
                    "completed_at",
                    "notes",
                ]
            )

        return {
            "status": "success",
            "message": "covid_qa_deepset embedding completed",
            "articles": int(result.get("articles", num_articles)),
            "index_c": index_c,
            "index_a": index_a,
            "index_b": index_b,
            "total": total,
            "extraction_dataset_path": result.get("extraction_dataset_path", ""),
        }

    except Exception as exc:
        logger.error(
            "Error processing covid_qa_deepset embedding: %s",
            exc,
            exc_info=True,
        )
        if job:
            job.status = EmbeddingJobStatus.FAILED
            job.error_messages = [str(exc)]
            job.completed_at = datetime.now(timezone.utc)
            job.save(update_fields=["status", "error_messages", "completed_at"])

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)

        return {
            "status": "error",
            "message": f"Failed covid_qa_deepset embedding: {str(exc)}",
            "total": 0,
        }
