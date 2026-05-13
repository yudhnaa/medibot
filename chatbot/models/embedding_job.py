from django.contrib.auth import get_user_model
from django.db import models

User = get_user_model()


class EmbeddingJobStatus(models.TextChoices):
    PENDING = "pending", "Pending"
    PROCESSING = "processing", "Processing"
    COMPLETED = "completed", "Completed"
    FAILED = "failed", "Failed"
    CANCELLED = "cancelled", "Cancelled"


class EmbeddingJob(models.Model):
    """Track all embedding-related async jobs (re-embedding, CSV uploads, provider changes)."""

    JOB_TYPE_CHOICES = [
        ("reembed_selected", "Re-embed Selected Documents"),
        ("reembed_section", "Re-embed by Section"),
        ("reembed_missing", "Re-embed Missing Embeddings"),
        ("change_provider", "Change Embedding Provider"),
        ("csv_upload", "CSV Upload"),
        ("url_ingest", "URL Crawl & Embed"),
        ("paste_ingest", "Paste Content & Embed"),
    ]

    # Tracking
    created_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, related_name="embedding_jobs"
    )
    job_type = models.CharField(
        max_length=50, choices=JOB_TYPE_CHOICES, default="reembed_selected"
    )
    status = models.CharField(
        max_length=20,
        choices=EmbeddingJobStatus.choices,
        default=EmbeddingJobStatus.PENDING,
    )

    # Job parameters
    provider = models.CharField(max_length=50, default="transformers")
    old_provider = models.CharField(max_length=50, blank=True, null=True)
    section_type = models.CharField(max_length=50, blank=True, null=True)
    document_ids = models.JSONField(default=list, blank=True)

    # Results
    total_documents = models.IntegerField(default=0)
    successful_documents = models.IntegerField(default=0)
    failed_documents = models.IntegerField(default=0)
    error_messages = models.JSONField(default=list, blank=True)

    # Timestamps
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    celery_task_id = models.CharField(max_length=255, blank=True, null=True)

    # Notes
    notes = models.TextField(blank=True, default="")

    class Meta:
        app_label = "chatbot"
        ordering = ["-started_at"]
        indexes = [
            models.Index(fields=["status", "-started_at"]),
            models.Index(fields=["job_type", "status"]),
            models.Index(fields=["celery_task_id"]),
        ]

    def __str__(self):
        label_map = dict(self.JOB_TYPE_CHOICES)
        label = label_map.get(self.job_type, self.job_type)
        return f"{label} - {self.status} ({self.started_at})"

    @property
    def is_complete(self):
        return self.status in [
            EmbeddingJobStatus.COMPLETED,
            EmbeddingJobStatus.FAILED,
            EmbeddingJobStatus.CANCELLED,
        ]

    @property
    def success_rate(self):
        if self.total_documents == 0:
            return 0
        return (self.successful_documents / self.total_documents) * 100

    @property
    def duration(self):
        if not self.completed_at:
            return None
        return self.completed_at - self.started_at
