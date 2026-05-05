from django.contrib.auth import get_user_model
from django.db import models

User = get_user_model()


class EmbeddingAuditLog(models.Model):
    """Audit trail for all embedding-related admin actions."""

    ACTION_CHOICES = [
        ("create", "Create Document"),
        ("update", "Update Document"),
        ("delete", "Delete Document"),
        ("reembed", "Re-embed Document"),
        ("bulk_delete", "Bulk Delete"),
        ("bulk_export", "Bulk Export"),
        ("bulk_metadata_edit", "Bulk Metadata Edit"),
    ]

    user = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, related_name="embedding_audit_logs"
    )
    action = models.CharField(max_length=50, choices=ACTION_CHOICES)
    document = models.ForeignKey(
        "MedicalDocument",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="audit_logs",
    )

    # What changed
    changes = models.JSONField(
        default=dict, blank=True, help_text="JSON object showing before/after values"
    )

    embedding_job = models.ForeignKey(
        "EmbeddingJob",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="audit_logs",
    )

    notes = models.CharField(max_length=255, blank=True, default="")
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "chatbot"
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["action", "-timestamp"]),
            models.Index(fields=["user", "-timestamp"]),
            models.Index(fields=["document"]),
        ]

    def __str__(self):
        label_map = dict(self.ACTION_CHOICES)
        label = label_map.get(self.action, self.action)
        return f"{label} - {self.user} at {self.timestamp}"
