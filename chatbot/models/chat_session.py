import uuid
from typing_extensions import override

from django.db import models

from authentication.models import Customer


class ChatSession(models.Model):
    """Represents a chat session for a customer."""

    customer = models.ForeignKey(
        Customer,
        on_delete=models.CASCADE,
        related_name="chat_sessions",
        verbose_name="Customer",
    )
    session_id = models.UUIDField(
        unique=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name="Session ID",
    )
    title = models.CharField(
        max_length=255,
        blank=True,
        verbose_name="Title",
        help_text="Auto-generated from first message or user-defined",
    )
    max_messages = models.PositiveIntegerField(
        default=50, help_text="Max messages to keep in context window"
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Created At")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Updated At")
    is_active = models.BooleanField(default=True, verbose_name="Is Active")

    class Meta:
        db_table = "chat_session"
        verbose_name = "Chat Session"
        verbose_name_plural = "Chat Sessions"
        ordering = ["-updated_at"]

    @override
    def __str__(self) -> str:
        return f"{self.customer.username} - {self.title or self.session_id}"
