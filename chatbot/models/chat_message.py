from django.db import models

from typing_extensions import override


class MessageRole(models.TextChoices):
    USER = "user", "User"
    ASSISTANT = "assistant", "Assistant"
    SYSTEM = "system", "System"
    TOOL = "tool", "Tool"


class ChatMessage(models.Model):
    """Represents a single message in a chat session."""

    session = models.ForeignKey(
        "chatbot.ChatSession",
        on_delete=models.CASCADE,
        related_name="messages",
        verbose_name="Session",
    )
    role = models.CharField(
        max_length=20,
        choices=MessageRole.choices,
        verbose_name="Role",
    )
    content = models.TextField(verbose_name="Content")
    response_time_ms = models.PositiveIntegerField(
        null=True, blank=True, help_text="Time taken to generate this response"
    )
    metadata = models.JSONField(
        default=dict,
        blank=True,
        verbose_name="Metadata",
        help_text="Additional data like retrieved documents, tokens used, etc.",
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Created At")

    class Meta:
        db_table = "chat_message"
        verbose_name = "Chat Message"
        verbose_name_plural = "Chat Messages"
        ordering = ["created_at"]

    @override
    def __str__(self) -> str:
        return f"[{self.role}] {self.content[:50]}..."
