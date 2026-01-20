from typing_extensions import override

from django.db import models


class ConfigCategory(models.TextChoices):
    """Categories for organizing configuration."""

    RAG = "rag", "RAG Pipeline"
    MODEL = "model", "Model Settings"
    RATE_LIMIT = "rate_limit", "Rate Limits"
    FEATURE = "feature", "Feature Flags"
    PROCESSING = "processing", "Text Processing"


class ChatbotConfig(models.Model):
    """Dynamic configuration for the chatbot, stored in database.

    Predefined keys:
    - RAG_CONFIG: RAG pipeline parameters (thresholds, weights, limits)
    - MODEL_CONFIG: LLM and embedding model settings
    - RATE_LIMITS: Embedding rate limits (RPM, TPM, RPD)
    - FEATURE_FLAGS: Feature toggles (VnCoreNLP, normalized embeddings, etc.)
    - PROCESSING_CONFIG: Text processing settings (chunk size, batch size)
    """

    key = models.CharField(
        max_length=100,
        unique=True,
        verbose_name="Configuration Key",
    )
    category = models.CharField(
        max_length=20,
        choices=ConfigCategory.choices,
        default=ConfigCategory.RAG,
        verbose_name="Category",
    )
    value = models.JSONField(
        null=True,
        verbose_name="Value",
        help_text="Configuration value (can be string, number, object, etc.)",
    )
    description = models.TextField(
        blank=True,
        verbose_name="Description",
        help_text="Description of what this configuration controls",
    )
    is_active = models.BooleanField(
        default=True,
        verbose_name="Is Active",
        help_text="Whether this configuration is currently in use",
    )
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Updated At")

    class Meta:
        db_table = "chatbot_config"
        verbose_name = "Chatbot Configuration"
        verbose_name_plural = "Chatbot Configurations"
        indexes = [
            models.Index(fields=["category"]),
            models.Index(fields=["is_active"]),
        ]

    @override
    def __str__(self) -> str:
        return f"{self.key} ({self.category}): {self.value}"

    @classmethod
    def get_config(cls, key: str, default: object = None) -> object:
        """Get a configuration value by key."""
        try:
            config = cls.objects.get(key=key, is_active=True)
            return config.value
        except cls.DoesNotExist:
            return default

    @classmethod
    def set_config(
        cls,
        key: str,
        value: object,
        category: str = ConfigCategory.RAG,
        description: str = "",
    ) -> "ChatbotConfig":
        """Set a configuration value."""
        config, _ = cls.objects.update_or_create(
            key=key,
            defaults={
                "value": value,
                "category": category,
                "description": description,
                "is_active": True,
            },
        )
        return config
