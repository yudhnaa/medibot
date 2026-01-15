from typing import override

from django.db import models

from authentication.models import Customer


class ResponseStyle(models.TextChoices):
    """Response style preferences."""

    CONCISE = "concise", "Concise"
    BALANCED = "balanced", "Balanced"
    DETAILED = "detailed", "Detailed"


class UserPreference(models.Model):
    """User-specific preferences for chatbot behavior.

    These settings override global defaults for individual users.
    """

    customer = models.OneToOneField(
        Customer,
        on_delete=models.CASCADE,
        related_name="chatbot_preference",
        verbose_name="Customer",
    )
    response_style = models.CharField(
        max_length=20,
        choices=ResponseStyle.choices,
        default=ResponseStyle.BALANCED,
        verbose_name="Response Style",
        help_text="Preferred level of detail in responses",
    )
    language = models.CharField(
        max_length=10,
        default="vi",
        verbose_name="Language",
        help_text="Preferred response language (vi, en)",
    )
    max_context_messages = models.PositiveIntegerField(
        default=50,
        verbose_name="Max Context Messages",
        help_text="Maximum messages to keep in conversation context",
    )
    enable_medical_disclaimer = models.BooleanField(
        default=True,
        verbose_name="Enable Medical Disclaimer",
        help_text="Show medical disclaimer in responses",
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Created At")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Updated At")

    class Meta:
        db_table = "user_preference"
        verbose_name = "User Preference"
        verbose_name_plural = "User Preferences"

    @override
    def __str__(self) -> str:
        return f"Preferences for {self.customer.username}"

    @classmethod
    def get_or_create_for_customer(cls, customer: Customer) -> "UserPreference":
        """Get or create preferences for a customer."""
        preference, _ = cls.objects.get_or_create(customer=customer)
        return preference
