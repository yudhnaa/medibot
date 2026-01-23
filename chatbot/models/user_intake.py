from django.db import models
from typing_extensions import override

from authentication.models import Customer


class UserIntake(models.Model):
    """
    User medical intake information - one per customer.
    Stores patient medical information for chatbot context.
    """

    customer = models.OneToOneField(
        Customer,
        on_delete=models.CASCADE,
        related_name="intake",
        primary_key=True,
        verbose_name="Customer",
    )

    # Medical Information (Session-Specific - Resetable)
    disease_name = models.CharField(
        max_length=200,
        blank=True,
        null=True,
        verbose_name="Disease Name",
        help_text="Current disease or condition being discussed",
    )
    symptoms = models.JSONField(
        default=list,
        verbose_name="Positive Symptoms",
        help_text="List of symptoms the patient is experiencing",
    )
    symptoms_negated = models.JSONField(
        default=list,
        verbose_name="Negated Symptoms",
        help_text="List of symptoms the patient explicitly does NOT have",
    )

    # Demographics (Persistent)
    age = models.PositiveIntegerField(
        null=True,
        blank=True,
        verbose_name="Age",
        help_text="Patient age in years",
    )
    sex = models.CharField(
        max_length=10,
        choices=[
            ("male", "Male"),
            ("female", "Female"),
            ("unknown", "Unknown"),
        ],
        default="unknown",
        verbose_name="Sex",
    )
    pregnancy_status = models.CharField(
        max_length=20,
        choices=[
            ("yes", "Yes"),
            ("no", "No"),
            ("unknown", "Unknown"),
            ("not_applicable", "N/A"),
        ],
        null=True,
        blank=True,
        verbose_name="Pregnancy Status",
    )
    location_country = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        verbose_name="Country",
        help_text="Patient's country of residence",
    )

    # Medical History (Persistent)
    chronic_conditions = models.JSONField(
        default=list,
        verbose_name="Chronic Conditions",
        help_text="Long-term medical conditions",
    )
    allergies = models.JSONField(
        default=list,
        verbose_name="Allergies",
        help_text="Known allergies",
    )

    # Current Episode (Resetable)
    onset_days = models.PositiveIntegerField(
        null=True,
        blank=True,
        verbose_name="Onset Days",
        help_text="Number of days since symptoms started",
    )
    meds = models.JSONField(
        default=list,
        verbose_name="Current Medications",
        help_text="Medications currently being taken",
    )

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Created At")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Updated At")

    # Fields that should be reset when chat session is cleared
    RESETABLE_FIELDS = [
        "disease_name",
        "symptoms",
        "symptoms_negated",
        "onset_days",
        "meds",
        "pregnancy_status",
    ]

    class Meta:
        db_table = "user_intake"
        verbose_name = "User Intake"
        verbose_name_plural = "User Intakes"

    @override
    def __str__(self) -> str:
        return f"Intake for {self.customer.username}"

    def reset_session_specific_fields(self) -> None:
        """
        Reset fields that should clear when chat session is cleared.
        Preserves persistent demographic and medical history information.
        """
        for field in self.RESETABLE_FIELDS:
            if field in ["symptoms", "symptoms_negated", "meds"]:
                # Reset list fields to empty list
                setattr(self, field, [])
            else:
                # Reset scalar fields to None
                setattr(self, field, None)
        self.save()
