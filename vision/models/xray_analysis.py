from django.conf import settings
from django.db import models

from pgvector.django import VectorField
from typing_extensions import override


class XRayAnalysis(models.Model):
    """Stores the classification and analysis results of an uploaded X-ray image."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="xray_analyses",
        verbose_name="User",
    )
    image = models.ImageField(
        upload_to="xrays/%Y/%m/%d/",
        verbose_name="X-ray Image",
        help_text="Chest X-ray image file (JPEG, PNG, DICOM).",
    )
    class_probs = models.JSONField(
        verbose_name="Class Probabilities",
        help_text="Classification probabilities per class.",
    )
    pred_label = models.CharField(
        max_length=100,
        verbose_name="Predicted Label",
        help_text="Predicted class label.",
    )
    findings = models.JSONField(
        verbose_name="Findings",
        help_text="Derived findings from heatmap analysis.",
    )
    heatmap_base64 = models.TextField(
        verbose_name="Heatmap Base64",
        help_text="Base64 encoded PNG of the Grad-CAM heatmap.",
        null=True,
        blank=True,
    )
    embedding = VectorField(
        dimensions=1024,
        verbose_name="Vector Embedding",
        help_text="1024-dimensional embedding",
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Created At")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Updated At")

    class Meta:
        db_table = "xray_analysis"
        verbose_name = "X-ray Analysis"
        verbose_name_plural = "X-ray Analyses"

    @override
    def __str__(self) -> str:
        return f"Analysis for {self.user.username} - {self.pred_label} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"
