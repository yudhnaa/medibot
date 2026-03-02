"""
Vision Serializers
DRF serializers for Vision API request/response validation.
"""

from rest_framework import serializers


class AnalyzeRequestSerializer(serializers.Serializer):
    """Validates X-ray image upload for analysis."""

    image = serializers.ImageField(
        help_text="Chest X-ray image file (JPEG, PNG, DICOM)."
    )


class AnalyzeResponseSerializer(serializers.Serializer):
    """Response from X-ray analysis."""

    class_probs = serializers.DictField(
        child=serializers.FloatField(),
        help_text="Classification probabilities per class.",
    )
    pred_label = serializers.CharField(help_text="Predicted class label.")
    findings = serializers.ListField(
        child=serializers.CharField(),
        help_text="Derived findings from heatmap analysis.",
    )
    heatmap_base64 = serializers.CharField(
        help_text="Grad-CAM heatmap overlay as base64-encoded PNG.",
        allow_null=True,
    )
    embedding = serializers.ListField(
        child=serializers.FloatField(),
        help_text="1024-dim feature embedding vector.",
    )


class EmbedRequestSerializer(serializers.Serializer):
    """Request for batch embedding ingestion."""

    batch_size = serializers.IntegerField(
        default=32,
        min_value=1,
        max_value=256,
        help_text="Number of images per batch.",
    )


class EmbedResponseSerializer(serializers.Serializer):
    """Response from batch embedding ingestion."""

    status = serializers.CharField()
    count = serializers.IntegerField(help_text="Number of embeddings ingested.")


class SimilarItemSerializer(serializers.Serializer):
    """Single similar image result."""

    image_id = serializers.CharField()
    label = serializers.CharField()
    metadata = serializers.DictField(allow_null=True)
    distance = serializers.FloatField()


class SimilarRequestSerializer(serializers.Serializer):
    """Request for similarity search."""

    embedding = serializers.ListField(
        child=serializers.FloatField(),
        help_text="Query embedding vector (1024-dim).",
    )
    k = serializers.IntegerField(
        default=5,
        min_value=1,
        max_value=50,
        help_text="Number of similar images to return.",
    )
