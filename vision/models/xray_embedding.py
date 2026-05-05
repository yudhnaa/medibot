from django.db import models

from pgvector.django import HnswIndex, VectorField
from typing_extensions import override


class XRayEmbedding(models.Model):
    """Manages the vector embeddings for X-ray images, primarily synced from Kaggle test sets."""

    image_id = models.CharField(max_length=255, verbose_name="Image ID")
    label = models.CharField(max_length=255, verbose_name="Label")
    embedding = VectorField(
        dimensions=1024,
        verbose_name="Vector Embedding",
        help_text="1024-dimensional embedding",
        null=True,
        blank=True,
    )
    metadata = models.JSONField(
        default=dict,
        blank=True,
        verbose_name="Metadata",
    )

    class Meta:
        db_table = "xray_embeddings"
        verbose_name = "X-ray Embedding"
        verbose_name_plural = "X-ray Embeddings"
        indexes = [
            HnswIndex(
                name="xray_embedding_hnsw_idx",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
        ]

    @override
    def __str__(self) -> str:
        return f"Embedding for {self.image_id} - {self.label}"
