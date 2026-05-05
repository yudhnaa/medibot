from django.db import models

from pgvector.django import HnswIndex, VectorField
from typing_extensions import override


class IndexType(models.TextChoices):
    A = "A", "Summary Index"
    B = "B", "Detail Index"
    C = "C", "Title Index"


class SectionType(models.TextChoices):
    SYMPTOM = "symptom", "Symptoms"
    AETIOLOGIES = "aetiologies", "Aetiologies"
    DIAGNOSE_AND_TREATY = "diagnose_and_treaty", "Diagnosis and Treatment"
    RISK = "risk", "Risk Factors"
    LIVING_AND_PREVENTIVE = "living_and_preventive", "Living and Prevention"
    GENERAL = "general", "General Information"


class MedicalDocument(models.Model):
    """Represents a medical document with vector embedding for similarity search."""

    title = models.CharField(max_length=500, verbose_name="Title")
    content = models.TextField(verbose_name="Content")
    embedding = VectorField(
        dimensions=768,
        null=True,
        blank=True,
        verbose_name="Vector Embedding",
        help_text="768-dimensional embedding (text-embedding-004)",
    )
    section_type = models.CharField(
        max_length=50,
        choices=SectionType.choices,
        default=SectionType.GENERAL,
        verbose_name="Section Type",
    )
    index_type = models.CharField(
        max_length=1,
        choices=IndexType.choices,
        default=IndexType.B,
        verbose_name="Index Type",
    )
    source = models.CharField(
        max_length=255,
        blank=True,
        verbose_name="Source",
        help_text="Source file or dataset name",
    )
    metadata = models.JSONField(
        default=dict,
        blank=True,
        verbose_name="Metadata",
    )
    # NEW FIELDS FOR EMBEDDING TRACKING
    embedding_provider = models.CharField(
        max_length=50,
        default="transformers",
        verbose_name="Embedding Provider",
        help_text="Provider used to generate this embedding: transformers, gemini, tei",
    )
    last_reembedded_at = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name="Last Re-embedded At",
        help_text="When this document was last re-embedded",
    )
    embedding_job = models.ForeignKey(
        "EmbeddingJob",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="documents",
        verbose_name="Embedding Job",
    )

    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Created At")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Updated At")

    class Meta:
        db_table = "medical_document"
        verbose_name = "Medical Document"
        verbose_name_plural = "Medical Documents"
        indexes = [
            models.Index(fields=["title"]),
            models.Index(fields=["section_type"]),
            models.Index(fields=["index_type"]),
            models.Index(fields=["embedding_provider"]),
            models.Index(fields=["-last_reembedded_at"]),
            HnswIndex(
                name="embedding_hnsw_idx",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
        ]

    @override
    def __str__(self) -> str:
        return f"{self.title} ({self.section_type})"

    @property
    def has_embedding(self):
        return self.embedding is not None

    @property
    def embedding_dimension(self):
        return len(self.embedding) if self.embedding else 0

    @property
    def embedding_norm(self):
        if not self.embedding:
            return None
        import math

        return math.sqrt(sum(x**2 for x in self.embedding))
