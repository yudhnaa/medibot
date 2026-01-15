from typing import override
from django.db import models
from pgvector.django import HnswIndex, VectorField


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
        help_text="768-dimensional embedding from text-embedding-004",
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
