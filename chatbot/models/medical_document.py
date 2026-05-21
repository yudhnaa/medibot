import math
from typing import Any, TypeAlias

from django.db import models

from pgvector.django import HnswIndex, VectorField
from typing_extensions import override

MEDICAL_DOCUMENTS_DISEASE_COLLECTION = "medical_documents_disease"
MEDICAL_DOCUMENTS_CHUNKS_COLLECTION = "medical_documents_chunks"
MEDICAL_DOCUMENTS_TITLES_COLLECTION = "medical_documents_titles"


class SectionType(models.TextChoices):
    SYMPTOM = "symptom", "Symptoms"
    AETIOLOGIES = "aetiologies", "Aetiologies"
    DIAGNOSE_AND_TREATY = "diagnose_and_treaty", "Diagnosis and Treatment"
    RISK = "risk", "Risk Factors"
    LIVING_AND_PREVENTIVE = "living_and_preventive", "Living and Prevention"
    GENERAL = "general", "General Information"


class BaseMedicalVectorDocument(models.Model):
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
    source = models.CharField(
        max_length=255,
        blank=True,
        verbose_name="Source",
        help_text="Source file or dataset name",
    )
    metadata = models.JSONField(default=dict, blank=True, verbose_name="Metadata")
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
        related_name="%(class)s_documents",
        verbose_name="Embedding Job",
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Created At")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Updated At")

    class Meta:
        abstract = True
        indexes = [
            models.Index(fields=["title"]),
            models.Index(fields=["section_type"]),
            models.Index(fields=["embedding_provider"]),
            models.Index(fields=["-last_reembedded_at"]),
        ]

    @override
    def __str__(self) -> str:
        return f"{self.title} ({self.section_type})"

    @property
    def has_embedding(self) -> bool:
        return self.embedding is not None

    @property
    def embedding_dimension(self) -> int:
        return len(self.embedding) if self.embedding else 0

    @property
    def embedding_norm(self) -> float | None:
        if not self.embedding:
            return None
        return math.sqrt(sum(x**2 for x in self.embedding))

    @property
    def collection_name(self) -> str:
        return self._meta.db_table


class MedicalDiseaseDocument(BaseMedicalVectorDocument):
    class Meta(BaseMedicalVectorDocument.Meta):
        db_table = MEDICAL_DOCUMENTS_DISEASE_COLLECTION
        verbose_name = "Medical Disease Document"
        verbose_name_plural = "Medical Disease Documents"
        indexes = BaseMedicalVectorDocument.Meta.indexes + [
            HnswIndex(
                name="med_disease_hnsw_idx",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
        ]


class MedicalDocumentChunk(BaseMedicalVectorDocument):
    class Meta(BaseMedicalVectorDocument.Meta):
        db_table = MEDICAL_DOCUMENTS_CHUNKS_COLLECTION
        verbose_name = "Medical Document Chunk"
        verbose_name_plural = "Medical Document Chunks"
        indexes = BaseMedicalVectorDocument.Meta.indexes + [
            HnswIndex(
                name="med_chunks_hnsw_idx",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
        ]


class MedicalDocumentTitle(BaseMedicalVectorDocument):
    class Meta(BaseMedicalVectorDocument.Meta):
        db_table = MEDICAL_DOCUMENTS_TITLES_COLLECTION
        verbose_name = "Medical Document Title"
        verbose_name_plural = "Medical Document Titles"
        indexes = BaseMedicalVectorDocument.Meta.indexes + [
            HnswIndex(
                name="med_titles_hnsw_idx",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
        ]


MedicalVectorDocument: TypeAlias = (
    MedicalDiseaseDocument | MedicalDocumentChunk | MedicalDocumentTitle
)
MedicalVectorDocumentModel: TypeAlias = type[
    MedicalDiseaseDocument | MedicalDocumentChunk | MedicalDocumentTitle
]

COLLECTION_MODEL_BY_NAME: dict[str, MedicalVectorDocumentModel] = {
    MEDICAL_DOCUMENTS_DISEASE_COLLECTION: MedicalDiseaseDocument,
    MEDICAL_DOCUMENTS_CHUNKS_COLLECTION: MedicalDocumentChunk,
    MEDICAL_DOCUMENTS_TITLES_COLLECTION: MedicalDocumentTitle,
}

COLLECTION_NAMES: tuple[str, ...] = tuple(COLLECTION_MODEL_BY_NAME.keys())


def get_collection_model(collection_name: str) -> MedicalVectorDocumentModel:
    try:
        return COLLECTION_MODEL_BY_NAME[collection_name]
    except KeyError as exc:
        allowed = ", ".join(COLLECTION_NAMES)
        raise ValueError(
            f"Unknown medical document collection: {collection_name}. Allowed: {allowed}"
        ) from exc


def iter_collection_models() -> list[tuple[str, MedicalVectorDocumentModel]]:
    return list(COLLECTION_MODEL_BY_NAME.items())


def collection_for_document_payload(payload: dict[str, Any]) -> str:
    collection_name = str(payload.get("collection_name") or "").strip()
    if not collection_name:
        metadata = payload.get("metadata") or {}
        if isinstance(metadata, dict):
            collection_name = str(metadata.get("collection") or "").strip()
    get_collection_model(collection_name)
    return collection_name
