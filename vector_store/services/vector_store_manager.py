"""
Vector Store Manager
Service for managing vector storage operations using pgvector (PostgreSQL).
Integrates with EmbeddingService and MedicalDocument model.
"""

import logging
from typing import Any

from django.db import transaction

import pandas as pd

from chatbot.models import IndexType, MedicalDocument, SectionType
from vector_store.services.embedding_service import EmbeddingService
from vector_store.services.embedding_docs_pipeline.csv_pipeline import (
    CSVEmbeddingPipeline,
)

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manager for vector storage operations using pgvector (PostgreSQL).
    Handles document embedding, storage, and similarity search.
    """

    def __init__(
        self,
        embedding_provider: str | None = None,
        batch_size: int = 100,
    ):
        """
        Initialize the Vector Store Manager.

        Args:
            embedding_provider: Provider override (None to use ChatbotConfig)
            batch_size: Batch size for bulk operations
        """
        self.embedding_service = EmbeddingService(provider=embedding_provider)
        self.batch_size = batch_size
        self.embedding_provider = self.embedding_service.get_provider_name()
        logger.info(
            f"VectorStoreManager initialized with provider: {self.embedding_provider}"
        )

    def _csv_pipeline(self) -> CSVEmbeddingPipeline:
        """Create a CSV pipeline instance using the same provider as the manager."""
        return CSVEmbeddingPipeline(self.embedding_provider)

    # -------------------------
    # Document Operations
    # -------------------------

    def add_document(
        self,
        content: str,
        title: str,
        section_type: str = SectionType.GENERAL,
        index_type: str = IndexType.B,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> MedicalDocument:
        """
        Add a single document with embedding to the vector store.

        Args:
            content: Document content text
            title: Document title
            section_type: Type of section (symptom, general, etc.)
            index_type: Index type (A, B, C)
            source: Source identifier
            metadata: Additional metadata

        Returns:
            Created MedicalDocument instance
        """
        embedding = self.embedding_service.embed_text(content)

        doc = MedicalDocument.objects.create(
            title=title,
            content=content,
            embedding=embedding,
            embedding_provider=self.embedding_provider,
            section_type=section_type,
            index_type=index_type,
            source=source,
            metadata=metadata or {},
        )
        logger.info(f"Added document: {title} (ID: {doc.pk})")
        return doc

    def add_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[MedicalDocument]:
        """
        Add multiple documents with embeddings in batches.

        Args:
            documents: List of dicts with keys: content, title, section_type, index_type, source, metadata

        Returns:
            List of created MedicalDocument instances
        """
        created_docs: list[MedicalDocument] = []

        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            contents = [doc["content"] for doc in batch]

            # Generate embeddings for batch
            embeddings = self.embedding_service.embed_documents(contents)

            # Create documents
            with transaction.atomic():
                for doc_data, embedding in zip(batch, embeddings):
                    doc = MedicalDocument.objects.create(
                        title=doc_data.get("title", ""),
                        content=doc_data["content"],
                        embedding=embedding,
                        embedding_provider=self.embedding_provider,
                        section_type=doc_data.get("section_type", SectionType.GENERAL),
                        index_type=doc_data.get("index_type", IndexType.B),
                        source=doc_data.get("source", ""),
                        metadata=doc_data.get("metadata", {}),
                    )
                    created_docs.append(doc)

            logger.info(
                f"Added batch {i // self.batch_size + 1}: {len(batch)} documents"
            )

        logger.info(f"Total documents added: {len(created_docs)}")
        return created_docs

    # -------------------------
    # Search Operations
    # -------------------------

    def search_similar(
        self,
        query: str,
        k: int = 5,
        index_type: str | None = None,
        section_type: str | None = None,
    ) -> list[MedicalDocument]:
        """
        Search for similar documents using cosine similarity.

        Args:
            query: Search query text
            k: Number of results to return
            index_type: Filter by index type (A, B, C)
            section_type: Filter by section type

        Returns:
            List of similar MedicalDocument instances
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Build WHERE clause
        conditions = ["embedding IS NOT NULL"]
        params: list[Any] = []

        if index_type:
            conditions.append("index_type = %s")
            params.append(index_type)
        if section_type:
            conditions.append("section_type = %s")
            params.append(section_type)

        where_clause = " AND ".join(conditions)
        params.append(k)

        # Use raw SQL for pgvector cosine distance
        sql = f"""
            SELECT id, title, content, section_type, index_type, source, metadata,
                   created_at, updated_at, (embedding <=> '{embedding_str}'::vector) AS distance
            FROM medical_document
            WHERE {where_clause}
            ORDER BY distance
            LIMIT %s
        """

        results = list(MedicalDocument.objects.raw(sql, params))
        logger.info(f"Found {len(results)} similar documents for query")
        return results

    # -------------------------
    # CSV Processing
    # -------------------------

    def _safe_parse_list(self, value: Any) -> list[str]:
        """Safely parse a Python list serialized as a string."""
        return self._csv_pipeline()._safe_parse_list(value)

    def _split_sentences(self, text: str) -> list[str]:
        """Simple sentence splitter for Vietnamese text."""
        return self._csv_pipeline()._split_sentences(text)

    def create_document_content(self, section: str, row: pd.Series) -> str | None:
        """Create content for a specific section of the medical CSV row."""
        return self._csv_pipeline().create_document_content(section, row)

    def _extract_items_from_value(self, val: Any, k: int = 3) -> list[str]:
        """Extract up to k items from a cell value (list string or comma-separated)."""
        return self._csv_pipeline()._extract_items_from_value(val, k)

    def _parse_items_string(self, s: str, k: int) -> list[str]:
        """Parse a string that may be a Python list literal or comma-separated values."""
        return self._csv_pipeline()._parse_items_string(s, k)

    def _build_summary_parts(self, row: pd.Series) -> list[str]:
        """Build the list of summary parts from row data."""
        return self._csv_pipeline()._build_summary_parts(row)

    def _build_index_a_summary(self, row: pd.Series) -> str | None:
        """Construct disease-level summary for Index A."""
        return self._csv_pipeline().build_index_a_summary(row)

    def process_csv_to_documents(
        self,
        csv_path: str,
        source: str,
        index_type: str = IndexType.B,
    ) -> list[dict[str, Any]]:
        """
        Process CSV file to document dictionaries.

        Args:
            csv_path: Path to CSV file
            source: Source identifier
            index_type: Index type for documents

        Returns:
            List of document dictionaries ready for add_documents()
        """
        documents = self._csv_pipeline().build_documents_from_csv(
            csv_path=csv_path,
            source=source,
            index_type=index_type,
        )
        logger.info(f"Processed {len(documents)} documents from {csv_path}")
        return documents

    # -------------------------
    # Statistics
    # -------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get vector store statistics."""
        count_a = MedicalDocument.objects.filter(index_type=IndexType.A).count()
        count_b = MedicalDocument.objects.filter(index_type=IndexType.B).count()
        count_c = MedicalDocument.objects.filter(index_type=IndexType.C).count()

        return {
            "total_documents": count_a + count_b + count_c,
            "index_a_count": count_a,
            "index_b_count": count_b,
            "index_c_count": count_c,
            "embedding_provider": self.embedding_service.get_provider_name(),
        }

    def clear_index(self, index_type: str) -> int:
        """
        Clear all documents of a specific index type.

        Args:
            index_type: Index type to clear (A, B, C)

        Returns:
            Number of deleted documents
        """
        deleted, _ = MedicalDocument.objects.filter(index_type=index_type).delete()
        logger.info(f"Cleared {deleted} documents from index {index_type}")
        return deleted
