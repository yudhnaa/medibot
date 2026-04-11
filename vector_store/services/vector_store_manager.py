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
from vector_store.services.embedding_docs_pipeline.article_schema import (
    normalize_article_record,
    parse_list_items,
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

    def _metadata_value_matches(self, actual: Any, expected: Any) -> bool:
        """Match a metadata value with case-insensitive scalar/list semantics."""
        if isinstance(expected, (list, tuple, set)):
            expected_values = {str(item).strip().lower() for item in expected if item is not None}
            if not expected_values:
                return False
            if isinstance(actual, list):
                actual_values = {str(item).strip().lower() for item in actual if item is not None}
                return bool(actual_values.intersection(expected_values))
            return str(actual).strip().lower() in expected_values

        expected_norm = str(expected).strip().lower()
        if isinstance(actual, list):
            return expected_norm in {
                str(item).strip().lower() for item in actual if item is not None
            }
        return str(actual).strip().lower() == expected_norm

    def _matches_metadata_filters(
        self, metadata: dict[str, Any], metadata_filters: dict[str, Any]
    ) -> bool:
        """Return True if metadata satisfies all requested filter keys."""
        for key, expected in metadata_filters.items():
            if key not in metadata:
                return False
            if not self._metadata_value_matches(metadata.get(key), expected):
                return False
        return True

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
        metadata_filters: dict[str, Any] | None = None,
        filter_oversample_factor: int = 5,
    ) -> list[MedicalDocument]:
        """
        Search for similar documents using cosine similarity.

        Args:
            query: Search query text
            k: Number of results to return
            index_type: Filter by index type (A, B, C)
            section_type: Filter by section type
            metadata_filters: Optional metadata exact-match filters
            filter_oversample_factor: Oversampling factor before metadata filtering

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
        sql_limit = k
        if metadata_filters:
            sql_limit = max(k, k * max(1, filter_oversample_factor))
        params.append(sql_limit)

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
        if metadata_filters:
            results = [
                doc
                for doc in results
                if self._matches_metadata_filters(
                    getattr(doc, "metadata", {}) or {},
                    metadata_filters,
                )
            ][:k]

        logger.info("Found %s similar documents for query", len(results))
        logger.info("Results: %s", [doc.content for doc in results])
        return results

    # -------------------------
    # CSV Processing
    # -------------------------

    def _safe_parse_list(self, value: Any) -> list[str]:
        """Safely parse list-like values from CSV cells."""
        if value is None:
            return []
        if not isinstance(value, (str, list, tuple, set)):
            return []
        return parse_list_items(value)

    def _split_sentences(self, text: str) -> list[str]:
        """Simple sentence splitter for Vietnamese text."""
        return self._csv_pipeline()._split_sentences(text)

    def create_document_content(self, section: str, row: pd.Series) -> str | None:
        """Create content for a specific section of the medical CSV row."""
        if section not in row:
            return None
        value = row.get(section)
        if value is None:
            return None
        if section == "general":
            content = str(value).strip()
            return content if content else None
        items = parse_list_items(value)
        if not items:
            return None
        return ", ".join(items)

    def _extract_items_from_value(self, val: Any, k: int = 3) -> list[str]:
        """Extract up to k items from a cell value (list string or comma-separated)."""
        return parse_list_items(val)[:k]

    def _parse_items_string(self, s: str, k: int) -> list[str]:
        """Parse a string that may be a Python list literal or comma-separated values."""
        return parse_list_items(s)[:k]

    def _build_summary_parts(self, row: pd.Series) -> list[str]:
        """Build the list of summary parts from row data."""
        return self._csv_pipeline()._build_summary_parts(row)

    def _build_index_a_summary(self, row: pd.Series) -> str | None:
        """Construct disease-level summary for Index A."""
        return self._csv_pipeline().build_index_a_summary(row)

    def process_csv_to_documents_with_report(
        self,
        csv_path: str,
        source: str,
        index_type: str = IndexType.B,
        ingestion_job_id: int | None = None,
    ) -> dict[str, Any]:
        """
        Process CSV file to index-ready documents with per-row validation report.

        Returns:
            {
              "documents": [...],
              "report": {"total_rows": int, "success_rows": int, "failed_rows": int, "errors": [...]}
            }
        """
        documents: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        df = pd.read_csv(csv_path)

        for row_index, row in df.iterrows():
            raw_payload = {
                "title": row.get("title", ""),
                "general": (
                    row.get("general", "")
                    if index_type != IndexType.C
                    else (row.get("general", "") or row.get("title", ""))
                ),
                "symptom": row.get("symptom", ""),
                "aetiologies": row.get("aetiologies", ""),
                "risk": row.get("risk", ""),
                "diagnose_and_treaty": row.get("diagnose_and_treaty", ""),
                "living_and_preventive": row.get("living_and_preventive", ""),
                "url": row.get("url", ""),
            }

            record, row_errors = normalize_article_record(
                raw_payload,
                source_url=str(row.get("url", "")).strip(),
                source_type="csv",
                source_name=source.lower(),
                row_index=int(row_index),
                ingestion_job_id=ingestion_job_id,
                ingestion_trace=f"csv_row_{row_index}",
            )
            if record is None:
                errors.append(
                    {
                        "row_index": int(row_index),
                        "errors": row_errors,
                    }
                )
                continue

            documents.extend(record.build_index_documents(index_type=index_type))

        report = {
            "total_rows": int(len(df)),
            "success_rows": int(len(df) - len(errors)),
            "failed_rows": int(len(errors)),
            "errors": errors,
        }
        return {"documents": documents, "report": report}

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
        result = self.process_csv_to_documents_with_report(
            csv_path=csv_path,
            source=source,
            index_type=index_type,
            ingestion_job_id=None,
        )
        documents = result["documents"]
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
