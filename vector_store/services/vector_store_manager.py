"""
Vector Store Manager
Service for managing vector storage operations using pgvector (PostgreSQL).
Integrates with EmbeddingService and MedicalDocument model.
"""

import ast
import logging
import re
from typing import Any

from django.db import transaction

import pandas as pd

from chatbot.models import IndexType, MedicalDocument, SectionType
from vector_store.services.constants import EMBEDDING_PROVIDER_TRANSFORMERS
from vector_store.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manager for vector storage operations using pgvector (PostgreSQL).
    Handles document embedding, storage, and similarity search.
    """

    def __init__(
        self,
        embedding_provider: str = EMBEDDING_PROVIDER_TRANSFORMERS,
        batch_size: int = 100,
    ):
        """
        Initialize the Vector Store Manager.

        Args:
            embedding_provider: Provider for embeddings ('gemini', 'tei', 'transformers')
            batch_size: Batch size for bulk operations
        """
        self.embedding_service = EmbeddingService(provider=embedding_provider)
        self.batch_size = batch_size
        logger.info(
            f"VectorStoreManager initialized with provider: {embedding_provider}"
        )

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
        if not value:
            return []
        if not isinstance(value, str):
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [
                    str(x).strip() for x in parsed if isinstance(x, (str, int, float))
                ]
            return []
        except Exception:
            return [s.strip() for s in value.split(",") if s.strip()]

    def _split_sentences(self, text: str) -> list[str]:
        """Simple sentence splitter for Vietnamese text."""
        if not text:
            return []
        text = " ".join(str(text).strip().split())
        parts = re.split(r"([.!?]+)\s+", text)
        sentences: list[str] = []
        for i in range(0, len(parts), 2):
            sent = parts[i]
            punct = parts[i + 1] if i + 1 < len(parts) else ""
            full = (sent + punct).strip()
            if full:
                sentences.append(full)
        return sentences

    def create_document_content(self, section: str, row: pd.Series) -> str | None:
        """Create content for a specific section of the medical CSV row."""
        if section not in row:
            return None
        val = row[section]
        # Check for NaN/empty - use explicit bool conversion for scalar check
        try:
            is_na = (
                bool(pd.isna(val))
                if not hasattr(val, "__len__") or isinstance(val, str)
                else False
            )
            if is_na:
                return None
        except (TypeError, ValueError):
            pass
        if not str(val).strip():
            return None

        try:
            if section != "general":
                items = self._safe_parse_list(val)
                content = ", ".join(item.strip() for item in items)
                return content if content else None
            else:
                content = str(val).strip()
                return content if content else None
        except Exception as e:
            logger.warning(f"Failed to parse section {section}: {e}")
            return None

    def _extract_items_from_value(self, val: Any, k: int = 3) -> list[str]:
        """Extract up to k items from a cell value (list string or comma-separated)."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return []
        s = str(val).strip()
        if not s or s.lower() in {"[]", "none", "null", "nan"}:
            return []
        return self._parse_items_string(s, k)

    def _parse_items_string(self, s: str, k: int) -> list[str]:
        """Parse a string that may be a Python list literal or comma-separated values."""
        try:
            if s.startswith("[") and s.endswith("]"):
                obj = ast.literal_eval(s)
                if isinstance(obj, list):
                    return [str(x).strip() for x in obj[:k] if str(x).strip()]
        except Exception:
            pass
        items = []
        for part in re.split(r"[,;•·\n]+", s):
            item = part.strip(" \t-•·")
            if item and len(items) < k:
                items.append(item)
        return items[:k]

    def _build_summary_parts(self, row: pd.Series) -> list[str]:
        """Build the list of summary parts from row data."""
        parts = []
        title = str(row.get("title", "")).strip()
        general = str(row.get("general", "")).strip()

        if title:
            parts.append(title)

        general_sents = self._split_sentences(general)
        if general_sents:
            parts.append(general_sents[0])

        sections = [
            ("symptom", "triệu chứng"),
            ("aetiologies", "nguyên nhân"),
            ("risk", "yếu tố nguy cơ"),
            ("diagnose_and_treaty", "chẩn đoán và điều trị"),
            ("living_and_preventive", "sinh hoạt và phòng ngừa"),
        ]
        for section, section_vn in sections:
            items = self._extract_items_from_value(row.get(section), k=3)
            if items:
                parts.append(f"{section_vn}: {', '.join(items)}")

        return parts

    def _build_index_a_summary(self, row: pd.Series) -> str | None:
        """Construct disease-level summary for Index A."""
        title = str(row.get("title", "")).strip()
        general = str(row.get("general", "")).strip()

        if not title and not general:
            return None

        parts = self._build_summary_parts(row)
        summary = "\n".join(parts).strip()
        return summary if summary else None

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
        documents: list[dict[str, Any]] = []
        df = pd.read_csv(csv_path)

        if index_type == "A":
            # Index A: disease-level summary
            for idx, row in df.iterrows():
                content = self._build_index_a_summary(row)
                if content:
                    documents.append(
                        {
                            "content": content.lower(),
                            "title": str(row.get("title", "")).strip().lower(),
                            "section_type": SectionType.GENERAL,
                            "index_type": IndexType.A,
                            "source": source.lower(),
                            "metadata": {
                                "row_index": idx,
                                "url": str(row.get("url", "")).lower(),
                            },
                        }
                    )

        elif index_type == "C":
            # Index C: title-only
            for idx, row in df.iterrows():
                title = str(row.get("title", "")).strip()
                if title:
                    documents.append(
                        {
                            "content": title.lower(),
                            "title": title.lower(),
                            "section_type": SectionType.GENERAL,
                            "index_type": IndexType.C,
                            "source": source.lower(),
                            "metadata": {
                                "row_index": idx,
                                "url": str(row.get("url", "")).lower(),
                            },
                        }
                    )

        else:
            # Index B: per-section documents
            sections = [
                "symptom",
                "aetiologies",
                "diagnose_and_treaty",
                "risk",
                "living_and_preventive",
                "general",
            ]
            section_map = {
                "symptom": SectionType.SYMPTOM,
                "aetiologies": SectionType.AETIOLOGIES,
                "diagnose_and_treaty": SectionType.DIAGNOSE_AND_TREATY,
                "risk": SectionType.RISK,
                "living_and_preventive": SectionType.LIVING_AND_PREVENTIVE,
                "general": SectionType.GENERAL,
            }

            for idx, row in df.iterrows():
                for section in sections:
                    content = self.create_document_content(section, row)
                    if content:
                        documents.append(
                            {
                                "content": content.strip().lower(),
                                "title": str(row.get("title", "")).strip().lower(),
                                "section_type": section_map.get(
                                    section, SectionType.GENERAL
                                ),
                                "index_type": IndexType.B,
                                "source": source.lower(),
                                "metadata": {
                                    "row_index": idx,
                                    "url": str(row.get("url", "")).lower(),
                                },
                            }
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
