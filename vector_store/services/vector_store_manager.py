"""
Vector Store Manager
Service for managing vector storage operations using pgvector (PostgreSQL).
"""

import logging
from typing import Any

from django.db import transaction

import pandas as pd

from chatbot.models import (
    MEDICAL_DOCUMENTS_CHUNKS_COLLECTION,
    MEDICAL_DOCUMENTS_DISEASE_COLLECTION,
    MEDICAL_DOCUMENTS_TITLES_COLLECTION,
    MedicalVectorDocument,
    SectionType,
    collection_for_document_payload,
    get_collection_model,
    iter_collection_models,
)
from vector_store.services.embedding_docs_pipeline.article_schema import (
    normalize_article_record,
    parse_list_items,
)
from vector_store.services.embedding_docs_pipeline.csv_pipeline import (
    CSVEmbeddingPipeline,
)
from vector_store.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manager for vector storage operations using pgvector collections."""

    def __init__(
        self,
        embedding_provider: str | None = None,
        batch_size: int = 100,
    ):
        self.embedding_service = EmbeddingService(provider=embedding_provider)
        self.batch_size = batch_size
        self.embedding_provider = self.embedding_service.get_provider_name()
        logger.info(
            f"VectorStoreManager initialized with provider: {self.embedding_provider}"
        )

    def _csv_pipeline(self) -> CSVEmbeddingPipeline:
        return CSVEmbeddingPipeline(self.embedding_provider)

    def _metadata_value_matches(self, actual: Any, expected: Any) -> bool:
        if isinstance(expected, (list, tuple, set)):
            expected_values = {
                str(item).strip().lower() for item in expected if item is not None
            }
            if not expected_values:
                return False
            if isinstance(actual, list):
                actual_values = {
                    str(item).strip().lower() for item in actual if item is not None
                }
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
        for key, expected in metadata_filters.items():
            if key not in metadata:
                return False
            if not self._metadata_value_matches(metadata.get(key), expected):
                return False
        return True

    def add_document_to_collection(
        self,
        *,
        collection_name: str,
        content: str,
        title: str,
        section_type: str = "general",
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> MedicalVectorDocument:
        embedding = self.embedding_service.embed_text(content)
        model = get_collection_model(collection_name)
        doc = model.objects.create(
            title=title,
            content=content,
            embedding=embedding,
            embedding_provider=self.embedding_provider,
            section_type=section_type,
            source=source,
            metadata=metadata or {},
        )
        logger.info("Added document to %s: %s (ID: %s)", collection_name, title, doc.pk)
        return doc

    def add_documents_to_collection(
        self,
        collection_name: str,
        documents: list[dict[str, Any]],
    ) -> list[MedicalVectorDocument]:
        model = get_collection_model(collection_name)
        created_docs: list[MedicalVectorDocument] = []

        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            contents = [doc["content"] for doc in batch]
            embeddings = self.embedding_service.embed_documents(contents)

            with transaction.atomic():
                for doc_data, embedding in zip(batch, embeddings):
                    doc = model.objects.create(
                        title=doc_data.get("title", ""),
                        content=doc_data["content"],
                        embedding=embedding,
                        embedding_provider=self.embedding_provider,
                        section_type=doc_data.get("section_type", SectionType.GENERAL),
                        source=doc_data.get("source", ""),
                        metadata=doc_data.get("metadata", {}),
                    )
                    created_docs.append(doc)

            logger.info(
                "Added batch %s to %s: %s documents",
                i // self.batch_size + 1,
                collection_name,
                len(batch),
            )

        logger.info(
            "Total documents added to %s: %s", collection_name, len(created_docs)
        )
        return created_docs

    def add_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[MedicalVectorDocument]:
        documents_by_collection: dict[str, list[dict[str, Any]]] = {}
        for document in documents:
            collection_name = collection_for_document_payload(document)
            documents_by_collection.setdefault(collection_name, []).append(document)

        created_docs: list[MedicalVectorDocument] = []
        for collection_name, collection_documents in documents_by_collection.items():
            created_docs.extend(
                self.add_documents_to_collection(collection_name, collection_documents)
            )
        return created_docs

    def search_collection(
        self,
        *,
        collection_name: str,
        query: str,
        k: int = 5,
        section_type: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
        filter_oversample_factor: int = 5,
    ) -> list[MedicalVectorDocument]:
        model = get_collection_model(collection_name)
        query_embedding = self.embedding_service.embed_text(query)
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        conditions = ["embedding IS NOT NULL"]
        params: list[Any] = []
        if section_type:
            conditions.append("section_type = %s")
            params.append(section_type)

        where_clause = " AND ".join(conditions)
        sql_limit = k
        if metadata_filters:
            sql_limit = max(k, k * max(1, filter_oversample_factor))
        params.append(sql_limit)

        table_name = model._meta.db_table
        if table_name != collection_name:
            raise ValueError(f"Collection/table mismatch: {collection_name}")
        sql = f"""
            SELECT id, title, content, section_type, source, metadata,
                   embedding_provider, last_reembedded_at, embedding_job_id,
                   created_at, updated_at, (embedding <=> '{embedding_str}'::vector) AS distance
            FROM {table_name}
            WHERE {where_clause}
            ORDER BY distance
            LIMIT %s
        """

        results = list(model.objects.raw(sql, params))
        if metadata_filters:
            results = [
                doc
                for doc in results
                if self._matches_metadata_filters(
                    getattr(doc, "metadata", {}) or {},
                    metadata_filters,
                )
            ][:k]

        logger.info("Found %s similar documents in %s", len(results), collection_name)
        return results

    def _safe_parse_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, (str, list, tuple, set)):
            return []
        return parse_list_items(value)

    def _split_sentences(self, text: str) -> list[str]:
        return self._csv_pipeline()._split_sentences(text)

    def create_document_content(self, section: str, row: pd.Series) -> str | None:
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
        return parse_list_items(val)[:k]

    def _parse_items_string(self, s: str, k: int) -> list[str]:
        return parse_list_items(s)[:k]

    def _build_summary_parts(self, row: pd.Series) -> list[str]:
        return self._csv_pipeline()._build_summary_parts(row)

    def _build_disease_collection_summary(self, row: pd.Series) -> str | None:
        return self._csv_pipeline().build_disease_collection_summary(row)

    def process_csv_to_documents_with_report(
        self,
        csv_path: str,
        source: str,
        collection_name: str,
        ingestion_job_id: int | None = None,
    ) -> dict[str, Any]:
        documents: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        df = pd.read_csv(csv_path)

        for row_index, row in df.iterrows():
            raw_payload = {
                "title": row.get("title", ""),
                "general": row.get("general", "") or row.get("title", ""),
                "aliases": row.get("aliases", ""),
                "symptom": row.get("symptom", ""),
                "aetiologies": row.get("aetiologies", ""),
                "risk": row.get("risk", ""),
                "diagnose_and_treaty": row.get("diagnose_and_treaty", ""),
                "living_and_preventive": row.get("living_and_preventive", ""),
                "url": row.get("url", ""),
            }

            row_number = int(str(row_index))

            record, row_errors = normalize_article_record(
                raw_payload,
                source_url=str(row.get("url", "")).strip(),
                source_type="csv",
                source_name=source.lower(),
                row_index=row_number,
                ingestion_job_id=ingestion_job_id,
                ingestion_trace=f"csv_row_{row_number}",
            )
            if record is None:
                errors.append({"row_index": row_number, "errors": row_errors})
                continue

            documents.extend(record.build_collection_documents(collection_name))

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
        collection_name: str,
    ) -> list[dict[str, Any]]:
        result = self.process_csv_to_documents_with_report(
            csv_path=csv_path,
            source=source,
            collection_name=collection_name,
            ingestion_job_id=None,
        )
        documents = result["documents"]
        logger.info("Processed %s documents from %s", len(documents), csv_path)
        return documents

    def get_stats(self) -> dict[str, Any]:
        collection_counts = {
            collection_name: model.objects.count()
            for collection_name, model in iter_collection_models()
        }
        return {
            "total_documents": sum(collection_counts.values()),
            "collections": collection_counts,
            "disease_count": collection_counts[MEDICAL_DOCUMENTS_DISEASE_COLLECTION],
            "chunks_count": collection_counts[MEDICAL_DOCUMENTS_CHUNKS_COLLECTION],
            "titles_count": collection_counts[MEDICAL_DOCUMENTS_TITLES_COLLECTION],
            "embedding_provider": self.embedding_service.get_provider_name(),
        }

    def clear_collection(self, collection_name: str) -> int:
        model = get_collection_model(collection_name)
        deleted, _ = model.objects.all().delete()
        logger.info("Cleared %s documents from %s", deleted, collection_name)
        return deleted
