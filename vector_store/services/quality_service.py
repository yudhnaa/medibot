"""
Embedding Quality Service
Detects issues: missing embeddings, duplicates, dimension mismatches, stale documents.
"""

import logging
from datetime import timedelta
from typing import Any

from django.db.models import Count, Q
from django.utils import timezone

import numpy as np

from chatbot.models import iter_collection_models

logger = logging.getLogger(__name__)


class QualityService:
    """Service for analyzing embedding quality across medical collections."""

    STALE_THRESHOLD_DAYS = 30
    DUPLICATION_THRESHOLD = 0.99

    @staticmethod
    def _collection_queryset_items():
        for collection_name, model in iter_collection_models():
            yield collection_name, model.objects.all()

    @staticmethod
    def check_missing_embeddings(page: int = 1, page_size: int = 50) -> dict[str, Any]:
        rows = []
        for collection_name, queryset in QualityService._collection_queryset_items():
            for doc in queryset.filter(embedding__isnull=True).order_by("title"):
                rows.append((collection_name, doc))
        total = len(rows)
        start = (page - 1) * page_size
        end = start + page_size

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "documents": [
                {
                    "id": doc.pk,
                    "collection_name": collection_name,
                    "title": doc.title,
                    "section_type": doc.section_type,
                    "created_at": doc.created_at,
                    "updated_at": doc.updated_at,
                }
                for collection_name, doc in rows[start:end]
            ],
        }

    @staticmethod
    def detect_duplicate_embeddings(
        similarity_threshold: float = DUPLICATION_THRESHOLD,
    ) -> dict[str, Any]:
        docs = []
        for collection_name, queryset in QualityService._collection_queryset_items():
            docs.extend(
                (collection_name, doc)
                for doc in queryset.filter(embedding__isnull=False).order_by("id")
            )
        if not docs:
            return {"duplicates": [], "count": 0}

        embeddings = QualityService._load_embeddings(docs)
        processed = set()
        duplicates = []
        for doc_key, emb1 in embeddings.items():
            if doc_key in processed:
                continue
            group = QualityService._find_duplicate_group(
                doc_key,
                emb1,
                embeddings,
                processed,
                similarity_threshold,
            )
            if len(group) > 1:
                processed.add(doc_key)
                duplicates.append(
                    {
                        "group_size": len(group),
                        "documents": QualityService._serialize_duplicate_docs(
                            docs, group
                        ),
                    }
                )
        return {"duplicates": duplicates, "count": len(duplicates)}

    @staticmethod
    def _load_embeddings(
        docs: list[tuple[str, Any]],
    ) -> dict[tuple[str, int], np.ndarray]:
        embeddings = {}
        for collection_name, doc in docs:
            doc_id = doc.pk
            if doc_id is not None:
                embeddings[(collection_name, int(doc_id))] = np.array(doc.embedding)
        return embeddings

    @staticmethod
    def _find_duplicate_group(
        doc_key: tuple[str, int],
        emb1: np.ndarray,
        embeddings: dict[tuple[str, int], np.ndarray],
        processed: set[tuple[str, int]],
        similarity_threshold: float,
    ) -> list[tuple[str, int]]:
        group = [doc_key]
        for other_key, emb2 in embeddings.items():
            if doc_key == other_key or other_key in processed:
                continue
            cosine_sim = QualityService._cosine_similarity(emb1, emb2)
            if cosine_sim >= similarity_threshold:
                group.append(other_key)
                processed.add(other_key)
        return group

    @staticmethod
    def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    @staticmethod
    def _serialize_duplicate_docs(
        docs: list[tuple[str, Any]], group: list[tuple[str, int]]
    ) -> list[dict[str, Any]]:
        group_set = set(group)
        return [
            {
                "id": doc.pk,
                "collection_name": collection_name,
                "title": doc.title,
                "section_type": doc.section_type,
                "created_at": doc.created_at,
            }
            for collection_name, doc in docs
            if doc.pk is not None and (collection_name, int(doc.pk)) in group_set
        ]

    @staticmethod
    def compute_embedding_stats() -> dict[str, Any]:
        embeddings = []
        for _, queryset in QualityService._collection_queryset_items():
            embeddings.extend(
                doc.embedding for doc in queryset.filter(embedding__isnull=False)
            )
        if not embeddings:
            return {"total_with_embedding": 0, "stats": None}

        embedding_array = np.array(embeddings)
        norms = np.linalg.norm(embedding_array, axis=1)
        return {
            "total_with_embedding": len(embeddings),
            "stats": {
                "mean_norm": float(np.mean(norms)),
                "std_norm": float(np.std(norms)),
                "min_norm": float(np.min(norms)),
                "max_norm": float(np.max(norms)),
                "median_norm": float(np.median(norms)),
                "dimensions": embedding_array.shape[1],
            },
        }

    @staticmethod
    def check_embedding_dimensions() -> dict[str, Any]:
        dimension_groups: dict[int, list[dict[str, Any]]] = {}
        for collection_name, queryset in QualityService._collection_queryset_items():
            for doc in queryset.filter(embedding__isnull=False):
                dim = len(doc.embedding)
                dimension_groups.setdefault(dim, [])
                if doc.pk is not None:
                    dimension_groups[dim].append(
                        {"collection_name": collection_name, "id": doc.pk}
                    )

        mismatches = {dim: ids for dim, ids in dimension_groups.items() if dim != 768}
        return {
            "expected_dimension": 768,
            "dimension_distribution": {
                str(k): len(v) for k, v in dimension_groups.items()
            },
            "mismatches": {str(k): len(v) for k, v in mismatches.items()},
            "mismatch_document_ids": {str(k): v for k, v in mismatches.items()},
        }

    @staticmethod
    def detect_stale_embeddings(days: int | None = None) -> dict[str, Any]:
        if days is None:
            days = QualityService.STALE_THRESHOLD_DAYS
        stale_date = timezone.now() - timedelta(days=days)
        stale_rows = []
        provider_counts: dict[str, int] = {}
        total_stale = 0
        for collection_name, queryset in QualityService._collection_queryset_items():
            stale = queryset.filter(
                Q(last_reembedded_at__isnull=True)
                | Q(last_reembedded_at__lt=stale_date)
            ).order_by("last_reembedded_at")
            for provider, count in (
                stale.values("embedding_provider")
                .annotate(count=Count("id"))
                .values_list("embedding_provider", "count")
            ):
                count_int = int(count)
                total_stale += count_int
                key = str(provider or "")
                provider_counts[key] = provider_counts.get(key, 0) + count_int
            stale_rows.extend((collection_name, doc) for doc in stale[:20])

        return {
            "threshold_days": days,
            "stale_date_threshold": stale_date,
            "total_stale": total_stale,
            "by_provider": provider_counts,
            "sample_documents": [
                {
                    "id": doc.pk,
                    "collection_name": collection_name,
                    "title": doc.title,
                    "last_reembedded_at": doc.last_reembedded_at,
                    "provider": doc.embedding_provider,
                    "created_at": doc.created_at,
                }
                for collection_name, doc in stale_rows[:20]
            ],
        }

    @staticmethod
    def get_quality_report(page: int = 1) -> dict[str, Any]:
        return {
            "missing_embeddings": QualityService.check_missing_embeddings(page=page),
            "duplicate_embeddings": QualityService.detect_duplicate_embeddings(),
            "embedding_stats": QualityService.compute_embedding_stats(),
            "dimension_check": QualityService.check_embedding_dimensions(),
            "stale_embeddings": QualityService.detect_stale_embeddings(),
        }
