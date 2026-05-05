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

from chatbot.models import MedicalDocument

logger = logging.getLogger(__name__)


class QualityService:
    """Service for analyzing embedding quality and detecting issues."""

    STALE_THRESHOLD_DAYS = 30  # Documents not reembedded in 30 days
    DUPLICATION_THRESHOLD = 0.99  # Cosine similarity threshold for duplicates

    @staticmethod
    def check_missing_embeddings(page: int = 1, page_size: int = 50) -> dict[str, Any]:
        """
        Find documents without embeddings.

        Args:
            page: Page number (1-indexed)
            page_size: Number of results per page

        Returns:
            Dict with total count, page info, and document list
        """
        queryset = MedicalDocument.objects.filter(embedding__isnull=True).order_by(
            "title"
        )
        total = queryset.count()

        start = (page - 1) * page_size
        end = start + page_size
        docs = queryset[start:end]

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "documents": [
                {
                    "id": doc.pk,
                    "title": doc.title,
                    "section_type": doc.section_type,
                    "created_at": doc.created_at,
                    "updated_at": doc.updated_at,
                }
                for doc in docs
            ],
        }

    @staticmethod
    def detect_duplicate_embeddings(
        similarity_threshold: float = 0.99,
    ) -> dict[str, Any]:
        """
        Detect documents with identical or near-identical embeddings.

        Args:
            similarity_threshold: Cosine similarity threshold (0-1)

        Returns:
            List of duplicate groups
        """
        docs = MedicalDocument.objects.filter(embedding__isnull=False).order_by("id")

        if not docs.exists():
            return {"duplicates": [], "count": 0}

        duplicates = []
        embeddings = {}
        for doc in docs:
            doc_id = doc.pk
            if doc_id is None:
                continue
            embeddings[doc_id] = np.array(doc.embedding)
        processed = set()

        for doc1_id, emb1 in embeddings.items():
            if doc1_id in processed:
                continue

            group = [doc1_id]
            norm1 = np.linalg.norm(emb1)

            for doc2_id, emb2 in embeddings.items():
                if doc1_id == doc2_id or doc2_id in processed:
                    continue

                norm2 = np.linalg.norm(emb2)
                if norm1 == 0 or norm2 == 0:
                    cosine_sim = 0
                else:
                    cosine_sim = np.dot(emb1, emb2) / (norm1 * norm2)

                if cosine_sim >= similarity_threshold:
                    group.append(doc2_id)
                    processed.add(doc2_id)

            if len(group) > 1:
                processed.add(doc1_id)
                docs_in_group = [
                    {
                        "id": doc.pk,
                        "title": doc.title,
                        "section_type": doc.section_type,
                        "created_at": doc.created_at,
                    }
                    for doc in docs.filter(id__in=group)
                ]
                duplicates.append(
                    {"group_size": len(group), "documents": docs_in_group}
                )

        return {"duplicates": duplicates, "count": len(duplicates)}

    @staticmethod
    def compute_embedding_stats() -> dict[str, Any]:
        """Compute statistics about embeddings."""
        docs = MedicalDocument.objects.filter(embedding__isnull=False)

        if not docs.exists():
            return {
                "total_with_embedding": 0,
                "stats": None,
            }

        embeddings = np.array([doc.embedding for doc in docs])
        norms = np.linalg.norm(embeddings, axis=1)

        return {
            "total_with_embedding": len(embeddings),
            "stats": {
                "mean_norm": float(np.mean(norms)),
                "std_norm": float(np.std(norms)),
                "min_norm": float(np.min(norms)),
                "max_norm": float(np.max(norms)),
                "median_norm": float(np.median(norms)),
                "dimensions": embeddings.shape[1],
            },
        }

    @staticmethod
    def check_embedding_dimensions() -> dict[str, Any]:
        """Verify all embeddings have correct dimensions."""
        docs = MedicalDocument.objects.filter(embedding__isnull=False)

        dimension_groups = {}
        for doc in docs:
            dim = len(doc.embedding)
            if dim not in dimension_groups:
                dimension_groups[dim] = []
            if doc.pk is not None:
                dimension_groups[dim].append(doc.pk)

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
        """
        Find documents not re-embedded recently.

        Args:
            days: Number of days (defaults to STALE_THRESHOLD_DAYS)

        Returns:
            List of stale documents
        """
        if days is None:
            days = QualityService.STALE_THRESHOLD_DAYS

        stale_date = timezone.now() - timedelta(days=days)

        # Documents with no last_reembedded_at or older than threshold
        stale = MedicalDocument.objects.filter(
            Q(last_reembedded_at__isnull=True) | Q(last_reembedded_at__lt=stale_date)
        ).order_by("last_reembedded_at")

        total = stale.count()

        return {
            "threshold_days": days,
            "stale_date_threshold": stale_date,
            "total_stale": total,
            "by_provider": dict(
                stale.values("embedding_provider")
                .annotate(count=Count("id"))
                .values_list("embedding_provider", "count")
            ),
            "sample_documents": [
                {
                    "id": doc.pk,
                    "title": doc.title,
                    "last_reembedded_at": doc.last_reembedded_at,
                    "provider": doc.embedding_provider,
                    "created_at": doc.created_at,
                }
                for doc in stale[:20]
            ],
        }

    @staticmethod
    def get_quality_report(page: int = 1) -> dict[str, Any]:
        """Get comprehensive quality report."""
        return {
            "missing_embeddings": QualityService.check_missing_embeddings(page=page),
            "duplicate_embeddings": QualityService.detect_duplicate_embeddings(),
            "embedding_stats": QualityService.compute_embedding_stats(),
            "dimension_check": QualityService.check_embedding_dimensions(),
            "stale_embeddings": QualityService.detect_stale_embeddings(),
        }
