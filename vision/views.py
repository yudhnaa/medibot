"""
Vision Views
DRF views for X-ray analysis, embedding ingestion, and similarity search.
"""

import base64
import logging
import os
import tempfile
from typing import Any, cast

from django.conf import settings

from rest_framework import permissions, status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from vision.serializers import (
    AnalyzeRequestSerializer,
    AnalyzeResponseSerializer,
    EmbedRequestSerializer,
    EmbedResponseSerializer,
    SimilarItemSerializer,
    SimilarRequestSerializer,
)
from vision.services.retrieval_pipeline import get_conn, query_similar
from vision.services.ingest_embeddings import ingest_embeddings
from vision.services.vision_service import analyze_xray
from vision.utils import load_config

logger = logging.getLogger(__name__)


from vision.models import XRayAnalysis


class AnalyzeView(APIView):
    """POST /api/v1/vision/analyze/

    Accepts an X-ray image upload, runs classification + Grad-CAM,
    and returns classification probabilities, predicted label, findings,
    heatmap as base64 PNG, and the feature embedding.
    """

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request: Request) -> Response:
        serializer = AnalyzeRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        uploaded: Any = request.FILES["image"]

        # Save to a temp file so vision_service can read it by path
        suffix: str = os.path.splitext(str(uploaded.name))[1] or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            for chunk in uploaded.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            result = analyze_xray(
                config_path=settings.VISION_CONFIG_PATH,
                image_path=tmp_path,
                checkpoint_path=settings.VISION_CHECKPOINT_PATH,
            )

            # Convert heatmap file to base64 if it exists
            heatmap_b64 = None
            heatmap_path: str | None = result.get("heatmap_path")  # type: ignore[assignment]
            if heatmap_path and os.path.isfile(heatmap_path):
                with open(heatmap_path, "rb") as f:
                    heatmap_b64 = base64.b64encode(f.read()).decode("utf-8")

            # Save the analysis to the database
            analysis = XRayAnalysis.objects.create(
                user=request.user,
                image=uploaded,
                class_probs=result["class_probs"],
                pred_label=result["pred_label"],
                findings=result["findings"],
                embedding=result["embedding"],
            )

            response_data = {
                "id": analysis.id,
                "class_probs": result["class_probs"],
                "pred_label": result["pred_label"],
                "findings": result["findings"],
                "heatmap_base64": heatmap_b64,
                "embedding": result["embedding"],
            }

            return Response(
                AnalyzeResponseSerializer(response_data).data,
                status=status.HTTP_200_OK,
            )
        except Exception:
            logger.exception("X-ray analysis failed")
            return Response(
                {"error": "Analysis failed. Please try again."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class EmbedView(APIView):
    """POST /api/v1/vision/embed/

    Batch-embeds images from the configured dataset into pgvector.
    """

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request: Request) -> Response:
        serializer = EmbedRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated = cast(dict[str, Any], serializer.validated_data)
        batch_size: int = validated["batch_size"]

        try:
            result = ingest_embeddings(
                config_path=settings.VISION_CONFIG_PATH,
                checkpoint_path=settings.VISION_CHECKPOINT_PATH,
                batch_size=batch_size,
            )
            return Response(
                EmbedResponseSerializer(result).data,
                status=status.HTTP_200_OK,
            )
        except Exception:
            logger.exception("Embedding ingestion failed")
            return Response(
                {"error": "Embedding ingestion failed."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class SimilarView(APIView):
    """POST /api/v1/vision/similar/

    Queries similar X-ray images by embedding vector.
    Expects JSON body with 'embedding' (list of floats) and optional 'k'.
    """

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request: Request) -> Response:
        serializer = SimilarRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated = cast(dict[str, Any], serializer.validated_data)
        embedding: list[float] = validated["embedding"]
        k: int = validated["k"]

        try:
            cfg = load_config(settings.VISION_CONFIG_PATH)
            conn = get_conn(cfg)
            rows = query_similar(conn, cfg["pgvector"]["table"], embedding, k=k)
            conn.close()

            results = [
                {
                    "image_id": row[0],
                    "label": row[1],
                    "metadata": row[2],
                    "distance": float(row[3]),
                }
                for row in rows
            ]

            return Response(
                SimilarItemSerializer(results, many=True).data,
                status=status.HTTP_200_OK,
            )
        except Exception:
            logger.exception("Similarity search failed")
            return Response(
                {"error": "Similarity search failed."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
