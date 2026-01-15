from typing import Any, cast

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from embeddings.serializers import EmbedDocumentsSerializer, EmbedTextSerializer
from embeddings.services.constants import EMBEDDING_PROVIDER_GEMINI
from embeddings.services.embedding_service import EmbeddingService


class EmbedTextView(APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request: Request):
        serializer = EmbedTextSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        validated_data = cast(dict[str, Any], serializer.validated_data)
        text = validated_data["text"]
        provider = validated_data.get("provider", EMBEDDING_PROVIDER_GEMINI)

        embedding_service = EmbeddingService(provider=provider)
        embedding = embedding_service.embed_text(text)

        return Response(
            {"embedding": embedding, "provider": provider},
            status=status.HTTP_200_OK,
        )


class EmbedDocumentsView(APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request: Request):
        serializer = EmbedDocumentsSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        validated_data = cast(dict[str, Any], serializer.validated_data)
        texts = validated_data["texts"]
        provider = validated_data.get("provider", EMBEDDING_PROVIDER_GEMINI)

        embedding_service = EmbeddingService(provider=provider)
        embeddings = embedding_service.embed_documents(texts)

        return Response(
            {"embeddings": embeddings, "count": len(embeddings), "provider": provider},
            status=status.HTTP_200_OK,
        )
