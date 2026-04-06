from typing import Any, cast

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from vector_store.serializers import EmbedDocumentsSerializer, EmbedTextSerializer
from vector_store.services.embedding_service import EmbeddingService


class EmbedTextView(APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request: Request):
        serializer = EmbedTextSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        validated_data = cast(dict[str, Any], serializer.validated_data)
        text = validated_data["text"]

        embedding_service = EmbeddingService()
        embedding = embedding_service.embed_text(text)
        provider = embedding_service.get_provider_name()

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

        embedding_service = EmbeddingService()
        embeddings = embedding_service.embed_documents(texts)
        provider = embedding_service.get_provider_name()

        return Response(
            {"embeddings": embeddings, "count": len(embeddings), "provider": provider},
            status=status.HTTP_200_OK,
        )
