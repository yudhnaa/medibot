from embeddings.services.constants import (
    EMBEDDING_PROVIDER_GEMINI,
    EMBEDDING_PROVIDER_TEI,
    EMBEDDING_PROVIDER_TRANSFORMERS,
)
from rest_framework import serializers


class EmbedTextSerializer(serializers.Serializer):
    text = serializers.CharField(required=True, allow_blank=False)
    provider = serializers.ChoiceField(
        choices=[
            EMBEDDING_PROVIDER_GEMINI,
            EMBEDDING_PROVIDER_TEI,
            EMBEDDING_PROVIDER_TRANSFORMERS,
        ],
        default=EMBEDDING_PROVIDER_TRANSFORMERS,
        required=False,
    )


class EmbedDocumentsSerializer(serializers.Serializer):
    texts = serializers.ListField(
        child=serializers.CharField(allow_blank=False),
        required=True,
        allow_empty=False,
    )
    provider = serializers.ChoiceField(
        choices=[
            EMBEDDING_PROVIDER_GEMINI,
            EMBEDDING_PROVIDER_TEI,
            EMBEDDING_PROVIDER_TRANSFORMERS,
        ],
        default=EMBEDDING_PROVIDER_TRANSFORMERS,
        required=False,
    )
