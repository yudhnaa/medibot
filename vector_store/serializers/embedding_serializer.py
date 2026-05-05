from rest_framework import serializers


class EmbedTextSerializer(serializers.Serializer):
    text = serializers.CharField(required=True, allow_blank=False)


class EmbedDocumentsSerializer(serializers.Serializer):
    texts = serializers.ListField(
        child=serializers.CharField(allow_blank=False),
        required=True,
        allow_empty=False,
    )
