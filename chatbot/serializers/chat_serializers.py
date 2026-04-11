"""
Chat Serializers
DRF serializers for chat API endpoints.
"""

from rest_framework import serializers

from chatbot.models import ChatMessage, ChatSession


class ChatSessionSerializer(serializers.ModelSerializer):
    """Serializer for ChatSession model."""

    message_count = serializers.SerializerMethodField()

    class Meta:
        model = ChatSession
        fields = [
            "id",
            "session_id",
            "title",
            "max_messages",
            "is_active",
            "created_at",
            "updated_at",
            "message_count",
        ]
        read_only_fields = [
            "id",
            "session_id",
            "created_at",
            "updated_at",
            "message_count",
        ]

    def get_message_count(self, obj: ChatSession) -> int:
        return ChatMessage.objects.filter(session=obj).count()


class ChatSessionCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating a chat session."""

    message_count = serializers.SerializerMethodField()

    class Meta:
        model = ChatSession
        fields = [
            "id",
            "session_id",
            "title",
            "max_messages",
            "is_active",
            "created_at",
            "updated_at",
            "message_count",
        ]
        read_only_fields = [
            "id",
            "session_id",
            "created_at",
            "updated_at",
            "message_count",
            "is_active",
        ]

    def get_message_count(self, obj: ChatSession) -> int:
        return 0  # New session starts with 0 messages


class ChatMessageSerializer(serializers.ModelSerializer):
    """Serializer for ChatMessage model."""

    class Meta:
        model = ChatMessage
        fields = [
            "id",
            "role",
            "content",
            "response_time_ms",
            "metadata",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]

    def to_representation(self, instance):
        """Clean metadata before sending to frontend."""
        ret = super().to_representation(instance)
        metadata = ret.get("metadata")
        if isinstance(metadata, dict) and "xray_analysis" in metadata:
            if isinstance(metadata["xray_analysis"], dict):
                metadata["xray_analysis"].pop("embedding", None)
        return ret


class ChatInputSerializer(serializers.Serializer):
    """Serializer for chat input."""

    session_id = serializers.UUIDField(required=True)
    message = serializers.CharField(required=True, max_length=10000)
    stream = serializers.BooleanField(default=False, required=False)
    xray_analysis_id = serializers.IntegerField(required=False, allow_null=True)
    image = serializers.ImageField(required=False, allow_null=True)


class ChatResponseSerializer(serializers.Serializer):
    """Serializer for chat response."""

    session_id = serializers.UUIDField()
    response = serializers.CharField()
    response_time_ms = serializers.IntegerField()
    source_urls = serializers.ListField(
        child=serializers.URLField(),
        required=False,
        allow_empty=True,
    )
    metadata = serializers.DictField(required=False)
    is_active = serializers.BooleanField(default=True, read_only=True)

    def to_representation(self, instance):
        """Clean metadata before sending to frontend."""
        ret = super().to_representation(instance)
        metadata = ret.get("metadata")
        if isinstance(metadata, dict) and "xray_analysis" in metadata:
            if isinstance(metadata["xray_analysis"], dict):
                metadata["xray_analysis"].pop("embedding", None)
        return ret


class UserIntakeInputSerializer(serializers.Serializer):
    """Serializer for user intake data."""

    disease_name = serializers.CharField(
        required=False, allow_blank=True, max_length=200
    )
    symptoms = serializers.ListField(
        child=serializers.CharField(max_length=100),
        required=False,
        allow_empty=True,
    )
    symptoms_negated = serializers.ListField(
        child=serializers.CharField(max_length=100),
        required=False,
        allow_empty=True,
    )
    age = serializers.IntegerField(
        required=False, min_value=0, max_value=150, allow_null=True
    )
    sex = serializers.ChoiceField(
        choices=["male", "female", "unknown"],
        required=False,
        default="unknown",
    )
    onset_days = serializers.IntegerField(
        required=False, min_value=0, max_value=3650, allow_null=True
    )
    pregnancy_status = serializers.CharField(
        required=False, allow_blank=True, allow_null=True
    )


class UserIntakeOutputSerializer(serializers.Serializer):
    """Serializer for user intake response."""

    disease_name = serializers.CharField(allow_null=True)
    symptoms = serializers.ListField(child=serializers.CharField())
    symptoms_negated = serializers.ListField(child=serializers.CharField())
    age = serializers.IntegerField(allow_null=True)
    sex = serializers.CharField()
    onset_days = serializers.IntegerField(allow_null=True)
    pregnancy_status = serializers.CharField(allow_null=True)


class RetrievedDocSerializer(serializers.Serializer):
    """Serializer for retrieved documents."""

    title = serializers.CharField(allow_null=True)
    section = serializers.CharField(allow_null=True)
    source = serializers.CharField(allow_null=True)
    url = serializers.URLField(allow_null=True, required=False)
    preview = serializers.CharField()
