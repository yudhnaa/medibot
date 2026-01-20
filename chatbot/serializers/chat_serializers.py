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


class ChatInputSerializer(serializers.Serializer):
    """Serializer for chat input."""

    session_id = serializers.UUIDField(required=True)
    message = serializers.CharField(required=True, max_length=10000)
    stream = serializers.BooleanField(default=False, required=False)


class ChatResponseSerializer(serializers.Serializer):
    """Serializer for chat response."""

    session_id = serializers.UUIDField()
    response = serializers.CharField()
    response_time_ms = serializers.IntegerField()
    metadata = serializers.DictField(required=False)


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
    preview = serializers.CharField()
