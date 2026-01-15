"""
Chatbot Serializers
DRF serializers for ChatSession and ChatMessage.
"""

from typing import Any
from rest_framework import serializers
from chatbot.models import ChatMessage, ChatSession, MedicalDocument


class ChatSessionSerializer(serializers.ModelSerializer):
    """Serializer for ChatSession."""

    class Meta:
        model = ChatSession
        fields = ["id", "session_id", "title", "created_at", "updated_at", "is_active"]
        read_only_fields = ["id", "session_id", "created_at", "updated_at"]

    def create(self, validated_data: dict[str, Any]):
        # Associate with current user from context
        user = self.context["request"].user

        # Link to Customer if the user is a Customer proxy or User model
        # Assuming user is Customer (since Customer inherits from User)
        # We need to ensure the user is authenticated
        if not user.is_authenticated:
            raise serializers.ValidationError(
                "Authentication required to create session"
            )

        validated_data["customer"] = user
        return super().create(validated_data)


class ChatMessageSerializer(serializers.ModelSerializer):
    """Serializer for ChatMessage."""

    class Meta:
        model = ChatMessage
        fields = ["id", "role", "content", "metadata", "created_at"]
        read_only_fields = ["id", "created_at"]


class MedicalDocumentSerializer(serializers.ModelSerializer):
    """Serializer for MedicalDocument."""

    class Meta:
        model = MedicalDocument
        fields = [
            "id",
            "title",
            "content",
            "section_type",
            "index_type",
            "source",
            "metadata",
        ]
