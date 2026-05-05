from rest_framework import serializers

from authentication.models import Customer
from chatbot.schema.constants import MAX_BIO_LENGTH, MAX_NAME_LENGTH


class UserProfileSerializer(serializers.ModelSerializer):
    """Serializer for User Profile (Customer) model."""

    class Meta:
        model = Customer
        fields = [
            "id",
            "username",
            "email",
            "first_name",
            "last_name",
            "bio",
            "birth_date",
            "address",
            "date_joined",
            "last_login",
        ]
        read_only_fields = ["id", "username", "date_joined", "last_login"]

    def validate_first_name(self, value: str | None) -> str | None:
        if value and len(value) > MAX_NAME_LENGTH:
            raise serializers.ValidationError(
                f"First name must not exceed {MAX_NAME_LENGTH} characters"
            )
        return value

    def validate_last_name(self, value: str | None) -> str | None:
        if value and len(value) > MAX_NAME_LENGTH:
            raise serializers.ValidationError(
                f"Last name must not exceed {MAX_NAME_LENGTH} characters"
            )
        return value

    def validate_bio(self, value: str | None) -> str | None:
        if value and len(value) > MAX_BIO_LENGTH:
            raise serializers.ValidationError(
                f"Bio must not exceed {MAX_BIO_LENGTH} characters"
            )
        return value
