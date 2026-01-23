from rest_framework import serializers

from chatbot.models import UserIntake
from chatbot.schema.constants import MAX_AGE, MAX_DISEASE_NAME_LENGTH, MAX_ONSET_DAYS


class UserIntakeSerializer(serializers.ModelSerializer):
    """Serializer for UserIntake model with validation."""

    class Meta:
        model = UserIntake
        fields = [
            "disease_name",
            "symptoms",
            "symptoms_negated",
            "age",
            "sex",
            "pregnancy_status",
            "location_country",
            "chronic_conditions",
            "allergies",
            "onset_days",
            "meds",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["created_at", "updated_at"]

    def validate_age(self, value: int | None) -> int | None:
        """Validate age is within acceptable range."""
        if value is not None and (value < 0 or value > MAX_AGE):
            raise serializers.ValidationError(f"Age must be between 0 and {MAX_AGE}")
        return value

    def validate_disease_name(self, value: str | None) -> str | None:
        """Validate disease name length."""
        if value and len(value) > MAX_DISEASE_NAME_LENGTH:
            raise serializers.ValidationError(
                f"Disease name must not exceed {MAX_DISEASE_NAME_LENGTH} characters"
            )
        return value

    def validate_onset_days(self, value: int | None) -> int | None:
        """Validate onset days is within acceptable range."""
        if value is not None and (value < 0 or value > MAX_ONSET_DAYS):
            raise serializers.ValidationError(
                f"Onset days must be between 0 and {MAX_ONSET_DAYS}"
            )
        return value

    def validate_symptoms(self, value: list) -> list:
        """Validate symptoms is a list of strings."""
        if not isinstance(value, list):
            raise serializers.ValidationError("Symptoms must be a list")
        if not all(isinstance(item, str) for item in value):
            raise serializers.ValidationError("All symptoms must be strings")
        return value

    def validate_symptoms_negated(self, value: list) -> list:
        """Validate negated symptoms is a list of strings."""
        if not isinstance(value, list):
            raise serializers.ValidationError("Negated symptoms must be a list")
        if not all(isinstance(item, str) for item in value):
            raise serializers.ValidationError("All negated symptoms must be strings")
        return value

    def validate_chronic_conditions(self, value: list) -> list:
        """Validate chronic conditions is a list of strings."""
        if not isinstance(value, list):
            raise serializers.ValidationError("Chronic conditions must be a list")
        if not all(isinstance(item, str) for item in value):
            raise serializers.ValidationError("All chronic conditions must be strings")
        return value

    def validate_meds(self, value: list) -> list:
        """Validate medications is a list of strings."""
        if not isinstance(value, list):
            raise serializers.ValidationError("Medications must be a list")
        if not all(isinstance(item, str) for item in value):
            raise serializers.ValidationError("All medications must be strings")
        return value

    def validate_allergies(self, value: list) -> list:
        """Validate allergies is a list of strings."""
        if not isinstance(value, list):
            raise serializers.ValidationError("Allergies must be a list")
        if not all(isinstance(item, str) for item in value):
            raise serializers.ValidationError("All allergies must be strings")
        return value
