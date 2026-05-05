from django.contrib.auth.password_validation import validate_password

from rest_framework import serializers
from rest_framework.validators import UniqueValidator

from authentication.models import Customer


class RegisterSerializer(serializers.ModelSerializer):
    email = serializers.EmailField(
        required=True, validators=[UniqueValidator(queryset=Customer.objects.all())]
    )

    password = serializers.CharField(
        write_only=True, required=True, validators=[validate_password]
    )

    class Meta:
        model = Customer
        fields = ("username", "email", "password")

    def create(self, validated_data):
        # Customer inherits from User via multi-table inheritance
        # This creates both auth_user and customer records
        customer = Customer.objects.create(**validated_data)

        customer.set_password(validated_data["password"])
        customer.save()

        return customer
