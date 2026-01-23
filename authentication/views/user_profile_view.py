from typing import override
from rest_framework import generics, permissions
from authentication.models import Customer
from authentication.serializers.user_profile_serializer import UserProfileSerializer


class UserProfileView(generics.RetrieveUpdateAPIView):
    """
    Get or update current user's profile information.

    GET: Retrieve user profile
    PATCH: Update user profile fields

    Permissions:
        - User must be authenticated
    """

    serializer_class = UserProfileSerializer
    permission_classes = [permissions.IsAuthenticated]

    @override
    def get_object(self) -> Customer:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return the authenticated user."""
        return self.request.user  # pyright: ignore[reportReturnType]
