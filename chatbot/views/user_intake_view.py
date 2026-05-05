from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response

from chatbot.models import UserIntake
from chatbot.serializers.user_intake_serializer import UserIntakeSerializer


@api_view(["GET", "PUT", "PATCH"])
@permission_classes([IsAuthenticated])
def user_intake_view(request: Request) -> Response:
    """
    Get or update current user's intake information.

    GET: Retrieve user's intake data
    PUT: Full update of intake data
    PATCH: Partial update of intake data

    Permissions:
        - User must be authenticated
        - User can only access their own intake
    """
    # Get or create intake for current user
    intake, created = UserIntake.objects.get_or_create(customer=request.user)

    if request.method == "GET":
        serializer = UserIntakeSerializer(intake)
        return Response(serializer.data)

    elif request.method in ["PUT", "PATCH"]:
        partial = request.method == "PATCH"
        serializer = UserIntakeSerializer(intake, data=request.data, partial=partial)

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def reset_intake_view(request: Request) -> Response:
    """
    Reset session-specific fields in user's intake.

    POST: Reset intake fields (disease_name, symptoms, etc.)

    Permissions:
        - User must be authenticated
        - User can only reset their own intake
    """
    try:
        intake = request.user.intake
        intake.reset_session_specific_fields()
        serializer = UserIntakeSerializer(intake)
        return Response(
            {
                "message": "Intake session fields reset successfully",
                "data": serializer.data,
            }
        )
    except UserIntake.DoesNotExist:
        return Response(
            {"error": "User intake not found"},
            status=status.HTTP_404_NOT_FOUND,
        )
