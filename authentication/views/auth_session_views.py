from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
from rest_framework_simplejwt.serializers import (
    TokenObtainPairSerializer,
    TokenRefreshSerializer,
)
from rest_framework_simplejwt.tokens import RefreshToken

from authentication.session import (
    clear_auth_cookies,
    get_refresh_token_from_request,
    set_auth_cookies,
)


class LoginView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request: Request) -> Response:
        serializer = TokenObtainPairSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        token_payload = dict(serializer.validated_data)
        response = Response(token_payload, status=status.HTTP_200_OK)
        set_auth_cookies(
            response,
            access_token=str(token_payload["access"]),
            refresh_token=str(token_payload["refresh"]),
        )
        return response


class RefreshView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request: Request) -> Response:
        refresh_token = get_refresh_token_from_request(request)
        if not refresh_token:
            raise InvalidToken("Refresh token was not provided")

        serializer = TokenRefreshSerializer(data={"refresh": refresh_token})
        serializer.is_valid(raise_exception=True)

        token_payload = dict(serializer.validated_data)
        rotated_refresh_token = str(token_payload.get("refresh") or refresh_token)

        response = Response(token_payload, status=status.HTTP_200_OK)
        set_auth_cookies(
            response,
            access_token=str(token_payload["access"]),
            refresh_token=rotated_refresh_token,
        )
        return response


class LogoutView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request: Request) -> Response:
        refresh_token = get_refresh_token_from_request(request)
        if refresh_token:
            self._try_blacklist_refresh_token(refresh_token)

        response = Response(
            {"message": "Logged out successfully"},
            status=status.HTTP_200_OK,
        )
        clear_auth_cookies(response)
        return response

    @staticmethod
    def _try_blacklist_refresh_token(refresh_token: str) -> None:
        try:
            RefreshToken(refresh_token).blacklist()
        except (AttributeError, InvalidToken, TokenError):
            # Always clear cookies even when token blacklisting fails.
            return
