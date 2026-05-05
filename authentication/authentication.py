from typing import Any

from rest_framework.request import Request

from rest_framework_simplejwt.authentication import JWTAuthentication

from authentication.session import get_access_token_from_request_cookie


class CookieJWTAuthentication(JWTAuthentication):
    """Authenticate JWTs from HttpOnly cookies only."""

    def authenticate(self, request: Request) -> tuple[Any, Any] | None:
        access_token = get_access_token_from_request_cookie(request)

        if not access_token:
            return None

        return self._authenticate_raw_token(access_token)

    def _authenticate_raw_token(self, raw_token: str) -> tuple[Any, Any]:
        validated_token = self.get_validated_token(raw_token.encode("utf-8"))
        return self.get_user(validated_token), validated_token
