from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from django.conf import settings

from rest_framework.request import Request
from rest_framework.response import Response

CookieSameSite = Literal["Lax", "Strict", "None", False] | None


@dataclass(frozen=True)
class AuthSessionConfig:
    access_cookie_name: str
    refresh_cookie_name: str
    access_cookie_path: str
    refresh_cookie_path: str
    cookie_domain: str | None
    cookie_secure: bool
    cookie_samesite: CookieSameSite
    access_cookie_max_age: int
    refresh_cookie_max_age: int


def _normalize_cookie_samesite(value: object) -> CookieSameSite:
    if value in ("Lax", "Strict", "None", False, None):
        return value

    msg = "AUTH_SESSION['COOKIE_SAMESITE'] must be one of 'Lax', 'Strict', 'None', false, or null."
    raise ValueError(msg)


def get_auth_session_config() -> AuthSessionConfig:
    session_settings = settings.AUTH_SESSION
    return AuthSessionConfig(
        access_cookie_name=str(session_settings["ACCESS_COOKIE_NAME"]),
        refresh_cookie_name=str(session_settings["REFRESH_COOKIE_NAME"]),
        access_cookie_path=str(session_settings["ACCESS_COOKIE_PATH"]),
        refresh_cookie_path=str(session_settings["REFRESH_COOKIE_PATH"]),
        cookie_domain=session_settings.get("COOKIE_DOMAIN"),
        cookie_secure=bool(session_settings["COOKIE_SECURE"]),
        cookie_samesite=_normalize_cookie_samesite(session_settings["COOKIE_SAMESITE"]),
        access_cookie_max_age=int(session_settings["ACCESS_COOKIE_MAX_AGE"]),
        refresh_cookie_max_age=int(session_settings["REFRESH_COOKIE_MAX_AGE"]),
    )


def set_auth_cookies(
    response: Response,
    *,
    access_token: str,
    refresh_token: str | None,
) -> None:
    config = get_auth_session_config()
    response.set_cookie(
        key=config.access_cookie_name,
        value=access_token,
        max_age=config.access_cookie_max_age,
        httponly=True,
        secure=config.cookie_secure,
        samesite=config.cookie_samesite,
        domain=config.cookie_domain,
        path=config.access_cookie_path,
    )

    if refresh_token:
        response.set_cookie(
            key=config.refresh_cookie_name,
            value=refresh_token,
            max_age=config.refresh_cookie_max_age,
            httponly=True,
            secure=config.cookie_secure,
            samesite=config.cookie_samesite,
            domain=config.cookie_domain,
            path=config.refresh_cookie_path,
        )


def clear_auth_cookies(response: Response) -> None:
    config = get_auth_session_config()
    response.delete_cookie(
        key=config.access_cookie_name,
        path=config.access_cookie_path,
        domain=config.cookie_domain,
        samesite=config.cookie_samesite,
    )
    response.delete_cookie(
        key=config.refresh_cookie_name,
        path=config.refresh_cookie_path,
        domain=config.cookie_domain,
        samesite=config.cookie_samesite,
    )


def get_access_token_from_request_cookie(request: Request) -> str | None:
    config = get_auth_session_config()
    token = request.COOKIES.get(config.access_cookie_name)
    return str(token) if token else None


def get_refresh_token_from_request(request: Request) -> str | None:
    request_data = request.data
    request_refresh_token = (
        request_data.get("refresh") if isinstance(request_data, Mapping) else None
    )
    if isinstance(request_refresh_token, str) and request_refresh_token:
        return request_refresh_token

    config = get_auth_session_config()
    cookie_refresh_token = request.COOKIES.get(config.refresh_cookie_name)
    return str(cookie_refresh_token) if cookie_refresh_token else None
