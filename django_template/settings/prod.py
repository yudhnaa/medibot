try:
    import sentry_sdk
    from sentry_sdk.integrations.django import DjangoIntegration
except ImportError:  # pragma: no cover - optional prod dependency
    sentry_sdk = None
    DjangoIntegration = None

# pylint: disable=wildcard-import,unused-wildcard-import
from .base import *  # noqa: F401,F403
from utils.logger import get_logging_config


def _split_csv_env(name):
    return [value.strip() for value in os.getenv(name, "").split(",") if value.strip()]


ALLOWED_HOSTS = _split_csv_env("ALLOWED_HOSTS") or ["localhost", "127.0.0.1"]
CORS_ALLOWED_ORIGINS = _split_csv_env("CORS_ALLOWED_ORIGINS")
CSRF_TRUSTED_ORIGINS = _split_csv_env("CSRF_TRUSTED_ORIGINS")
CORS_ALLOW_CREDENTIALS = (
    os.getenv("CORS_ALLOW_CREDENTIALS", "True").lower() in ("true", "1", "yes")
)

_auth_cookie_samesite = os.getenv("AUTH_COOKIE_SAMESITE", AUTH_SESSION["COOKIE_SAMESITE"])
if _auth_cookie_samesite not in {"Lax", "Strict", "None"}:
    _auth_cookie_samesite = AUTH_SESSION["COOKIE_SAMESITE"]

AUTH_SESSION = {
    **AUTH_SESSION,
    "COOKIE_SECURE": os.getenv("AUTH_COOKIE_SECURE", "True").lower()
    in ("true", "1", "yes"),
    "COOKIE_SAMESITE": _auth_cookie_samesite,
}

DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

STATIC_ROOT = os.path.join(os.path.dirname(BASE_DIR), "static")

# Database
# https://docs.djangoproject.com/en/2.0/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.getenv("DB_NAME", "prod_db"),
        "USER": os.getenv("DB_USER", "postgres"),
        "PASSWORD": os.getenv("DB_PASSWORD", ""),
        "HOST": os.getenv("DB_HOST", "127.0.0.1"),
        "PORT": os.getenv("DB_PORT", "5432"),
    }
}

# Sentry
SENTRY_DSN = os.getenv("SENTRY_DNS")
if sentry_sdk is not None and DjangoIntegration is not None and SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        enable_tracing=True,
        integrations=[
            DjangoIntegration(
                transaction_style="url",
                middleware_spans=True,
                signals_spans=False,
                cache_spans=False,
            ),
        ],
    )

LOGGING = get_logging_config(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    enable_colors=False,
    enable_django_debug=False,
    enable_sql_debug=ENABLE_SQL_DEBUG_LOGGING,
)
