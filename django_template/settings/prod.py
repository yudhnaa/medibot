import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration

# pylint: disable=wildcard-import,unused-wildcard-import
from .base import *  # noqa: F401,F403
from utils.logger import get_logging_config

ALLOWED_HOSTS = []

DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

STATIC_ROOT = os.path.join(BASE_DIR, "static")

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
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DNS"),
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
