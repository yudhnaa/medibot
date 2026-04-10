# pylint: disable=wildcard-import,unused-wildcard-import
from .base import *  # noqa: F401,F403
from utils.logger import get_logging_config

ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

# CORS Settings for Development
CORS_ALLOWED_ORIGINS = ["http://localhost:3001", "http://127.0.0.1:3001"]

# Allow credentials (cookies, authorization headers)
CORS_ALLOW_CREDENTIALS = True

STATIC_ROOT = os.path.join(BASE_DIR, "static_root")

# Database
# https://docs.djangoproject.com/en/2.0/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.getenv("DB_DEV_NAME", "dev_db"),
        "USER": os.getenv("DB_DEV_USER", "postgres"),
        "PASSWORD": os.getenv("DB_DEV_PASSWORD", ""),
        "HOST": os.getenv("DB_DEV_HOST", "127.0.0.1"),
        "PORT": os.getenv("DB_DEV_PORT", "5432"),
    }
}

if DEBUG:
    try:
        import debug_toolbar  # NOQA  # pyright: ignore[reportUnusedImport]
    except ImportError:
        pass
    else:
        INSTALLED_APPS.append("debug_toolbar")
        INTERNAL_IPS = ["127.0.0.1"]
        MIDDLEWARE.insert(
            MIDDLEWARE.index("django.middleware.common.CommonMiddleware") + 1,
            "debug_toolbar.middleware.DebugToolbarMiddleware",
        )

LOGGING = get_logging_config(
    log_level=os.getenv("LOG_LEVEL", "DEBUG"),
    enable_colors=True,
    enable_django_debug=DEBUG,
    enable_sql_debug=ENABLE_SQL_DEBUG_LOGGING,
)
