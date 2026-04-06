"""
config.py - Compatibility layer for Django bootstrap & DB-backed settings.

This module is rarely used directly. Instead, import from embed_contexts.py
which handles Django setup properly. This module serves legacy entry points.
"""

import os
from typing import cast

import django
from dotenv import load_dotenv

# Setup environment & Django
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_template.settings.dev")
django.setup()

from chatbot.models import ChatbotConfig  # noqa: E402
from django.conf import settings  # noqa: E402
from vector_store.services.constants import DEFAULT_EMBEDDING_PROVIDER  # noqa: E402

GOOGLE_API_KEY = cast(str, getattr(settings, "GOOGLE_API_KEY", ""))
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY is not set. "
        "Please add it to your .env file or export it as an environment variable."
    )

# DB-driven settings (read from ChatbotConfig)
EMBEDDING_DIMENSIONS: int = cast(
    int, ChatbotConfig.get_config("VECTOR_DIMENSIONS", 768)
)
CHUNK_SIZE: int = cast(int, ChatbotConfig.get_config("CHUNK_SIZE", 1000))
CHUNK_OVERLAP: int = cast(int, ChatbotConfig.get_config("CHUNK_OVERLAP", 200))
EMBEDDING_PROVIDER: str = str(
    ChatbotConfig.get_config("EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER)
)

# Database settings for LangChain PGVector
DEFAULT_DB = settings.DATABASES.get("default", {})
DB_NAME = cast(str, DEFAULT_DB.get("NAME", "medibot"))
DB_USER = cast(str, DEFAULT_DB.get("USER", "postgres"))
DB_PASSWORD = cast(str, DEFAULT_DB.get("PASSWORD", "postgres"))
DB_HOST = cast(str, DEFAULT_DB.get("HOST", "localhost"))
DB_PORT = str(DEFAULT_DB.get("PORT", "5432"))

PG_CONNECTION_STRING = (
    f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
