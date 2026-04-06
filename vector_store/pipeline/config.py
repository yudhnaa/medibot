"""
config.py - Shared configuration & Django bootstrap for the COVID-QA embedding pipeline.

Bootstraps Django settings so scripts can use the ORM (MedicalDocument),
loads .env for API keys, and defines all shared constants.

Model-related settings (VECTOR_DIMENSIONS, CHUNK_SIZE, CHUNK_OVERLAP) are
read from ChatbotConfig (DB) at runtime so they stay in sync with the rest
of the application.  Operational / rate-limit constants remain env-based.
"""

import os
import sys

# ============================================================================
# Django Bootstrap
# ============================================================================
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_template.settings.dev")

import django  # noqa: E402

django.setup()

# ============================================================================
# Google API Key — used by the LLM (disease extraction / summarisation).
# Embedding providers read this themselves; we only need it here for the LLM.
# ============================================================================
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY is not set. "
        "Please add it to your .env file or export it as an environment variable."
    )

# ============================================================================
# DB-driven settings (read from ChatbotConfig)
# ============================================================================
# Import after Django setup so the ORM is available.
from chatbot.models import ChatbotConfig  # noqa: E402
from vector_store.services.constants import DEFAULT_EMBEDDING_PROVIDER  # noqa: E402

# Embedding vector size — must match VectorField(dimensions=?) in MedicalDocument.
EMBEDDING_DIMENSIONS: int = int(
    ChatbotConfig.get_config("VECTOR_DIMENSIONS", 768)  # ty:ignore[invalid-argument-type] # type: ignore
)

# Text splitting settings.
CHUNK_SIZE: int = int(
    ChatbotConfig.get_config("CHUNK_SIZE", 1000)  # ty:ignore[invalid-argument-type] # type: ignore
)
CHUNK_OVERLAP: int = int(
    ChatbotConfig.get_config("CHUNK_OVERLAP", 200)  # ty:ignore[invalid-argument-type] # type: ignore
)

# Default embedding provider — which backend to use when embed_contexts() is called
EMBEDDING_PROVIDER: str = str(
    ChatbotConfig.get_config("EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER)  # type: ignore[arg-type]
)

# ============================================================================
# Data Processing Constants
# ============================================================================

# Number of unique articles to embed.
# Change this to scale: 5 = quick test, 147 = full dataset.
NUM_DOCS_TO_PROCESS: int = 5

# ============================================================================
# LLM Configuration (for extraction / summarisation)
# ============================================================================
LLM_MODEL: str = "gemini-2.5-flash"

# Max Q&A pairs for evaluation dataset generation
QA_SAMPLE_SIZE: int = 200
QA_SAMPLE_MIN: int = 100

# ============================================================================
# Rate Limiting Configuration
# ============================================================================

# Sleep between embedding API calls (seconds)
EMBEDDING_SLEEP_SECONDS: float = 1.0

# Sleep between translation API calls (seconds)
TRANSLATION_SLEEP_SECONDS: float = 1.5

# Tenacity retry config for 429 (Too Many Requests)
RETRY_MAX_ATTEMPTS: int = 6
RETRY_WAIT_MIN: float = 2.0  # seconds
RETRY_WAIT_MAX: float = 60.0  # seconds

# ============================================================================
# Output Paths
# ============================================================================
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DATASET_OUTPUT_PATH = os.path.join(PIPELINE_DIR, "evaluation_dataset_vi.json")

# Source tag used to mark documents in MedicalDocument table
DOCUMENT_SOURCE_TAG: str = "covid_qa_deepset"

# ============================================================================
# Database (PostgreSQL connection string for LangChain PGVector)
# ============================================================================
DB_NAME = os.environ.get("DB_DEV_NAME", "medibot")
DB_USER = os.environ.get("DB_DEV_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_DEV_PASSWORD", "postgres")
DB_HOST = os.environ.get("DB_DEV_HOST", "localhost")
DB_PORT = os.environ.get("DB_DEV_PORT", "5432")

PG_CONNECTION_STRING = (
    f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
