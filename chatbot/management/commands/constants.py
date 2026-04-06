"""
Default configuration values for chatbot seed data.
These are used by the seed_config management command.
"""

# =============================================================================
# RAG Pipeline Configuration
# =============================================================================
DEFAULT_RAG_CONFIG: dict[str, float | int] = {
    "RAG_THRESH_C": 0.75,
    "RAG_B_TOPK": 20,
    "RAG_TITLE_TOP_M": 5,
    "RAG_FINAL_TITLES": 5,
}

# =============================================================================
# Model Configuration
# =============================================================================
DEFAULT_MODEL_CONFIG: dict[str, str | float | bool | int] = {
    "IS_USE_PAID_GOOGLE_GEMINI_API_KEY": False,
    "LLM_MODEL": "gemini-2.5-flash",
    "EMBEDDING_PROVIDER": "openrouter",  # Active embedding provider: gemini | tei | transformers | openrouter
    "GEMINI_EMBEDDING_MODEL": "models/gemini-embedding-001",
    "TRANSFORMERS_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B",
    "OPENROUTER_EMBEDDING_MODEL": "text-embedding-3-small",
    "TEMPERATURE": 0,
    "VECTOR_DIMENSIONS": 768,
}

# =============================================================================
# Rate Limits Configuration
# =============================================================================
DEFAULT_RATE_LIMITS: dict[str, bool | int | None] = {}

# =============================================================================
# Feature Flags
# =============================================================================
DEFAULT_FEATURE_FLAGS: dict[str, bool] = {
    "VNCORENLP_ENABLED": True,
}

# =============================================================================
# Processing Configuration
# =============================================================================
DEFAULT_PROCESSING_CONFIG: dict[str, int] = {
    "CHUNK_SIZE": 1000,
    "CHUNK_OVERLAP": 200,
}

# =============================================================================
# Config Descriptions
# =============================================================================
CONFIG_DESCRIPTIONS: dict[str, str] = {
    "RAG_CONFIG": "RAG pipeline parameters (thresholds, weights, limits)",
    "MODEL_CONFIG": "LLM and embedding model settings",
    "RATE_LIMITS": "Embedding API rate limits",
    "FEATURE_FLAGS": "Feature toggles for optional functionality",
    "PROCESSING_CONFIG": "Text processing and chunking settings",
}
