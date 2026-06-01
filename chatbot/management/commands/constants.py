"""
Default configuration values for chatbot seed data.
These are used by the seed_config management command.
"""

# =============================================================================
# RAG Pipeline Configuration
# =============================================================================
DEFAULT_RAG_CONFIG: dict[str, float | int] = {
    "RAG_TITLES_COLLECTION_THRESHOLD": 0.65,
    "RAG_CHUNKS_COLLECTION_TOPK": 20,
    "RAG_MERGED_LIMIT": 40,
    "RAG_TITLE_TOP_M": 5,
    "RAG_NEG_SYM_SIM_THRESH": 0.75,
    "RAG_PENALTY_ALPHA": 0.5,
    "RAG_FINAL_TITLES": 5,
    "RAG_MERGE_WEIGHT_ENTITIES": 0.5,
    "RAG_MERGE_WEIGHT_QUERY": 0.5,
}

# =============================================================================
# Model Configuration
# =============================================================================
DEFAULT_MODEL_CONFIG: dict[str, str | float | bool | int] = {
    "IS_USE_PAID_GOOGLE_GEMINI_API_KEY": False,
    "LLM_PROVIDER": "openrouter",  # Active LLM provider: gemini | openrouter
    "LLM_MODEL": "google/gemini-2.5-flash",
    "EMBEDDING_PROVIDER": "openrouter",  # Active embedding provider: gemini | tei | transformers | openrouter
    "EMBEDDING_MODEL": "openai/text-embedding-3-small",
    "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
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
    "ENABLE_LOCAL_NLP_FALLBACK": False,
    "RAG_LOG_STAGE1_PREPROCESS": True,
    "RAG_LOG_STAGE2_GATE": True,
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
