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
    "RAG_MERGED_LIMIT": 40,
    "RAG_MERGE_WEIGHT_ENTITIES": 0.5,
    "RAG_MERGE_WEIGHT_QUERY": 0.5,
    "RAG_TITLE_TOP_M": 5,
    "RAG_FINAL_TITLES": 5,
    "RAG_PENALTY_ALPHA": 0.5,
    "RAG_NEG_SYM_SIM_THRESH": 0.75,
}

# =============================================================================
# Model Configuration
# =============================================================================
DEFAULT_MODEL_CONFIG: dict[str, str | float | bool | int] = {
    "IS_USE_PAID_GOOGLE_GEMINI_API_KEY": False,
    "LLM_MODEL": "gemini-2.5-flash",
    "EMBEDDING_MODEL": "models/gemini-embedding-001",
    "TEMPERATURE": 0,
    "EMBEDDINGS_NORMALIZED": True,
    "VECTOR_DIMENSIONS": 3072,
}

# =============================================================================
# Rate Limits Configuration
# =============================================================================
DEFAULT_RATE_LIMITS: dict[str, bool | int | None] = {
    "EMBEDDING_RATE_LIMIT_ENABLE": True,
    "EMBEDDING_RPM": 100,
    "EMBEDDING_TPM": 30000,
    "EMBEDDING_RPD": 1000,
    "PAID_EMBEDDING_RPM": 3000,
    "PAID_EMBEDDING_TPM": 1000000,
    "PAID_EMBEDDING_RPD": None,
}

# =============================================================================
# Feature Flags
# =============================================================================
DEFAULT_FEATURE_FLAGS: dict[str, bool] = {
    "VNCORENLP_ENABLED": True,
    "MEDICAL_DISCLAIMER_ENABLED": True,
    "CHAT_HISTORY_ENABLED": True,
}

# =============================================================================
# Processing Configuration
# =============================================================================
DEFAULT_PROCESSING_CONFIG: dict[str, int] = {
    "CHUNK_SIZE": 1000,
    "CHUNK_OVERLAP": 200,
    "BATCH_SIZE": 100,
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
