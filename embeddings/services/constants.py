"""
Constants for embedding services.
"""

DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_PROVIDER_GEMINI = "gemini"
EMBEDDING_PROVIDER_TEI = "tei"
EMBEDDING_PROVIDER_TRANSFORMERS = "transformers"

DEFAULT_TRANSFORMERS_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_TRANSFORMERS_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

GOOGLE_API_KEY_ENV_NAME = "GOOGLE_API_KEY"
TEI_ENDPOINT_ENV_NAME = "TEI_ENDPOINT_URL"
TEI_EMBEDDING_ENDPOINT_SUFFIX = "/v1/embeddings"
