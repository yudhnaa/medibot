"""
Shared constants for the COVID-QA embedding pipeline.
"""

import os
from typing import cast

from chatbot.models import ChatbotConfig
from chatbot.models.medical_document import SectionType
from django.conf import settings
from vector_store.services.constants import DEFAULT_EMBEDDING_PROVIDER

# =============================================================================
# Prompt constants for LLM extraction
# =============================================================================

# Valid section types accepted by the search code (SECTION_ORDER in chatbot/services/constants.py)
VALID_SECTIONS = [
    "general",
    "symptom",
    "aetiologies",
    "risk",
    "diagnose_and_treaty",
    "living_and_preventive",
]

PROMPT_EXTRACT_DISEASES = """\
You are a medical NLP specialist. Extract ALL disease names, syndrome names, \
and pathogen names mentioned in the following medical text.

Rules:
- Return ONLY a JSON array of strings, no other text.
- Use the most common/standard name for each entity.
- Include pathogens (e.g. "SARS-CoV-2") as separate entries from the diseases they cause.
- Deduplicate: do not include "COVID-19" and "covid-19" separately.
- If no diseases are found, return an empty array [].

Example output: ["COVID-19", "SARS-CoV-2", "ARDS", "Pneumonia"]

Medical text:
---
{text}
---

JSON array:"""

PROMPT_EXTRACT_SUMMARY = """\
You are a medical data specialist. From the following medical article, extract a \
structured summary for the disease/condition "{disease_name}".

You MUST return valid JSON with exactly these keys (use empty string "" if not found):
{{
  "disease_name": "the disease name in lowercase",
  "general": "1-2 sentence definition/overview",
  "triệu chứng": "comma-separated list of key symptoms",
  "nguyên nhân": "comma-separated list of causes/etiology",
  "yếu tố nguy cơ": "comma-separated risk factors",
  "chẩn đoán và điều trị": "comma-separated diagnosis & treatment methods",
  "sinh hoạt và phòng ngừa": "comma-separated lifestyle & prevention tips"
}}

Return ONLY the JSON object. No markdown, no explanation.

Medical text:
---
{text}
---

JSON:"""

PROMPT_CLASSIFY_SECTIONS = """\
You are a medical text classifier. Classify the following medical text into sections.

You MUST return valid JSON with exactly these keys. Each value is the relevant text \
extracted from the article for that section. Use empty string "" if no relevant text found.

{{
  "general": "overview, definition, epidemiology text",
  "symptom": "clinical manifestations, signs, symptoms text",
  "aetiologies": "causes, pathogenesis, etiology, transmission text",
  "risk": "risk factors, vulnerable populations text",
  "diagnose_and_treaty": "diagnosis, treatment, management, therapy text",
  "living_and_preventive": "prevention, lifestyle, public health measures text"
}}

Rules:
- Extract actual text from the article, not just labels.
- Each section can contain multiple paragraphs.
- Text can appear in multiple sections if relevant.
- Return ONLY the JSON object.

Medical text:
---
{text}
---

JSON:"""

# =============================================================================
# Data Processing Constants
# =============================================================================

# Number of unique articles to embed.
# Change this to scale: 5 = quick test, 147 = full dataset.
NUM_DOCS_TO_PROCESS: int = 5

# =============================================================================
# LLM Configuration (for extraction / summarisation)
# =============================================================================
LLM_MODEL: str = "gemini-2.5-flash"

# Max Q&A pairs for evaluation dataset generation
QA_SAMPLE_SIZE: int = 200
QA_SAMPLE_MIN: int = 100

# =============================================================================
# Rate Limiting Configuration
# =============================================================================

# Sleep between embedding API calls (seconds)
EMBEDDING_SLEEP_SECONDS: float = 1.0

# Sleep between translation API calls (seconds)
TRANSLATION_SLEEP_SECONDS: float = 1.5

# Tenacity retry config for 429 (Too Many Requests)
RETRY_MAX_ATTEMPTS: int = 6
RETRY_WAIT_MIN: float = 2.0  # seconds
RETRY_WAIT_MAX: float = 60.0  # seconds

# =============================================================================
# Output Paths
# =============================================================================

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DATASET_OUTPUT_PATH = os.path.join(PIPELINE_DIR, "evaluation_dataset_vi.json")

# Source tag used to mark documents in MedicalDocument table
DOCUMENT_SOURCE_TAG: str = "covid_qa_deepset"

# =============================================================================
# Runtime configuration (from Django settings + ChatbotConfig)
# =============================================================================

GOOGLE_API_KEY = cast(str, getattr(settings, "GOOGLE_API_KEY", ""))

EMBEDDING_DIMENSIONS: int = cast(
    int, ChatbotConfig.get_config("VECTOR_DIMENSIONS", 768)
)
CHUNK_SIZE: int = cast(int, ChatbotConfig.get_config("CHUNK_SIZE", 1000))
CHUNK_OVERLAP: int = cast(int, ChatbotConfig.get_config("CHUNK_OVERLAP", 200))
EMBEDDING_PROVIDER: str = str(
    ChatbotConfig.get_config("EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER)
)

SECTION_TYPE_MAP = {
    "general": SectionType.GENERAL,
    "symptom": SectionType.SYMPTOM,
    "aetiologies": SectionType.AETIOLOGIES,
    "risk": SectionType.RISK,
    "diagnose_and_treaty": SectionType.DIAGNOSE_AND_TREATY,
    "living_and_preventive": SectionType.LIVING_AND_PREVENTIVE,
}
