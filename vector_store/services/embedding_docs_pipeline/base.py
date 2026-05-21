"""
Base classes and shared logic for the embedding pipelines.
"""

import hashlib
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from django.conf import settings
from django.core.exceptions import SynchronousOnlyOperation

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from chatbot.models import (
    MEDICAL_DOCUMENTS_CHUNKS_COLLECTION,
    MEDICAL_DOCUMENTS_DISEASE_COLLECTION,
    MEDICAL_DOCUMENTS_TITLES_COLLECTION,
    get_collection_model,
    iter_collection_models,
)
from chatbot.models.chatbot_config import ChatbotConfig
from chatbot.models.medical_document import SectionType
from vector_store.services.constants import (
    OPENROUTER_API_KEY_ENV_NAME,
    OPENROUTER_BASE_URL_ENV_NAME,
)
from vector_store.services.embedding_docs_pipeline.article_schema import (
    parse_list_items,
)
from vector_store.services.embedding_docs_pipeline.constants import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COVID_QA_DATASET_NAME,
    DOCUMENT_SOURCE_TAG,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_PROVIDER,
    EMBEDDING_SLEEP_SECONDS,
    EXTRACTION_DATASET_OUTPUT_PATH,
    LLM_MODEL,
    LLM_PROVIDER,
    OPENROUTER_LLM_BASE_URL,
    PROMPT_CLASSIFY_SECTIONS,
    PROMPT_EXTRACT_DISEASES,
    PROMPT_EXTRACT_SUMMARY,
    RETRY_MAX_ATTEMPTS,
    RETRY_WAIT_MAX,
    RETRY_WAIT_MIN,
    SECTION_TYPE_MAP,
    VALID_SECTIONS,
)
from vector_store.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class BaseEmbeddingPipeline(ABC):
    """Abstract embedding pipeline base class."""

    def __init__(
        self,
        embedding_provider: str | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        **provider_kwargs,
    ) -> None:
        self.embedding_dimensions = int(
            self.get_runtime_config("VECTOR_DIMENSIONS", EMBEDDING_DIMENSIONS)
        )
        self.chunk_size = int(self.get_runtime_config("CHUNK_SIZE", CHUNK_SIZE))
        self.chunk_overlap = int(
            self.get_runtime_config("CHUNK_OVERLAP", CHUNK_OVERLAP)
        )
        self.embedding_provider = embedding_provider or str(
            self.get_runtime_config("EMBEDDING_PROVIDER", EMBEDDING_PROVIDER)
        )
        self.llm_provider = (
            (llm_provider or str(self.get_runtime_config("LLM_PROVIDER", LLM_PROVIDER)))
            .strip()
            .lower()
        )
        self.llm_model_override = llm_model
        self.llm_base_url_override = llm_base_url
        self.llm_api_key_override = llm_api_key
        self.provider_kwargs = provider_kwargs
        self.embedding_service = self.create_embedding_service(
            self.embedding_provider, **provider_kwargs
        )
        self.llm = self.create_llm()
        self.extraction_dataset_output_path = EXTRACTION_DATASET_OUTPUT_PATH
        self.extraction_records: dict[str, dict[str, Any]] = {}
        self._qa_pairs_by_context: dict[str, list[dict[str, Any]]] = {}
        self._titles_collection_count: int = 0

    def get_runtime_config(self, key: str, default: Any) -> Any:
        """Read config from DB at runtime with safe fallback for async startup."""
        try:
            return ChatbotConfig.get_config(key, default)
        except SynchronousOnlyOperation:
            logger.warning(
                "Skipping DB config lookup for %s in async context; using default.",
                key,
            )
            return default
        except Exception as exc:
            logger.warning(
                "Failed to load config %s from DB (%s); using default.",
                key,
                exc,
            )
            return default

    def create_llm(self) -> ChatGoogleGenerativeAI | ChatOpenAI:
        """Create LLM instance for extraction tasks."""
        provider = self.llm_provider

        if provider == "openrouter":
            model_name = self.llm_model_override or str(
                self.get_runtime_config("LLM_MODEL", "openai/gpt-4.1-mini")
            )
            self.llm_model = model_name
            base_url_default = (
                os.getenv(OPENROUTER_BASE_URL_ENV_NAME) or OPENROUTER_LLM_BASE_URL
            )
            base_url = (
                self.llm_base_url_override
                or str(
                    self.get_runtime_config("OPENROUTER_BASE_URL", base_url_default)
                ).strip()
            )
            api_key_default = os.getenv(OPENROUTER_API_KEY_ENV_NAME, "")
            api_key = (
                self.llm_api_key_override
                or str(
                    self.get_runtime_config(
                        OPENROUTER_API_KEY_ENV_NAME, api_key_default
                    )
                ).strip()
            )
            if not api_key:
                raise ValueError(
                    f"{OPENROUTER_API_KEY_ENV_NAME} is required for openrouter LLM provider"
                )

            return ChatOpenAI(
                model=model_name,
                api_key=SecretStr(api_key),
                base_url=base_url,
                temperature=0.1,
            )

        model_name = self.llm_model_override or str(
            self.get_runtime_config("LLM_MODEL", LLM_MODEL)
        )
        self.llm_model = model_name
        google_api_key_default = (
            os.getenv("GOOGLE_API_KEY")
            or str(getattr(settings, "GOOGLE_API_KEY", "")).strip()
        )
        google_api_key = (
            self.llm_api_key_override
            or str(
                self.get_runtime_config("GOOGLE_API_KEY", google_api_key_default)
            ).strip()
        )
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=SecretStr(google_api_key),
            temperature=0.1,
        )

    def create_embedding_service(
        self, provider: str | None = None, **provider_kwargs
    ) -> EmbeddingService:
        """Create the embedding service for the requested provider."""
        return EmbeddingService(provider=provider, **provider_kwargs)

    @abstractmethod
    def load_contexts(
        self, num_docs: int | None = None, start_article: int = 1
    ) -> list[dict]:
        """Load source contexts for the pipeline."""

    def load_qa_pairs(self, contexts: list[dict]) -> list[dict[str, Any]]:
        """Load QA pairs for provided contexts. Subclasses can override."""
        return []

    def _initialize_extraction_records(self, contexts: list[dict]) -> None:
        """Initialize per-article extraction records for dataset export."""
        records: dict[str, dict[str, Any]] = {}

        for ctx in contexts:
            context_id = str(ctx.get("context_id", ""))
            article_id = str(ctx.get("article_id") or context_id)
            records[article_id] = {
                "article_id": article_id,
                "context_id": context_id,
                "dataset": COVID_QA_DATASET_NAME,
                "llm_provider": self.llm_provider,
                "llm_model": self.llm_model,
                "stage_1": {
                    "diseases": [],
                    "route_units": [],
                },
                "stage_2": {
                    "faqs": [],
                },
                "stage_3": {
                    "sections": {},
                },
            }

        self.extraction_records = records

    def _persist_extraction_dataset(self) -> str:
        """Persist per-article extraction artifacts to JSON file."""
        output_dir = os.path.dirname(self.extraction_dataset_output_path)
        os.makedirs(output_dir, exist_ok=True)

        payload = {
            "dataset": COVID_QA_DATASET_NAME,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "article_count": len(self.extraction_records),
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "articles": list(self.extraction_records.values()),
        }

        with open(self.extraction_dataset_output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info(
            "Saved extraction dataset: %s (articles=%s)",
            self.extraction_dataset_output_path,
            len(self.extraction_records),
        )
        return self.extraction_dataset_output_path

    def _context_id_to_article_id_map(self, contexts: list[dict]) -> dict[str, str]:
        """Build context_id -> article_id mapping for metadata tracking."""
        mapping: dict[str, str] = {}
        for ctx in contexts:
            context_id = str(ctx.get("context_id", ""))
            if not context_id:
                continue
            mapping[context_id] = str(ctx.get("article_id") or context_id)
        return mapping

    def _group_qa_pairs_by_context(
        self, qa_pairs: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Group QA rows by context_id."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        for qa in qa_pairs:
            context_id = str(qa.get("context_id", "")).strip()
            if not context_id:
                continue
            grouped.setdefault(context_id, []).append(qa)
        return grouped

    def _normalize_text(self, text: str) -> str:
        """Normalize free text for deterministic storage and hashing."""
        lowered = str(text or "").strip().lower()
        lowered = lowered.replace("covid 19", "covid-19")
        lowered = re.sub(r"\s+", " ", lowered)
        return lowered

    def _build_qa_id(self, context_id: str, question: str, answer: str) -> str:
        """Build deterministic QA ID from context/question/answer."""
        payload = "|".join(
            [
                self._normalize_text(context_id),
                self._normalize_text(question),
                self._normalize_text(answer),
            ]
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]

    def _infer_intent(self, question: str, answer: str = "") -> str:
        """Infer a coarse intent label from QA text."""
        text = f"{question} {answer}".lower()

        rules = [
            (
                "symptom",
                [
                    "triệu chứng",
                    "dấu hiệu",
                    "symptom",
                    "manifestation",
                    "biểu hiện",
                ],
            ),
            (
                "cause_transmission",
                [
                    "nguyên nhân",
                    "lây",
                    "transmission",
                    "cause",
                    "etiolog",
                    "pathogenesis",
                ],
            ),
            (
                "risk_factor",
                [
                    "yếu tố nguy cơ",
                    "nguy cơ",
                    "risk factor",
                    "susceptible",
                    "vulnerable",
                ],
            ),
            (
                "diagnosis_treatment",
                [
                    "chẩn đoán",
                    "điều trị",
                    "thuốc",
                    "xét nghiệm",
                    "treatment",
                    "diagnos",
                    "therapy",
                    "manage",
                ],
            ),
            (
                "prevention_lifestyle",
                [
                    "phòng ngừa",
                    "phòng bệnh",
                    "prevent",
                    "lifestyle",
                    "hygiene",
                    "mask",
                    "vaccine",
                ],
            ),
            (
                "prognosis_complication",
                [
                    "biến chứng",
                    "tiên lượng",
                    "prognosis",
                    "complication",
                    "mortality",
                ],
            ),
            (
                "epidemiology",
                [
                    "dịch tễ",
                    "tỷ lệ",
                    "prevalence",
                    "incidence",
                    "outbreak",
                    "pandemic",
                ],
            ),
            (
                "definition_overview",
                [
                    "là gì",
                    "what is",
                    "định nghĩa",
                    "overview",
                    "general",
                ],
            ),
        ]

        for intent, keywords in rules:
            if any(keyword in text for keyword in keywords):
                return intent
        return "other_medical"

    def _build_context_aliases(
        self, disease_to_contexts: dict[str, set[str]]
    ) -> dict[str, list[str]]:
        """Build context_id -> alias list mapping from Stage 1 results."""
        context_aliases: dict[str, set[str]] = {}
        for disease_name, context_ids in disease_to_contexts.items():
            alias = str(disease_name).strip().lower()
            if not alias:
                continue
            for context_id in context_ids:
                context_aliases.setdefault(context_id, set()).add(alias)
        return {
            context_id: sorted(list(aliases))
            for context_id, aliases in context_aliases.items()
        }

    def _canonical_title_score(self, alias: str) -> float:
        """Score alias for canonical title selection."""
        score = float(len(alias))
        canonical_keywords = (
            "infection",
            "disease",
            "syndrome",
            "condition",
            "illness",
            "disorder",
        )
        if any(keyword in alias for keyword in canonical_keywords):
            score += 100.0
        return score

    def _select_canonical_title(self, aliases: list[str]) -> str:
        """Select a canonical title from alias list for disease and chunk collection storage."""
        cleaned = sorted({str(alias).strip().lower() for alias in aliases if alias})
        if not cleaned:
            return ""
        return max(
            cleaned,
            key=lambda alias: (self._canonical_title_score(alias), len(alias), alias),
        )

    def _build_route_text(
        self,
        canonical_title: str,
        aliases: list[str],
        intent: str,
        sample_questions: list[str],
    ) -> str:
        """Compose route unit text for titles collection embedding."""
        deduped_aliases = sorted(
            {
                self._normalize_text(alias)
                for alias in aliases
                if self._normalize_text(alias)
                and self._normalize_text(alias) != canonical_title
            }
        )
        question_block = " | ".join(
            [
                self._normalize_text(question)
                for question in sample_questions[:3]
                if question
            ]
        )
        alias_block = ", ".join(deduped_aliases)
        parts = [canonical_title, intent]
        if alias_block:
            parts.append(f"aliases: {alias_block}")
        if question_block:
            parts.append(f"query_examples: {question_block}")
        return " | ".join(parts)

    def extract_diseases_from_context(self, context_text: str) -> list[str]:
        """Extract disease/syndrome/pathogen names from a single article."""
        text = context_text[:8000] if len(context_text) > 8000 else context_text
        prompt = PROMPT_EXTRACT_DISEASES.format(text=text)

        try:
            response = self._llm_invoke(self.llm, prompt)
            diseases = self._parse_json_response(response)
            if isinstance(diseases, list):
                return [str(d).strip() for d in diseases if str(d).strip()]
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Disease extraction failed: {e}")

        return []

    def build_summary_text(self, summary_data: dict) -> str:
        """Convert structured summary data to text."""
        lines = []

        disease_name = summary_data.get("disease_name", "").strip()
        if disease_name:
            lines.append(disease_name)

        general = summary_data.get("general", "").strip()
        if general:
            lines.append(general)

        section_labels = [
            ("triệu chứng", summary_data.get("triệu chứng", "")),
            ("nguyên nhân", summary_data.get("nguyên nhân", "")),
            ("yếu tố nguy cơ", summary_data.get("yếu tố nguy cơ", "")),
            ("chẩn đoán và điều trị", summary_data.get("chẩn đoán và điều trị", "")),
            (
                "sinh hoạt và phòng ngừa",
                summary_data.get("sinh hoạt và phòng ngừa", ""),
            ),
        ]

        for label, value in section_labels:
            if value and value.strip():
                lines.append(f"{label}: {value.strip()}")

        return "\n".join(lines)

    def classify_sections(self, context_text: str) -> dict[str, str]:
        """Classify a context into section-specific text blocks."""
        text = context_text[:8000] if len(context_text) > 8000 else context_text
        prompt = PROMPT_CLASSIFY_SECTIONS.format(text=text)

        response = self._llm_invoke(self.llm, prompt)
        sections = self._parse_json_response(response)
        if not isinstance(sections, dict):
            raise ValueError("LLM returned non-dict section payload")
        return sections

    def load_csv(self, csv_path: str):
        """Load a CSV file into a DataFrame."""
        import pandas as pd

        return pd.read_csv(csv_path)

    def create_document_content(self, section: str, row) -> str | None:
        """Create content for a specific section of a CSV row."""
        if section not in row:
            return None

        val = row[section]
        try:
            import pandas as pd

            if pd.isna(val):
                return None
        except Exception:
            pass

        if not str(val).strip():
            return None

        try:
            if section != "general":
                items = self._safe_parse_list(val)
                content = ", ".join(item.strip() for item in items)
                return content if content else None
            content = str(val).strip()
            return content if content else None
        except Exception as e:
            logger.warning(f"Failed to parse section {section}: {e}")
            return None

    def _safe_parse_list(self, value) -> list[str]:
        """Safely parse a Python list serialized as a string."""
        import ast

        if not value or not isinstance(value, str):
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [
                    str(x).strip() for x in parsed if isinstance(x, (str, int, float))
                ]
        except Exception:
            return [s.strip() for s in value.split(",") if s.strip()]
        return []

    def _split_sentences(self, text: str) -> list[str]:
        """Simple sentence splitter for Vietnamese text."""
        import re

        if not text:
            return []
        text = " ".join(str(text).strip().split())
        parts = re.split(r"([.!?]+)\s+", text)
        sentences: list[str] = []
        for i in range(0, len(parts), 2):
            sent = parts[i]
            punct = parts[i + 1] if i + 1 < len(parts) else ""
            full = (sent + punct).strip()
            if full:
                sentences.append(full)
        return sentences

    def _extract_items_from_value(self, val, k: int = 3) -> list[str]:
        """Extract up to k items from a cell value (list string or comma-separated)."""
        if val is None:
            return []
        try:
            import pandas as pd

            if isinstance(val, float) and pd.isna(val):
                return []
        except Exception:
            pass
        s = str(val).strip()
        if not s or s.lower() in {"[]", "none", "null", "nan"}:
            return []
        return self._parse_items_string(s, k)

    def _parse_items_string(self, s: str, k: int) -> list[str]:
        """Parse a string that may be a Python list literal or comma-separated values."""
        import ast
        import re

        try:
            if s.startswith("[") and s.endswith("]"):
                obj = ast.literal_eval(s)
                if isinstance(obj, list):
                    return [str(x).strip() for x in obj[:k] if str(x).strip()]
        except Exception:
            pass
        items = []
        for part in re.split(r"[,;•·\n]+", s):
            item = part.strip(" \t-•·")
            if item and len(items) < k:
                items.append(item)
        return items[:k]

    def _build_summary_parts(self, row) -> list[str]:
        """Build the list of summary parts from row data."""
        parts = []
        title = str(row.get("title", "")).strip()
        general = str(row.get("general", "")).strip()

        if title:
            parts.append(title)

        general_sents = self._split_sentences(general)
        if general_sents:
            parts.append(general_sents[0])

        sections = [
            ("symptom", "triệu chứng"),
            ("aetiologies", "nguyên nhân"),
            ("risk", "yếu tố nguy cơ"),
            ("diagnose_and_treaty", "chẩn đoán và điều trị"),
            ("living_and_preventive", "sinh hoạt và phòng ngừa"),
        ]
        for section, section_vn in sections:
            items = self._extract_items_from_value(row.get(section), k=3)
            if items:
                parts.append(f"{section_vn}: {', '.join(items)}")

        return parts

    def build_disease_collection_summary(self, row) -> str | None:
        """Construct disease-level summary for disease collection."""
        title = str(row.get("title", "")).strip()
        general = str(row.get("general", "")).strip()

        if not title and not general:
            return None

        parts = self._build_summary_parts(row)
        summary = "\n".join(parts).strip()
        return summary if summary else None

    def build_documents_from_csv(
        self,
        csv_path: str,
        source: str,
        collection_name: str = MEDICAL_DOCUMENTS_CHUNKS_COLLECTION,
    ) -> list[dict[str, object]]:
        """Build document dictionaries from a CSV file."""
        documents: list[dict[str, object]] = []
        df = self.load_csv(csv_path)

        if collection_name == MEDICAL_DOCUMENTS_DISEASE_COLLECTION:
            for idx, row in df.iterrows():
                content = self.build_disease_collection_summary(row)
                if content:
                    documents.append(
                        {
                            "content": content.lower(),
                            "title": str(row.get("title", "")).strip().lower(),
                            "section_type": SectionType.GENERAL,
                            "collection_name": collection_name,
                            "source": source.lower(),
                            "metadata": {
                                "row_index": idx,
                                "url": str(row.get("url", "")).lower(),
                            },
                        }
                    )
            return documents

        if collection_name == MEDICAL_DOCUMENTS_TITLES_COLLECTION:
            for idx, row in df.iterrows():
                title = str(row.get("title", "")).strip()
                aliases = parse_list_items(row.get("aliases"))
                title_aliases = [title.lower()]
                title_aliases.extend(alias.lower() for alias in aliases)
                title_aliases = list(
                    dict.fromkeys(alias for alias in title_aliases if alias)
                )
                if title:
                    documents.append(
                        {
                            "content": " | ".join(title_aliases),
                            "title": title.lower(),
                            "section_type": SectionType.GENERAL,
                            "collection_name": collection_name,
                            "source": source.lower(),
                            "metadata": {
                                "row_index": idx,
                                "url": str(row.get("url", "")).lower(),
                                "title_aliases": title_aliases,
                            },
                        }
                    )
            return documents

        for idx, row in df.iterrows():
            for section in VALID_SECTIONS:
                content = self.create_document_content(section, row)
                if content:
                    documents.append(
                        {
                            "content": content.strip().lower(),
                            "title": str(row.get("title", "")).strip().lower(),
                            "section_type": SECTION_TYPE_MAP.get(
                                section, SectionType.GENERAL
                            ),
                            "collection_name": MEDICAL_DOCUMENTS_CHUNKS_COLLECTION,
                            "source": source.lower(),
                            "metadata": {
                                "row_index": idx,
                                "url": str(row.get("url", "")).lower(),
                            },
                        }
                    )

        return documents

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents through the configured embedding service."""
        return self._embed_texts_with_retry(self.embedding_service, texts)

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text through the configured embedding service."""
        return self.embedding_service.embed_text(text)

    def run(
        self,
        num_docs: int | None = None,
        start_article: int = 1,
        clear_existing: bool = True,
    ) -> dict:
        """Run the three-stage pipeline."""
        contexts = self.load_contexts(num_docs=num_docs, start_article=start_article)
        if not contexts:
            logger.warning("No contexts loaded. Pipeline cannot continue.")
            return {
                "titles_collection": 0,
                "disease_collection": 0,
                "chunks_collection": 0,
            }

        self._initialize_extraction_records(contexts)
        qa_pairs = self.load_qa_pairs(contexts)
        self._qa_pairs_by_context = self._group_qa_pairs_by_context(qa_pairs)
        if clear_existing:
            clear_covid_qa_documents()

        disease_to_contexts = self.stage_1_titles_collection(contexts)
        if not disease_to_contexts:
            logger.error("Stage 1 failed — no diseases extracted. Aborting.")
            extraction_path = self._persist_extraction_dataset()
            return {
                "articles": len(contexts),
                "titles_collection": 0,
                "disease_collection": 0,
                "chunks_collection": 0,
                "extraction_dataset_path": extraction_path,
            }
        if self._titles_collection_count == 0:
            self._titles_collection_count = len(disease_to_contexts)

        disease_collection_count = self.stage_2_disease_collection(
            contexts, disease_to_contexts
        )
        chunks_collection_count = self.stage_3_chunks_collection(
            contexts, disease_to_contexts
        )
        extraction_path = self._persist_extraction_dataset()

        return {
            "articles": len(contexts),
            "titles_collection": self._titles_collection_count,
            "disease_collection": disease_collection_count,
            "chunks_collection": chunks_collection_count,
            "total": self._titles_collection_count
            + disease_collection_count
            + chunks_collection_count,
            "extraction_dataset_path": extraction_path,
        }

    def stage_1_titles_collection(self, contexts: list[dict]) -> dict[str, set[str]]:
        """Stage 1: Extract diseases, build route units, and populate titles collection."""
        logger.info("%s", "=" * 60)
        logger.info("STAGE 1: Titles collection — Router Units (Disease + Intent)")
        logger.info("%s", "=" * 60)

        disease_to_contexts = self._stage_1_collect_disease_contexts(contexts)
        if not disease_to_contexts:
            logger.warning("No diseases extracted. Pipeline cannot continue.")
            self._titles_collection_count = 0
            return {}

        logger.info(
            f"Extracted {len(disease_to_contexts)} unique diseases: "
            f"{list(disease_to_contexts.keys())}"
        )

        context_to_article = self._context_id_to_article_id_map(contexts)
        context_aliases = self._build_context_aliases(disease_to_contexts)
        route_units = self._stage_1_route_units(
            context_aliases=context_aliases,
            context_to_article=context_to_article,
        )
        route_payloads = self._stage_1_route_payloads(route_units)
        self._store_titles_collection_route_payloads(route_payloads)

        self._titles_collection_count = len(route_payloads)
        logger.info(
            "Titles collection: %s route entries stored", self._titles_collection_count
        )
        return disease_to_contexts

    def _stage_1_collect_disease_contexts(
        self,
        contexts: list[dict],
    ) -> dict[str, set[str]]:
        disease_to_contexts: dict[str, set[str]] = {}
        for ctx in tqdm(contexts, desc="Stage 1: Extracting diseases"):
            context_id = str(ctx.get("context_id", ""))
            context_text = str(ctx.get("context", ""))
            article_id = str(ctx.get("article_id") or context_id)

            diseases = self.extract_diseases_from_context(context_text)
            unique_diseases: set[str] = set()
            for disease in diseases:
                normalized = disease.strip().lower()
                if normalized:
                    disease_to_contexts.setdefault(normalized, set()).add(context_id)
                    unique_diseases.add(normalized)

            if article_id in self.extraction_records:
                self.extraction_records[article_id]["stage_1"]["diseases"] = sorted(
                    unique_diseases
                )

            time.sleep(EMBEDDING_SLEEP_SECONDS)
        return disease_to_contexts

    def _stage_1_route_units(
        self,
        *,
        context_aliases: dict[str, list[str]],
        context_to_article: dict[str, str],
    ) -> dict[str, dict[str, Any]]:
        route_units: dict[str, dict[str, Any]] = {}
        for context_id, aliases in context_aliases.items():
            canonical_title = self._select_canonical_title(aliases)
            if not canonical_title:
                continue

            qa_pairs = self._qa_pairs_by_context.get(context_id, [])
            intents = {
                self._infer_intent(
                    str(qa.get("question", "")),
                    str(qa.get("answer", "")),
                )
                for qa in qa_pairs
                if str(qa.get("question", "")).strip()
            }
            if not intents:
                intents = {"other_medical"}

            sample_questions = [
                str(qa.get("question", "")).strip()
                for qa in qa_pairs
                if str(qa.get("question", "")).strip()
            ]

            article_id = context_to_article.get(context_id, context_id)
            for intent in intents:
                route_unit_id = f"{canonical_title}|{intent}"
                route_unit = route_units.setdefault(
                    route_unit_id,
                    {
                        "canonical_title": canonical_title,
                        "aliases": set(),
                        "intent": intent,
                        "context_ids": set(),
                        "article_ids": set(),
                        "questions": [],
                    },
                )
                route_unit["aliases"].update(aliases)
                route_unit["context_ids"].add(context_id)
                route_unit["article_ids"].add(article_id)
                route_unit["questions"].extend(sample_questions[:3])

                if article_id in self.extraction_records:
                    route_item = {
                        "route_unit_id": route_unit_id,
                        "canonical_title": canonical_title,
                        "intent": intent,
                    }
                    stage_1_route_units = self.extraction_records[article_id][
                        "stage_1"
                    ]["route_units"]
                    if route_item not in stage_1_route_units:
                        stage_1_route_units.append(route_item)
        return route_units

    def _stage_1_route_payloads(
        self,
        route_units: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        route_payloads: list[dict[str, Any]] = []
        for route_unit_id, unit in route_units.items():
            canonical_title = str(unit["canonical_title"])
            aliases = sorted(
                {str(alias) for alias in unit["aliases"] if str(alias).strip()}
            )
            intent = str(unit["intent"])
            questions = [
                self._normalize_text(question)
                for question in unit["questions"]
                if self._normalize_text(question)
            ]
            route_text = self._build_route_text(
                canonical_title=canonical_title,
                aliases=aliases,
                intent=intent,
                sample_questions=questions,
            )
            route_payloads.append(
                {
                    "route_unit_id": route_unit_id,
                    "title": canonical_title,
                    "content": route_text,
                    "metadata": {
                        "dataset": COVID_QA_DATASET_NAME,
                        "canonical_title": canonical_title,
                        "title_aliases": aliases,
                        "primary_intent": intent,
                        "intent_variants": [intent],
                        "source_context_ids": sorted(
                            {str(context_id) for context_id in unit["context_ids"]}
                        ),
                        "source_article_ids": sorted(
                            {str(article_id) for article_id in unit["article_ids"]}
                        ),
                        "route_unit_id": route_unit_id,
                    },
                }
            )
        return route_payloads

    def _store_titles_collection_route_payloads(
        self,
        route_payloads: list[dict[str, Any]],
    ) -> None:
        embeddings = self.embed_documents(
            [str(payload["content"]).lower() for payload in route_payloads]
        )
        for payload, embedding in zip(route_payloads, embeddings):
            get_collection_model(MEDICAL_DOCUMENTS_TITLES_COLLECTION).objects.create(
                title=str(payload["title"]),
                content=str(payload["content"]).lower(),
                embedding=embedding,
                section_type=SectionType.GENERAL,
                source=DOCUMENT_SOURCE_TAG,
                metadata=payload["metadata"],
            )

    def stage_2_disease_collection(
        self,
        contexts: list[dict],
        disease_to_contexts: dict[str, set[str]],
    ) -> int:
        """Stage 2: Build FAQ question vectors and answer payloads in disease collection."""
        logger.info("%s", "=" * 60)
        logger.info("STAGE 2: Disease collection — FAQ Question Embedding")
        logger.info("%s", "=" * 60)

        ctx_lookup = {
            str(ctx["context_id"]): ctx for ctx in contexts if ctx.get("context_id")
        }
        context_aliases = self._build_context_aliases(disease_to_contexts)
        count = 0
        qa_seen: set[str] = set()

        for context_id, aliases in tqdm(
            context_aliases.items(), desc="Stage 2: Building FAQs"
        ):
            ctx_data = ctx_lookup.get(context_id)
            if not ctx_data:
                continue

            canonical_title = self._select_canonical_title(aliases)
            if not canonical_title:
                continue

            article_id = str(ctx_data.get("article_id") or context_id)
            qa_pairs = self._qa_pairs_by_context.get(context_id, [])
            if not qa_pairs:
                count += self._stage_2_create_summary_fallback(
                    ctx_data=ctx_data,
                    context_id=context_id,
                    article_id=article_id,
                    canonical_title=canonical_title,
                    aliases=aliases,
                )
                continue

            docs_to_create = self._stage_2_qa_docs(
                context_id=context_id,
                article_id=article_id,
                canonical_title=canonical_title,
                aliases=aliases,
                qa_pairs=qa_pairs,
                qa_seen=qa_seen,
            )

            if not docs_to_create:
                continue

            count += self._store_disease_collection_docs(docs_to_create)
            time.sleep(EMBEDDING_SLEEP_SECONDS)

        logger.info("Disease collection: %s FAQ entries stored", count)
        return count

    def _stage_2_create_summary_fallback(
        self,
        *,
        ctx_data: dict,
        context_id: str,
        article_id: str,
        canonical_title: str,
        aliases: list[str],
    ) -> int:
        context_text = str(ctx_data.get("context", ""))
        if not context_text.strip():
            return 0
        try:
            summary_text = self._stage_2_summary_text(
                context_text=context_text,
                canonical_title=canonical_title,
            )
            if not summary_text.strip():
                return 0
            self._store_disease_collection_summary_doc(
                context_id=context_id,
                article_id=article_id,
                canonical_title=canonical_title,
                aliases=aliases,
                summary_text=summary_text,
            )
        except Exception as exc:
            logger.warning(
                "Summary fallback failed for %s (context %s): %s",
                canonical_title,
                context_id,
                exc,
            )
            return 0
        return 1

    def _stage_2_summary_text(
        self,
        *,
        context_text: str,
        canonical_title: str,
    ) -> str:
        text = context_text[:8000] if len(context_text) > 8000 else context_text
        prompt = PROMPT_EXTRACT_SUMMARY.format(
            disease_name=canonical_title,
            text=text,
        )
        response = self._llm_invoke(self.llm, prompt)
        summary_data = self._parse_json_response(response)
        if not isinstance(summary_data, dict):
            return ""
        summary_data["disease_name"] = canonical_title
        return self.build_summary_text(summary_data)

    def _store_disease_collection_summary_doc(
        self,
        *,
        context_id: str,
        article_id: str,
        canonical_title: str,
        aliases: list[str],
        summary_text: str,
    ) -> None:
        qa_id = self._build_qa_id(context_id, canonical_title, summary_text)
        embedding = self.embed_documents([summary_text.lower()])[0]
        get_collection_model(MEDICAL_DOCUMENTS_DISEASE_COLLECTION).objects.create(
            title=canonical_title,
            content=summary_text.lower(),
            embedding=embedding,
            section_type=SectionType.GENERAL,
            source=DOCUMENT_SOURCE_TAG,
            metadata={
                "context_id": context_id,
                "article_id": article_id,
                "canonical_title": canonical_title,
                "title_aliases": aliases,
                "primary_intent": "definition_overview",
                "secondary_intents": [],
                "answer_text": summary_text.strip(),
                "qa_id": qa_id,
                "dataset": COVID_QA_DATASET_NAME,
            },
        )
        self._record_stage_2_faq(
            article_id=article_id,
            canonical_title=canonical_title,
            aliases=aliases,
            qa_id=qa_id,
            question=canonical_title,
            intent="definition_overview",
        )

    def _stage_2_qa_docs(
        self,
        *,
        context_id: str,
        article_id: str,
        canonical_title: str,
        aliases: list[str],
        qa_pairs: list[dict],
        qa_seen: set[str],
    ) -> list[dict[str, Any]]:
        docs_to_create: list[dict[str, Any]] = []
        for qa in qa_pairs:
            doc = self._stage_2_qa_doc(
                context_id=context_id,
                article_id=article_id,
                canonical_title=canonical_title,
                aliases=aliases,
                qa=qa,
                qa_seen=qa_seen,
            )
            if doc is not None:
                docs_to_create.append(doc)
        return docs_to_create

    def _stage_2_qa_doc(
        self,
        *,
        context_id: str,
        article_id: str,
        canonical_title: str,
        aliases: list[str],
        qa: dict,
        qa_seen: set[str],
    ) -> dict[str, Any] | None:
        question = self._normalize_text(str(qa.get("question", "")))
        answer = str(qa.get("answer", "")).strip()
        if not question or not answer:
            return None

        qa_id = self._build_qa_id(context_id, question, answer)
        if qa_id in qa_seen:
            return None
        qa_seen.add(qa_id)

        intent = self._infer_intent(question, answer)
        self._record_stage_2_faq(
            article_id=article_id,
            canonical_title=canonical_title,
            aliases=aliases,
            qa_id=qa_id,
            question=question,
            intent=intent,
        )
        return {
            "title": canonical_title,
            "content": question,
            "metadata": {
                "qa_id": qa_id,
                "context_id": context_id,
                "article_id": article_id,
                "canonical_title": canonical_title,
                "title_aliases": aliases,
                "primary_intent": intent,
                "secondary_intents": [],
                "answer_text": answer,
                "answer_start": qa.get("answer_start"),
                "dataset": COVID_QA_DATASET_NAME,
            },
        }

    def _record_stage_2_faq(
        self,
        *,
        article_id: str,
        canonical_title: str,
        aliases: list[str],
        qa_id: str,
        question: str,
        intent: str,
    ) -> None:
        if article_id not in self.extraction_records:
            return
        stage_2 = self.extraction_records[article_id]["stage_2"]
        stage_2["canonical_title"] = canonical_title
        stage_2["aliases"] = aliases
        stage_2["faqs"].append(
            {
                "qa_id": qa_id,
                "question": question,
                "intent": intent,
            }
        )

    def _store_disease_collection_docs(
        self, docs_to_create: list[dict[str, Any]]
    ) -> int:
        embeddings = self.embed_documents([d["content"] for d in docs_to_create])
        for payload, embedding in zip(docs_to_create, embeddings):
            get_collection_model(MEDICAL_DOCUMENTS_DISEASE_COLLECTION).objects.create(
                title=payload["title"],
                content=payload["content"],
                embedding=embedding,
                section_type=SectionType.GENERAL,
                source=DOCUMENT_SOURCE_TAG,
                metadata=payload["metadata"],
            )
        return len(docs_to_create)

    def stage_3_chunks_collection(
        self,
        contexts: list[dict],
        disease_to_contexts: dict[str, set[str]],
    ) -> int:
        """Stage 3: Classify text into sections, chunk, and populate chunks collection."""
        logger.info("%s", "=" * 60)
        logger.info("STAGE 3: Chunks collection — Section-Aware Chunking")
        logger.info("%s", "=" * 60)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
        )

        ctx_lookup = {
            str(ctx["context_id"]): ctx for ctx in contexts if ctx.get("context_id")
        }
        count = 0
        context_aliases = self._build_context_aliases(disease_to_contexts)

        for context_id, aliases in tqdm(
            context_aliases.items(), desc="Stage 3: Section chunking"
        ):
            ctx_data = ctx_lookup.get(context_id)
            stage_context = self._stage_3_context(
                ctx_data=ctx_data,
                context_id=context_id,
                aliases=aliases,
            )
            if stage_context is None:
                continue

            sections = self._stage_3_classify_sections(
                context_id=context_id,
                context_text=stage_context["context_text"],
                article_id=stage_context["article_id"],
                canonical_title=stage_context["canonical_title"],
                aliases=aliases,
            )
            if sections is None:
                time.sleep(EMBEDDING_SLEEP_SECONDS)
                continue

            count += self._store_stage_3_sections(
                splitter=splitter,
                context_id=context_id,
                article_id=stage_context["article_id"],
                canonical_title=stage_context["canonical_title"],
                aliases=aliases,
                intent_hints=stage_context["intent_hints"],
                sections=sections,
            )

        logger.info(f"Chunks collection: {count} chunk entries stored")
        return count

    def _stage_3_context(
        self,
        *,
        ctx_data: dict | None,
        context_id: str,
        aliases: list[str],
    ) -> dict[str, Any] | None:
        if not ctx_data:
            return None
        context_text = str(ctx_data.get("context", ""))
        if not context_text.strip():
            return None
        canonical_title = self._select_canonical_title(aliases)
        if not canonical_title:
            return None

        return {
            "context_text": context_text,
            "article_id": str(ctx_data.get("article_id") or context_id),
            "canonical_title": canonical_title,
            "intent_hints": self._stage_3_intent_hints(context_id),
        }

    def _stage_3_intent_hints(self, context_id: str) -> list[str]:
        qa_pairs = self._qa_pairs_by_context.get(context_id, [])
        return sorted(
            {
                self._infer_intent(
                    str(qa.get("question", "")),
                    str(qa.get("answer", "")),
                )
                for qa in qa_pairs
            }
        )

    def _stage_3_classify_sections(
        self,
        *,
        context_id: str,
        context_text: str,
        article_id: str,
        canonical_title: str,
        aliases: list[str],
    ) -> dict[str, Any] | None:
        try:
            sections = self.classify_sections(context_text)
        except Exception as exc:
            logger.warning(f"Section classification failed for {context_id}: {exc}")
            return None

        if article_id in self.extraction_records:
            stage_3 = self.extraction_records[article_id]["stage_3"]
            stage_3["canonical_title"] = canonical_title
            stage_3["aliases"] = aliases
            stage_3["sections"] = sections
        return sections

    def _store_stage_3_sections(
        self,
        *,
        splitter: RecursiveCharacterTextSplitter,
        context_id: str,
        article_id: str,
        canonical_title: str,
        aliases: list[str],
        intent_hints: list[str],
        sections: dict[str, Any],
    ) -> int:
        count = 0
        for section_key in VALID_SECTIONS:
            count += self._store_stage_3_section_chunks(
                splitter=splitter,
                context_id=context_id,
                article_id=article_id,
                canonical_title=canonical_title,
                aliases=aliases,
                intent_hints=intent_hints,
                section_key=section_key,
                section_text=str(sections.get(section_key, "")).strip(),
            )
        return count

    def _store_stage_3_section_chunks(
        self,
        *,
        splitter: RecursiveCharacterTextSplitter,
        context_id: str,
        article_id: str,
        canonical_title: str,
        aliases: list[str],
        intent_hints: list[str],
        section_key: str,
        section_text: str,
    ) -> int:
        if not section_text:
            return 0
        section_type = SECTION_TYPE_MAP.get(section_key, SectionType.GENERAL)
        chunks = splitter.split_text(section_text)
        if not chunks:
            return 0

        try:
            chunks_lower = [chunk.lower() for chunk in chunks]
            embeddings = self.embed_documents(chunks_lower)
        except Exception as exc:
            logger.warning(f"Embedding failed for {context_id}/{section_key}: {exc}")
            return 0

        for index, (chunk_text, embedding) in enumerate(zip(chunks_lower, embeddings)):
            get_collection_model(MEDICAL_DOCUMENTS_CHUNKS_COLLECTION).objects.create(
                title=canonical_title,
                content=chunk_text,
                embedding=embedding,
                section_type=section_type,
                source=DOCUMENT_SOURCE_TAG,
                metadata=self._stage_3_chunk_metadata(
                    context_id=context_id,
                    article_id=article_id,
                    canonical_title=canonical_title,
                    aliases=aliases,
                    section_key=section_key,
                    chunk_index=index,
                    total_chunks=len(chunks),
                    intent_hints=intent_hints,
                ),
            )
        time.sleep(EMBEDDING_SLEEP_SECONDS)
        return len(chunks)

    def _stage_3_chunk_metadata(
        self,
        *,
        context_id: str,
        article_id: str,
        canonical_title: str,
        aliases: list[str],
        section_key: str,
        chunk_index: int,
        total_chunks: int,
        intent_hints: list[str],
    ) -> dict[str, Any]:
        return {
            "context_id": context_id,
            "article_id": article_id,
            "disease": canonical_title,
            "canonical_title": canonical_title,
            "title_aliases": aliases,
            "evidence_id": f"{context_id}:{section_key}:{chunk_index}",
            "section": section_key,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "intent_hints": intent_hints,
            "dataset": COVID_QA_DATASET_NAME,
        }

    def _parse_json_response(self, response: str) -> dict | list:
        """Parse JSON from LLM response, handling markdown code blocks."""
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()
        return json.loads(text)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
        reraise=True,
    )
    def _llm_invoke(self, llm: Any, prompt: str) -> str:
        """Call LLM with retry. Returns raw text response."""
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else response
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts).strip()
        return str(content)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
        reraise=True,
    )
    def _embed_texts_with_retry(
        self, service: EmbeddingService, texts: list[str]
    ) -> list[list[float]]:
        return service.embed_documents(texts)


def clear_covid_qa_documents() -> int:
    """Remove all COVID-QA documents from physical collections."""
    deleted_count = 0
    for _, model in iter_collection_models():
        result = model.objects.filter(source=DOCUMENT_SOURCE_TAG).delete()
        deleted_count += result[0] if isinstance(result, tuple) else int(result)
    logger.info(f"Deleted {deleted_count} COVID-QA documents")
    return deleted_count
