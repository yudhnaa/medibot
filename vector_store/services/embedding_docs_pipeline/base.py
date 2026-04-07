"""
Base classes and shared logic for the embedding pipelines.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from django.core.exceptions import SynchronousOnlyOperation
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from chatbot.models import MedicalDocument
from chatbot.models.chatbot_config import ChatbotConfig
from chatbot.models.medical_document import IndexType, SectionType
from vector_store.services.embedding_docs_pipeline.constants import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_SOURCE_TAG,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_PROVIDER,
    EMBEDDING_SLEEP_SECONDS,
    GOOGLE_API_KEY,
    LLM_MODEL,
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
        self.provider_kwargs = provider_kwargs
        self.embedding_service = self.create_embedding_service(
            self.embedding_provider, **provider_kwargs
        )
        self.llm = self.create_llm()

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

    def create_llm(self) -> ChatGoogleGenerativeAI:
        """Create LLM instance for extraction tasks."""
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=SecretStr(GOOGLE_API_KEY),
            temperature=0.1,
        )

    def create_embedding_service(
        self, provider: str | None = None, **provider_kwargs
    ) -> EmbeddingService:
        """Create the embedding service for the requested provider."""
        return EmbeddingService(provider=provider, **provider_kwargs)

    @abstractmethod
    def load_contexts(self, num_docs: int | None = None) -> list[dict]:
        """Load source contexts for the pipeline."""

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

    def build_index_a_summary(self, row) -> str | None:
        """Construct disease-level summary for Index A."""
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
        index_type: str = IndexType.B,
    ) -> list[dict[str, object]]:
        """Build document dictionaries from a CSV file."""
        documents: list[dict[str, object]] = []
        df = self.load_csv(csv_path)

        if index_type == IndexType.A:
            for idx, row in df.iterrows():
                content = self.build_index_a_summary(row)
                if content:
                    documents.append(
                        {
                            "content": content.lower(),
                            "title": str(row.get("title", "")).strip().lower(),
                            "section_type": SectionType.GENERAL,
                            "index_type": IndexType.A,
                            "source": source.lower(),
                            "metadata": {
                                "row_index": idx,
                                "url": str(row.get("url", "")).lower(),
                            },
                        }
                    )
            return documents

        if index_type == IndexType.C:
            for idx, row in df.iterrows():
                title = str(row.get("title", "")).strip()
                if title:
                    documents.append(
                        {
                            "content": title.lower(),
                            "title": title.lower(),
                            "section_type": SectionType.GENERAL,
                            "index_type": IndexType.C,
                            "source": source.lower(),
                            "metadata": {
                                "row_index": idx,
                                "url": str(row.get("url", "")).lower(),
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
                            "index_type": IndexType.B,
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

    def run(self, num_docs: int | None = None) -> dict:
        """Run the three-stage pipeline."""
        contexts = self.load_contexts(num_docs=num_docs)
        if not contexts:
            logger.warning("No contexts loaded. Pipeline cannot continue.")
            return {"index_c": 0, "index_a": 0, "index_b": 0}

        clear_covid_qa_documents()

        disease_to_contexts = self.stage_1_index_c(contexts)
        if not disease_to_contexts:
            logger.error("Stage 1 failed — no diseases extracted. Aborting.")
            return {"index_c": 0, "index_a": 0, "index_b": 0}

        index_a_count = self.stage_2_index_a(contexts, disease_to_contexts)
        index_b_count = self.stage_3_index_b(contexts, disease_to_contexts)

        return {
            "index_c": len(disease_to_contexts),
            "index_a": index_a_count,
            "index_b": index_b_count,
            "total": len(disease_to_contexts) + index_a_count + index_b_count,
        }

    def stage_1_index_c(self, contexts: list[dict]) -> dict[str, set[str]]:
        """Stage 1: Extract disease names and populate Index C."""
        logger.info("%s", "=" * 60)
        logger.info("STAGE 1: Index C — Disease Name Extraction")
        logger.info("%s", "=" * 60)

        disease_to_contexts: dict[str, set[str]] = {}

        for ctx in tqdm(contexts, desc="Stage 1: Extracting diseases"):
            context_id = ctx["context_id"]
            context_text = ctx["context"]

            diseases = self.extract_diseases_from_context(context_text)
            for disease in diseases:
                normalized = disease.strip().lower()
                if normalized:
                    disease_to_contexts.setdefault(normalized, set()).add(context_id)

            time.sleep(EMBEDDING_SLEEP_SECONDS)

        if not disease_to_contexts:
            logger.warning("No diseases extracted. Pipeline cannot continue.")
            return {}

        logger.info(
            f"Extracted {len(disease_to_contexts)} unique diseases: "
            f"{list(disease_to_contexts.keys())}"
        )

        disease_names = list(disease_to_contexts.keys())
        embeddings = self.embed_documents(disease_names)

        for disease_name, embedding in zip(disease_names, embeddings):
            MedicalDocument.objects.create(
                title=disease_name,
                content=disease_name,
                embedding=embedding,
                section_type=SectionType.GENERAL,
                index_type=IndexType.C,
                source=DOCUMENT_SOURCE_TAG,
                metadata={
                    "dataset": "deepset/covid_qa_deepset",
                    "source_contexts": list(disease_to_contexts[disease_name]),
                },
            )

        logger.info(f"Index C: {len(disease_names)} disease entries stored")
        return disease_to_contexts

    def stage_2_index_a(
        self,
        contexts: list[dict],
        disease_to_contexts: dict[str, set[str]],
    ) -> int:
        """Stage 2: Extract structured summaries and populate Index A."""
        logger.info("%s", "=" * 60)
        logger.info("STAGE 2: Index A — Structured Summary Extraction")
        logger.info("%s", "=" * 60)

        ctx_lookup = {ctx["context_id"]: ctx["context"] for ctx in contexts}
        count = 0

        for disease_name, context_ids in tqdm(
            disease_to_contexts.items(), desc="Stage 2: Building summaries"
        ):
            for context_id in context_ids:
                context_text = ctx_lookup.get(context_id, "")
                if not context_text:
                    continue

                text = context_text[:8000] if len(context_text) > 8000 else context_text
                prompt = PROMPT_EXTRACT_SUMMARY.format(
                    disease_name=disease_name, text=text
                )

                try:
                    response = self._llm_invoke(self.llm, prompt)
                    summary_data = self._parse_json_response(response)
                    if not isinstance(summary_data, dict):
                        logger.warning(
                            f"Non-dict summary for {disease_name}: {type(summary_data)}"
                        )
                        continue

                    summary_data["disease_name"] = disease_name

                    summary_text = self.build_summary_text(summary_data)
                    if not summary_text.strip():
                        continue

                    embedding = self.embed_documents([summary_text.lower()])[0]

                    MedicalDocument.objects.create(
                        title=disease_name,
                        content=summary_text.lower(),
                        embedding=embedding,
                        section_type=SectionType.GENERAL,
                        index_type=IndexType.A,
                        source=DOCUMENT_SOURCE_TAG,
                        metadata={
                            "context_id": context_id,
                            "dataset": "deepset/covid_qa_deepset",
                        },
                    )
                    count += 1

                except Exception as e:
                    logger.warning(
                        f"Summary extraction failed for {disease_name} "
                        f"(context {context_id}): {e}"
                    )

                time.sleep(EMBEDDING_SLEEP_SECONDS)

        logger.info(f"Index A: {count} summary entries stored")
        return count

    def stage_3_index_b(
        self,
        contexts: list[dict],
        disease_to_contexts: dict[str, set[str]],
    ) -> int:
        """Stage 3: Classify text into sections, chunk, and populate Index B."""
        logger.info("%s", "=" * 60)
        logger.info("STAGE 3: Index B — Section-Aware Chunking")
        logger.info("%s", "=" * 60)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
        )

        ctx_lookup = {ctx["context_id"]: ctx["context"] for ctx in contexts}
        count = 0

        context_to_diseases: dict[str, list[str]] = {}
        for disease_name, context_ids in disease_to_contexts.items():
            for cid in context_ids:
                context_to_diseases.setdefault(cid, []).append(disease_name)

        for context_id, diseases in tqdm(
            context_to_diseases.items(), desc="Stage 3: Section chunking"
        ):
            context_text = ctx_lookup.get(context_id, "")
            if not context_text:
                continue

            try:
                sections = self.classify_sections(context_text)
            except Exception as e:
                logger.warning(f"Section classification failed for {context_id}: {e}")
                time.sleep(EMBEDDING_SLEEP_SECONDS)
                continue

            for section_key in VALID_SECTIONS:
                section_text = str(sections.get(section_key, "")).strip()
                if not section_text:
                    continue

                section_type = SECTION_TYPE_MAP.get(section_key, SectionType.GENERAL)
                chunks = splitter.split_text(section_text)
                if not chunks:
                    continue

                try:
                    chunks_lower = [c.lower() for c in chunks]
                    embeddings = self.embed_documents(chunks_lower)
                except Exception as e:
                    logger.warning(
                        f"Embedding failed for {context_id}/{section_key}: {e}"
                    )
                    continue

                for disease_name in diseases:
                    for i, (chunk_text, embedding) in enumerate(
                        zip(chunks_lower, embeddings)
                    ):
                        MedicalDocument.objects.create(
                            title=disease_name,
                            content=chunk_text,
                            embedding=embedding,
                            section_type=section_type,
                            index_type=IndexType.B,
                            source=DOCUMENT_SOURCE_TAG,
                            metadata={
                                "context_id": context_id,
                                "disease": disease_name,
                                "section": section_key,
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                                "dataset": "deepset/covid_qa_deepset",
                            },
                        )
                        count += 1

                time.sleep(EMBEDDING_SLEEP_SECONDS)

        logger.info(f"Index B: {count} chunk entries stored")
        return count

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
    def _llm_invoke(self, llm: ChatGoogleGenerativeAI, prompt: str) -> str:
        """Call LLM with retry. Returns raw text response."""
        response = llm.invoke(prompt)
        return str(response.content) if hasattr(response, "content") else str(response)

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
    """Remove all COVID-QA documents from MedicalDocument table."""
    result = MedicalDocument.objects.filter(source=DOCUMENT_SOURCE_TAG).delete()
    deleted_count = result[0] if isinstance(result, tuple) else result
    logger.info(f"🗑️  Deleted {deleted_count} COVID-QA documents")
    return deleted_count
