from __future__ import annotations

import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator

from chatbot.models import ChatbotConfig, get_collection_model
from chatbot.services.gemini_manager import get_gemini_manager
from rag_benchmark.services.constants import (
    NEGATION_TOKENS,
    NOISY_STYLES,
    SECTION_PATTERN,
    SUPPORTED_PROVIDERS,
    TITLE_PATTERN,
)
from vector_store.services.constants import (
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_MODEL,
    OPENROUTER_API_KEY_ENV_NAME,
    OPENROUTER_BASE_URL_ENV_NAME,
)


@dataclass(slots=True)
class GeneratedBenchmarkDatasetReport:
    output_path: str
    total_cases: int
    dev_cases: int
    test_cases: int
    unique_gold_titles: int
    source_titles: int


class RagasBenchmarkDatasetGenerator:
    """
    Generate benchmark dataset JSONL from current vector-store documents via Ragas.

    The generated payload follows the benchmark importer schema in this repository.
    """

    def generate_jsonl(
        self,
        *,
        output_path: str,
        testset_size: int,
        dataset_version: str = "v1",
        dev_ratio: float = 0.6,
        seed: int = 42,
        provider: str = "auto",
        llm_model: str | None = None,
        embedding_model: str | None = None,
        llm_context: str = (
            "Generate realistic Vietnamese end-user medical questions that match "
            "the provided disease context. Keep questions concise and practical."
        ),
        collections: list[str] | None = None,
        min_document_words: int = 100,
    ) -> GeneratedBenchmarkDatasetReport:
        if testset_size < 2:
            raise ValueError(
                "testset_size must be >= 2 to include both dev and test split"
            )

        selected_collections = [
            item.strip() for item in (collections or ["medical_documents_chunks"])
        ]
        for collection_name in selected_collections:
            get_collection_model(collection_name)
        if not selected_collections:
            raise ValueError("collections must include at least one collection name")

        source_docs, source_titles = self._build_source_documents(
            collections=selected_collections,
            min_document_words=min_document_words,
        )
        if not source_docs:
            raise ValueError(
                "No vector-store documents meet minimum length for Ragas generation"
            )

        testset_records = self._generate_ragas_records(
            documents=source_docs,
            testset_size=testset_size,
            provider=self._resolve_provider(provider),
            llm_model=llm_model,
            embedding_model=embedding_model,
            llm_context=llm_context,
        )

        cases = self._build_cases(
            rows=testset_records,
            dataset_version=dataset_version,
            dev_ratio=dev_ratio,
            seed=seed,
            fallback_title=source_docs[0].metadata.get("title", ""),
        )
        if len(cases) < 2:
            raise ValueError(
                "Ragas generated too few valid rows. Increase testset_size or corpus coverage."
            )

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            for row in cases:
                fp.write(json.dumps(row, ensure_ascii=False))
                fp.write("\n")

        dev_cases = sum(1 for case in cases if case["split"] == "dev")
        test_cases = sum(1 for case in cases if case["split"] == "test")
        unique_titles = {
            title
            for case in cases
            for title in case.get("gold_titles", [])
            if str(title).strip()
        }
        return GeneratedBenchmarkDatasetReport(
            output_path=str(path),
            total_cases=len(cases),
            dev_cases=dev_cases,
            test_cases=test_cases,
            unique_gold_titles=len(unique_titles),
            source_titles=source_titles,
        )

    def _build_source_documents(
        self,
        *,
        collections: list[str],
        min_document_words: int,
    ) -> tuple[list[Document], int]:
        docs_by_title: dict[str, list[Any]] = defaultdict(list)
        for collection_name in collections:
            model = get_collection_model(collection_name)
            queryset = model.objects.exclude(content="").order_by(
                "title", "section_type", "id"
            )
            for record in queryset:
                title = str(record.title or "").strip()
                if not title:
                    continue
                docs_by_title[title].append(record)

        source_documents: list[Document] = []
        for title, records in docs_by_title.items():
            lines: list[str] = [f"Title: {title}"]
            sections_seen: set[str] = set()
            for record in records:
                content = str(record.content or "").strip()
                if not content:
                    continue
                section = str(record.section_type or "general").strip().lower()
                sections_seen.add(section)
                lines.append(f"Section[{section}]: {content}")

            page_content = "\n".join(lines).strip()
            if len(page_content.split()) < min_document_words:
                continue

            source_documents.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "title": title,
                        "sections": sorted(sections_seen),
                        "doc_count": len(records),
                    },
                )
            )

        return source_documents, len(docs_by_title)

    def _generate_ragas_records(
        self,
        *,
        documents: list[Document],
        testset_size: int,
        provider: str,
        llm_model: str | None,
        embedding_model: str | None,
        llm_context: str,
    ) -> list[dict[str, Any]]:
        llm, embeddings = self._create_ragas_models(
            provider=provider,
            llm_model=llm_model,
            embedding_model=embedding_model,
        )

        generator = TestsetGenerator.from_langchain(
            llm=llm,
            embedding_model=embeddings,
            llm_context=llm_context,
        )
        testset = generator.generate_with_langchain_docs(
            documents,
            testset_size=testset_size,
            run_config=RunConfig(
                timeout=180,
                max_retries=3,
                max_wait=30,
                max_workers=4,
            ),
            raise_exceptions=False,
        )
        dataframe = cast(Any, testset).to_pandas()
        return cast(list[dict[str, Any]], dataframe.to_dict(orient="records"))

    def _resolve_provider(self, provider: str) -> str:
        normalized = str(provider or "").strip().lower()
        if not normalized or normalized == "auto":
            configured = ChatbotConfig.get_config("LLM_PROVIDER", "gemini")
            normalized = str(configured or "gemini").strip().lower()
        if normalized not in SUPPORTED_PROVIDERS:
            supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
            raise ValueError(
                f"Unsupported provider `{normalized}`. Supported: {supported}"
            )
        return normalized

    def _create_ragas_models(
        self,
        *,
        provider: str,
        llm_model: str | None,
        embedding_model: str | None,
    ) -> tuple[Any, Any]:
        if provider == "openrouter":
            return self._create_openrouter_models(
                llm_model=llm_model,
                embedding_model=embedding_model,
            )
        return self._create_gemini_models(
            llm_model=llm_model,
            embedding_model=embedding_model,
        )

    def _create_gemini_models(
        self,
        *,
        llm_model: str | None,
        embedding_model: str | None,
    ) -> tuple[ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings]:
        manager = get_gemini_manager()
        api_key = manager.get_current_key()
        llm = ChatGoogleGenerativeAI(
            model=llm_model or "gemini-2.5-flash",
            api_key=SecretStr(api_key),
            temperature=0.2,
            max_tokens=2048,
            streaming=False,
        )
        embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model or "models/gemini-embedding-001",
            api_key=SecretStr(api_key),
        )
        return llm, embeddings

    def _create_openrouter_models(
        self,
        *,
        llm_model: str | None,
        embedding_model: str | None,
    ) -> tuple[ChatOpenAI, OpenAIEmbeddings]:
        api_key_raw = ChatbotConfig.get_config(OPENROUTER_API_KEY_ENV_NAME, None)
        api_key = (
            api_key_raw.strip()
            if isinstance(api_key_raw, str) and api_key_raw.strip()
            else os.getenv(OPENROUTER_API_KEY_ENV_NAME)
        )
        if not api_key:
            raise ValueError(
                f"{OPENROUTER_API_KEY_ENV_NAME} is required for provider=openrouter"
            )

        db_base_url = ChatbotConfig.get_config("OPENROUTER_BASE_URL", None)
        base_url = (
            db_base_url.strip()
            if isinstance(db_base_url, str) and db_base_url.strip()
            else os.getenv(OPENROUTER_BASE_URL_ENV_NAME) or DEFAULT_OPENROUTER_BASE_URL
        )

        default_llm_model = str(
            ChatbotConfig.get_config("LLM_MODEL", "openai/gpt-4.1-mini")
        )
        default_embedding_model = str(
            ChatbotConfig.get_config("EMBEDDING_MODEL", DEFAULT_OPENROUTER_MODEL)
        )

        llm = ChatOpenAI(
            model=(llm_model or default_llm_model),
            api_key=SecretStr(api_key),
            base_url=base_url,
            temperature=0.2,
            max_completion_tokens=2048,
        )
        embeddings = OpenAIEmbeddings(
            model=(embedding_model or default_embedding_model),
            api_key=SecretStr(api_key),
            base_url=base_url,
        )
        return llm, embeddings

    def _build_cases(
        self,
        *,
        rows: list[dict[str, Any]],
        dataset_version: str,
        dev_ratio: float,
        seed: int,
        fallback_title: str,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for row in rows:
            question = str(row.get("user_input") or "").strip()
            if not question:
                continue
            normalized.append(row)

        if len(normalized) < 2:
            return []

        randomizer = random.Random(seed)
        randomizer.shuffle(normalized)

        total = len(normalized)
        dev_count = max(1, min(total - 1, int(round(total * dev_ratio))))

        cases: list[dict[str, Any]] = []
        for idx, row in enumerate(normalized, start=1):
            split = "dev" if idx <= dev_count else "test"
            case = self._build_case_payload(
                row=row,
                case_no=idx,
                split=split,
                dataset_version=dataset_version,
                fallback_title=fallback_title,
            )
            cases.append(case)
        return cases

    def _build_case_payload(
        self,
        *,
        row: dict[str, Any],
        case_no: int,
        split: str,
        dataset_version: str,
        fallback_title: str,
    ) -> dict[str, Any]:
        question = str(row.get("user_input") or "").strip()
        query_style = str(row.get("query_style") or "").strip().upper()
        query_length = str(row.get("query_length") or "").strip().upper()
        synthesizer_name = str(row.get("synthesizer_name") or "").strip()

        contexts_raw = row.get("reference_contexts")
        contexts: list[str] = []
        if isinstance(contexts_raw, list):
            contexts = [str(item).strip() for item in contexts_raw if str(item).strip()]

        title = self._extract_title_from_contexts(contexts, fallback=fallback_title)
        sections = self._extract_sections_from_contexts(contexts)
        scenario = self._infer_scenario(
            question=question,
            query_style=query_style,
            query_length=query_length,
        )
        expected_behavior = self._infer_expected_behavior(scenario)
        gold_titles = [title] if title else []
        expected_mode = self._infer_expected_mode(
            expected_behavior=expected_behavior,
            gold_titles=gold_titles,
        )

        must_have_sections = sections if expected_mode == "single-disease" else []
        if expected_mode == "single-disease" and not must_have_sections:
            must_have_sections = ["general", "symptom"]

        case_id = f"{split}-{scenario}-{case_no:03d}"
        reference_answer = str(row.get("reference") or "").strip()
        question, reference_answer = self._normalize_question_reference_pair(
            question=question,
            reference_answer=reference_answer,
            query_style=query_style,
        )

        return {
            "case_id": case_id,
            "dataset_version": dataset_version,
            "split": split,
            "question": question,
            "intake_payload": {},
            "scenario": scenario,
            "expected_mode": expected_mode,
            "gold_titles": gold_titles,
            "forbidden_titles": [],
            "must_have_sections": must_have_sections,
            "expected_behavior": expected_behavior,
            "reference_answer": reference_answer,
            "notes": (
                "Generated by ragas testset generator "
                f"(synthesizer={synthesizer_name}, style={query_style}, length={query_length})"
            ),
            "gold_analysis": {},
            "gold_primary_title": gold_titles[0] if gold_titles else "",
            "must_not_sections": [],
            "reference_context_ids": [],
            "requires_followup_topic": "",
            "risk_level": "medium",
            "annotation_metadata": {
                "source": "ragas_testset_generator",
                "synthesizer_name": synthesizer_name,
                "query_style": query_style,
                "query_length": query_length,
            },
        }

    def _normalize_question_reference_pair(
        self,
        *,
        question: str,
        reference_answer: str,
        query_style: str,
    ) -> tuple[str, str]:
        normalized_question = re.sub(r"\s+", " ", str(question or "")).strip()
        normalized_reference = re.sub(r"\s+", " ", str(reference_answer or "")).strip()
        question_lower = normalized_question.lower()
        reference_lower = normalized_reference.lower()

        asks_sars_cause = bool(
            re.search(
                r"\bwhat\s+causes?\s+sars[\s-]?cov[\s-]?2\b",
                question_lower,
            )
            or re.search(
                r"\bwhat\s+cause\s+sars[\s-]?cov[\s-]?2\b",
                question_lower,
            )
        )
        reference_answers_covid_cause = (
            "cause of covid-19" in reference_lower
            or "caused by the sars-cov-2 virus" in reference_lower
        )
        if asks_sars_cause and reference_answers_covid_cause:
            if query_style in NOISY_STYLES:
                normalized_question = "What cause covid-19?"
            else:
                normalized_question = "What causes COVID-19?"

        asks_covid_cause_with_virus_name = (
            "main cause" in question_lower
            and "covid-19" in question_lower
            and "virus" in question_lower
            and "sars-cov-2" in question_lower
        )
        reference_contains_extra_general_clause = (
            "most people experience mild to moderate respiratory illness"
            in reference_lower
        )
        if asks_covid_cause_with_virus_name and reference_contains_extra_general_clause:
            normalized_reference = "COVID-19 is caused by the SARS-CoV-2 virus."

        return (
            normalized_question or str(question or "").strip(),
            normalized_reference or str(reference_answer or "").strip(),
        )

    def _extract_title_from_contexts(
        self, contexts: list[str], *, fallback: str
    ) -> str:
        for context in contexts:
            first_line = context.splitlines()[0] if context else ""
            match = TITLE_PATTERN.match(first_line.strip())
            if match:
                title = str(match.group("title")).strip()
                if title:
                    return title
        return str(fallback).strip()

    def _extract_sections_from_contexts(self, contexts: list[str]) -> list[str]:
        sections: list[str] = []
        for context in contexts:
            for match in SECTION_PATTERN.finditer(context):
                section = str(match.group("section")).strip().lower()
                if section and section not in sections:
                    sections.append(section)
        return sections[:3]

    def _infer_scenario(
        self,
        *,
        question: str,
        query_style: str,
        query_length: str,
    ) -> str:
        normalized_question = f" {question.lower()} "
        if any(token in normalized_question for token in NEGATION_TOKENS):
            return "negation"
        if query_style in NOISY_STYLES:
            return "noisy_query"
        if query_style == "WEB_SEARCH_LIKE":
            return "paraphrase"
        if query_length == "SHORT" and len(question.split()) <= 6:
            return "insufficient_info"
        return "single_clear"

    def _infer_expected_behavior(self, scenario: str) -> str:
        if scenario == "insufficient_info":
            return "ask_followup"
        if scenario == "out_of_scope":
            return "abstain"
        return "answer"

    def _infer_expected_mode(
        self,
        *,
        expected_behavior: str,
        gold_titles: list[str],
    ) -> str:
        if expected_behavior != "answer":
            return "multi-disease-v2"
        if len(gold_titles) == 1:
            return "single-disease"
        return "multi-disease-v2"
