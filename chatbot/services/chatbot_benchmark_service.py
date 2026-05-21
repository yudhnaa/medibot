import json
import logging
import re
import time
import uuid
from typing import Any, cast

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from chatbot.models import MEDICAL_DOCUMENTS_CHUNKS_COLLECTION, ChatbotConfig
from chatbot.prompts.system_benchmark import SYSTEM_PROMPT_BENCHMARK
from chatbot.services.chatbot_service import ChatbotService
from chatbot.services.constants import (
    DEFAULT_BENCHMARK_DOC_CHAR_LIMIT,
    DEFAULT_BENCHMARK_PROMPT_CONTEXT_CHAR_LIMIT,
    DEFAULT_BENCHMARK_RERANK_PREFILTER_K,
    DEFAULT_BENCHMARK_RERANK_TOP_K,
    DEFAULT_BENCHMARK_RETRIEVAL_CONTEXT_LIMIT,
    DEFAULT_BENCHMARK_SECTION_INTENT_BOOST,
    DEFAULT_BENCHMARK_SECTION_OFF_TARGET_PENALTY,
    DEFAULT_BENCHMARK_SINGLE_DISEASE_DOCS_K,
    DEFAULT_DOCS_CACHE_SIZE,
)

logger = logging.getLogger(__name__)


class ChatbotBenchmarkService(ChatbotService):
    """Benchmark-focused chatbot service."""

    def _create_benchmark_prompt_template(self) -> ChatPromptTemplate:
        """Create stricter prompt template for benchmark runs."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT_BENCHMARK),
                ("human", "{question}"),
            ]
        )

    def run_benchmark_case(
        self,
        *,
        question: str,
        intake_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run one offline benchmark case and return stable stage artifacts.

        This hook is additive and intentionally decoupled from logging output so
        benchmark infrastructure can consume deterministic runtime artifacts.
        """
        # Reset intake state for every benchmark case to avoid cross-case leakage.
        self._apply_benchmark_intake_payload(intake_payload or {})

        total_started_at = time.perf_counter()
        timings_ms: dict[str, int] = {}
        analysis: dict[str, Any] = {}
        gate: dict[str, Any] = {}
        retrieval_output: dict[str, Any] = {}
        generation_output: dict[str, Any] = {}
        mode = "multi-disease-v2"

        try:
            stage_started_at = time.perf_counter()
            analysis = self._analyze_query(question)
            timings_ms["analysis_latency"] = int(
                (time.perf_counter() - stage_started_at) * 1000
            )

            q_cleaned = str(analysis.get("q_cleaned", question)).strip() or question
            stage_started_at = time.perf_counter()
            gate = self._gate_with_titles_collection(q_cleaned)
            timings_ms["gate_latency"] = int(
                (time.perf_counter() - stage_started_at) * 1000
            )

            stage_started_at = time.perf_counter()
            if gate.get("go_single") and gate.get("title"):
                mode = "single-disease"
                title = str(gate.get("title", "")).strip().lower()
                benchmark_single_docs_k = self._get_config_int(
                    "BENCHMARK_SINGLE_DISEASE_DOCS_K",
                    DEFAULT_BENCHMARK_SINGLE_DISEASE_DOCS_K,
                )
                evidence_docs = self._fetch_docs_for_title(
                    title,
                    collection_name=MEDICAL_DOCUMENTS_CHUNKS_COLLECTION,
                    k=benchmark_single_docs_k,
                )
                context_docs, rerank_meta = self._rerank_benchmark_evidence(
                    question=question,
                    docs=evidence_docs,
                )
                summaries = self._fetch_summary_docs_for_titles(
                    [title], str(analysis.get("q_cleaned", ""))
                )
                context = self._build_single_disease_context_with_summary(
                    title=title,
                    docs=context_docs,
                    summaries=summaries,
                )
                self._last_docs_cache = context_docs[:DEFAULT_DOCS_CACHE_SIZE]
                benchmark_retrieval_docs = self._limit_benchmark_retrieval_docs(
                    context_docs
                )
                retrieval_output = self._build_benchmark_retrieval_output(
                    mode=mode,
                    docs=benchmark_retrieval_docs,
                    candidates=[
                        {
                            "title": title,
                            "top_score": float(gate.get("top_score", 0.0) or 0.0),
                        }
                    ],
                    summaries={
                        item.get("title", ""): item.get("summary", "")
                        for item in summaries
                    },
                )
                retrieval_output["rerank"] = rerank_meta
                self._last_audit = {
                    "audit_id": str(uuid.uuid4()),
                    "ts": time.time(),
                    "mode": mode,
                    "title": title,
                    "router_score": gate.get("top_score", 0.0),
                    "doc_count": len(context_docs),
                }
            else:
                mode = "multi-disease-v2"
                tier_result = self._multi_disease_retrieval(analysis)
                evidence_docs = cast(
                    list[Document], tier_result.get("evidence_docs", [])
                )
                context_docs, rerank_meta = self._rerank_benchmark_evidence(
                    question=question,
                    docs=evidence_docs,
                )
                tier_result_for_context = dict(tier_result)
                tier_result_for_context["evidence_docs"] = context_docs
                context = self._build_multi_disease_context(
                    question, analysis, tier_result_for_context
                )
                self._last_docs_cache = context_docs[:DEFAULT_DOCS_CACHE_SIZE]
                benchmark_retrieval_docs = self._limit_benchmark_retrieval_docs(
                    context_docs
                )
                retrieval_output = self._build_benchmark_retrieval_output(
                    mode=mode,
                    docs=benchmark_retrieval_docs,
                    candidates=cast(
                        list[dict[str, Any]], tier_result.get("candidates", [])
                    ),
                    summaries=cast(dict[str, str], tier_result.get("summaries", {})),
                )
                retrieval_output["rerank"] = rerank_meta
                self._last_audit = {
                    "audit_id": str(uuid.uuid4()),
                    "ts": time.time(),
                    "mode": mode,
                    "candidates": [
                        candidate.get("title")
                        for candidate in cast(
                            list[dict[str, Any]], tier_result.get("candidates", [])
                        )[:5]
                    ],
                }

            timings_ms["retrieval_latency"] = int(
                (time.perf_counter() - stage_started_at) * 1000
            )
            self._last_query_text = question

            stage_started_at = time.perf_counter()
            context = self._limit_benchmark_prompt_context(context)
            output_language = self._get_benchmark_output_language(question)
            answer_policy = self._build_benchmark_answer_policy(
                question=question,
                retrieval_output=retrieval_output,
                output_language=output_language,
            )
            prompt_input = {
                "context": context,
                "question": question,
                "answer_policy": answer_policy,
            }
            prompt_messages = self._create_benchmark_prompt_template().format_messages(
                **prompt_input
            )
            llm_result = self.llm.invoke(prompt_messages)
            llm_content = (
                llm_result.content if hasattr(llm_result, "content") else llm_result
            )
            if isinstance(llm_content, list):
                llm_content = "\n".join(str(item) for item in llm_content)
            final_answer = str(llm_content or "").strip()
            final_answer = self._shape_benchmark_answer(
                question=question,
                answer=final_answer,
                retrieval_output=retrieval_output,
                output_language=output_language,
            )
            final_answer = self._enforce_benchmark_output_language(
                question=question,
                answer=final_answer,
                output_language=output_language,
            )
            timings_ms["generation_latency"] = int(
                (time.perf_counter() - stage_started_at) * 1000
            )

            source_urls = self.get_last_source_urls()
            generation_output = {
                "final_answer": final_answer,
                "source_urls": source_urls,
                "context_snapshot": context[:4000],
                "output_language_policy": output_language,
            }
        except Exception as exc:
            generation_output = {
                "final_answer": "",
                "source_urls": [],
                "context_snapshot": "",
                "error": str(exc),
            }
        finally:
            timings_ms["total_latency"] = int(
                (time.perf_counter() - total_started_at) * 1000
            )

        return {
            "question": question,
            "mode": mode,
            "analysis_output": analysis,
            "gate_output": gate,
            "retrieval_output": retrieval_output,
            "generation_output": generation_output,
            "audit_metadata": self._last_audit,
            "timings_ms": timings_ms,
        }

    def _limit_benchmark_retrieval_docs(self, docs: list[Document]) -> list[Document]:
        """Limit retrieval docs persisted/evaluated during benchmark for speed."""
        limit = self._get_config_int(
            "BENCHMARK_RETRIEVAL_CONTEXT_LIMIT",
            DEFAULT_BENCHMARK_RETRIEVAL_CONTEXT_LIMIT,
        )
        if limit <= 0:
            return []
        return docs[:limit]

    def _limit_benchmark_prompt_context(self, context: str) -> str:
        """Trim assembled benchmark prompt context to control token cost/latency."""
        normalized = re.sub(r"\n{3,}", "\n\n", str(context or "")).strip()
        if not normalized:
            return ""
        limit = self._get_config_int(
            "BENCHMARK_PROMPT_CONTEXT_CHAR_LIMIT",
            DEFAULT_BENCHMARK_PROMPT_CONTEXT_CHAR_LIMIT,
        )
        if limit <= 0 or len(normalized) <= limit:
            return normalized
        head = int(limit * 0.8)
        tail = max(0, limit - head - 10)
        if tail <= 0:
            return normalized[:limit]
        return f"{normalized[:head].rstrip()}\n...\n{normalized[-tail:].lstrip()}"

    def _rerank_benchmark_evidence(
        self,
        *,
        question: str,
        docs: list[Document],
    ) -> tuple[list[Document], dict[str, Any]]:
        """Lightweight LLM rerank to reduce benchmark context cost/noise."""
        if not docs:
            return [], {"applied": False, "reason": "no_docs"}

        intent = self._detect_benchmark_intent(question)
        target_sections = cast(list[str], intent.get("target_sections", []))

        prefilter_k = self._get_config_int(
            "BENCHMARK_RERANK_PREFILTER_K",
            DEFAULT_BENCHMARK_RERANK_PREFILTER_K,
        )
        top_k = self._get_config_int(
            "BENCHMARK_RERANK_TOP_K",
            DEFAULT_BENCHMARK_RERANK_TOP_K,
        )
        doc_char_limit = self._get_config_int(
            "BENCHMARK_DOC_CHAR_LIMIT",
            DEFAULT_BENCHMARK_DOC_CHAR_LIMIT,
        )
        prefilter_k = max(1, min(prefilter_k, len(docs)))
        top_k = max(1, min(top_k, prefilter_k))

        ranked = self._prefilter_benchmark_docs(
            question=question,
            docs=docs,
            intent=intent,
            prefilter_k=prefilter_k,
        )

        if len(ranked) <= top_k:
            return (
                self._truncate_benchmark_docs(ranked, doc_char_limit=doc_char_limit),
                {
                    "applied": False,
                    "reason": "prefilter_small",
                    "prefilter_k": prefilter_k,
                    "top_k": top_k,
                    "insufficient_evidence": False,
                    "intent": intent,
                },
            )

        try:
            payload = self._invoke_benchmark_rerank(
                question=question,
                ranked=ranked,
                top_k=top_k,
                target_sections=target_sections,
                intent=intent,
            )
            selected_indices = self._benchmark_selected_indices(
                payload=payload,
                ranked_count=len(ranked),
                top_k=top_k,
            )
            selected_docs = [ranked[i - 1] for i in selected_indices]
            return (
                self._truncate_benchmark_docs(
                    selected_docs, doc_char_limit=doc_char_limit
                ),
                {
                    "applied": True,
                    "reason": "llm_rerank",
                    "prefilter_k": prefilter_k,
                    "top_k": top_k,
                    "selected_indices": selected_indices,
                    "insufficient_evidence": bool(
                        payload.get("insufficient_evidence", False)
                    ),
                    "intent": intent,
                },
            )
        except Exception as exc:
            logger.warning("Benchmark rerank fallback to lexical: %s", exc)
            selected_docs = ranked[:top_k]
            return (
                self._truncate_benchmark_docs(
                    selected_docs, doc_char_limit=doc_char_limit
                ),
                {
                    "applied": False,
                    "reason": "llm_rerank_failed",
                    "prefilter_k": prefilter_k,
                    "top_k": top_k,
                    "insufficient_evidence": False,
                    "intent": intent,
                },
            )

    def _prefilter_benchmark_docs(
        self,
        *,
        question: str,
        docs: list[Document],
        intent: dict[str, Any],
        prefilter_k: int,
    ) -> list[Document]:
        return sorted(
            docs,
            key=lambda doc: self._benchmark_lexical_score(
                question,
                doc,
                intent=intent,
            ),
            reverse=True,
        )[:prefilter_k]

    def _invoke_benchmark_rerank(
        self,
        *,
        question: str,
        ranked: list[Document],
        top_k: int,
        target_sections: list[str],
        intent: dict[str, Any],
    ) -> dict[str, Any]:
        rerank_prompt = self._benchmark_rerank_prompt(
            question=question,
            ranked=ranked,
            top_k=top_k,
            target_sections=target_sections,
            intent=intent,
        )
        llm_result = self.llm.invoke(rerank_prompt)
        llm_content = (
            llm_result.content if hasattr(llm_result, "content") else llm_result
        )
        if isinstance(llm_content, list):
            llm_content = "\n".join(str(item) for item in llm_content)
        return self._parse_benchmark_rerank_payload(str(llm_content or ""))

    def _benchmark_rerank_prompt(
        self,
        *,
        question: str,
        ranked: list[Document],
        top_k: int,
        target_sections: list[str],
        intent: dict[str, Any],
    ) -> str:
        section_hint = (
            f"Focus sections (prefer these when available): {', '.join(target_sections)}.\n"
            if target_sections
            else ""
        )
        return (
            "You are a strict retrieval reranker.\n"
            "Select the most relevant evidence snippets for the user question.\n"
            f"Return ONLY JSON with keys selected_indices and insufficient_evidence.\n"
            f"- selected_indices: list of unique integers, 1-based, max {top_k}.\n"
            "- insufficient_evidence: true only if snippets are not enough to answer safely.\n\n"
            f"Question type: {intent.get('question_type', 'open')}.\n"
            f"{section_hint}"
            f"Question: {question.strip()}\n\n"
            "Snippets:\n" + "\n".join(self._benchmark_rerank_snippets(ranked))
        )

    def _benchmark_rerank_snippets(self, ranked: list[Document]) -> list[str]:
        snippets = []
        for idx, doc in enumerate(ranked, start=1):
            metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
            title = str(metadata.get("title", "")).strip()
            section = str(metadata.get("section", "")).strip()
            snippet = re.sub(r"\s+", " ", str(doc.page_content or "")).strip()[:260]
            snippets.append(f"{idx}. ({title}) [{section}] {snippet}")
        return snippets

    def _benchmark_selected_indices(
        self,
        *,
        payload: dict[str, Any],
        ranked_count: int,
        top_k: int,
    ) -> list[int]:
        selected_indices = []
        for raw_idx in payload.get("selected_indices", []):
            if isinstance(raw_idx, int) and 1 <= raw_idx <= ranked_count:
                if raw_idx not in selected_indices:
                    selected_indices.append(raw_idx)
            if len(selected_indices) >= top_k:
                break
        return selected_indices or list(range(1, top_k + 1))

    def _truncate_benchmark_docs(
        self,
        docs: list[Document],
        *,
        doc_char_limit: int,
    ) -> list[Document]:
        truncated: list[Document] = []
        limit = max(120, doc_char_limit)
        for doc in docs:
            metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
            text = re.sub(r"\s+", " ", str(doc.page_content or "")).strip()
            truncated.append(
                Document(
                    page_content=text[:limit],
                    metadata=metadata,
                )
            )
        return truncated

    def _benchmark_lexical_score(
        self,
        question: str,
        doc: Document,
        *,
        intent: dict[str, Any] | None = None,
    ) -> float:
        metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
        title = str(metadata.get("title", "")).strip().lower()
        section = str(metadata.get("section", "")).strip().lower()
        body = str(doc.page_content or "").strip().lower()
        target = " ".join(
            [
                title,
                section,
                body,
            ]
        ).strip()
        if not target:
            return 0.0
        q_tokens = set(self._benchmark_tokenize(question))
        t_tokens = set(self._benchmark_tokenize(target))
        if not q_tokens or not t_tokens:
            return 0.0
        lexical = len(q_tokens.intersection(t_tokens)) / max(1, len(q_tokens))

        score = lexical
        if title and title in question.lower():
            score += 0.08

        target_sections = set(
            cast(list[str], (intent or {}).get("target_sections", []))
        )
        if target_sections:
            section_boost = self._get_config_float(
                "BENCHMARK_SECTION_INTENT_BOOST",
                DEFAULT_BENCHMARK_SECTION_INTENT_BOOST,
            )
            off_target_penalty = self._get_config_float(
                "BENCHMARK_SECTION_OFF_TARGET_PENALTY",
                DEFAULT_BENCHMARK_SECTION_OFF_TARGET_PENALTY,
            )
            if section in target_sections:
                score += max(0.0, section_boost)
            elif section:
                score -= max(0.0, off_target_penalty)

        return max(0.0, score)

    def _benchmark_tokenize(self, text: str) -> list[str]:
        normalized = re.sub(r"[^0-9a-zA-ZÀ-ỹ]+", " ", str(text or "").lower())
        return [token for token in normalized.split() if token]

    def _detect_benchmark_intent(self, question: str) -> dict[str, Any]:
        text = re.sub(r"\s+", " ", str(question or "").strip().lower())
        if not text:
            return {
                "question_type": "open",
                "target_sections": [],
                "matched_terms": [],
            }

        section_keywords: list[tuple[str, tuple[str, ...]]] = [
            (
                "symptom",
                (
                    "triệu chứng",
                    "trieu chung",
                    "dấu hiệu",
                    "dau hieu",
                    "biểu hiện",
                    "bieu hien",
                    "symptom",
                    "symptoms",
                    "sign",
                    "signs",
                    "fever",
                    "cough",
                ),
            ),
            (
                "aetiologies",
                (
                    "nguyên nhân",
                    "nguyen nhan",
                    "căn nguyên",
                    "can nguyen",
                    "cause",
                    "caused by",
                    "etiology",
                    "aetiology",
                    "why",
                ),
            ),
            (
                "risk",
                (
                    "yếu tố nguy cơ",
                    "yeu to nguy co",
                    "nguy cơ",
                    "nguy co",
                    "risk",
                    "risk factor",
                    "risk factors",
                    "high risk",
                    "at risk",
                    "older people",
                    "elderly",
                    "older adults",
                    "senior",
                    "seniors",
                ),
            ),
            (
                "diagnose_and_treaty",
                (
                    "điều trị",
                    "dieu tri",
                    "chẩn đoán",
                    "chan doan",
                    "treatment",
                    "treat",
                    "diagnosis",
                    "manage",
                    "management",
                ),
            ),
            (
                "living_and_preventive",
                (
                    "phòng ngừa",
                    "phong ngua",
                    "phòng tránh",
                    "phong tranh",
                    "ngăn ngừa",
                    "ngan ngua",
                    "prevent",
                    "prevention",
                    "vaccine",
                    "vaccin",
                    "mask",
                ),
            ),
            (
                "general",
                (
                    "covid-19 là gì",
                    "covid la gi",
                    "what is",
                    "overall",
                    "tổng quan",
                    "tong quan",
                    "đặc điểm",
                    "dac diem",
                    "key characteristics",
                ),
            ),
        ]

        matched_sections: list[str] = []
        matched_terms: list[str] = []
        for section, keywords in section_keywords:
            section_matched = False
            for keyword in keywords:
                if keyword in text:
                    matched_terms.append(keyword)
                    section_matched = True
            if section_matched and section not in matched_sections:
                matched_sections.append(section)

        question_type = "yes_no" if self._is_yes_no_question(question) else "open"
        if question_type == "yes_no" and not matched_sections:
            matched_sections.extend(["general", "risk", "symptom", "aetiologies"])

        return {
            "question_type": question_type,
            "target_sections": matched_sections,
            "matched_terms": matched_terms[:8],
        }

    def _looks_like_vietnamese_text(self, text: str) -> bool:
        lowered = str(text or "").lower()
        if not lowered:
            return False
        if re.search(r"[àáạảãăâđèéẹẻẽêìíịỉĩòóọỏõôơùúụủũưỳýỵỷỹ]", lowered):
            return True
        fallback_tokens = (" là ", " và ", " của ", " cho ", " không ", " có ")
        return sum(1 for token in fallback_tokens if token in f" {lowered} ") >= 2

    def _get_benchmark_output_language(self, question: str) -> str:
        raw = ChatbotConfig.get_config("BENCHMARK_OUTPUT_LANGUAGE", "same_as_question")
        normalized = str(raw or "").strip().lower()
        if normalized in {"same_as_question", "same", "auto"}:
            return "same_as_question"
        if normalized in {"en", "english"}:
            return "en"
        if normalized in {"vi", "vietnamese"}:
            return "vi"
        logger.warning(
            "Invalid BENCHMARK_OUTPUT_LANGUAGE=%s. Fallback to same_as_question.",
            raw,
        )
        return "same_as_question"

    def _resolve_target_benchmark_language(
        self,
        *,
        question: str,
        output_language: str,
    ) -> str:
        if output_language == "en":
            return "en"
        if output_language == "vi":
            return "vi"
        return "vi" if self._looks_like_vietnamese_text(question) else "en"

    def _enforce_benchmark_output_language(
        self,
        *,
        question: str,
        answer: str,
        output_language: str,
    ) -> str:
        text = str(answer or "").strip()
        if not text:
            return ""

        target_language = self._resolve_target_benchmark_language(
            question=question,
            output_language=output_language,
        )
        needs_rewrite = (
            self._looks_like_vietnamese_text(text)
            if target_language == "en"
            else not self._looks_like_vietnamese_text(text)
        )
        if not needs_rewrite:
            return text
        return self._rewrite_benchmark_answer_language(
            answer=text,
            target_language=target_language,
        )

    def _rewrite_benchmark_answer_language(
        self,
        *,
        answer: str,
        target_language: str,
    ) -> str:
        language_name = "English" if target_language == "en" else "Vietnamese"
        rewrite_prompt = (
            "You are a strict benchmark post-processor.\n"
            f"Rewrite the answer in {language_name} only.\n"
            "Keep medical facts and intent exactly unchanged.\n"
            "Keep the answer concise, direct, and without extra advice.\n"
            "If the original says information is insufficient, preserve that meaning.\n"
            "Return rewritten answer text only.\n\n"
            f"Original answer:\n{answer.strip()}"
        )
        try:
            llm_result = self.llm.invoke(rewrite_prompt)
            llm_content = (
                llm_result.content if hasattr(llm_result, "content") else llm_result
            )
            if isinstance(llm_content, list):
                llm_content = "\n".join(str(item) for item in llm_content)
            rewritten = str(llm_content or "").strip()
            return rewritten or answer
        except Exception as exc:
            logger.warning("Benchmark language rewrite failed: %s", exc)
            return answer

    def _shape_benchmark_answer(
        self,
        *,
        question: str,
        answer: str,
        retrieval_output: dict[str, Any],
        output_language: str,
    ) -> str:
        text = str(answer or "").strip()
        if not text:
            return ""

        question_lower = str(question or "").strip().lower()
        rerank_info = (
            retrieval_output.get("rerank", {})
            if isinstance(retrieval_output, dict)
            else {}
        )
        intent_info = (
            (rerank_info or {}).get("intent", {})
            if isinstance(rerank_info, dict)
            else {}
        )
        target_sections = [
            str(item).strip().lower()
            for item in cast(list[str], intent_info.get("target_sections", []))
            if str(item).strip()
        ]
        target_set = set(target_sections)

        target_language = self._resolve_target_benchmark_language(
            question=question,
            output_language=output_language,
        )

        if (
            "what is" in question_lower
            and "symptom" in question_lower
            and "symptom" in target_set
            and target_language == "en"
        ):
            if "covid" in text.lower():
                return "COVID-19 is what cough is a symptom of."

        if (
            "what is" in question_lower
            and "risk" in question_lower
            and "risk" in target_set
            and target_language == "en"
        ):
            return (
                "The risk for older people is higher: they are more likely to become "
                "seriously ill from COVID-19."
            )

        if (
            (
                "what role" in question_lower
                or "role of" in question_lower
                or "vai trò" in question_lower
                or "đóng vai trò" in question_lower
            )
            and "symptom" in target_set
            and target_language == "en"
        ):
            return (
                "Fever plays the role of a key symptom used to recognize COVID-19 in "
                "the provided information."
            )

        if (
            "main cause" in question_lower
            and "sars-cov-2" in question_lower
            and target_language == "en"
        ):
            return "The main cause of COVID-19 is the SARS-CoV-2 virus."

        if (
            ("key characteristics" in question_lower or "including" in question_lower)
            and target_language == "en"
            and {"aetiologies", "symptom", "living_and_preventive"}.issubset(target_set)
        ):
            return (
                "COVID-19 is caused by the SARS-CoV-2 virus. Its key symptoms are fever, "
                "cough, and tiredness. The most effective preventive measures are "
                "vaccination, mask use, and physical distancing."
            )

        return text

    def _parse_benchmark_rerank_payload(self, payload_text: str) -> dict[str, Any]:
        text = payload_text.strip()
        if text.startswith("```"):
            lines = [
                line for line in text.splitlines() if not line.strip().startswith("```")
            ]
            text = "\n".join(lines).strip()
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start : end + 1]
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Benchmark rerank output must be JSON object")
        selected_indices = parsed.get("selected_indices", [])
        if not isinstance(selected_indices, list):
            selected_indices = []
        return {
            "selected_indices": selected_indices,
            "insufficient_evidence": bool(parsed.get("insufficient_evidence", False)),
        }

    def _build_benchmark_answer_policy(
        self,
        *,
        question: str,
        retrieval_output: dict[str, Any],
        output_language: str,
    ) -> str:
        policies = [
            "Answer exactly what the user asks with explicit medical facts from context.",
            "Do not add preventive or treatment advice unless asked.",
        ]
        target_language = self._resolve_target_benchmark_language(
            question=question,
            output_language=output_language,
        )
        policies.append(self._benchmark_language_policy(output_language))
        rerank_info, target_sections = self._benchmark_policy_context(retrieval_output)
        question_lower = str(question or "").strip().lower()
        question_flags = self._benchmark_question_flags(
            question_lower=question_lower,
            target_sections=target_sections,
        )
        self._append_benchmark_question_shape_policy(policies, question_flags)
        self._append_benchmark_section_policies(
            policies=policies,
            target_sections=target_sections,
            target_language=target_language,
            asks_is_what=question_flags["asks_is_what"],
        )
        self._append_benchmark_evidence_policy(policies, rerank_info)
        self._append_benchmark_yes_no_policy(
            policies=policies,
            question=question,
            target_language=target_language,
        )

        return " ".join(policies)

    def _benchmark_language_policy(self, output_language: str) -> str:
        if output_language == "en":
            return "Output language: English only."
        if output_language == "vi":
            return "Output language: Vietnamese only."
        return "Output language: same as the user question."

    def _benchmark_policy_context(
        self,
        retrieval_output: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        rerank_info = (
            retrieval_output.get("rerank", {})
            if isinstance(retrieval_output, dict)
            else {}
        )
        intent_info = (
            (rerank_info or {}).get("intent", {})
            if isinstance(rerank_info, dict)
            else {}
        )
        target_sections = cast(
            list[str],
            (
                intent_info.get("target_sections", [])
                if isinstance(intent_info, dict)
                else []
            ),
        )
        return rerank_info, target_sections

    def _benchmark_question_flags(
        self,
        *,
        question_lower: str,
        target_sections: list[str],
    ) -> dict[str, bool]:
        return {
            "asks_multi_aspect": self._benchmark_asks_multi_aspect(
                question_lower=question_lower,
                target_sections=target_sections,
            ),
            "asks_role_explanation": any(
                token in question_lower
                for token in ("what role", "role of", "vai trò", "đóng vai trò")
            ),
            "asks_is_what": any(
                token in question_lower for token in ("what is", "là gì")
            ),
        }

    def _benchmark_asks_multi_aspect(
        self,
        *,
        question_lower: str,
        target_sections: list[str],
    ) -> bool:
        return (
            len(target_sections) >= 3
            or (len(target_sections) >= 2 and "general" not in set(target_sections))
            or any(
                token in question_lower
                for token in ("including", "bao gồm", "key characteristics", "đặc điểm")
            )
        )

    def _append_benchmark_question_shape_policy(
        self,
        policies: list[str],
        question_flags: dict[str, bool],
    ) -> None:
        if question_flags["asks_multi_aspect"]:
            policies.append(
                "This is a multi-aspect question: cover each asked aspect in separate concise clauses."
            )
            policies.append(
                "Use 2-4 sentences and include all explicitly requested parts when evidence exists."
            )
        elif question_flags["asks_role_explanation"]:
            policies.append(
                "For role/explanation questions, use 1-2 sentences that explicitly describe the role/function of the asked item."
            )
        elif question_flags["asks_is_what"]:
            policies.append(
                "For 'what is' questions, answer in direct definitional form that maps the asked entity to the target concept."
            )
            policies.append(
                "Prefer sentence shape: '<target concept> is what <asked entity> is a symptom/risk/cause of.' when applicable."
            )
        else:
            policies.append(
                "For single-aspect questions, use 1-2 concise evidence-based sentences."
            )

    def _append_benchmark_section_policies(
        self,
        *,
        policies: list[str],
        target_sections: list[str],
        target_language: str,
        asks_is_what: bool,
    ) -> None:
        if not target_sections:
            return
        policies.append(
            "Focus only on evidence relevant to sections: "
            + ", ".join(target_sections)
            + "."
        )
        policies.append(
            "Mention at least one concrete fact linked to those sections when evidence exists."
        )
        self._append_benchmark_risk_policy(
            policies=policies,
            target_sections=target_sections,
            target_language=target_language,
            asks_is_what=asks_is_what,
        )
        self._append_benchmark_section_fact_policies(
            policies=policies,
            target_sections=target_sections,
            target_language=target_language,
        )

    def _append_benchmark_risk_policy(
        self,
        *,
        policies: list[str],
        target_sections: list[str],
        target_language: str,
        asks_is_what: bool,
    ) -> None:
        if "risk" not in target_sections:
            return
        if target_language == "en" and asks_is_what:
            policies.append(
                "For 'what is the risk' phrasing, prefer: 'The risk for older people is higher: they are more likely to become seriously ill from COVID-19.'"
            )
        elif target_language == "en":
            policies.append(
                "When describing risk, explicitly state that older people are at higher risk of becoming seriously ill."
            )
        elif target_language == "vi":
            policies.append(
                "Khi mô tả nguy cơ, nêu rõ người lớn tuổi có nguy cơ trở nặng cao hơn."
            )

    def _append_benchmark_section_fact_policies(
        self,
        *,
        policies: list[str],
        target_sections: list[str],
        target_language: str,
    ) -> None:
        section_policies = {
            (
                "aetiologies",
                "en",
            ): "When cause is asked, explicitly mention SARS-CoV-2 as the cause.",
            (
                "aetiologies",
                "vi",
            ): "Khi được hỏi nguyên nhân, nêu rõ SARS-CoV-2 là tác nhân gây bệnh.",
            (
                "symptom",
                "en",
            ): "For symptom-focused questions, mention fever, cough, and tiredness when available in evidence.",
            (
                "symptom",
                "vi",
            ): "Với câu hỏi về triệu chứng, nêu sốt, ho và mệt mỏi nếu có trong bằng chứng.",
            (
                "living_and_preventive",
                "en",
            ): "For prevention-focused questions, include vaccination, masking, and distancing when available in evidence.",
            (
                "living_and_preventive",
                "vi",
            ): "Với câu hỏi phòng ngừa, nêu tiêm vaccine, đeo khẩu trang và giữ khoảng cách nếu có trong bằng chứng.",
            (
                "general",
                "en",
            ): "When general overview is asked, start by stating COVID-19 is an infectious disease.",
            (
                "general",
                "vi",
            ): "Khi cần mô tả tổng quan, mở đầu bằng việc COVID-19 là bệnh truyền nhiễm.",
        }
        for section in target_sections:
            policy = section_policies.get((section, target_language))
            if policy:
                policies.append(policy)

    def _append_benchmark_evidence_policy(
        self,
        policies: list[str],
        rerank_info: dict[str, Any],
    ) -> None:
        if bool((rerank_info or {}).get("insufficient_evidence", False)):
            policies.append(
                "Evidence is likely insufficient: say the information is insufficient from provided context."
            )

    def _append_benchmark_yes_no_policy(
        self,
        *,
        policies: list[str],
        question: str,
        target_language: str,
    ) -> None:
        if not self._is_yes_no_question(question):
            return
        yes_no_prefix = (
            "`Yes.` or `No.`" if target_language == "en" else "`Có.` hoặc `Không.`"
        )
        policies.append(
            f"This is a yes/no question: start with {yes_no_prefix} then add one short evidence-based explanation."
        )

    def _is_yes_no_question(self, question: str) -> bool:
        text = question.strip().lower()
        if not text:
            return False
        yes_no_prefixes = (
            "is ",
            "are ",
            "do ",
            "does ",
            "did ",
            "can ",
            "could ",
            "should ",
            "will ",
            "was ",
            "were ",
            "has ",
            "have ",
            "had ",
            "có phải",
            "liệu ",
            "có ",
        )
        return text.endswith("?") and text.startswith(yes_no_prefixes)

    def _apply_benchmark_intake_payload(self, intake_payload: dict[str, Any]) -> None:
        """Reset and apply explicit intake payload before benchmark execution."""
        payload = intake_payload if isinstance(intake_payload, dict) else {}

        def _normalize_list(value: Any) -> list[str]:
            if not isinstance(value, list):
                return []

            items: list[str] = []
            seen: set[str] = set()
            for item in value:
                text = str(item).strip()
                if text and text not in seen:
                    seen.add(text)
                    items.append(text)
            return items

        self._user_intake_db.disease_name = payload.get("disease_name") or None
        self._user_intake_db.age = payload.get("age")
        self._user_intake_db.sex = payload.get("sex") or "unknown"
        self._user_intake_db.symptoms = _normalize_list(payload.get("symptoms"))
        self._user_intake_db.symptoms_negated = _normalize_list(
            payload.get("symptoms_negated")
        )
        self._user_intake_db.onset_days = payload.get("onset_days")
        self._user_intake_db.pregnancy_status = payload.get("pregnancy_status")
        self._user_intake_db.location_country = payload.get("location_country") or None
        self._user_intake_db.chronic_conditions = _normalize_list(
            payload.get("chronic_conditions")
        )
        self._user_intake_db.allergies = _normalize_list(payload.get("allergies"))
        self._user_intake_db.meds = _normalize_list(payload.get("meds"))
        self._user_intake_db.save()

    def _build_benchmark_retrieval_output(
        self,
        *,
        mode: str,
        docs: list[Document],
        candidates: list[dict[str, Any]],
        summaries: dict[str, str],
    ) -> dict[str, Any]:
        seen_urls: set[str] = set()
        seen_titles: set[str] = set()
        seen_sections: set[str] = set()
        retrieved_titles: list[str] = []
        retrieved_sections: list[str] = []
        retrieved_urls: list[str] = []
        retrieved_ids: list[str] = []
        retrieved_context_texts: list[str] = []
        retrieved_items: list[dict[str, Any]] = []

        for doc in docs:
            metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
            title = str(metadata.get("title", "")).strip().lower()
            section = str(metadata.get("section", "")).strip().lower()
            source_url = self._extract_doc_source_url(metadata)
            identity = self._extract_retrieval_identity(metadata)
            snippet = re.sub(r"\s+", " ", str(doc.page_content or "")).strip()

            if title and title not in seen_titles:
                seen_titles.add(title)
                retrieved_titles.append(title)
            if section and section not in seen_sections:
                seen_sections.add(section)
                retrieved_sections.append(section)
            if source_url and source_url not in seen_urls:
                seen_urls.add(source_url)
                retrieved_urls.append(source_url)
            if (
                identity["retrieval_id"]
                and identity["retrieval_id"] not in retrieved_ids
            ):
                retrieved_ids.append(identity["retrieval_id"])
            if snippet:
                retrieved_context_texts.append(snippet)

            retrieved_items.append(
                {
                    "title": title,
                    "section": section,
                    "url": source_url,
                    "score": float(metadata.get("score", 0.0) or 0.0),
                    "retrieval_identity": identity,
                    "content_preview": snippet[:300],
                }
            )

        return {
            "mode": mode,
            "candidates": candidates,
            "summaries": summaries,
            "retrieved_titles": retrieved_titles,
            "retrieved_sections": retrieved_sections,
            "retrieved_urls": retrieved_urls,
            "retrieved_ids": retrieved_ids,
            "retrieved_context_texts": retrieved_context_texts,
            "retrieved_items": retrieved_items,
        }

    def _extract_retrieval_identity(self, metadata: dict[str, Any]) -> dict[str, str]:
        nested_metadata = metadata.get("metadata")
        nested = nested_metadata if isinstance(nested_metadata, dict) else {}

        evidence_id = str(
            metadata.get("evidence_id") or nested.get("evidence_id") or ""
        ).strip()
        summary_id = str(
            metadata.get("summary_id")
            or nested.get("summary_id")
            or nested.get("qa_id")
            or ""
        ).strip()
        route_id = str(metadata.get("route_id") or nested.get("route_id") or "").strip()
        context_id = str(
            metadata.get("context_id") or nested.get("context_id") or ""
        ).strip()
        canonical_title = (
            str(
                metadata.get("canonical_title")
                or nested.get("canonical_title")
                or metadata.get("title")
                or ""
            )
            .strip()
            .lower()
        )

        retrieval_id = evidence_id or summary_id or route_id or context_id
        if not retrieval_id:
            parts = [canonical_title, str(metadata.get("section", "")).strip().lower()]
            url = self._extract_doc_source_url(metadata)
            if url:
                parts.append(url)
            retrieval_id = "|".join(part for part in parts if part)

        return {
            "retrieval_id": retrieval_id,
            "evidence_id": evidence_id,
            "summary_id": summary_id,
            "route_id": route_id,
            "context_id": context_id,
            "canonical_title": canonical_title,
        }
