"""
Chatbot Service
Main service for medical RAG chatbot with NER, negation detection, and streaming support.
"""

import logging
import json
import math
import re
import time
import uuid
from collections.abc import AsyncGenerator, Generator
from typing import Any, cast

from asgiref.sync import sync_to_async
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from chatbot.models import (
    ChatbotConfig,
    ChatMessage,
    ChatSession,
    MessageRole,
    UserIntake,
)
from chatbot.prompts.system_vi import SYSTEM_PROMPT_VI
from vision.models import XRayAnalysis
from vision.serializers import XRayAnalysisDisplaySerializer
from chatbot.services.constants import (
    DEFAULT_DOCS_CACHE_SIZE,
    DEFAULT_DOC_PREVIEW_LENGTH,
    DEFAULT_INDEX_B_K,
    DEFAULT_RAG_B_TOPK,
    DEFAULT_RAG_FINAL_TITLES,
    DEFAULT_RAG_MERGED_LIMIT,
    DEFAULT_RAG_MERGE_WEIGHT_ENTITIES,
    DEFAULT_RAG_MERGE_WEIGHT_QUERY,
    DEFAULT_RAG_NEG_SYM_SIM_THRESH,
    DEFAULT_RAG_PENALTY_ALPHA,
    DEFAULT_RAG_THRESH_C,
    DEFAULT_RAG_TITLE_TOP_M,
    DEFAULT_SECTION_ITEMS_LIMIT,
    DEFAULT_SINGLE_DISEASE_DOCS_K,
    HEADER_EVIDENCE_BLOCK,
    HEADER_FAQ_MATCH,
    HEADER_MULTI_DISEASE_ANALYSIS,
    HEADER_MULTI_DISEASE_CANDIDATES,
    HEADER_PATIENT_INFO,
    HEADER_SINGLE_DISEASE,
    HEADER_SINGLE_DISEASE_SUBTITLE,
    HEADER_XRAY_ANALYSIS_END,
    HEADER_XRAY_ANALYSIS_START,
    HEADER_XRAY_FINDINGS,
    HEADER_XRAY_PREDICTION,
    HEADER_XRAY_PROBABILITIES,
    GENERIC_SYMPTOM_TERMS,
    MSG_ANALYSIS_ERROR,
    MSG_CONTEXT_HINT_MULTI,
    MSG_CONTEXT_HINT_SINGLE,
    QUERY_ANALYZER_PROMPT,
    MSG_NO_DOCS_FOR_TITLE,
    MSG_PROCESSING_ERROR,
    MSG_STREAMING_ERROR,
    MSG_XRAY_INSTRUCTION,
    SECTION_HEADERS,
    SECTION_ORDER,
    SYNONYM_MAP,
    XRAY_RESPIRATORY_DOMAIN_CONTEXT_VI,
)
from chatbot.services.gemini_manager import get_gemini_manager
from vector_store.services import VectorStoreManager
from nlp.services.runtime import get_shared_integrator

logger = logging.getLogger(__name__)


class ChatbotService:
    """
    Medical RAG Chatbot service with database persistence.
    Provides chat functionality with NER, negation detection, and streaming.
    """

    def __init__(self, session: ChatSession) -> None:
        """
        Initialize ChatbotService for a specific session.

        Args:
            session: ChatSession model instance
        """
        self.session = session
        self.session_id = str(session.session_id)

        # User intake tracking (database model)
        self._user_intake_db, _ = UserIntake.objects.get_or_create(
            customer=self.session.customer
        )

        # Document cache for UI
        self._last_docs_cache: list[Document] = []
        self._last_query_text = ""
        self._last_audit: dict[str, Any] = {}

        # Initialize managers
        self._init_managers()
        self._setup_chain()

    def _init_managers(self) -> None:
        """Initialize API and vector managers."""
        self.llm_manager = get_gemini_manager()
        self.vector_manager = VectorStoreManager()
        self.llm = self.llm_manager.create_llm()
        self.integrator = None
        if self._get_config_bool("ENABLE_LOCAL_NLP_FALLBACK", False):
            try:
                self.integrator = get_shared_integrator()
            except Exception as exc:
                logger.warning("Local NLP fallback unavailable: %s", exc)
        logger.info(f"ChatbotService initialized for session: {self.session_id}")

    def _get_config_bool(self, key: str, default: bool) -> bool:
        """Read boolean config with safe coercion from JSON/string values."""
        value = ChatbotConfig.get_config(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    def _get_config_int(self, key: str, default: int) -> int:
        """Read integer config with fallback on invalid values."""
        value = ChatbotConfig.get_config(key, default)
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return default
        return default

    def _get_config_float(self, key: str, default: float) -> float:
        """Read float config with fallback on invalid values."""
        value = ChatbotConfig.get_config(key, default)
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return default
        return default

    def _setup_chain(self) -> None:
        """Setup the conversational chain."""
        prompt = self._create_prompt_template()
        get_context = self._create_context_function()

        self._chain = (
            RunnablePassthrough.assign(context=get_context)
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Streaming chain (direct LLM in pipeline)
        self._streaming_chain = (
            RunnablePassthrough.assign(context=get_context)
            | prompt
            | self.llm
            | StrOutputParser()
        )
        logger.info("Conversational chain setup complete")

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create prompt template using system prompt."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT_VI),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

    def _get_chat_history(self) -> list[HumanMessage | AIMessage]:
        """Load chat history from database."""
        messages: list[HumanMessage | AIMessage] = []
        db_messages = (
            ChatMessage.objects.filter(session=self.session)
            .order_by("created_at")
            .values("role", "content")
        )

        for msg in db_messages[: self.session.max_messages]:
            if msg["role"] == MessageRole.USER:
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == MessageRole.ASSISTANT:
                messages.append(AIMessage(content=msg["content"]))

        return messages

    def _save_message(
        self,
        role: str,
        content: str,
        response_time_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """Save a message to the database."""
        return ChatMessage.objects.create(
            session=self.session,
            role=role,
            content=content,
            response_time_ms=response_time_ms,
            metadata=metadata or {},
        )

    def _create_context_function(self):
        """Create the context-building function for RAG."""
        return RunnableLambda(self._build_context)

    async def _build_context(self, inputs: dict[str, Any]) -> str:
        """Build the final RAG context for the active request."""
        question = inputs.get("question") or ""
        audit_id = str(uuid.uuid4())

        try:
            analysis_input = inputs.get("analysis")
            if isinstance(analysis_input, dict):
                analysis = cast(dict[str, Any], analysis_input)
            else:
                analysis = await sync_to_async(self._analyze_query)(question)
            q_cleaned = cast(str, analysis.get("q_cleaned", question))
            gate = await sync_to_async(self._gate_with_index_c)(q_cleaned)
            await sync_to_async(self._log_stage2_gate)(
                question=question,
                q_cleaned=q_cleaned,
                gate=gate,
            )

            if gate.get("go_single") and gate.get("title"):
                title = str(gate.get("title", ""))
                evidence_docs = await sync_to_async(self._fetch_docs_for_title)(
                    title, index="B", k=DEFAULT_SINGLE_DISEASE_DOCS_K
                )
                summaries = await sync_to_async(self._fetch_summary_docs_for_titles)(
                    [title], cast(str, analysis.get("q_cleaned", ""))
                )
                context = self._build_single_disease_context_with_summary(
                    title=title,
                    docs=evidence_docs,
                    summaries=summaries,
                )

                self._last_docs_cache = evidence_docs[:DEFAULT_DOCS_CACHE_SIZE]
                self._last_audit = {
                    "audit_id": audit_id,
                    "ts": time.time(),
                    "mode": "single-disease",
                    "title": title,
                    "router_score": gate.get("top_score", 0.0),
                    "doc_count": len(evidence_docs),
                }
            else:
                tier_result = await sync_to_async(self._multi_disease_retrieval)(
                    analysis
                )
                context = self._build_multi_disease_context(
                    question, analysis, tier_result
                )
                self._last_docs_cache = cast(
                    list[Document], tier_result.get("evidence_docs", [])
                )[:DEFAULT_DOCS_CACHE_SIZE]
                self._last_audit = {
                    "audit_id": audit_id,
                    "ts": time.time(),
                    "mode": "multi-disease-v2",
                    "candidates": [
                        c.get("title") for c in tier_result.get("candidates", [])[:5]
                    ],
                }

            self._last_query_text = question

        except Exception as ex:
            logger.warning(f"Context build error: {ex}")
            context = MSG_ANALYSIS_ERROR
            self._last_query_text = question

        xray_context = str(inputs.get("xray_context") or "")
        if not xray_context and inputs.get("xray_analysis_id"):
            xray_context = await sync_to_async(self._get_xray_context_by_id)(
                inputs["xray_analysis_id"]
            )

        intake_ctx = await sync_to_async(self._get_intake_context)()
        context_parts = [context]
        if xray_context:
            context_parts.append(xray_context)
        if intake_ctx:
            context_parts.append(intake_ctx)
        return "\n".join(part for part in context_parts if part)

    # TODO: Remove this method
    def query(self, question: str) -> str:
        """
        Process a medical question and return response.

        Args:
            question: User's question

        Returns:
            AI response string
        """
        start_time = time.time()

        try:
            # Save user message
            self._save_message(MessageRole.USER, question)

            # Analyze once per turn and update intake
            analysis = self._analyze_query(question)
            self._apply_analysis_to_intake(question, analysis=analysis)

            # Get chat history
            chat_history = self._get_chat_history()

            # Run chain
            response = self._chain.invoke(
                {
                    "question": question,
                    "chat_history": chat_history,
                    "analysis": analysis,
                }
            )
            response_text = str(response)
            source_urls = self.get_last_source_urls()

            response_time_ms = int((time.time() - start_time) * 1000)

            # Save assistant message
            assistant_metadata: dict[str, Any] = {
                "audit_id": self._last_audit.get("audit_id"),
                "mode": self._last_audit.get("mode"),
            }
            if source_urls:
                assistant_metadata["source_urls"] = source_urls
            self._save_message(
                MessageRole.ASSISTANT,
                response_text,
                response_time_ms=response_time_ms,
                metadata=assistant_metadata,
            )

            logger.info(f"Query processed in {response_time_ms}ms")
            return response_text

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_msg = MSG_PROCESSING_ERROR.format(error=str(e))
            self._save_message(MessageRole.ASSISTANT, error_msg)
            return error_msg

    async def aquery(self, question: str, xray_analysis_id: int | None = None) -> str:
        """
        Async process a medical question and return response.
        """
        start_time = time.time()

        try:
            xray_payload = await sync_to_async(self._get_xray_analysis_payload)(
                xray_analysis_id
            )
            user_metadata = dict(xray_payload["user_metadata"])

            # Save user message
            await sync_to_async(self._save_message)(
                MessageRole.USER, question, metadata=user_metadata
            )

            # Analyze once per turn and update intake
            analysis = await sync_to_async(self._analyze_query)(question)
            await sync_to_async(self._apply_analysis_to_intake)(
                question, analysis=analysis
            )

            # Get chat history
            chat_history = await sync_to_async(self._get_chat_history)()

            # Run chain
            response = await self._chain.ainvoke(
                {
                    "question": question,
                    "chat_history": chat_history,
                    "xray_analysis_id": xray_analysis_id,
                    "xray_context": xray_payload["context"],
                    "analysis": analysis,
                }
            )
            response_text = str(response)
            source_urls = self.get_last_source_urls()

            response_time_ms = int((time.time() - start_time) * 1000)
            assistant_metadata = {
                "audit_id": self._last_audit.get("audit_id"),
                "mode": self._last_audit.get("mode"),
            }
            if xray_payload["serialized"] is not None:
                assistant_metadata["xray_analysis"] = xray_payload["serialized"]
            if source_urls:
                assistant_metadata["source_urls"] = source_urls

            # Save assistant message
            await sync_to_async(self._save_message)(
                MessageRole.ASSISTANT,
                response_text,
                response_time_ms=response_time_ms,
                metadata=assistant_metadata,
            )

            logger.info(f"Query processed in {response_time_ms}ms")
            return response_text

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_msg = MSG_PROCESSING_ERROR.format(error=str(e))
            await sync_to_async(self._save_message)(MessageRole.ASSISTANT, error_msg)
            return error_msg

    # TODO: Remove this method
    def stream_response(self, question: str) -> Generator[str, None, None]:
        """
        Stream response tokens for a question.

        Args:
            question: User's question

        Yields:
            Response text chunks
        """
        # Save user message
        self._save_message(MessageRole.USER, question)

        # Analyze once per turn and update intake
        analysis = self._analyze_query(question)
        self._apply_analysis_to_intake(question, analysis=analysis)

        # Get history
        chat_history = self._get_chat_history()

        start_time = time.time()
        full_response = ""

        try:
            for chunk in self._streaming_chain.stream(
                {
                    "question": question,
                    "chat_history": chat_history,
                    "analysis": analysis,
                }
            ):
                # StrOutputParser always returns str
                full_response += chunk
                yield chunk

            response_with_sources = full_response
            source_urls = self.get_last_source_urls()

            # Save complete response
            response_time_ms = int((time.time() - start_time) * 1000)
            assistant_metadata: dict[str, Any] = {}
            if source_urls:
                assistant_metadata["source_urls"] = source_urls
            self._save_message(
                MessageRole.ASSISTANT,
                response_with_sources,
                response_time_ms=response_time_ms,
                metadata=assistant_metadata,
            )

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            error_msg = MSG_STREAMING_ERROR.format(error=str(e))
            self._save_message(MessageRole.ASSISTANT, error_msg)
            yield error_msg

    async def astream_response(
        self, question: str, xray_analysis_id: int | None = None
    ) -> AsyncGenerator[str, None]:
        """
        Async stream response tokens for a question.

        Args:
            question: User's question
            xray_analysis_id: Optional ID of an X-ray analysis to include in context.

        Yields:
            Response text chunks
        """
        xray_payload = await sync_to_async(self._get_xray_analysis_payload)(
            xray_analysis_id
        )
        user_metadata = dict(xray_payload["user_metadata"])

        # Save user message (DB op -> async)
        await sync_to_async(self._save_message)(
            MessageRole.USER, question, metadata=user_metadata
        )

        # Analyze once per turn and update intake
        analysis = await sync_to_async(self._analyze_query)(question)
        await sync_to_async(self._apply_analysis_to_intake)(question, analysis=analysis)

        # Get history (DB op -> async)
        chat_history = await sync_to_async(self._get_chat_history)()

        start_time = time.time()
        full_response = ""

        try:
            async for chunk in self._streaming_chain.astream(
                {
                    "question": question,
                    "chat_history": chat_history,
                    "xray_analysis_id": xray_analysis_id,
                    "xray_context": xray_payload["context"],
                    "analysis": analysis,
                }
            ):
                # StrOutputParser always returns str
                full_response += chunk
                yield chunk

            response_with_sources = full_response
            source_urls = self.get_last_source_urls()

            # Prepare metadata for assistant message
            assistant_metadata = {
                "audit_id": self._last_audit.get("audit_id"),
                "mode": self._last_audit.get("mode"),
            }
            if xray_payload["serialized"] is not None:
                assistant_metadata["xray_analysis"] = xray_payload["serialized"]
            if source_urls:
                assistant_metadata["source_urls"] = source_urls

            # Save complete response (DB op -> async)
            response_time_ms = int((time.time() - start_time) * 1000)
            await sync_to_async(self._save_message)(
                MessageRole.ASSISTANT,
                response_with_sources,
                response_time_ms=response_time_ms,
                metadata=assistant_metadata,
            )

        except Exception as e:
            logger.error(f"Async streaming error: {e}")
            error_msg = MSG_STREAMING_ERROR.format(error=str(e))
            await sync_to_async(self._save_message)(MessageRole.ASSISTANT, error_msg)
            yield error_msg

    def _serialize_xray_analysis(self, analysis_record: XRayAnalysis) -> dict[str, Any]:
        """Serialize X-ray analysis for message metadata."""
        return cast(
            dict[str, Any],
            XRayAnalysisDisplaySerializer(analysis_record).data,
        )

    def _build_xray_context(self, analysis_record: XRayAnalysis) -> str:
        """Build the prompt context block for a stored X-ray analysis."""
        class_probs = analysis_record.class_probs
        probs = class_probs if isinstance(class_probs, dict) else {}
        top_probs = sorted(
            probs.items(),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:5]
        probs_str = (
            ", ".join(f"{label}: {float(score):.3f}" for label, score in top_probs)
            if top_probs
            else "Không có dữ liệu"
        )

        findings = analysis_record.findings
        findings_list = findings if isinstance(findings, list) else []
        findings_str = (
            ", ".join(str(item) for item in findings_list if str(item).strip())
            or "Không có phát hiện nổi bật"
        )

        return "".join(
            [
                HEADER_XRAY_ANALYSIS_START,
                XRAY_RESPIRATORY_DOMAIN_CONTEXT_VI,
                MSG_XRAY_INSTRUCTION,
                HEADER_XRAY_PREDICTION.format(pred_label=analysis_record.pred_label),
                HEADER_XRAY_PROBABILITIES.format(probs_str=probs_str),
                HEADER_XRAY_FINDINGS.format(findings_str=findings_str),
                HEADER_XRAY_ANALYSIS_END,
            ]
        )

    def _get_xray_analysis_payload(
        self, xray_analysis_id: int | None
    ) -> dict[str, Any]:
        """Load X-ray metadata/context once for the current request."""
        payload: dict[str, Any] = {
            "context": "",
            "serialized": None,
            "user_metadata": {},
        }
        if not xray_analysis_id:
            return payload

        try:
            analysis_record = XRayAnalysis.objects.get(id=xray_analysis_id)
        except XRayAnalysis.DoesNotExist:
            logger.warning("XRayAnalysis not found for id=%s", xray_analysis_id)
            return payload
        except Exception as exc:
            logger.warning(
                "Failed to load XRayAnalysis id=%s: %s", xray_analysis_id, exc
            )
            return payload

        payload["context"] = self._build_xray_context(analysis_record)
        payload["serialized"] = self._serialize_xray_analysis(analysis_record)
        if analysis_record.image:
            payload["user_metadata"] = {"attachment": analysis_record.image.url}
        return payload

    def _get_xray_context_by_id(self, xray_analysis_id: int | None) -> str:
        """Resolve X-ray context from an analysis id."""
        payload = self._get_xray_analysis_payload(xray_analysis_id)
        return cast(str, payload["context"])

    # ---------------------------
    # RAG Helper Methods
    # ---------------------------

    def _analyze_query(self, question: str) -> dict[str, Any]:
        """Analyze query via LLM JSON extraction with safe fallback."""
        normalized_question = re.sub(r"\s+", " ", str(question or "").strip()).lower()
        intake = self._user_intake_db
        known_symptoms = ", ".join(
            str(sym).strip().lower()
            for sym in (intake.symptoms or [])
            if str(sym).strip()
        )

        prompt = QUERY_ANALYZER_PROMPT.format(
            question=normalized_question,
            age=intake.age if intake.age is not None else "unknown",
            sex=intake.sex or "unknown",
            symptoms=known_symptoms or "none",
        )

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else response
            if isinstance(content, list):
                content = "\n".join(str(part) for part in content)
            raw_payload_text = str(content)
            payload = self._parse_query_analyzer_payload(raw_payload_text)
            entities_by_type = payload.get("entities_by_type", {}) or {}

            disease_mentions = [
                str(item).strip().lower()
                for item in payload.get("disease_mentions", [])
                if str(item).strip()
            ]
            disease_set = set(disease_mentions)

            symptom_positive_raw = self._expand_synonyms(
                [
                    str(item).strip().lower()
                    for item in payload.get("symptom_positive", [])
                    if str(item).strip()
                ]
            )
            symptom_negative_raw = self._expand_synonyms(
                [
                    str(item).strip().lower()
                    for item in payload.get("symptom_negative", [])
                    if str(item).strip()
                ]
            )
            symptom_positive = self._filter_generic_symptom_terms(
                symptom_positive_raw,
                disease_mentions=disease_set,
            )
            symptom_negative = self._filter_generic_symptom_terms(
                symptom_negative_raw,
                disease_mentions=disease_set,
            )
            etiology_terms = [
                str(item).strip().lower()
                for item in entities_by_type.get("aetiology", [])
                if str(item).strip()
            ]
            risk_terms = [
                str(item).strip().lower()
                for item in entities_by_type.get("risk", [])
                if str(item).strip()
            ]

            q_cleaned = str(payload.get("q_cleaned", "")).strip().lower()
            if not q_cleaned:
                q_cleaned = normalized_question
                for neg in symptom_negative:
                    q_cleaned = q_cleaned.replace(neg, "").strip()
                q_cleaned = re.sub(r"\s+", " ", q_cleaned).strip()

            q_symptom_parts = list(symptom_positive) + disease_mentions
            q_symptom_parts.extend(
                str(sym).strip().lower()
                for sym in (intake.symptoms or [])
                if str(sym).strip()
            )
            q_symptom = ", ".join(dict.fromkeys(q_symptom_parts))
            if not q_symptom:
                q_symptom = q_cleaned or normalized_question

            patient_state_extract = payload.get("patient_state_extract", {}) or {}
            analysis_result = {
                "original": question,
                "processed_text": str(
                    payload.get("normalized_query", normalized_question)
                ),
                "positives": {
                    "SYMPTOM": symptom_positive,
                    "ETIOLOGY": etiology_terms,
                    "RISK": risk_terms,
                },
                "negatives": {"SYMPTOM": symptom_negative},
                "has_negation": bool(symptom_negative),
                "entities_by_type": entities_by_type,
                "disease_mentions": disease_mentions,
                "patient_state_extract": patient_state_extract,
                "q_cleaned": q_cleaned,
                "q_symptom": q_symptom,
            }
            self._log_stage1_preprocess(
                question=question,
                raw_payload_text=raw_payload_text,
                parsed_payload=payload,
                analysis_result=analysis_result,
                fallback=False,
            )
            return analysis_result
        except Exception as exc:
            logger.warning("LLM query analyzer failed, using fallback: %s", exc)
            fallback_result = self._fallback_query_analysis(question)
            self._log_stage1_preprocess(
                question=question,
                raw_payload_text="",
                parsed_payload={},
                analysis_result=fallback_result,
                fallback=True,
                error=str(exc),
            )
            return fallback_result

    def _parse_query_analyzer_payload(self, payload_text: str) -> dict[str, Any]:
        text = payload_text.strip()
        if text.startswith("```"):
            lines = [
                line for line in text.splitlines() if not line.strip().startswith("```")
            ]
            text = "\n".join(lines).strip()
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Query analyzer output must be JSON object")
        return parsed

    def _fallback_query_analysis(self, question: str) -> dict[str, Any]:
        """Fallback with NER + negation detection, then heuristic."""
        try:
            from nlp.services.integrator import NERNegationIntegrator

            integrator = NERNegationIntegrator()
            ner_result = integrator.process_text(question)

            negated_entities = ner_result["entities"]["negated"]
            non_negated_entities = ner_result["entities"]["non_negated"]

            symptom_negative = [
                str(e.get("span", "")).strip().lower()
                for e in negated_entities
                if e.get("label") == "SYMPTOM" and str(e.get("span", "")).strip()
            ]
            symptom_positive = [
                str(e.get("span", "")).strip().lower()
                for e in non_negated_entities
                if e.get("label") == "SYMPTOM" and str(e.get("span", "")).strip()
            ]
            disease_mentions = [
                str(e.get("span", "")).strip().lower()
                for e in non_negated_entities
                if e.get("label") == "DISEASE" and str(e.get("span", "")).strip()
            ]
            etiology_terms = [
                str(e.get("span", "")).strip().lower()
                for e in non_negated_entities
                if e.get("label") in ("AETIOLOGY", "ETIOLOGY")
                and str(e.get("span", "")).strip()
            ]
            risk_terms = [
                str(e.get("span", "")).strip().lower()
                for e in non_negated_entities
                if e.get("label") == "RISK" and str(e.get("span", "")).strip()
            ]

            text = re.sub(r"\s+", " ", str(question or "").strip()).lower()
            q_cleaned = text
            for neg in symptom_negative:
                q_cleaned = q_cleaned.replace(neg, "").strip()
            q_cleaned = re.sub(r"\s+", " ", q_cleaned).strip()

            q_symptom_parts = list(symptom_positive) + disease_mentions
            q_symptom_parts.extend(
                str(sym).strip().lower()
                for sym in (self._user_intake_db.symptoms or [])
                if str(sym).strip()
            )
            q_symptom = ", ".join(dict.fromkeys(q_symptom_parts))
            if not q_symptom:
                q_symptom = q_cleaned or text

            return {
                "original": question,
                "processed_text": text,
                "positives": {
                    "SYMPTOM": symptom_positive,
                    "ETIOLOGY": etiology_terms,
                    "RISK": risk_terms,
                },
                "negatives": {"SYMPTOM": symptom_negative},
                "has_negation": bool(symptom_negative),
                "entities_by_type": {},
                "disease_mentions": disease_mentions,
                "patient_state_extract": {},
                "q_cleaned": q_cleaned or text,
                "q_symptom": q_symptom or text,
            }
        except Exception as exc:
            logger.warning("NER fallback failed: %s, using heuristic", exc)
            return self._fallback_query_analysis_heuristic(question)

    def _fallback_query_analysis_heuristic(self, question: str) -> dict[str, Any]:
        """Simple heuristic fallback using Vietnamese negation cues."""
        text = re.sub(r"\s+", " ", str(question or "").strip()).lower()
        neg_candidates: list[str] = []
        neg_cues = ["không", "chưa", "không có", "không bị"]
        for cue in neg_cues:
            if cue in text:
                trailing = text.split(cue, 1)[1].strip()
                if trailing:
                    neg_candidates.append(trailing.split(",")[0].split(".")[0].strip())

        q_cleaned = text
        for neg in neg_candidates:
            q_cleaned = q_cleaned.replace(neg, "").strip()
        q_cleaned = re.sub(r"\s+", " ", q_cleaned).strip()

        q_symptom_parts = list(self._user_intake_db.symptoms or [])
        if text:
            q_symptom_parts.append(text)
        q_symptom = ", ".join(
            dict.fromkeys(
                str(item).strip().lower()
                for item in q_symptom_parts
                if str(item).strip()
            )
        )
        return {
            "original": question,
            "processed_text": text,
            "positives": {"SYMPTOM": [], "ETIOLOGY": [], "RISK": []},
            "negatives": {"SYMPTOM": neg_candidates},
            "has_negation": bool(neg_candidates),
            "entities_by_type": {},
            "disease_mentions": [],
            "patient_state_extract": {},
            "q_cleaned": q_cleaned or text,
            "q_symptom": q_symptom or text,
        }

    def _log_stage1_preprocess(
        self,
        *,
        question: str,
        raw_payload_text: str,
        parsed_payload: dict[str, Any],
        analysis_result: dict[str, Any],
        fallback: bool,
        error: str = "",
    ) -> None:
        """Log Stage 1 preprocessing output from LLM (or fallback)."""
        should_log = self._get_config_bool("RAG_LOG_STAGE1_PREPROCESS", True)
        if not should_log:
            return

        question_preview = re.sub(r"\s+", " ", str(question or "").strip())
        if len(question_preview) > 300:
            question_preview = question_preview[:300] + "...(truncated)"

        raw_preview = re.sub(r"\s+", " ", str(raw_payload_text or "").strip())
        if len(raw_preview) > 1500:
            raw_preview = raw_preview[:1500] + "...(truncated)"

        safe_parsed = {
            "normalized_query": parsed_payload.get("normalized_query", ""),
            "disease_mentions": parsed_payload.get("disease_mentions", []),
            "symptom_positive": parsed_payload.get("symptom_positive", []),
            "symptom_negative": parsed_payload.get("symptom_negative", []),
            "patient_state_extract": parsed_payload.get("patient_state_extract", {}),
            "q_cleaned": parsed_payload.get("q_cleaned", ""),
            "q_symptom": parsed_payload.get("q_symptom", ""),
        }
        safe_analysis = {
            "processed_text": analysis_result.get("processed_text", ""),
            "q_cleaned": analysis_result.get("q_cleaned", ""),
            "q_symptom": analysis_result.get("q_symptom", ""),
            "positives": (analysis_result.get("positives", {}) or {}).get(
                "SYMPTOM", []
            ),
            "negatives": (analysis_result.get("negatives", {}) or {}).get(
                "SYMPTOM", []
            ),
            "disease_mentions": analysis_result.get("disease_mentions", []),
            "patient_state_extract": analysis_result.get("patient_state_extract", {}),
        }

        logger.info(
            "[STAGE_1_PREPROCESS] fallback=%s question=%s error=%s raw_llm=%s parsed=%s analysis=%s",
            fallback,
            question_preview,
            error,
            raw_preview,
            json.dumps(safe_parsed, ensure_ascii=False),
            json.dumps(safe_analysis, ensure_ascii=False),
        )

    def _log_stage2_gate(
        self,
        *,
        question: str,
        q_cleaned: str,
        gate: dict[str, Any],
    ) -> None:
        """Log Stage 2 gate decision for single-disease short-circuit."""
        should_log = self._get_config_bool("RAG_LOG_STAGE2_GATE", True)
        if not should_log:
            return

        question_preview = re.sub(r"\s+", " ", str(question or "").strip())
        if len(question_preview) > 300:
            question_preview = question_preview[:300] + "...(truncated)"

        q_cleaned_preview = re.sub(r"\s+", " ", str(q_cleaned or "").strip())
        if len(q_cleaned_preview) > 300:
            q_cleaned_preview = q_cleaned_preview[:300] + "...(truncated)"

        top_score = 0.0
        try:
            top_score = float(gate.get("top_score", 0.0) or 0.0)
        except Exception:
            top_score = 0.0

        logger.info(
            "[STAGE_2_GATE] question=%s q_cleaned=%s go_single=%s reason=%s top_score=%.4f threshold=%s title=%s",
            question_preview,
            q_cleaned_preview,
            bool(gate.get("go_single", False)),
            gate.get("reason", ""),
            top_score,
            gate.get("threshold", ""),
            gate.get("title", ""),
        )

    def _expand_synonyms(self, items: list[str]) -> list[str]:
        """Expand items with synonyms."""
        out: list[str] = []
        seen: set[str] = set()
        for it in items:
            it_l = str(it).lower()
            if it_l not in seen:
                out.append(it_l)
                seen.add(it_l)
            if it_l in SYNONYM_MAP:
                for syn in SYNONYM_MAP[it_l]:
                    if syn not in seen:
                        out.append(syn)
                        seen.add(syn)
        return out

    def _filter_generic_symptom_terms(
        self,
        items: list[str],
        disease_mentions: set[str] | None = None,
    ) -> list[str]:
        """Remove generic symptom placeholders and keep concrete symptom phrases."""
        filtered: list[str] = []
        seen: set[str] = set()
        disease_set = disease_mentions or set()

        for item in items:
            normalized = re.sub(r"\s+", " ", str(item).strip().lower())
            if not normalized:
                continue
            normalized = re.sub(
                r"^(triệu chứng|trieu chung|dấu hiệu|dau hieu|biểu hiện|bieu hien)\s*[:\-]?\s*",
                "",
                normalized,
            ).strip(" ,.;:-")
            if not normalized:
                continue
            if normalized in GENERIC_SYMPTOM_TERMS:
                continue
            if normalized in disease_set:
                continue
            if normalized not in seen:
                filtered.append(normalized)
                seen.add(normalized)

        return filtered

    def _gate_with_index_c(
        self,
        q_cleaned: str,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Stage 2 title gate using Index C and threshold RAG_THRESH_C."""
        fallback_threshold = (
            threshold if threshold is not None else DEFAULT_RAG_THRESH_C
        )
        score_threshold = self._get_config_float(
            "RAG_THRESH_C",
            fallback_threshold,
        )
        result: dict[str, Any] = {
            "go_single": False,
            "reason": "no_candidates",
            "top_score": 0.0,
            "title": "",
            "threshold": score_threshold,
        }

        try:
            docs = self.vector_manager.search_similar(q_cleaned, k=1, index_type="C")
            if not docs:
                return result

            top_doc = docs[0]
            score = self._score_from_distance(getattr(top_doc, "distance", 1.0))
            metadata = getattr(top_doc, "metadata", {}) or {}
            title = (
                str(
                    metadata.get("canonical_title")
                    or metadata.get("title")
                    or top_doc.title
                    or top_doc.content
                    or ""
                )
                .strip()
                .lower()
            )

            result["top_score"] = score
            result["title"] = title
            if score >= score_threshold and title:
                result["go_single"] = True
                result["reason"] = f"above_{score_threshold}"
            else:
                result["reason"] = f"below_{score_threshold}"
        except Exception as ex:
            logger.warning(f"Gate error: {ex}")
            result["reason"] = "gate_error"

        return result

    def _score_from_distance(self, distance: Any) -> float:
        try:
            return 1.0 - (float(distance) / 2.0)
        except Exception:
            return 0.0

    def _fetch_docs_for_title(
        self, title: str, index: str = "B", k: int = DEFAULT_INDEX_B_K
    ) -> list[Document]:
        """Fetch documents for a disease title."""
        try:
            normalized_title = title.strip().lower()
            docs = self.vector_manager.search_similar(
                title,
                k=k,
                index_type=index,
                metadata_filters={"canonical_title": normalized_title},
            )
            if not docs:
                docs = self.vector_manager.search_similar(title, k=k, index_type=index)

            filtered_docs = [
                d
                for d in docs
                if (d.title or "").lower() == normalized_title
                or str(
                    (getattr(d, "metadata", {}) or {}).get("canonical_title", "")
                ).lower()
                == normalized_title
            ]
            return [
                Document(
                    page_content=d.content,
                    metadata={
                        "title": d.title,
                        "section": d.section_type,
                        "source": d.source,
                        "url": (
                            (getattr(d, "metadata", {}) or {}).get("url")
                            or (getattr(d, "metadata", {}) or {}).get("source_url")
                            or ""
                        ),
                        "score": self._score_from_distance(getattr(d, "distance", 1.0)),
                        "evidence_id": (getattr(d, "metadata", {}) or {}).get(
                            "evidence_id"
                        ),
                        "metadata": getattr(d, "metadata", {}) or {},
                    },
                )
                for d in filtered_docs
            ]
        except Exception as ex:
            logger.warning(f"Fetch docs error: {ex}")
            return []

    def _fetch_summary_docs_for_titles(
        self,
        titles: list[str],
        q_cleaned: str,
    ) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for title in titles:
            normalized_title = str(title).strip().lower()
            if not normalized_title:
                continue
            docs = self.vector_manager.search_similar(
                q_cleaned or normalized_title,
                k=1,
                index_type="A",
                metadata_filters={"canonical_title": normalized_title},
            )
            if not docs:
                continue
            doc = docs[0]
            summaries.append(
                {
                    "title": normalized_title,
                    "summary": str(doc.content or "").strip(),
                    "score": self._score_from_distance(getattr(doc, "distance", 1.0)),
                    "metadata": getattr(doc, "metadata", {}) or {},
                }
            )
        return summaries

    def _build_single_disease_context_with_summary(
        self,
        title: str,
        docs: list[Document],
        summaries: list[dict[str, Any]],
    ) -> str:
        if not summaries:
            return self._build_single_disease_context(title, docs)

        lines = [
            HEADER_SINGLE_DISEASE.format(title=title),
            HEADER_SINGLE_DISEASE_SUBTITLE,
            f"- {HEADER_FAQ_MATCH}:",
        ]
        for summary in summaries[:1]:
            summary_text = re.sub(r"\s+", " ", summary.get("summary", "")).strip()
            lines.append(f"  {summary_text}")
        lines.append("")
        lines.append(f"- {HEADER_EVIDENCE_BLOCK}:")
        for idx, doc in enumerate(docs[:DEFAULT_SECTION_ITEMS_LIMIT], start=1):
            section = str(doc.metadata.get("section", "")).strip()
            snippet = re.sub(r"\s+", " ", doc.page_content).strip()
            lines.append(f"  {idx}. [{section}] {snippet}")
        lines.append("")
        lines.append(MSG_CONTEXT_HINT_SINGLE)
        return "\n".join(lines)

    def _build_single_disease_context(self, title: str, docs: list[Document]) -> str:
        """Build context for single disease mode."""
        if not docs:
            return MSG_NO_DOCS_FOR_TITLE.format(title=title)

        by_section: dict[str, list[str]] = {s: [] for s in SECTION_ORDER}
        for d in docs:
            sec = (d.metadata.get("section") or "").lower()
            txt = (d.page_content or "").strip()
            if sec in by_section and txt:
                by_section[sec].append(txt)

        lines = [
            HEADER_SINGLE_DISEASE.format(title=title),
            HEADER_SINGLE_DISEASE_SUBTITLE,
        ]

        for sec in SECTION_ORDER:
            items = by_section.get(sec, [])
            if not items:
                continue
            lines.append(f"- {SECTION_HEADERS.get(sec, sec.title())}:")
            if sec == "general":
                lines.append(f"  {items[0]}")
            else:
                for it in items[:DEFAULT_SECTION_ITEMS_LIMIT]:
                    brief = re.sub(r"\s+", " ", it).strip()
                    lines.append(f"  * {brief}")
            lines.append("")

        lines.append(MSG_CONTEXT_HINT_SINGLE)
        return "\n".join(lines)

    def _compose_query_with_state(self, q_cleaned: str) -> str:
        intake = self._user_intake_db
        parts = [q_cleaned]
        if intake.age is not None:
            parts.append(f"age {intake.age}")
        if intake.sex and intake.sex != "unknown":
            parts.append(str(intake.sex))
        for symptom in intake.symptoms or []:
            if str(symptom).strip():
                parts.append(str(symptom).strip().lower())
        return ", ".join(dict.fromkeys(part for part in parts if str(part).strip()))

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        a = re.sub(r"\s+", " ", text_a.strip().lower())
        b = re.sub(r"\s+", " ", text_b.strip().lower())
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0

        # Fast lexical overlap fallback.
        tokens_a = set(a.split())
        tokens_b = set(b.split())
        lexical = len(tokens_a.intersection(tokens_b)) / max(
            1, len(tokens_a.union(tokens_b))
        )

        # Optional embedding similarity if available.
        try:
            emb_service = getattr(self.vector_manager, "embedding_service", None)
            if emb_service is None:
                return lexical
            vec_a = emb_service.embed_text(a)
            vec_b = emb_service.embed_text(b)
            dot = sum(float(x) * float(y) for x, y in zip(vec_a, vec_b))
            norm_a = math.sqrt(sum(float(x) ** 2 for x in vec_a))
            norm_b = math.sqrt(sum(float(y) ** 2 for y in vec_b))
            if norm_a <= 0 or norm_b <= 0:
                return lexical
            return max(lexical, dot / (norm_a * norm_b))
        except Exception:
            return lexical

    def _multi_disease_retrieval(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Stage 3-6 retrieval: dual Index-B search -> merge -> negation penalty -> Index-A summary."""
        k = self._get_config_int("RAG_B_TOPK", DEFAULT_RAG_B_TOPK)
        merged_limit = self._get_config_int(
            "RAG_MERGED_LIMIT",
            DEFAULT_RAG_MERGED_LIMIT,
        )
        top_m = self._get_config_int("RAG_TITLE_TOP_M", DEFAULT_RAG_TITLE_TOP_M)
        final_titles_limit = self._get_config_int(
            "RAG_FINAL_TITLES",
            DEFAULT_RAG_FINAL_TITLES,
        )
        merge_w_symptom = self._get_config_float(
            "RAG_MERGE_WEIGHT_ENTITIES",
            DEFAULT_RAG_MERGE_WEIGHT_ENTITIES,
        )
        merge_w_query = self._get_config_float(
            "RAG_MERGE_WEIGHT_QUERY",
            DEFAULT_RAG_MERGE_WEIGHT_QUERY,
        )
        neg_alpha = self._get_config_float(
            "RAG_PENALTY_ALPHA",
            DEFAULT_RAG_PENALTY_ALPHA,
        )
        neg_thresh = self._get_config_float(
            "RAG_NEG_SYM_SIM_THRESH",
            DEFAULT_RAG_NEG_SYM_SIM_THRESH,
        )

        try:
            q_symptom = str(analysis.get("q_symptom", "")).strip()
            q_cleaned = str(analysis.get("q_cleaned", "")).strip()
            composed_state_query = self._compose_query_with_state(q_cleaned)
            docs_2a = self.vector_manager.search_similar(q_symptom, k=k, index_type="B")
            docs_2b = self.vector_manager.search_similar(
                composed_state_query,
                k=k,
                index_type="B",
            )

            title_map: dict[str, dict[str, Any]] = {}
            for source_key, docs in [("score_2a", docs_2a), ("score_2b", docs_2b)]:
                for doc in docs:
                    metadata = getattr(doc, "metadata", {}) or {}
                    title = (
                        str(
                            metadata.get("canonical_title")
                            or metadata.get("title")
                            or doc.title
                            or ""
                        )
                        .strip()
                        .lower()
                    )
                    if not title:
                        continue
                    score = self._score_from_distance(getattr(doc, "distance", 1.0))
                    group = title_map.setdefault(
                        title,
                        {
                            "docs": [],
                            "scores": [],
                            "score_2a": 0.0,
                            "score_2b": 0.0,
                            "source": doc.source,
                        },
                    )
                    group["docs"].append(doc)
                    group["scores"].append(score)
                    group[source_key] = max(float(group[source_key]), score)

            merged_items: list[tuple[str, dict[str, Any]]] = []
            for title, group in title_map.items():
                score_merge = merge_w_symptom * float(
                    group.get("score_2a", 0.0)
                ) + merge_w_query * float(group.get("score_2b", 0.0))
                group["score_merge"] = score_merge
                merged_items.append((title, group))

            merged_items = sorted(
                merged_items,
                key=lambda item: float(item[1]["score_merge"]),
                reverse=True,
            )[:merged_limit]

            neg_symptoms = [
                str(sym).strip().lower()
                for sym in (analysis.get("negatives", {}) or {}).get("SYMPTOM", [])
                if str(sym).strip()
            ]
            candidates: list[dict[str, Any]] = []
            evidence_docs: list[Document] = []

            for title, group in merged_items:
                scores = sorted(
                    [float(score) for score in group.get("scores", [])], reverse=True
                )
                avg_score = sum(scores[:top_m]) / max(1, min(len(scores), top_m))

                symptom_texts: list[str] = []
                raw_docs = cast(list[Any], group.get("docs", []))
                for raw_doc in raw_docs:
                    md = getattr(raw_doc, "metadata", {}) or {}
                    section = str(
                        md.get("section") or raw_doc.section_type or ""
                    ).lower()
                    url = str(md.get("url") or md.get("source_url") or "").strip()
                    if section in {"symptom", "symptoms"}:
                        symptom_texts.append(str(raw_doc.content))
                    evidence_docs.append(
                        Document(
                            page_content=str(raw_doc.content),
                            metadata={
                                "title": title,
                                "section": section,
                                "source": raw_doc.source,
                                "url": url,
                                "score": self._score_from_distance(
                                    getattr(raw_doc, "distance", 1.0)
                                ),
                                "metadata": md,
                            },
                        )
                    )

                neg_matches = 0
                for neg_symptom in neg_symptoms:
                    if any(
                        self._text_similarity(neg_symptom, symptom) >= neg_thresh
                        for symptom in symptom_texts
                    ):
                        neg_matches += 1

                neg_frac = neg_matches / max(1, len(symptom_texts))
                final_score = avg_score * (1 - neg_alpha * neg_frac)
                candidates.append(
                    {
                        "title": title,
                        "avg_score": avg_score,
                        "merge_score": float(group.get("score_merge", 0.0)),
                        "final_score": final_score,
                        "neg_frac": neg_frac,
                        "doc_count": len(raw_docs),
                        "source": group.get("source"),
                    }
                )

            candidates = sorted(
                candidates,
                key=lambda item: float(item["final_score"]),
                reverse=True,
            )[:final_titles_limit]

            top_titles = [str(candidate["title"]) for candidate in candidates]
            summary_docs = self._fetch_summary_docs_for_titles(top_titles, q_cleaned)
            summary_by_title = {
                str(item["title"]): str(item["summary"])
                for item in summary_docs
                if str(item.get("title", "")).strip()
            }
            for candidate in candidates:
                candidate["summary"] = summary_by_title.get(str(candidate["title"]), "")

            return {
                "candidates": candidates,
                "summaries": summary_by_title,
                "evidence_docs": evidence_docs,
            }
        except Exception as ex:
            logger.error(f"Multi-disease retrieval error: {ex}")
            return {"candidates": [], "summaries": {}, "evidence_docs": []}

    def _build_multi_disease_context(
        self,
        question: str,
        analysis: dict[str, Any],
        tier_result: dict[str, Any],
    ) -> str:
        """Build context for multi-disease response with summary + evidence."""
        lines = [HEADER_MULTI_DISEASE_ANALYSIS]

        pos = analysis.get("positives", {})
        neg = analysis.get("negatives", {})
        q_cleaned = str(analysis.get("q_cleaned", "")).strip()
        q_symptom = str(analysis.get("q_symptom", "")).strip()
        if question:
            lines.append(f"- Query gốc: {question.strip()}")
        if q_cleaned:
            lines.append(f"- q_cleaned: {q_cleaned}")
        if q_symptom:
            lines.append(f"- q_symptom: {q_symptom}")

        if pos.get("SYMPTOM"):
            lines.append(f"- Triệu chứng (+): {', '.join(pos['SYMPTOM'])}")
        if neg.get("SYMPTOM"):
            lines.append(f"- Triệu chứng (-): {', '.join(neg['SYMPTOM'])}")
        if pos.get("ETIOLOGY"):
            lines.append(f"- Căn nguyên: {', '.join(pos['ETIOLOGY'])}")

        lines.append(HEADER_MULTI_DISEASE_CANDIDATES)
        for i, c in enumerate(tier_result.get("candidates", [])):
            lines.append(
                f"{i + 1}. {c['title']} - điểm: {c['final_score']:.3f} "
                f"(merge={c.get('merge_score', 0.0):.3f}, neg={c.get('neg_frac', 0.0):.2f})"
            )
            summary = re.sub(r"\s+", " ", str(c.get("summary", "")).strip())
            if summary:
                lines.append(f"   Tóm tắt: {summary[:320]}")

        evidence_docs = cast(list[Document], tier_result.get("evidence_docs", []))
        if evidence_docs:
            lines.append(f"\n- {HEADER_EVIDENCE_BLOCK}:")
            for idx, doc in enumerate(
                evidence_docs[:DEFAULT_SECTION_ITEMS_LIMIT], start=1
            ):
                section = str(doc.metadata.get("section", "")).strip()
                title = str(doc.metadata.get("title", "")).strip()
                snippet = re.sub(r"\s+", " ", str(doc.page_content)).strip()
                lines.append(f"  {idx}. ({title}) [{section}] {snippet[:240]}")

        lines.append(MSG_CONTEXT_HINT_MULTI)
        return "\n".join(lines)

    # ---------------------------
    # Intake Management
    # ---------------------------

    def _apply_analysis_to_intake(
        self,
        question: str,
        analysis: dict[str, Any] | None = None,
    ) -> None:
        """Update intake from analyzed question and persist to database."""
        analysis_result = (
            analysis if isinstance(analysis, dict) else self._analyze_query(question)
        )
        pos = analysis_result.get("positives", {}) or {}
        neg = analysis_result.get("negatives", {}) or {}
        patient_state = analysis_result.get("patient_state_extract", {}) or {}
        disease_mentions = analysis_result.get("disease_mentions", []) or []

        symptoms_updated = False

        # Update positive symptoms
        for sym in pos.get("SYMPTOM", []):
            if sym and sym not in self._user_intake_db.symptoms:
                self._user_intake_db.symptoms.append(sym)
                symptoms_updated = True

        # Update negated symptoms
        for sym in neg.get("SYMPTOM", []):
            if sym and sym not in self._user_intake_db.symptoms_negated:
                self._user_intake_db.symptoms_negated.append(sym)
                symptoms_updated = True

        # Remove negated from positive
        neg_set = {s.lower() for s in self._user_intake_db.symptoms_negated}
        filtered_symptoms = [
            s for s in self._user_intake_db.symptoms if s.lower() not in neg_set
        ]
        if filtered_symptoms != self._user_intake_db.symptoms:
            self._user_intake_db.symptoms = filtered_symptoms
            symptoms_updated = True

        # Update disease mention if explicit and not conflicting.
        if disease_mentions:
            first_disease = str(disease_mentions[0]).strip()
            if first_disease and first_disease != (
                self._user_intake_db.disease_name or ""
            ):
                self._user_intake_db.disease_name = first_disease
                symptoms_updated = True

        # Update age/sex if extracted by analyzer.
        extracted_age = patient_state.get("age")
        if isinstance(extracted_age, int) and extracted_age > 0:
            if self._user_intake_db.age != extracted_age:
                self._user_intake_db.age = extracted_age
                symptoms_updated = True

        extracted_sex = str(patient_state.get("sex", "")).strip().lower()
        if extracted_sex in {"male", "female", "unknown"}:
            if self._user_intake_db.sex != extracted_sex:
                self._user_intake_db.sex = extracted_sex
                symptoms_updated = True

        # Save to database if changed
        if symptoms_updated:
            self._user_intake_db.save()

    def _get_intake_context(self) -> str:
        """Get intake context string for LLM from database."""
        # Refresh to get latest data
        self._user_intake_db.refresh_from_db()
        intake = self._user_intake_db
        info: list[str] = []

        if intake.disease_name:
            info.append(f"Bệnh: {intake.disease_name}")
        if intake.symptoms:
            info.append(f"Triệu chứng (+): {', '.join(intake.symptoms)}")
        if intake.symptoms_negated:
            info.append(f"Triệu chứng (-): {', '.join(intake.symptoms_negated)}")
        if intake.age is not None:
            info.append(f"Tuổi: {intake.age}")
        if intake.sex and intake.sex != "unknown":
            info.append(f"Giới tính: {intake.sex}")

        if info:
            return HEADER_PATIENT_INFO + "\n".join(info)
        return ""

    def update_intake(self, **kwargs: Any) -> UserIntake:
        """Update user intake fields and save to database."""
        # Update scalar fields
        for key in [
            "disease_name",
            "age",
            "sex",
            "onset_days",
            "pregnancy_status",
            "location_country",
        ]:
            if key in kwargs and kwargs[key] is not None:
                setattr(self._user_intake_db, key, kwargs[key])

        # Update list fields (append mode)
        if "symptoms" in kwargs:
            for s in kwargs["symptoms"] or []:
                if s and s not in self._user_intake_db.symptoms:
                    self._user_intake_db.symptoms.append(s)

        if "symptoms_negated" in kwargs:
            for s in kwargs["symptoms_negated"] or []:
                if s and s not in self._user_intake_db.symptoms_negated:
                    self._user_intake_db.symptoms_negated.append(s)

        if "chronic_conditions" in kwargs:
            for c in kwargs["chronic_conditions"] or []:
                if c and c not in self._user_intake_db.chronic_conditions:
                    self._user_intake_db.chronic_conditions.append(c)

        if "allergies" in kwargs:
            for a in kwargs["allergies"] or []:
                if a and a not in self._user_intake_db.allergies:
                    self._user_intake_db.allergies.append(a)

        if "meds" in kwargs:
            for m in kwargs["meds"] or []:
                if m and m not in self._user_intake_db.meds:
                    self._user_intake_db.meds.append(m)

        # Save to database
        self._user_intake_db.save()
        return self._user_intake_db

    def get_intake(self) -> UserIntake:
        """Get current user intake from database."""
        self._user_intake_db.refresh_from_db()
        return self._user_intake_db

    def _extract_doc_source_url(self, metadata: dict[str, Any]) -> str:
        """Extract source URL from normalized or nested metadata."""
        candidates: list[Any] = [
            metadata.get("url"),
            metadata.get("source_url"),
        ]
        nested_metadata = metadata.get("metadata")
        if isinstance(nested_metadata, dict):
            candidates.extend(
                [
                    nested_metadata.get("url"),
                    nested_metadata.get("source_url"),
                ]
            )

        for candidate in candidates:
            url = str(candidate or "").strip()
            if url.startswith(("http://", "https://")):
                return url
        return ""

    def get_last_source_urls(
        self, max_items: int = DEFAULT_DOCS_CACHE_SIZE
    ) -> list[str]:
        """Get unique source URLs from the last retrieved evidence docs."""
        docs = cast(list[Document], getattr(self, "_last_docs_cache", []) or [])
        urls: list[str] = []
        seen: set[str] = set()
        for doc in docs[:max_items]:
            metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
            url = self._extract_doc_source_url(metadata)
            if url and url not in seen:
                urls.append(url)
                seen.add(url)
        return urls

    def get_last_docs(
        self, max_items: int = DEFAULT_DOCS_CACHE_SIZE
    ) -> list[dict[str, Any]]:
        """Get last retrieved documents for UI display."""
        out = []
        for d in self._last_docs_cache[:max_items]:
            metadata = d.metadata if isinstance(d.metadata, dict) else {}
            out.append(
                {
                    "title": metadata.get("title"),
                    "section": metadata.get("section"),
                    "source": metadata.get("source"),
                    "url": self._extract_doc_source_url(metadata) or None,
                    "preview": (
                        d.page_content[:DEFAULT_DOC_PREVIEW_LENGTH] + "..."
                        if len(d.page_content) > DEFAULT_DOC_PREVIEW_LENGTH
                        else d.page_content
                    ),
                }
            )
        return out

    def get_last_audit(self) -> dict[str, Any]:
        """Get last audit info for metadata."""
        return self._last_audit

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
        if intake_payload:
            self._apply_benchmark_intake_payload(intake_payload)

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
            gate = self._gate_with_index_c(q_cleaned)
            timings_ms["gate_latency"] = int(
                (time.perf_counter() - stage_started_at) * 1000
            )

            stage_started_at = time.perf_counter()
            if gate.get("go_single") and gate.get("title"):
                mode = "single-disease"
                title = str(gate.get("title", "")).strip().lower()
                evidence_docs = self._fetch_docs_for_title(
                    title, index="B", k=DEFAULT_SINGLE_DISEASE_DOCS_K
                )
                summaries = self._fetch_summary_docs_for_titles(
                    [title], str(analysis.get("q_cleaned", ""))
                )
                context = self._build_single_disease_context_with_summary(
                    title=title,
                    docs=evidence_docs,
                    summaries=summaries,
                )
                self._last_docs_cache = evidence_docs[:DEFAULT_DOCS_CACHE_SIZE]
                retrieval_output = self._build_benchmark_retrieval_output(
                    mode=mode,
                    docs=evidence_docs,
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
                self._last_audit = {
                    "audit_id": str(uuid.uuid4()),
                    "ts": time.time(),
                    "mode": mode,
                    "title": title,
                    "router_score": gate.get("top_score", 0.0),
                    "doc_count": len(evidence_docs),
                }
            else:
                mode = "multi-disease-v2"
                tier_result = self._multi_disease_retrieval(analysis)
                context = self._build_multi_disease_context(
                    question, analysis, tier_result
                )
                evidence_docs = cast(
                    list[Document], tier_result.get("evidence_docs", [])
                )
                self._last_docs_cache = evidence_docs[:DEFAULT_DOCS_CACHE_SIZE]
                retrieval_output = self._build_benchmark_retrieval_output(
                    mode=mode,
                    docs=evidence_docs,
                    candidates=cast(
                        list[dict[str, Any]], tier_result.get("candidates", [])
                    ),
                    summaries=cast(dict[str, str], tier_result.get("summaries", {})),
                )
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
            prompt_input = {
                "context": context,
                "chat_history": [],
                "question": question,
            }
            prompt_messages = self._create_prompt_template().format_messages(
                **prompt_input
            )
            llm_result = self.llm.invoke(prompt_messages)
            llm_content = (
                llm_result.content if hasattr(llm_result, "content") else llm_result
            )
            if isinstance(llm_content, list):
                llm_content = "\n".join(str(item) for item in llm_content)
            final_answer = str(llm_content or "").strip()
            timings_ms["generation_latency"] = int(
                (time.perf_counter() - stage_started_at) * 1000
            )

            source_urls = self.get_last_source_urls()
            generation_output = {
                "final_answer": final_answer,
                "source_urls": source_urls,
                "context_snapshot": context[:4000],
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

    def _apply_benchmark_intake_payload(self, intake_payload: dict[str, Any]) -> None:
        """Reset and apply explicit intake payload before benchmark execution."""
        self._user_intake_db.reset_session_specific_fields()
        updates = {
            "disease_name": intake_payload.get("disease_name"),
            "age": intake_payload.get("age"),
            "sex": intake_payload.get("sex"),
            "symptoms": intake_payload.get("symptoms", []),
            "symptoms_negated": intake_payload.get("symptoms_negated", []),
            "onset_days": intake_payload.get("onset_days"),
            "pregnancy_status": intake_payload.get("pregnancy_status"),
            "location_country": intake_payload.get("location_country"),
            "chronic_conditions": intake_payload.get("chronic_conditions", []),
            "allergies": intake_payload.get("allergies", []),
            "meds": intake_payload.get("meds", []),
        }
        safe_updates = {
            key: value for key, value in updates.items() if value is not None
        }
        if safe_updates:
            self.update_intake(**safe_updates)

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
