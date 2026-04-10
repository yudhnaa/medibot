"""
Chatbot Service
Main service for medical RAG chatbot with NER, negation detection, and streaming support.
"""

import logging
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
    DEFAULT_RAG_A_ANSWER_MIN,
    DEFAULT_RAG_A_EVIDENCE_MIN,
    DEFAULT_RAG_A_TOPK,
    DEFAULT_RAG_B_TOPK_ENRICH,
    DEFAULT_RAG_B_TOPK_FALLBACK,
    DEFAULT_RAG_C_HIGH,
    DEFAULT_RAG_C_LOW,
    DEFAULT_RAG_C_MARGIN,
    DEFAULT_RAG_C_TOPK,
    DEFAULT_DOCS_CACHE_SIZE,
    DEFAULT_DOC_PREVIEW_LENGTH,
    DEFAULT_INDEX_B_K,
    DEFAULT_RAG_B_TOPK,
    DEFAULT_RAG_FINAL_TITLES,
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
    LABEL_ALIASES,
    MSG_ANALYSIS_ERROR,
    MSG_CONTEXT_HINT_MULTI,
    MSG_CONTEXT_HINT_SINGLE,
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
from nlp.services.runtime import get_shared_integrator
from vector_store.services import VectorStoreManager

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
        self.integrator = get_shared_integrator()
        logger.info(f"ChatbotService initialized for session: {self.session_id}")

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
            analysis = await sync_to_async(self._analyze_query)(question)
            intent_info = self._detect_query_intent(question, analysis)
            analysis["intent"] = intent_info.get("primary")
            analysis["intent_confidence"] = intent_info.get("confidence")
            analysis["intent_secondary"] = intent_info.get("secondary", [])

            gate = await sync_to_async(self._gate_with_index_c)(
                question,
                cast(str, analysis.get("intent", "")),
            )

            if gate.get("go_single") and gate.get("title"):
                title = gate.get("title", "")
                faq_hit = await sync_to_async(self._faq_retrieval_index_a)(
                    question, gate, analysis
                )
                evidence_docs = await sync_to_async(self._evidence_retrieval_index_b)(
                    question,
                    gate,
                    analysis,
                    faq_hit=faq_hit,
                )
                if faq_hit.get("accepted"):
                    context = self._build_faq_context(title, faq_hit, evidence_docs)
                else:
                    docs = evidence_docs or await sync_to_async(self._fetch_docs_for_title)(
                        title, index="B", k=DEFAULT_SINGLE_DISEASE_DOCS_K
                    )
                    context = self._build_single_disease_context(title, docs)
                    evidence_docs = docs

                self._last_docs_cache = evidence_docs[:DEFAULT_DOCS_CACHE_SIZE]
                self._last_audit = {
                    "audit_id": audit_id,
                    "ts": time.time(),
                    "mode": "single-disease-cab",
                    "title": title,
                    "intent": gate.get("intent", ""),
                    "router_score": gate.get("top_score", 0.0),
                    "route_margin": gate.get("route_margin", 0.0),
                    "faq_score": faq_hit.get("score", 0.0),
                    "doc_count": len(evidence_docs),
                }
            else:
                tier_result = await sync_to_async(self._multi_disease_retrieval)(analysis)
                context = self._build_multi_disease_context(question, analysis, tier_result)
                self._last_audit = {
                    "audit_id": audit_id,
                    "ts": time.time(),
                    "mode": "multi-disease",
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

            # Update intake from question
            self._apply_analysis_to_intake(question)

            # Get chat history
            chat_history = self._get_chat_history()

            # Run chain
            response = self._chain.invoke(
                {
                    "question": question,
                    "chat_history": chat_history,
                }
            )

            response_time_ms = int((time.time() - start_time) * 1000)

            # Save assistant message
            self._save_message(
                MessageRole.ASSISTANT,
                response,
                response_time_ms=response_time_ms,
                metadata={
                    "audit_id": self._last_audit.get("audit_id"),
                    "mode": self._last_audit.get("mode"),
                },
            )

            logger.info(f"Query processed in {response_time_ms}ms")
            return response

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

            # Update intake
            await sync_to_async(self._apply_analysis_to_intake)(question)

            # Get chat history
            chat_history = await sync_to_async(self._get_chat_history)()

            # Run chain
            response = await self._chain.ainvoke(
                {
                    "question": question,
                    "chat_history": chat_history,
                    "xray_analysis_id": xray_analysis_id,
                    "xray_context": xray_payload["context"],
                }
            )

            response_time_ms = int((time.time() - start_time) * 1000)
            assistant_metadata = {
                "audit_id": self._last_audit.get("audit_id"),
                "mode": self._last_audit.get("mode"),
            }
            if xray_payload["serialized"] is not None:
                assistant_metadata["xray_analysis"] = xray_payload["serialized"]

            # Save assistant message
            await sync_to_async(self._save_message)(
                MessageRole.ASSISTANT,
                response,
                response_time_ms=response_time_ms,
                metadata=assistant_metadata,
            )

            logger.info(f"Query processed in {response_time_ms}ms")
            return response

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

        # Update intake
        self._apply_analysis_to_intake(question)

        # Get history
        chat_history = self._get_chat_history()

        start_time = time.time()
        full_response = ""

        try:
            for chunk in self._streaming_chain.stream(
                {
                    "question": question,
                    "chat_history": chat_history,
                }
            ):
                # StrOutputParser always returns str
                full_response += chunk
                yield chunk

            # Save complete response
            response_time_ms = int((time.time() - start_time) * 1000)
            self._save_message(
                MessageRole.ASSISTANT,
                full_response,
                response_time_ms=response_time_ms,
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

        # Update intake (DB/Analysis -> async)
        await sync_to_async(self._apply_analysis_to_intake)(question)

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
                }
            ):
                # StrOutputParser always returns str
                full_response += chunk
                yield chunk

            # Prepare metadata for assistant message
            assistant_metadata = {
                "audit_id": self._last_audit.get("audit_id"),
                "mode": self._last_audit.get("mode"),
            }
            if xray_payload["serialized"] is not None:
                assistant_metadata["xray_analysis"] = xray_payload["serialized"]

            # Save complete response (DB op -> async)
            response_time_ms = int((time.time() - start_time) * 1000)
            await sync_to_async(self._save_message)(
                MessageRole.ASSISTANT,
                full_response,
                response_time_ms=response_time_ms,
                metadata=assistant_metadata,
            )

        except Exception as e:
            logger.error(f"Async streaming error: {e}")
            error_msg = MSG_STREAMING_ERROR.format(error=str(e))
            await sync_to_async(self._save_message)(MessageRole.ASSISTANT, error_msg)
            yield error_msg

    def _serialize_xray_analysis(
        self, analysis_record: XRayAnalysis
    ) -> dict[str, Any]:
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

    def _get_xray_analysis_payload(self, xray_analysis_id: int | None) -> dict[str, Any]:
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
            logger.warning("Failed to load XRayAnalysis id=%s: %s", xray_analysis_id, exc)
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
        """Analyze query with NER and negation detection."""
        result = self.integrator.process_text(question, include_debug=False)

        def by_labels(labels: set[str], include_negated: bool = True) -> list[str]:
            items: list[str] = []
            for ent in result.get("entities", {}).get("all", []):
                if str(ent.get("label", "")).upper() in labels:
                    if include_negated or ent.get("is_negated") != "negated":
                        txt = ent.get("text", "")
                        if txt:
                            items.append(txt)
            return items

        positives = {
            key: by_labels(aliases, include_negated=False)
            for key, aliases in LABEL_ALIASES.items()
        }
        negatives = {
            "SYMPTOM": [
                e.get("text", "")
                for e in result.get("entities", {}).get("negated", [])
                if str(e.get("label", "")).upper() in LABEL_ALIASES["SYMPTOM"]
            ],
            "ETIOLOGY": [
                e.get("text", "")
                for e in result.get("entities", {}).get("negated", [])
                if str(e.get("label", "")).upper() in LABEL_ALIASES["ETIOLOGY"]
            ],
        }

        # Expand synonyms
        positives = {k: self._expand_synonyms(v) for k, v in positives.items()}
        negatives = {k: self._expand_synonyms(v) for k, v in negatives.items()}

        return {
            "original": question,
            "processed_text": result.get("processed_text", question),
            "positives": positives,
            "negatives": negatives,
            "has_negation": result.get("negation_info", {}).get("has_negation", False),
        }

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

    def _detect_query_intent(
        self, question: str, analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Infer primary/secondary intent from question and extracted entities."""
        text = str(question or "").lower()
        positives = analysis.get("positives", {}) or {}

        intent_rules: dict[str, list[str]] = {
            "symptom": ["triệu chứng", "dấu hiệu", "symptom", "biểu hiện"],
            "cause_transmission": ["nguyên nhân", "lây", "transmission", "etiolog", "cause"],
            "risk_factor": ["nguy cơ", "risk", "rủi ro", "risk factor"],
            "diagnosis_treatment": [
                "chẩn đoán",
                "điều trị",
                "thuốc",
                "xét nghiệm",
                "diagnos",
                "treatment",
                "therapy",
            ],
            "prevention_lifestyle": [
                "phòng ngừa",
                "phòng bệnh",
                "vaccine",
                "prevent",
                "lifestyle",
            ],
            "prognosis_complication": ["biến chứng", "tiên lượng", "prognosis", "complication"],
            "epidemiology": ["dịch tễ", "prevalence", "incidence", "outbreak", "pandemic"],
            "definition_overview": ["là gì", "what is", "định nghĩa", "overview"],
        }

        score_by_intent: dict[str, float] = {intent: 0.0 for intent in intent_rules}
        for intent, keywords in intent_rules.items():
            if any(keyword in text for keyword in keywords):
                score_by_intent[intent] += 1.0

        if positives.get("SYMPTOM"):
            score_by_intent["symptom"] += 0.5
        if positives.get("ETIOLOGY"):
            score_by_intent["cause_transmission"] += 0.5
        if positives.get("RISK"):
            score_by_intent["risk_factor"] += 0.5

        ranked = sorted(score_by_intent.items(), key=lambda kv: kv[1], reverse=True)
        primary, primary_score = ranked[0]
        secondary = [intent for intent, score in ranked[1:] if score > 0.0][:2]
        confidence = 0.9 if primary_score >= 1.5 else (0.7 if primary_score >= 1.0 else 0.55)
        if primary_score <= 0.0:
            primary = "other_medical"
            confidence = 0.5
            secondary = []

        return {
            "primary": primary,
            "secondary": secondary,
            "confidence": confidence,
        }

    def _gate_with_index_c(
        self,
        question: str,
        query_intent: str | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Index-C router/gate with top-k + margin for disease+intent."""
        c_topk = int(ChatbotConfig.get_config("RAG_C_TOPK", DEFAULT_RAG_C_TOPK))
        c_high = float(
            ChatbotConfig.get_config(
                "RAG_C_HIGH",
                threshold if threshold is not None else DEFAULT_RAG_C_HIGH,
            )
        )
        c_low = float(ChatbotConfig.get_config("RAG_C_LOW", DEFAULT_RAG_C_LOW))
        c_margin = float(ChatbotConfig.get_config("RAG_C_MARGIN", DEFAULT_RAG_C_MARGIN))
        query_intent = (query_intent or "").strip().lower()

        result: dict[str, Any] = {
            "go_single": False,
            "reason": "no_index_c_candidates",
            "top_score": 0.0,
            "title": "",
            "intent": query_intent or "other_medical",
            "route_margin": 0.0,
            "gate_level": "low",
        }

        try:
            query_text = question.strip()
            if query_intent:
                query_text = f"{question.strip()} | {query_intent}"

            docs = self.vector_manager.search_similar(
                query_text, k=max(2, c_topk), index_type="C"
            )
            if not docs:
                result["reason"] = "no_candidates"
                return result

            scored: list[tuple[Any, float, str, str]] = []
            for doc in docs:
                distance = float(getattr(doc, "distance", 1.0))
                score = 1.0 - (distance / 2.0)
                metadata = getattr(doc, "metadata", {}) or {}
                title = str(
                    metadata.get("canonical_title")
                    or doc.title
                    or doc.content
                    or ""
                ).strip().lower()
                intent = str(metadata.get("primary_intent") or "").strip().lower()
                scored.append((doc, score, title, intent))

            if query_intent:
                scored = sorted(
                    scored,
                    key=lambda item: (
                        1 if item[3] == query_intent else 0,
                        item[1],
                    ),
                    reverse=True,
                )
            else:
                scored = sorted(scored, key=lambda item: item[1], reverse=True)

            top_doc, top_score, top_title, top_intent = scored[0]
            second_score = scored[1][1] if len(scored) > 1 else 0.0
            route_margin = top_score - second_score

            result["top_score"] = float(top_score)
            result["title"] = top_title
            result["intent"] = top_intent or query_intent or "other_medical"
            result["route_margin"] = float(route_margin)

            if float(top_score) >= c_high and route_margin >= c_margin:
                result["go_single"] = True
                result["reason"] = "high_confidence_route"
                result["gate_level"] = "high"
            elif float(top_score) >= c_low:
                result["go_single"] = True
                result["reason"] = "medium_confidence_route"
                result["gate_level"] = "medium"
            else:
                result["reason"] = "low_confidence_route"
                result["gate_level"] = "low"

        except Exception as ex:
            logger.warning(f"Gate error: {ex}")
            result["reason"] = "gate_error"

        return result

    def _faq_retrieval_index_a(
        self,
        question: str,
        gate: dict[str, Any],
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """Retrieve best FAQ hit from Index A using disease+intent-aware filters."""
        title = str(gate.get("title", "")).strip().lower()
        intent = str(gate.get("intent", "")).strip().lower()
        if not title:
            return {"accepted": False, "score": 0.0}

        a_topk = int(ChatbotConfig.get_config("RAG_A_TOPK", DEFAULT_RAG_A_TOPK))
        answer_min = float(
            ChatbotConfig.get_config("RAG_A_ANSWER_MIN", DEFAULT_RAG_A_ANSWER_MIN)
        )
        evidence_min = float(
            ChatbotConfig.get_config("RAG_A_EVIDENCE_MIN", DEFAULT_RAG_A_EVIDENCE_MIN)
        )

        metadata_filters: dict[str, Any] = {"canonical_title": title}
        if intent:
            metadata_filters["primary_intent"] = [intent, "other_medical"]

        docs = self.vector_manager.search_similar(
            question,
            k=a_topk,
            index_type="A",
            metadata_filters=metadata_filters,
        )
        if not docs:
            return {
                "accepted": False,
                "score": 0.0,
                "reason": "no_faq_candidates",
            }

        best: dict[str, Any] = {"accepted": False, "score": 0.0}
        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            distance = float(getattr(doc, "distance", 1.0))
            score = 1.0 - (distance / 2.0)
            if intent and str(metadata.get("primary_intent", "")).lower() == intent:
                score += 0.03

            answer_text = str(metadata.get("answer_text", "")).strip()
            if len(answer_text) < 20:
                score -= 0.02

            if score > best["score"]:
                best = {
                    "accepted": score >= evidence_min and bool(answer_text),
                    "score": float(score),
                    "direct_answer": score >= answer_min and bool(answer_text),
                    "need_evidence": score < answer_min,
                    "question": doc.content,
                    "answer": answer_text,
                    "qa_id": metadata.get("qa_id"),
                    "context_id": metadata.get("context_id"),
                    "article_id": metadata.get("article_id"),
                    "intent": metadata.get("primary_intent", intent),
                    "title": title,
                }

        return best

    def _evidence_retrieval_index_b(
        self,
        question: str,
        gate: dict[str, Any],
        analysis: dict[str, Any],
        faq_hit: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Retrieve evidence chunks from Index B for fallback or enrichment."""
        title = str(gate.get("title", "")).strip().lower()
        intent = str(gate.get("intent", "")).strip().lower()
        positives = analysis.get("positives", {}) or {}

        if faq_hit and faq_hit.get("accepted"):
            top_k = int(ChatbotConfig.get_config("RAG_B_TOPK_ENRICH", DEFAULT_RAG_B_TOPK_ENRICH))
        else:
            top_k = int(
                ChatbotConfig.get_config("RAG_B_TOPK_FALLBACK", DEFAULT_RAG_B_TOPK_FALLBACK)
            )

        query_terms: list[str] = [question.strip()]
        if title:
            query_terms.append(title)
        if intent:
            query_terms.append(intent)
        for label in ["SYMPTOM", "ETIOLOGY", "RISK"]:
            query_terms.extend(positives.get(label, []) or [])
        query_text = ", ".join(term for term in query_terms if str(term).strip())

        metadata_filters: dict[str, Any] = {}
        if title:
            metadata_filters["canonical_title"] = title
        if intent:
            metadata_filters["intent_hints"] = [intent, "other_medical"]

        docs = self.vector_manager.search_similar(
            query_text,
            k=top_k,
            index_type="B",
            metadata_filters=metadata_filters or None,
        )

        out_docs: list[Document] = []
        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            out_docs.append(
                Document(
                    page_content=doc.content,
                    metadata={
                        "title": doc.title,
                        "section": doc.section_type,
                        "source": doc.source,
                        "score": 1.0 - (float(getattr(doc, "distance", 1.0)) / 2.0),
                        "context_id": metadata.get("context_id"),
                        "article_id": metadata.get("article_id"),
                        "evidence_id": metadata.get("evidence_id"),
                    },
                )
            )
        return out_docs

    def _build_faq_context(
        self,
        title: str,
        faq_hit: dict[str, Any],
        evidence_docs: list[Document],
    ) -> str:
        """Build context that combines FAQ payload and optional evidence snippets."""
        lines = [
            HEADER_SINGLE_DISEASE.format(title=title),
            HEADER_SINGLE_DISEASE_SUBTITLE,
            f"- {HEADER_FAQ_MATCH}:",
            f"  Câu hỏi: {faq_hit.get('question', '')}",
            f"  Trả lời: {faq_hit.get('answer', '')}",
            "",
        ]

        if evidence_docs:
            lines.append(f"- {HEADER_EVIDENCE_BLOCK}:")
            for idx, doc in enumerate(evidence_docs[:DEFAULT_SECTION_ITEMS_LIMIT], start=1):
                section = str(doc.metadata.get("section", "")).strip()
                snippet = re.sub(r"\s+", " ", doc.page_content).strip()
                lines.append(f"  {idx}. [{section}] {snippet}")
            lines.append("")

        lines.append(MSG_CONTEXT_HINT_SINGLE)
        return "\n".join(lines)

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
                or str((getattr(d, "metadata", {}) or {}).get("canonical_title", "")).lower()
                == normalized_title
            ]
            return [
                Document(
                    page_content=d.content,
                    metadata={
                        "title": d.title,
                        "section": d.section_type,
                        "source": d.source,
                        "score": 1.0 - (float(getattr(d, "distance", 1.0)) / 2.0),
                        "evidence_id": (getattr(d, "metadata", {}) or {}).get(
                            "evidence_id"
                        ),
                    },
                )
                for d in filtered_docs
            ]
        except Exception as ex:
            logger.warning(f"Fetch docs error: {ex}")
            return []

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

    def _multi_disease_retrieval(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Multi-disease retrieval with dual search and negative penalty."""
        k = cast(int, ChatbotConfig.get_config("RAG_B_TOPK", DEFAULT_RAG_B_TOPK))

        try:
            pos = analysis.get("positives", {}) or {}

            # Build query terms
            positive_terms: list[str] = []
            for label in ["SYMPTOM", "ETIOLOGY", "RISK"]:
                positive_terms.extend(pos.get(label, []) or [])

            # Dedupe
            seen: set[str] = set()
            positive_terms = [
                t for t in positive_terms if not (t in seen or seen.add(t))
            ]  # type: ignore[func-returns-value]

            query_text = (
                ", ".join(positive_terms)
                if positive_terms
                else analysis.get("processed_text", "")
            )

            # Search Index B
            docs = self.vector_manager.search_similar(query_text, k=k, index_type="B")

            # Group by title
            title_groups: dict[str, dict[str, Any]] = {}
            for doc in docs:
                title = (doc.title or "").strip().lower()
                if not title:
                    continue
                grp = title_groups.setdefault(
                    title,
                    {
                        "docs": [],
                        "scores": [],
                        "source": doc.source,
                    },
                )
                grp["docs"].append(doc)
                distance = getattr(doc, "distance", 1.0)
                score = 1.0 - (distance / 2.0)
                grp["scores"].append(float(score))

            # Calculate average scores
            top_m = cast(
                int,
                ChatbotConfig.get_config("RAG_TITLE_TOP_M", DEFAULT_RAG_TITLE_TOP_M),
            )
            for grp in title_groups.values():
                scs = sorted(grp["scores"], reverse=True)
                grp["avg_score"] = sum(scs[:top_m]) / max(1, min(len(scs), top_m))
                grp["final_score"] = grp["avg_score"]

            # Sort and return top candidates
            max_titles = cast(
                int,
                ChatbotConfig.get_config("RAG_FINAL_TITLES", DEFAULT_RAG_FINAL_TITLES),
            )
            ranked = sorted(
                title_groups.items(),
                key=lambda x: x[1]["final_score"],
                reverse=True,
            )[:max_titles]

            candidates = [
                {
                    "title": t,
                    "avg_score": grp["avg_score"],
                    "final_score": grp["final_score"],
                    "doc_count": len(grp["docs"]),
                    "source": grp["source"],
                }
                for t, grp in ranked
            ]

            return {"candidates": candidates}

        except Exception as ex:
            logger.error(f"Multi-disease retrieval error: {ex}")
            return {"candidates": []}

    def _build_multi_disease_context(
        self,
        _question: str,
        analysis: dict[str, Any],
        tier_result: dict[str, Any],
    ) -> str:
        """Build context for multi-disease response."""
        lines = [HEADER_MULTI_DISEASE_ANALYSIS]

        pos = analysis.get("positives", {})
        neg = analysis.get("negatives", {})

        if pos.get("SYMPTOM"):
            lines.append(f"- Triệu chứng (+): {', '.join(pos['SYMPTOM'])}")
        if neg.get("SYMPTOM"):
            lines.append(f"- Triệu chứng (-): {', '.join(neg['SYMPTOM'])}")
        if pos.get("ETIOLOGY"):
            lines.append(f"- Căn nguyên: {', '.join(pos['ETIOLOGY'])}")

        lines.append(HEADER_MULTI_DISEASE_CANDIDATES)
        for i, c in enumerate(tier_result.get("candidates", [])):
            lines.append(f"{i + 1}. {c['title']} - điểm: {c['final_score']:.3f}")

        lines.append(MSG_CONTEXT_HINT_MULTI)
        return "\n".join(lines)

    # ---------------------------
    # Intake Management
    # ---------------------------

    def _apply_analysis_to_intake(self, question: str) -> None:
        """Update intake from analyzed question and persist to database."""
        analysis = self._analyze_query(question)
        pos = analysis.get("positives", {}) or {}
        neg = analysis.get("negatives", {}) or {}

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

    def get_last_docs(
        self, max_items: int = DEFAULT_DOCS_CACHE_SIZE
    ) -> list[dict[str, Any]]:
        """Get last retrieved documents for UI display."""
        out = []
        for d in self._last_docs_cache[:max_items]:
            out.append(
                {
                    "title": d.metadata.get("title"),
                    "section": d.metadata.get("section"),
                    "source": d.metadata.get("source"),
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
