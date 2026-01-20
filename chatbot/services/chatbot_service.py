"""
Chatbot Service
Main service for medical RAG chatbot with NER, negation detection, and streaming support.
"""

import logging
import re
import time
import uuid
from collections.abc import Generator
from typing import Any, cast

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from chatbot.models import ChatbotConfig, ChatMessage, ChatSession, MessageRole
from chatbot.prompts.system_vi import SYSTEM_PROMPT_VI
from chatbot.schema.user_intake_message import UserIntakeMessage
from chatbot.services.constants import (
    DEFAULT_RAG_B_TOPK,
    DEFAULT_RAG_FINAL_TITLES,
    DEFAULT_RAG_THRESH_C,
    DEFAULT_RAG_TITLE_TOP_M,
    LABEL_ALIASES,
    SECTION_HEADERS,
    SECTION_ORDER,
    SYNONYM_MAP,
)
from chatbot.services.gemini_manager import get_gemini_manager
from nlp.services import NERNegationIntegrator
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

        # User intake tracking
        self._user_intake = UserIntakeMessage(content="")

        # Document cache for UI
        self._last_docs_cache: list[Document] = []
        self._last_query_text = ""
        self._last_audit: dict[str, Any] = {}

        # Initialize managers
        self._init_managers()
        self._setup_chain()

    def _init_managers(self) -> None:
        """Initialize API and vector managers."""
        self.gemini_manager = get_gemini_manager()
        self.vector_manager = VectorStoreManager()
        self.llm = self.gemini_manager.create_llm()
        self.integrator = NERNegationIntegrator(enable_text_normalization=True)
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

        def get_context(inputs: dict[str, Any]) -> str:
            question = inputs.get("question") or ""
            audit_id = str(uuid.uuid4())

            try:
                # Step 1: Analyze query with NER + negation
                analysis = self._analyze_query(question)

                # Step 2: Gate for single disease
                gate = self._gate_with_index_c(question)

                if gate.get("go_single") and gate.get("title"):
                    title = gate.get("title", "")
                    docs = self._fetch_docs_for_title(title, index="B", k=200)
                    context = self._build_single_disease_context(title, docs)

                    # Cache documents
                    self._last_docs_cache = docs[:10]
                    self._last_audit = {
                        "audit_id": audit_id,
                        "ts": time.time(),
                        "mode": "single-disease",
                        "title": title,
                        "doc_count": len(docs),
                    }
                else:
                    # Multi-disease retrieval
                    tier_result = self._multi_disease_retrieval(analysis)
                    context = self._build_multi_disease_context(
                        question, analysis, tier_result
                    )
                    self._last_audit = {
                        "audit_id": audit_id,
                        "ts": time.time(),
                        "mode": "multi-disease",
                        "candidates": [
                            c.get("title")
                            for c in tier_result.get("candidates", [])[:5]
                        ],
                    }

                self._last_query_text = question

            except Exception as ex:
                logger.warning(f"Context build error: {ex}")
                context = "Không thể phân tích đầy đủ truy vấn. Vui lòng cung cấp thêm thông tin."
                self._last_query_text = question

            # Append intake context
            intake_ctx = self._get_intake_context()
            return context + ("\n" + intake_ctx if intake_ctx else "")

        return RunnableLambda(get_context)

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
            error_msg = f"Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn: {e!s}"
            self._save_message(MessageRole.ASSISTANT, error_msg)
            return error_msg

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
            error_msg = f"Lỗi: {e!s}"
            self._save_message(MessageRole.ASSISTANT, error_msg)
            yield error_msg

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

    def _gate_with_index_c(
        self, question: str, threshold: float | None = None
    ) -> dict[str, Any]:
        """Index-C gate to decide single vs multi-disease mode."""
        config_thresh_raw = ChatbotConfig.get_config(
            "RAG_THRESH_C", DEFAULT_RAG_THRESH_C
        )
        config_thresh = (
            float(cast(float, config_thresh_raw))
            if config_thresh_raw is not None
            else DEFAULT_RAG_THRESH_C
        )
        thresh = threshold if threshold is not None else config_thresh

        result: dict[str, Any] = {
            "go_single": False,
            "reason": "no_index_c",
            "top_score": 0.0,
            "title": "",
        }

        try:
            docs = self.vector_manager.search_similar(question, k=1, index_type="C")
            if not docs:
                result["reason"] = "no_candidates"
                return result

            doc = docs[0]
            score = getattr(doc, "score", 0.0) or 0.0
            title = (doc.title or doc.content or "").strip().lower()

            result["top_score"] = float(score)
            result["title"] = title

            if float(score) >= thresh:
                result["go_single"] = True
                result["reason"] = f"threshold_{thresh}"
                logger.info(
                    f"Gate PASSED: score={score:.3f} >= {thresh}, title='{title}'"
                )
            else:
                result["reason"] = f"below_{thresh}"
                logger.info(f"Gate FAILED: score={score:.3f} < {thresh}")

        except Exception as ex:
            logger.warning(f"Gate error: {ex}")
            result["reason"] = "gate_error"

        return result

    def _fetch_docs_for_title(
        self, title: str, index: str = "B", k: int = 100
    ) -> list[Document]:
        """Fetch documents for a disease title."""
        try:
            docs = self.vector_manager.search_similar(title, k=k, index_type=index)
            # Filter by title
            filtered_docs = [
                d for d in docs if (d.title or "").lower() == title.strip().lower()
            ]
            return [
                Document(
                    page_content=d.content,
                    metadata={
                        "title": d.title,
                        "section": d.section_type,
                        "source": d.source,
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
            return f"Không tìm thấy tài liệu cho bệnh: {title}"

        by_section: dict[str, list[str]] = {s: [] for s in SECTION_ORDER}
        for d in docs:
            sec = (d.metadata.get("section") or "").lower()
            txt = (d.page_content or "").strip()
            if sec in by_section and txt:
                by_section[sec].append(txt)

        lines = [
            f"THÔNG TIN CHI TIẾT VỀ BỆNH: {title}",
            "(Tổng hợp từ cơ sở dữ liệu y khoa)\n",
        ]

        for sec in SECTION_ORDER:
            items = by_section.get(sec, [])
            if not items:
                continue
            lines.append(f"- {SECTION_HEADERS.get(sec, sec.title())}:")
            if sec == "general":
                lines.append(f"  {items[0]}")
            else:
                for it in items[:10]:
                    brief = re.sub(r"\s+", " ", it).strip()
                    lines.append(f"  * {brief}")
            lines.append("")

        lines.append(
            "\nGỢI Ý: Tóm tắt triệu chứng, nguyên nhân, điều trị. Không suy diễn ngoài tài liệu."
        )
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
                score = getattr(doc, "score", 0.5) or 0.5
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
        lines = ["PHÂN TÍCH TRUY VẤN (NER & phủ định):"]

        pos = analysis.get("positives", {})
        neg = analysis.get("negatives", {})

        if pos.get("SYMPTOM"):
            lines.append(f"- Triệu chứng (+): {', '.join(pos['SYMPTOM'])}")
        if neg.get("SYMPTOM"):
            lines.append(f"- Triệu chứng (-): {', '.join(neg['SYMPTOM'])}")
        if pos.get("ETIOLOGY"):
            lines.append(f"- Căn nguyên: {', '.join(pos['ETIOLOGY'])}")

        lines.append("\nCÁC BỆNH CÓ KHẢ NĂNG:")
        for i, c in enumerate(tier_result.get("candidates", [])):
            lines.append(f"{i + 1}. {c['title']} - điểm: {c['final_score']:.3f}")

        lines.append(
            "\nGỢI Ý: Trình bày danh sách bệnh có tổ chức. Khuyến nghị cung cấp thêm thông tin."
        )
        return "\n".join(lines)

    # ---------------------------
    # Intake Management
    # ---------------------------

    def _apply_analysis_to_intake(self, question: str) -> None:
        """Update intake from analyzed question."""
        analysis = self._analyze_query(question)
        pos = analysis.get("positives", {}) or {}
        neg = analysis.get("negatives", {}) or {}

        # Update symptoms
        for sym in pos.get("SYMPTOM", []):
            if sym and sym not in self._user_intake.symptoms:
                self._user_intake.symptoms.append(sym)

        for sym in neg.get("SYMPTOM", []):
            if sym and sym not in self._user_intake.symptoms_negated:
                self._user_intake.symptoms_negated.append(sym)

        # Remove negated from positive
        neg_set = {s.lower() for s in self._user_intake.symptoms_negated}
        self._user_intake.symptoms = [
            s for s in self._user_intake.symptoms if s.lower() not in neg_set
        ]

    def _get_intake_context(self) -> str:
        """Get intake context string for LLM."""
        intake = self._user_intake
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
            return "\nTHÔNG TIN BỆNH NHÂN:\n" + "\n".join(info)
        return ""

    def update_intake(self, **kwargs: Any) -> UserIntakeMessage:
        """Update user intake fields."""
        for key in ["disease_name", "age", "sex", "onset_days", "pregnancy_status"]:
            if key in kwargs and kwargs[key] is not None:
                setattr(self._user_intake, key, kwargs[key])

        if "symptoms" in kwargs:
            for s in kwargs["symptoms"] or []:
                if s and s not in self._user_intake.symptoms:
                    self._user_intake.symptoms.append(s)

        if "symptoms_negated" in kwargs:
            for s in kwargs["symptoms_negated"] or []:
                if s and s not in self._user_intake.symptoms_negated:
                    self._user_intake.symptoms_negated.append(s)

        return self._user_intake

    def get_intake(self) -> UserIntakeMessage:
        """Get current user intake."""
        return self._user_intake

    def get_last_docs(self, max_items: int = 10) -> list[dict[str, Any]]:
        """Get last retrieved documents for UI display."""
        out = []
        for d in self._last_docs_cache[:max_items]:
            out.append(
                {
                    "title": d.metadata.get("title"),
                    "section": d.metadata.get("section"),
                    "source": d.metadata.get("source"),
                    "preview": (
                        d.page_content[:300] + "..."
                        if len(d.page_content) > 300
                        else d.page_content
                    ),
                }
            )
        return out

    def get_last_audit(self) -> dict[str, Any]:
        """Get last audit info for metadata."""
        return self._last_audit
