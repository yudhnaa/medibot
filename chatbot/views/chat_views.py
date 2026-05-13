"""
Chat Views
DRF views for chat API endpoints.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, cast
from uuid import UUID

from django.db import transaction
from django.db.models import QuerySet
from django.http import StreamingHttpResponse

from rest_framework import permissions, serializers, status, viewsets
from rest_framework.decorators import action
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from asgiref.sync import sync_to_async
from typing_extensions import override

from chatbot.models import ChatMessage, ChatSession, UserIntake
from chatbot.serializers import (
    ChatInputSerializer,
    ChatMessageSerializer,
    ChatResponseSerializer,
    ChatSessionCreateSerializer,
    ChatSessionSerializer,
    RetrievedDocSerializer,
    UserIntakeInputSerializer,
    UserIntakeOutputSerializer,
    UserIntakeSerializer,
)
from chatbot.services.chatbot_service import ChatbotService
from vision.models import XRayAnalysis

logger = logging.getLogger(__name__)


class ChatSessionViewSet(viewsets.ModelViewSet):
    """ViewSet for managing chat sessions."""

    serializer_class = ChatSessionSerializer
    permission_classes = [permissions.IsAuthenticated]

    @override
    def get_queryset(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> QuerySet[ChatSession]:
        """Return sessions for the current user."""
        # Use customer_id since Customer extends User with multi-table inheritance
        # and they share the same primary key
        return ChatSession.objects.filter(customer_id=self.request.user.pk)

    @override
    def get_serializer_class(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        if self.action == "create":
            return ChatSessionCreateSerializer
        return ChatSessionSerializer

    @override
    def perform_create(self, serializer: ChatSessionCreateSerializer) -> None:
        """Create a new session, deactivating any existing active sessions."""
        user = self.request.user
        validated_data = cast(dict[str, Any], serializer.validated_data)
        intake_data = validated_data.pop("intake", None)

        with transaction.atomic():
            if intake_data is not None:
                self._save_session_intake(user, intake_data)

            ChatSession.objects.filter(customer_id=user.pk, is_active=True).update(
                is_active=False
            )
            serializer.save(customer_id=user.pk, is_active=True)

    def _save_session_intake(self, user: Any, intake_data: dict[str, Any]) -> None:
        intake, _ = UserIntake.objects.get_or_create(customer=user)
        serializer = UserIntakeSerializer(intake, data=intake_data, partial=True)
        if not serializer.is_valid():
            raise serializers.ValidationError({"intake": serializer.errors})
        serializer.save()

    @action(detail=True, methods=["get"])
    def messages(
        self, _request: Request, pk: str | None = None
    ) -> Response:  # noqa: ARG002
        """Get all messages for a session."""
        session = self.get_object()
        messages = ChatMessage.objects.filter(session=session).order_by(
            "created_at"
        )  # pyright: ignore[reportUnreachable]
        serializer = ChatMessageSerializer(messages, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=["post"])
    def clear(
        self, _request: Request, pk: str | None = None
    ) -> Response:  # noqa: ARG002
        """Clear all messages in a session."""
        session = self.get_object()
        count = ChatMessage.objects.filter(session=session).delete()[
            0
        ]  # pyright: ignore[reportUnreachable]
        return Response({"deleted_count": count})


class ChatView(APIView):
    """View for sending chat messages."""

    permission_classes = [permissions.IsAuthenticated]

    @override  # type: ignore[override]
    async def dispatch(  # type: ignore[reportIncompatibleMethodOverride]
        self, request: Request, *args: Any, **kwargs: Any
    ) -> Response | StreamingHttpResponse:
        """
        Custom async dispatch to handle DRF's sync authentication safely and support async handlers.
        Replaces APIView.dispatch logic with async-aware steps.
        """
        self.args = args
        self.kwargs = kwargs

        # 1. Initialize Request (Standard DRF)
        drf_request = self.initialize_request(request, *args, **kwargs)
        self.request = drf_request  # type: ignore[assignment]
        self.headers = (
            self.default_response_headers
        )  # dep: APIView usually sets this if available

        try:
            # 2. Perform initialization (Auth, Perms, Throttling) in a thread safe way!
            # sync_to_async protects against SynchronousOnlyOperation during DB auth
            await sync_to_async(self.initial)(drf_request, *args, **kwargs)

            # 3. Get Handler
            if drf_request.method.lower() in self.http_method_names:
                handler = getattr(
                    self, drf_request.method.lower(), self.http_method_not_allowed
                )
            else:
                handler = self.http_method_not_allowed

            # 4. Execute Handler
            # Since handler (post) is async def, this returns a coroutine
            response = handler(cast(Any, drf_request), *args, **kwargs)

            # Await the coroutine to get actual Response
            if asyncio.iscoroutine(response) or asyncio.isfuture(response):
                response = await response

            # 5. Finalize Response
            self.response = self.finalize_response(
                drf_request, response, *args, **kwargs
            )  # type: ignore[arg-type]
            return self.response  # type: ignore[return-value]

        except Exception as exc:
            # Handle exceptions (e.g. Auth failed)
            response = self.handle_exception(exc)
            self.response = self.finalize_response(
                drf_request, response, *args, **kwargs
            )  # type: ignore[arg-type]
            return self.response  # type: ignore[return-value]

    async def post(self, request: Request) -> Response | StreamingHttpResponse:
        """Send a message and get a response."""
        validated_or_response = self._validated_chat_input(request)
        if isinstance(validated_or_response, Response):
            return validated_or_response
        validated = validated_or_response
        session_id: UUID = validated["session_id"]
        message: str = validated["message"]
        use_stream: bool = validated.get("stream", False)
        xray_analysis_id: int | None = validated.get("xray_analysis_id")
        uploaded_image = self._uploaded_chat_image(request, validated)

        logger.info(
            "Chat request received. session=%s stream=%s xray=%s has_image=%s",
            session_id,
            use_stream,
            xray_analysis_id,
            bool(uploaded_image),
        )
        logger.debug(
            "Chat request payload keys: %s",
            sorted(cast(dict[str, Any], request.data).keys()),  # type: ignore[arg-type]
        )

        # --- Inline X-Ray Analysis logic ---
        if uploaded_image and not xray_analysis_id:
            xray_result = await self._create_xray_analysis_from_upload(
                request=request,
                uploaded_image=uploaded_image,
            )
            if isinstance(xray_result, Response):
                return xray_result
            xray_analysis_id = xray_result
        # --- End X-Ray Analysis logic ---

        # Get session
        session_or_response = await self._active_chat_session(request, session_id)
        if isinstance(session_or_response, Response):
            return session_or_response
        session = session_or_response

        xray_analysis_or_response = await self._authorized_xray_analysis(
            session, xray_analysis_id
        )
        if isinstance(xray_analysis_or_response, Response):
            return xray_analysis_or_response

        # Process with chatbot service
        start_time = time.time()
        # ChatbotService.__init__ touches DB (Config), so we must run it in a thread
        chatbot = await sync_to_async(ChatbotService)(session)

        if use_stream:
            # Return streaming response
            return self._streaming_response(
                chatbot,
                message,
                session_id,
                xray_analysis_id,
                xray_analysis_or_response,
            )

        # Regular response (Async)
        response_text = await chatbot.aquery(message, xray_analysis_id=xray_analysis_id)
        return self._regular_chat_response(
            chatbot=chatbot,
            session=session,
            session_id=session_id,
            response_text=response_text,
            start_time=start_time,
        )

    def _validated_chat_input(self, request: Request) -> dict[str, Any] | Response:
        serializer = ChatInputSerializer(data=request.data)
        if serializer.is_valid():
            return cast(dict[str, Any], serializer.validated_data)
        logger.error(f"Invalid chat input: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def _uploaded_chat_image(self, request: Request, validated: dict[str, Any]) -> Any:
        uploaded_image = validated.get("image")
        if uploaded_image is not None:
            return uploaded_image
        raw_request = getattr(request, "_request", request)
        return getattr(raw_request, "FILES", {}).get("image")

    async def _create_xray_analysis_from_upload(
        self,
        *,
        request: Request,
        uploaded_image: Any,
    ) -> int | Response:
        import os
        import tempfile

        from django.conf import settings

        from vision.services.vision_service import analyze_xray

        suffix: str = os.path.splitext(str(uploaded_image.name))[1] or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            for chunk in uploaded_image.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            result = await sync_to_async(analyze_xray)(
                config_path=settings.VISION_CONFIG_PATH,
                image_path=tmp_path,
                checkpoint_path=settings.VISION_CHECKPOINT_PATH,
            )
            analysis = await sync_to_async(XRayAnalysis.objects.create)(
                user=request.user,
                image=uploaded_image,
                class_probs=result["class_probs"],
                pred_label=result["pred_label"],
                findings=result["findings"],
                embedding=result["embedding"],
                heatmap_base64=self._xray_heatmap_base64(result),
            )
            logger.info("Generated new XRayAnalysis ID: %s", analysis.pk)
            return analysis.pk
        except Exception as exc:
            logger.exception("X-ray analysis failed during chat request")
            return Response(
                {"error": f"Image analysis failed: {str(exc)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _xray_heatmap_base64(self, result: dict[str, Any]) -> str | None:
        import base64
        import os

        heatmap_path: str | None = result.get("heatmap_path")
        if not heatmap_path or not os.path.isfile(heatmap_path):
            return None
        with open(heatmap_path, "rb") as file_obj:
            return base64.b64encode(file_obj.read()).decode("utf-8")

    async def _active_chat_session(
        self,
        request: Request,
        session_id: UUID,
    ) -> ChatSession | Response:
        try:
            session = await ChatSession.objects.aget(
                session_id=session_id,
                customer=request.user,
            )
        except ChatSession.DoesNotExist:
            return Response(
                {"error": "Session not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        if session.is_active:
            return session
        return Response(
            {"error": "This session is inactive. Please start a new session."},
            status=status.HTTP_403_FORBIDDEN,
        )

    async def _authorized_xray_analysis(
        self,
        session: ChatSession,
        xray_analysis_id: int | None,
    ) -> XRayAnalysis | None | Response:
        if not xray_analysis_id:
            return None
        try:
            return await XRayAnalysis.objects.aget(
                id=xray_analysis_id,
                user_id=session.customer_id,
            )
        except XRayAnalysis.DoesNotExist:
            return Response(
                {"error": "X-ray analysis not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

    def _regular_chat_response(
        self,
        *,
        chatbot: ChatbotService,
        session: ChatSession,
        session_id: UUID,
        response_text: str,
        start_time: float,
    ) -> Response:
        response_data = {
            "session_id": session_id,
            "response": response_text,
            "response_time_ms": int((time.time() - start_time) * 1000),
            "source_urls": chatbot.get_last_source_urls(),
            "metadata": {
                "mode": chatbot.get_last_audit().get("mode"),
            },
            "is_active": session.is_active,
        }
        return Response(
            ChatResponseSerializer(response_data).data,
            status=status.HTTP_200_OK,
        )

    def _streaming_response(
        self,
        chatbot: ChatbotService,
        message: str,
        session_id: UUID,
        xray_analysis_id: int | None = None,
        xray_analysis: XRayAnalysis | None = None,
    ) -> StreamingHttpResponse:
        """Generate a streaming response."""

        async def event_stream():
            try:
                if xray_analysis:
                    metadata_event = await self._xray_metadata_event(xray_analysis)
                    if metadata_event:
                        yield metadata_event

                chunk_count = 0
                async for chunk in chatbot.astream_response(
                    message, xray_analysis_id=xray_analysis_id
                ):
                    if not chunk:
                        continue
                    chunk_count += 1
                    payload = self._chat_sse_data_payload(chunk)

                    logger.debug(
                        "Streaming chunk sent. session=%s chunk=%s bytes=%s",
                        session_id,
                        chunk_count,
                        len(payload),
                    )
                    yield payload
                logger.info(
                    "Streaming complete. session=%s chunks=%s",
                    session_id,
                    chunk_count,
                )
                raw_source_urls = await sync_to_async(chatbot.get_last_source_urls)()
                source_event = self._source_urls_event(raw_source_urls)
                if source_event:
                    yield source_event
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error("Streaming error for session=%s: %s", session_id, e)
                yield f"data: [ERROR] {e!s}\n\n"

        response = StreamingHttpResponse(
            event_stream(), content_type="text/event-stream"
        )
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"  # Disable nginx buffering
        response["X-Session-Id"] = str(session_id)
        return response

    async def _xray_metadata_event(self, analysis_record: XRayAnalysis) -> str:
        import json

        from vision.serializers import XRayAnalysisDisplaySerializer

        try:
            data = await sync_to_async(
                lambda: XRayAnalysisDisplaySerializer(analysis_record).data
            )()
            return f"event: metadata\ndata: {json.dumps(data)}\n\n"
        except Exception as exc:
            logger.warning(
                f"Failed to serialize XRayAnalysis streaming metadata: {exc}"
            )
            return ""

    def _chat_sse_data_payload(self, chunk: str) -> str:
        if "\n" not in chunk:
            return f"data: {chunk}\n\n"
        formatted_lines = [f"data: {line}" for line in chunk.splitlines()]
        return "\n".join(formatted_lines) + "\n\n"

    def _source_urls_event(self, raw_source_urls: Any) -> str:
        import json

        source_urls = raw_source_urls if isinstance(raw_source_urls, list) else []
        source_urls = [str(url).strip() for url in source_urls if str(url).strip()]
        if not source_urls:
            return ""
        return (
            "event: sources\n"
            f"data: {json.dumps({'source_urls': source_urls}, ensure_ascii=False)}\n\n"
        )


class UserIntakeView(APIView):
    """View for managing user intake data."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request: Request, session_id: str) -> Response:
        """Get current user intake for a session."""
        from chatbot.services.chatbot_service import ChatbotService

        try:
            session = ChatSession.objects.get(
                session_id=session_id, customer=request.user
            )
        except ChatSession.DoesNotExist:
            return Response(
                {"error": "Session not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        chatbot = ChatbotService(session)
        intake = chatbot.get_intake()

        data = {
            "disease_name": intake.disease_name,
            "symptoms": intake.symptoms,
            "symptoms_negated": intake.symptoms_negated,
            "age": intake.age,
            "sex": intake.sex,
            "onset_days": intake.onset_days,
            "pregnancy_status": intake.pregnancy_status,
        }
        return Response(UserIntakeOutputSerializer(data).data)

    def patch(self, request: Request, session_id: str) -> Response:
        """Update user intake for a session."""
        from chatbot.services.chatbot_service import ChatbotService

        try:
            session = ChatSession.objects.get(
                session_id=session_id, customer=request.user
            )
        except ChatSession.DoesNotExist:
            return Response(
                {"error": "Session not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = UserIntakeInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated = cast(dict[str, Any], serializer.validated_data)
        chatbot = ChatbotService(session)
        intake = chatbot.update_intake(**validated)

        data = {
            "disease_name": intake.disease_name,
            "symptoms": intake.symptoms,
            "symptoms_negated": intake.symptoms_negated,
            "age": intake.age,
            "sex": intake.sex,
            "onset_days": intake.onset_days,
            "pregnancy_status": intake.pregnancy_status,
        }
        return Response(UserIntakeOutputSerializer(data).data)


class RetrievedDocsView(APIView):
    """View for getting retrieved documents from last query."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request: Request, session_id: str) -> Response:
        """Get last retrieved documents for a session."""
        from chatbot.services.chatbot_service import ChatbotService

        try:
            session = ChatSession.objects.get(
                session_id=session_id, customer=request.user
            )
        except ChatSession.DoesNotExist:
            return Response(
                {"error": "Session not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        chatbot = ChatbotService(session)
        docs = chatbot.get_last_docs()

        return Response(RetrievedDocSerializer(docs, many=True).data)
