"""
Chat Views
DRF views for chat API endpoints.
"""

import logging
import time
import asyncio
from typing import Any, cast
from typing_extensions import override
from uuid import UUID

from django.db import transaction
from django.db.models import QuerySet
from django.http import StreamingHttpResponse
from asgiref.sync import sync_to_async

from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from vision.models import XRayAnalysis
from vision.services.vision_service import analyze_xray

from chatbot.models import ChatMessage, ChatSession
from chatbot.serializers import (
    ChatInputSerializer,
    ChatMessageSerializer,
    ChatResponseSerializer,
    ChatSessionCreateSerializer,
    ChatSessionSerializer,
    RetrievedDocSerializer,
    UserIntakeInputSerializer,
    UserIntakeOutputSerializer,
)
from chatbot.services import ChatbotService

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

        with transaction.atomic():
            # Deactivate all current active sessions for this user
            ChatSession.objects.filter(customer_id=user.pk, is_active=True).update(
                is_active=False
            )

            # Create new active session
            serializer.save(customer_id=user.pk, is_active=True)

    @action(detail=True, methods=["get"])
    def messages(self, _request: Request, pk: str | None = None) -> Response:  # noqa: ARG002
        """Get all messages for a session."""
        session = self.get_object()
        messages = ChatMessage.objects.filter(session=session).order_by("created_at")  # pyright: ignore[reportUnreachable]
        serializer = ChatMessageSerializer(messages, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=["post"])
    def clear(self, _request: Request, pk: str | None = None) -> Response:  # noqa: ARG002
        """Clear all messages in a session."""
        session = self.get_object()
        count = ChatMessage.objects.filter(session=session).delete()[0]  # pyright: ignore[reportUnreachable]
        return Response({"deleted_count": count})


class ChatView(APIView):
    """View for sending chat messages."""

    permission_classes = [permissions.IsAuthenticated]

    @override
    async def dispatch(
        self, request: Request, *args: Any, **kwargs: Any
    ) -> Response | StreamingHttpResponse:
        """
        Custom async dispatch to handle DRF's sync authentication safely and support async handlers.
        Replaces APIView.dispatch logic with async-aware steps.
        """
        self.args = args
        self.kwargs = kwargs

        # 1. Initialize Request (Standard DRF)
        request = self.initialize_request(request, *args, **kwargs)
        self.request = request
        self.headers = (
            self.default_response_headers
        )  # dep: APIView usually sets this if available

        try:
            # 2. Perform initialization (Auth, Perms, Throttling) in a thread safe way!
            # sync_to_async protects against SynchronousOnlyOperation during DB auth
            await sync_to_async(self.initial)(request, *args, **kwargs)

            # 3. Get Handler
            if request.method.lower() in self.http_method_names:
                handler = getattr(
                    self, request.method.lower(), self.http_method_not_allowed
                )
            else:
                handler = self.http_method_not_allowed

            # 4. Execute Handler
            # Since handler (post) is async def, this returns a coroutine
            response = handler(request, *args, **kwargs)

            # Await the coroutine to get actual Response
            if asyncio.iscoroutine(response) or asyncio.isfuture(response):
                response = await response

            # 5. Finalize Response
            self.response = self.finalize_response(request, response, *args, **kwargs)
            return self.response

        except Exception as exc:
            # Handle exceptions (e.g. Auth failed)
            response = self.handle_exception(exc)
            self.response = self.finalize_response(request, response, *args, **kwargs)
            return self.response

    async def post(self, request: Request) -> Response | StreamingHttpResponse:
        """Send a message and get a response."""
        serializer = ChatInputSerializer(data=request.data)
        if not serializer.is_valid():
            logger.error(f"Invalid chat input: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated = cast(dict[str, Any], serializer.validated_data)
        session_id: UUID = validated["session_id"]
        message: str = validated["message"]
        use_stream: bool = validated.get("stream", False)
        xray_analysis_id: int | None = validated.get("xray_analysis_id")
        uploaded_image = validated.get("image")
        if uploaded_image is None:
            raw_request = getattr(request, "_request", request)
            uploaded_image = getattr(raw_request, "FILES", {}).get("image")

        logger.info(
            "Chat request received. session=%s stream=%s xray=%s has_image=%s",
            session_id,
            use_stream,
            xray_analysis_id,
            bool(uploaded_image),
        )
        logger.debug(
            "Chat request payload keys: %s",
            sorted(request.data.keys()),
        )

        # --- Inline X-Ray Analysis logic ---
        if uploaded_image and not xray_analysis_id:
            import os
            import tempfile
            from django.conf import settings

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

                import base64

                heatmap_b64 = None
                heatmap_path: str | None = result.get("heatmap_path")  # type: ignore[assignment]
                if heatmap_path and os.path.isfile(heatmap_path):
                    with open(heatmap_path, "rb") as f:
                        heatmap_b64 = base64.b64encode(f.read()).decode("utf-8")

                # Save the analysis to the database
                analysis = await sync_to_async(XRayAnalysis.objects.create)(
                    user=request.user,
                    image=uploaded_image,
                    class_probs=result["class_probs"],
                    pred_label=result["pred_label"],
                    findings=result["findings"],
                    embedding=result["embedding"],
                    heatmap_base64=heatmap_b64,
                )
                xray_analysis_id = analysis.pk
                logger.info("Generated new XRayAnalysis ID: %s", xray_analysis_id)
            except Exception as e:
                logger.exception("X-ray analysis failed during chat request")
                return Response(
                    {"error": f"Image analysis failed: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        # --- End X-Ray Analysis logic ---

        # Get session
        try:
            session = await ChatSession.objects.aget(
                session_id=session_id, customer=request.user
            )
        except ChatSession.DoesNotExist:
            return Response(
                {"error": "Session not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        if not session.is_active:
            return Response(
                {"error": "This session is inactive. Please start a new session."},
                status=status.HTTP_403_FORBIDDEN,
            )

        # Process with chatbot service
        start_time = time.time()
        # ChatbotService.__init__ touches DB (Config), so we must run it in a thread
        chatbot = await sync_to_async(ChatbotService)(session)

        if use_stream:
            # Return streaming response
            return self._streaming_response(
                chatbot, message, session_id, xray_analysis_id
            )

        # Regular response (Async)
        response_text = await chatbot.aquery(message, xray_analysis_id=xray_analysis_id)
        response_time_ms = int((time.time() - start_time) * 1000)

        response_data = {
            "session_id": session_id,
            "response": response_text,
            "response_time_ms": response_time_ms,
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
    ) -> StreamingHttpResponse:
        """Generate a streaming response."""

        async def event_stream():
            try:
                if xray_analysis_id:
                    import json
                    from vision.models import XRayAnalysis
                    from vision.serializers import XRayAnalysisDisplaySerializer

                    try:
                        analysis_record = await sync_to_async(XRayAnalysis.objects.get)(
                            id=xray_analysis_id
                        )
                        data = await sync_to_async(
                            lambda: XRayAnalysisDisplaySerializer(analysis_record).data
                        )()
                        yield f"event: metadata\ndata: {json.dumps(data)}\n\n"
                    except Exception as e:
                        logger.warning(
                            f"Failed to fetch XRayAnalysis for streaming metadata: {e}"
                        )

                chunk_count = 0
                async for chunk in chatbot.astream_response(
                    message, xray_analysis_id=xray_analysis_id
                ):
                    if not chunk:
                        continue
                    chunk_count += 1
                    # Format as Server-Sent Event
                    if "\n" in chunk:
                        formatted_lines = [
                            f"data: {line}" for line in chunk.splitlines()
                        ]
                        payload = "\n".join(formatted_lines) + "\n\n"
                    else:
                        payload = f"data: {chunk}\n\n"

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


class UserIntakeView(APIView):
    """View for managing user intake data."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request: Request, session_id: str) -> Response:
        """Get current user intake for a session."""
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
