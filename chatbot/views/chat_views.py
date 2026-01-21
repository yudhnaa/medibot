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

from django.db.models import QuerySet
from django.http import StreamingHttpResponse
from asgiref.sync import sync_to_async

from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

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
        """Create a new session for the current user."""
        # Use customer_id since Customer extends User with multi-table inheritance
        serializer.save(customer_id=self.request.user.pk)

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

        logger.info(
            f"Chat request received. Session: {session_id}, Stream: {use_stream}"
        )
        logger.debug(f"Request data: {request.data}")

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

        # Process with chatbot service
        start_time = time.time()
        # ChatbotService.__init__ touches DB (Config), so we must run it in a thread
        chatbot = await sync_to_async(ChatbotService)(session)

        if use_stream:
            # Return streaming response
            return self._streaming_response(chatbot, message, session_id)

        # Regular response (Async)
        response_text = await chatbot.aquery(message)
        response_time_ms = int((time.time() - start_time) * 1000)

        response_data = {
            "session_id": session_id,
            "response": response_text,
            "response_time_ms": response_time_ms,
            "metadata": {
                "mode": chatbot.get_last_audit().get("mode"),
            },
        }

        return Response(
            ChatResponseSerializer(response_data).data,
            status=status.HTTP_200_OK,
        )

    def _streaming_response(
        self, chatbot: ChatbotService, message: str, session_id: UUID
    ) -> StreamingHttpResponse:
        """Generate a streaming response."""

        async def event_stream():
            try:
                chunk_count = 0
                async for chunk in chatbot.astream_response(message):
                    chunk_count += 1
                    logger.info(f"Streaming chunk {chunk_count}: {repr(chunk)}")
                    # Format as Server-Sent Event
                    if "\n" in chunk:
                        # Handle multi-line chunks
                        payload = ""
                        # Use splitlines to handle various newline formats safely
                        formatted_lines = [
                            f"data: {line}" for line in chunk.splitlines()
                        ]
                        payload = "\n".join(formatted_lines) + "\n\n"
                    else:
                        payload = f"data: {chunk}\n\n"

                    logger.info(f"Sent event: {repr(payload)}")
                    yield payload
                logger.info(f"Streaming complete: {chunk_count} chunks")
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: [ERROR] {e!s}\n\n"

        response = StreamingHttpResponse(
            event_stream(), content_type="text/event-stream"
        )
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"  # Disable nginx buffering
        response["Connection"] = "keep-alive"
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
