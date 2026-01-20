"""
Chat WebSocket Consumer
Handles WebSocket connections for real-time chat with streaming responses.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from typing_extensions import override

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

from chatbot.models import ChatSession

if TYPE_CHECKING:
    from chatbot.services import ChatbotService

logger = logging.getLogger(__name__)


class ChatConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for real-time chat with streaming responses."""

    session_id: str = ""
    user: object = None
    session: ChatSession | None = None

    @override
    async def connect(self) -> None:
        """Handle WebSocket connection."""
        url_route = self.scope.get("url_route", {})
        kwargs = url_route.get("kwargs", {}) if url_route else {}
        self.session_id = str(kwargs.get("session_id", ""))
        self.user = self.scope.get("user")

        # Authenticate
        if not self.user or not getattr(self.user, "is_authenticated", False):
            await self.close(code=4001)
            return

        # Verify session belongs to user
        session = await self._get_session()
        if not session:
            await self.close(code=4004)
            return
        self.session = session

        await self.accept()
        logger.info(f"WebSocket connected: session={self.session_id}")

    @override
    async def disconnect(self, code: int) -> None:
        """Handle WebSocket disconnection."""
        session_id = getattr(self, "session_id", "unknown")
        logger.info(f"WebSocket disconnected: session={session_id}, code={code}")

    @override
    async def receive(
        self, text_data: str | None = None, bytes_data: bytes | None = None
    ) -> None:
        """Handle incoming message and stream response."""
        if not text_data:
            return

        try:
            data = json.loads(text_data)
            message = data.get("message", "")

            if not message:
                await self.send(
                    json.dumps(
                        {
                            "type": "error",
                            "message": "Empty message",
                        }
                    )
                )
                return

            # Send acknowledgment
            await self.send(
                json.dumps(
                    {
                        "type": "ack",
                        "message": "Processing...",
                    }
                )
            )

            # Process and stream response
            await self._process_and_stream(message)

        except json.JSONDecodeError:
            await self.send(
                json.dumps(
                    {
                        "type": "error",
                        "message": "Invalid JSON",
                    }
                )
            )
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await self.send(
                json.dumps(
                    {
                        "type": "error",
                        "message": str(e),
                    }
                )
            )

    async def _process_and_stream(self, message: str) -> None:
        """Process message and stream response tokens."""
        try:
            # Create chatbot service (sync operation wrapped)
            chatbot = await self._create_chatbot_service()

            # Stream response
            full_response = ""
            async for chunk in self._stream_response_async(chatbot, message):
                full_response += chunk
                await self.send(
                    json.dumps(
                        {
                            "type": "chunk",
                            "content": chunk,
                        }
                    )
                )

            # Send completion
            await self.send(
                json.dumps(
                    {
                        "type": "done",
                        "full_response": full_response,
                    }
                )
            )

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            await self.send(
                json.dumps(
                    {
                        "type": "error",
                        "message": str(e),
                    }
                )
            )

    @database_sync_to_async
    def _get_session(self) -> ChatSession | None:
        """Get session from database."""
        try:
            return ChatSession.objects.get(
                session_id=self.session_id,
                customer=self.user,
            )
        except ChatSession.DoesNotExist:
            return None

    @database_sync_to_async
    def _create_chatbot_service(self) -> "ChatbotService":
        """Create chatbot service instance."""
        from chatbot.services import ChatbotService

        assert self.session is not None
        return ChatbotService(self.session)

    async def _stream_response_async(
        self, chatbot: "ChatbotService", message: str
    ) -> AsyncGenerator[str, None]:
        """Wrap sync generator in async iterator."""

        def sync_generator() -> list[str]:
            return list(chatbot.stream_response(message))

        # Run sync generator in thread pool
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, sync_generator)

        for chunk in chunks:
            yield chunk
