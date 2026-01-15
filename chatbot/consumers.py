"""
Chatbot Consumers
WebSocket consumers for handling real-time chat interactions.
"""

import json
import logging

from typing import Any, override
from channels.generic.websocket import AsyncWebsocketConsumer

from django.contrib.auth.models import AbstractBaseUser, AnonymousUser
from chatbot.services.rag_service import RAGChatbotService

logger = logging.getLogger("chatbot")

# Global service instance to avoid reloading models/vectors on every connection
# In production, this might need better lifecycle management or be process-global
_rag_service = None


def get_rag_service():
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGChatbotService()
    return _rag_service


class ChatConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for RAG Chatbot.
    URL: /ws/chat/<session_id>/
    """

    session_id: str | None = None
    user: AbstractBaseUser | AnonymousUser | None = None

    async def connect(self):
        # Initialize instance variables with type hints
        self.session_id: str | None = None
        self.user: AbstractBaseUser | AnonymousUser | None = None

        if "url_route" in self.scope:
            self.session_id = self.scope["url_route"]["kwargs"].get("session_id")  # type: ignore

        if "user" in self.scope:
            self.user = self.scope["user"]

        # Basic Auth Check
        # If user is not authenticated, we might reject or allow specific flows
        # For now, allow but ensure session exists or create it linked to user if Auth

        # Verify session existence
        try:
            # Sync database access needs to be wrapped for async
            pass
        except Exception:
            await self.close()
            return

        await self.accept()

    @override
    async def disconnect(self, code: int):
        pass

    async def receive(
        self, text_data: str | None = None, bytes_data: bytes | None = None
    ):
        if text_data is None:
            return

        try:
            data: dict[str, Any] = json.loads(text_data)
            message: str = data.get("message", "").strip()

            if not message:
                return

            if not self.session_id:
                await self.send(json.dumps({"error": "No session ID"}))
                return

            service = get_rag_service()

            # Stream response
            # Since service.stream_response yields chunks synchronously (or via sync generator),
            # we need to run it in a way compatible with async.
            # RAGChatbotService uses sync LangChain.
            # We should run it in a threadpool or make it async.
            # Currently RAGChatbotService is synchronous.

            # Using sync_to_async for the blocking generator is tricky.
            # Easiest is to iterate the generator in a thread.

            # Helper to run the stream in a thread and send chunks back
            await self.stream_response_in_thread(service, message)

        except json.JSONDecodeError:
            await self.send(text_data=json.dumps({"error": "Invalid JSON"}))
        except Exception as e:
            logger.error(f"Error in consumer: {e}")
            await self.send(text_data=json.dumps({"error": str(e)}))

    async def stream_response_in_thread(self, service: RAGChatbotService, message: str):
        # We'll use a wrapper to iterate the sync generator and send messages
        # But self.send is async. We can't call it from the sync thread directly without async_to_sync which is messy loops.

        # Better approach:
        # Define a sync function that returns the FULL response for now (non-streaming MVP)
        # OR
        # Change RAGChatbotService to be async (requires async DB/LangChain).

        # For True Streaming with Sync logic in Django Channels:
        # Iterate the sync generator in a thread, putting chunks into a queue?
        # Or just use sync_to_async on the `next()` call of the generator?

        # Let's try iterating the generator step-by-step wrapped in sync_to_async

        # Helper to ensure session_id is string
        if not self.session_id:
            return

        iterator = service.stream_response(message, self.session_id)

        from asgiref.sync import sync_to_async

        # Fixing strict typing for sync_to_async wrapper
        def get_next_chunk(it) -> tuple[str | None, bool]:
            try:
                return next(it), False
            except StopIteration:
                return None, True

        async_next = sync_to_async(get_next_chunk, thread_sensitive=False)

        while True:
            chunk, done = await async_next(iterator)
            if done:
                break

            await self.send(
                text_data=json.dumps({"type": "chat.message.chunk", "content": chunk})
            )

        # Send done message
        await self.send(
            text_data=json.dumps(
                {"type": "chat.message.done", "session_id": self.session_id}
            )
        )
