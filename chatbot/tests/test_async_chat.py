import json
import uuid
from unittest.mock import MagicMock, patch, AsyncMock

from django.test import RequestFactory, SimpleTestCase
from rest_framework.request import Request as DRFRequest
from chatbot.views.chat_views import ChatView


class AsyncStreamingTest(SimpleTestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.username = "testuser"
        # Mock User
        self.user = MagicMock()
        self.user.username = self.username
        self.user.pk = 1

        self.session_id = uuid.uuid4()
        # Mock Session
        self.session = MagicMock()
        self.session.session_id = self.session_id
        self.session.customer = self.user
        self.session.title = "Test Session"

    async def test_streaming_response(self):
        view = ChatView()

        data = {
            "session_id": str(self.session_id),
            "message": "Hello AI",
            "stream": True,
        }

        request = self.factory.post(
            "/chatbot/chat/", data=data, content_type="application/json"
        )
        request.user = self.user

        # WRAP in DRF Request
        # APIView.initialize_request normally does this.
        drf_request = DRFRequest(request)
        # We manually set data because parsing is complex to setup without parsers context
        # normally validators run on parsers.
        # Since we passed content_type=application/json, RequestFactory put body in input stream.
        # But DRF Request needs parsers to read it.
        # Easiest is to just mock .data
        drf_request._full_data = data

        # Patch ChatSession.objects.aget
        with patch("chatbot.views.chat_views.ChatSession") as MockSessionCls:
            # Mock the manager and aget method
            mock_manager = MagicMock()
            mock_manager.aget = AsyncMock(return_value=self.session)
            MockSessionCls.objects = mock_manager

            # Patch ChatbotService to avoid real LLM call
            with patch("chatbot.views.chat_views.ChatbotService") as MockServiceCls:
                mock_service = MockServiceCls.return_value

                # Make sure we initialize ChatbotService with our mock session
                # The view does: chatbot = ChatbotService(session)
                # MockServiceCls(session) returns mock_service

                # Define async generator for astream_response
                async def mock_astream(question):
                    yield "Part 1"
                    yield "\nPart 2"

                mock_service.astream_response = mock_astream

                # We must use await on the view because it is async
                response = await view.post(drf_request)

                # StreamingHttpResponse status_code is 200 by default
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.streaming)

                # Consume content
                chunks = []
                if response.streaming_content:
                    async for chunk in response.streaming_content:
                        chunks.append(chunk.decode("utf-8"))

                full_text = "".join(chunks)
                print(f"DEBUG RESPONSE: {full_text}")

                # Check SSE format
                # "data: Part 1\n\n"
                # "data: Part 2\n\n"
                self.assertIn("data: Part 1", full_text)
                self.assertIn("data: Part 2", full_text)
