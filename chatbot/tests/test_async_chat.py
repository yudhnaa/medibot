import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from django.test import RequestFactory, SimpleTestCase

from rest_framework.request import Request as DRFRequest

from chatbot.services.chatbot_service import ChatbotService
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
                async def mock_astream(question, **kwargs):
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

    async def test_streaming_response_with_xray_analysis_id(self):
        """Test that xray_analysis_id is passed through to astream_response."""
        view = ChatView()

        xray_id = 42
        data = {
            "session_id": str(self.session_id),
            "message": "Phân tích ảnh X-quang này",
            "stream": True,
            "xray_analysis_id": xray_id,
        }

        request = self.factory.post(
            "/chatbot/chat/", data=data, content_type="application/json"
        )
        request.user = self.user

        drf_request = DRFRequest(request)
        drf_request._full_data = data
        analysis = MagicMock()
        analysis.id = xray_id

        with patch("chatbot.views.chat_views.ChatSession") as MockSessionCls:
            mock_manager = MagicMock()
            mock_manager.aget = AsyncMock(return_value=self.session)
            MockSessionCls.objects = mock_manager

            with patch("chatbot.views.chat_views.XRayAnalysis") as MockAnalysisCls:
                MockAnalysisCls.DoesNotExist = Exception
                analysis_manager = MagicMock()
                analysis_manager.aget = AsyncMock(return_value=analysis)
                MockAnalysisCls.objects = analysis_manager

                with patch("chatbot.views.chat_views.ChatbotService") as MockServiceCls:
                    mock_service = MockServiceCls.return_value

                    captured_kwargs = {}

                    async def mock_astream(question, **kwargs):
                        captured_kwargs.update(kwargs)
                        yield "X-ray analysis: COVID detected"

                    mock_service.astream_response = mock_astream

                    response = await view.post(drf_request)

                    self.assertEqual(response.status_code, 200)
                    self.assertTrue(response.streaming)

                    chunks = []
                    if response.streaming_content:
                        async for chunk in response.streaming_content:
                            chunks.append(chunk.decode("utf-8"))

                    full_text = "".join(chunks)

                    analysis_manager.aget.assert_awaited_once_with(
                        id=xray_id,
                        user_id=self.session.customer_id,
                    )
                    self.assertEqual(captured_kwargs.get("xray_analysis_id"), xray_id)
                    self.assertIn("data: X-ray analysis: COVID detected", full_text)

    async def test_chat_rejects_unowned_xray_analysis_id(self):
        view = ChatView()
        xray_id = 99
        data = {
            "session_id": str(self.session_id),
            "message": "Phân tích ảnh X-quang này",
            "stream": True,
            "xray_analysis_id": xray_id,
        }
        request = self.factory.post(
            "/chatbot/chat/", data=data, content_type="application/json"
        )
        request.user = self.user
        drf_request = DRFRequest(request)
        drf_request._full_data = data

        class MissingAnalysis(Exception):
            pass

        with patch("chatbot.views.chat_views.ChatSession") as MockSessionCls:
            mock_manager = MagicMock()
            mock_manager.aget = AsyncMock(return_value=self.session)
            MockSessionCls.objects = mock_manager

            with patch("chatbot.views.chat_views.XRayAnalysis") as MockAnalysisCls:
                MockAnalysisCls.DoesNotExist = MissingAnalysis
                analysis_manager = MagicMock()
                analysis_manager.aget = AsyncMock(side_effect=MissingAnalysis)
                MockAnalysisCls.objects = analysis_manager

                with patch("chatbot.views.chat_views.ChatbotService") as MockServiceCls:
                    response = await view.post(drf_request)

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.data, {"error": "X-ray analysis not found"})
        MockServiceCls.assert_not_called()
        analysis_manager.aget.assert_awaited_once_with(
            id=xray_id,
            user_id=self.session.customer_id,
        )

    async def test_streaming_metadata_uses_authorized_analysis(self):
        view = ChatView()
        analysis = MagicMock()
        analysis.id = 7
        analysis.class_probs = {"normal": 0.9}
        analysis.pred_label = "normal"
        analysis.findings = []
        analysis.heatmap_base64 = None

        event = await view._xray_metadata_event(analysis)

        self.assertIn("event: metadata", event)
        self.assertIn('"id": 7', event)


class XRayPayloadOwnershipTest(SimpleTestCase):
    def test_xray_payload_filters_by_session_customer(self):
        service = ChatbotService.__new__(ChatbotService)
        service.session = MagicMock()
        service.session.customer = MagicMock()
        service._build_xray_context = MagicMock(return_value="xray context")
        service._serialize_xray_analysis = MagicMock(return_value={"id": 42})
        analysis = MagicMock()
        analysis.image = None

        with patch("chatbot.services.chatbot_service.XRayAnalysis") as MockAnalysisCls:
            manager = MagicMock()
            manager.get.return_value = analysis
            MockAnalysisCls.objects = manager

            payload = service._get_xray_analysis_payload(42)

        manager.get.assert_called_once_with(id=42, user=service.session.customer)
        self.assertEqual(payload["context"], "xray context")
        self.assertEqual(payload["serialized"], {"id": 42})

    def test_xray_payload_omits_unowned_or_missing_analysis(self):
        service = ChatbotService.__new__(ChatbotService)
        service.session = MagicMock()
        service.session.customer = MagicMock()

        with patch("chatbot.services.chatbot_service.XRayAnalysis") as MockAnalysisCls:
            MockAnalysisCls.DoesNotExist = Exception
            manager = MagicMock()
            manager.get.side_effect = Exception
            MockAnalysisCls.objects = manager

            payload = service._get_xray_analysis_payload(42)

        manager.get.assert_called_once_with(id=42, user=service.session.customer)
        self.assertEqual(
            payload, {"context": "", "serialized": None, "user_metadata": {}}
        )
