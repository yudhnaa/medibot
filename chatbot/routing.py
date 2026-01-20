"""
Chatbot WebSocket Routing
"""

from chatbot.consumers import ChatConsumer

websocket_urlpatterns = [
    (r"ws/chat/(?P<session_id>[0-9a-f-]+)/$", ChatConsumer.as_asgi()),
]
