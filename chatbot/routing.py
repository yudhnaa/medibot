from django.urls import path
from .consumers import ChatConsumer

websocket_urlpatterns = [
    path("ws/chat/<str:session_id>/", ChatConsumer.as_asgi()),  # type: ignore
]
