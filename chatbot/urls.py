"""
Chatbot URL Configuration
"""

from django.urls import include, path

from rest_framework.routers import DefaultRouter

from chatbot.views import (
    ChatSessionViewSet,
    ChatView,
    RetrievedDocsView,
    UserIntakeView,
)

router = DefaultRouter()
router.register(r"sessions", ChatSessionViewSet, basename="chat-session")

urlpatterns = [
    path("", include(router.urls)),
    path("chat/", ChatView.as_view(), name="chat"),
    path(
        "sessions/<uuid:session_id>/intake/",
        UserIntakeView.as_view(),
        name="session-intake",
    ),
    path(
        "sessions/<uuid:session_id>/docs/",
        RetrievedDocsView.as_view(),
        name="session-docs",
    ),
]
