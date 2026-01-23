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
from chatbot.views.user_intake_view import reset_intake_view, user_intake_view

router = DefaultRouter()
router.register(r"sessions", ChatSessionViewSet, basename="chat-session")

urlpatterns = [
    path("", include(router.urls)),
    path("chat/", ChatView.as_view(), name="chat"),
    # New intake endpoints
    path("intake/", user_intake_view, name="user-intake"),
    path("intake/reset/", reset_intake_view, name="user-intake-reset"),
    # Legacy intake endpoint (session-specific, may need refactor)
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
