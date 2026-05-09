"""
Vision URL Configuration
"""

from django.urls import path

from vision.views import AnalyzeView, EmbedView

urlpatterns = [
    path("analyze/", AnalyzeView.as_view(), name="vision-analyze"),
    path("embed/", EmbedView.as_view(), name="vision-embed"),
]
