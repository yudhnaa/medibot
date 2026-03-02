"""
Vision URL Configuration
"""

from django.urls import path

from vision.views import AnalyzeView, EmbedView, SimilarView

urlpatterns = [
    path("analyze/", AnalyzeView.as_view(), name="vision-analyze"),
    path("embed/", EmbedView.as_view(), name="vision-embed"),
    path("similar/", SimilarView.as_view(), name="vision-similar"),
]
