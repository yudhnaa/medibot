from django.urls import path

from vector_store.views import EmbedDocumentsView, EmbedTextView

urlpatterns = [
    path("text/", EmbedTextView.as_view(), name="embed_text"),
    path("documents/", EmbedDocumentsView.as_view(), name="embed_documents"),
]
