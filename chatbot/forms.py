"""Forms for chatbot application."""

from django import forms

from chatbot.models import IndexType
from vector_store.services.constants import (
    EMBEDDING_PROVIDER_GEMINI,
    EMBEDDING_PROVIDER_TEI,
    EMBEDDING_PROVIDER_TRANSFORMERS,
)


class CsvUploadForm(forms.Form):
    """Form for uploading CSV files to embed into vector database."""

    PROVIDER_CHOICES = [
        (EMBEDDING_PROVIDER_TRANSFORMERS, "Transformers (Local)"),
        (EMBEDDING_PROVIDER_TEI, "TEI (Text Embeddings Inference)"),
        (EMBEDDING_PROVIDER_GEMINI, "Google Gemini"),
    ]

    csv_file = forms.FileField(
        label="CSV File",
        help_text=(
            "Upload a CSV file containing medical documents. "
            "Example: processed_data/test_dataset/test_dataset.csv"
        ),
        widget=forms.FileInput(attrs={"accept": ".csv"}),
    )

    embedding_provider = forms.ChoiceField(
        label="Embedding Provider",
        choices=PROVIDER_CHOICES,
        initial=EMBEDDING_PROVIDER_TRANSFORMERS,
        help_text="Select the embedding model provider to use for document vectorization.",
    )

    index_types = forms.MultipleChoiceField(
        label="Index Types",
        choices=IndexType.choices,
        initial=[IndexType.B],
        widget=forms.CheckboxSelectMultiple,
        help_text=(
            "Select one or more index types to create. "
            "A=Summary Index (disease-level), "
            "B=Detail Index (per-section), "
            "C=Title Index (title-only)"
        ),
    )

    def clean_csv_file(self):
        """Validate that uploaded file is a CSV."""
        csv_file = self.cleaned_data.get("csv_file")

        if not csv_file:
            raise forms.ValidationError("No file uploaded.")

        # Check file extension
        if not csv_file.name.endswith(".csv"):
            raise forms.ValidationError(
                "Invalid file type. Please upload a CSV file (.csv)"
            )

        # Check file size (max 50MB)
        if csv_file.size > 50 * 1024 * 1024:
            raise forms.ValidationError(
                (
                    "File too large. Maximum file size is 50MB. "
                    "For larger files, please contact system administrator."
                )
            )

        return csv_file
