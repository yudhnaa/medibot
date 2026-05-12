"""Forms for chatbot application."""

from urllib.parse import urlparse

from django import forms
from django.contrib.admin.widgets import FilteredSelectMultiple

from chatbot.models import IndexType, MedicalDocument, SectionType


class CsvUploadForm(forms.Form):
    """Form for uploading CSV files to embed into vector database."""

    csv_file = forms.FileField(
        label="CSV File",
        help_text=(
            "Upload a CSV file containing medical documents. "
            "Example: processed_data/test_dataset/test_dataset.csv"
        ),
        widget=forms.FileInput(attrs={"accept": ".csv"}),
    )

    index_types = forms.MultipleChoiceField(
        label="Index Types",
        choices=IndexType.choices,
        initial=[IndexType.C, IndexType.A, IndexType.B],
        widget=forms.CheckboxSelectMultiple,
        help_text=(
            "Select one or more index types to create. "
            "A=medical_documents_disease (summary per disease), "
            "B=medical_documents_chunks (section-level detail), "
            "C=medical_documents_titles (disease title gate)"
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


class ArticleUrlEmbedForm(forms.Form):
    """Form for triggering URL crawl + LLM extraction + embedding."""

    url = forms.URLField(
        label="Article URL",
        max_length=2048,
        help_text="Paste an article URL to crawl and index into C/A/B collections.",
    )

    def clean_url(self):
        raw_url = str(self.cleaned_data.get("url", "")).strip()
        parsed = urlparse(raw_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise forms.ValidationError("Only valid http/https URLs are supported.")
        return raw_url


class DocumentEditForm(forms.ModelForm):
    """Form for editing document fields with preview."""

    class Meta:
        model = MedicalDocument
        fields = ["title", "content", "section_type", "metadata"]
        widgets = {
            "title": forms.TextInput(attrs={"size": 80}),
            "content": forms.Textarea(attrs={"rows": 10, "cols": 80}),
            "section_type": forms.Select(),
            "metadata": forms.Textarea(attrs={"rows": 5, "cols": 80}),
        }


class DocumentBulkActionForm(forms.Form):
    """Form for selecting documents and choosing bulk action."""

    ACTION_CHOICES = [
        ("reembed", "Re-embed Selected Documents"),
        ("delete", "Delete Selected Documents"),
        ("export", "Export as JSON"),
        ("change_section", "Change Section Type"),
        ("change_index", "Change Index Type"),
    ]

    action = forms.ChoiceField(choices=ACTION_CHOICES, label="Action")
    new_section_type = forms.ChoiceField(
        choices=SectionType.choices,
        label="New Section Type",
        required=False,
        help_text="Only applies to change section action",
    )
    new_index_type = forms.ChoiceField(
        choices=IndexType.choices,
        label="New Index Type",
        required=False,
        help_text="Only applies to change index action",
    )
    confirm = forms.BooleanField(
        required=False,
        label="I understand this action cannot be undone",
    )

    def clean(self):
        cleaned_data = super().clean() or {}
        action = cleaned_data.get("action")
        confirm = cleaned_data.get("confirm")

        if action in ["delete", "reembed"] and not confirm:
            raise forms.ValidationError("You must confirm to proceed with this action.")

        return cleaned_data


class VectorSearchForm(forms.Form):
    """Form for vector similarity search."""

    query_text = forms.CharField(
        label="Search Query",
        widget=forms.Textarea(attrs={"rows": 4, "cols": 60}),
        help_text="Enter text to find similar documents",
    )
    k = forms.IntegerField(
        label="Number of Results (Top-K)",
        initial=5,
        min_value=1,
        max_value=100,
    )
    section_types = forms.MultipleChoiceField(
        choices=SectionType.choices,
        required=False,
        widget=FilteredSelectMultiple("Section Types", is_stacked=False),
        label="Filter by Section Type (optional)",
    )
    min_similarity = forms.FloatField(
        label="Minimum Similarity Score",
        initial=0.5,
        min_value=0.0,
        max_value=1.0,
        required=False,
        help_text="Filter results by minimum cosine similarity",
    )


class ReembeddingForm(forms.Form):
    """Form for re-embedding operations."""

    REEMBED_TYPE_CHOICES = [
        ("section", "Re-embed by Section Type"),
        ("missing", "Re-embed Missing Embeddings"),
    ]

    reembed_type = forms.ChoiceField(
        choices=REEMBED_TYPE_CHOICES,
        label="Re-embedding Type",
    )
    section_type = forms.ChoiceField(
        choices=SectionType.choices,
        label="Section Type",
        required=False,
        help_text="Only applies to 'by section' re-embedding",
    )
    batch_size = forms.IntegerField(
        initial=100,
        min_value=1,
        max_value=1000,
        label="Batch Size",
    )
    run_async = forms.BooleanField(
        required=False,
        initial=True,
        label="Run as Background Job (recommended for large batches)",
    )
    confirm = forms.BooleanField(
        required=False,
        label="Confirm re-embedding",
    )

    def clean(self):
        cleaned_data = super().clean() or {}
        if not cleaned_data.get("confirm"):
            raise forms.ValidationError("You must confirm to start re-embedding.")
        if cleaned_data.get("reembed_type") == "section" and not cleaned_data.get(
            "section_type"
        ):
            self.add_error("section_type", "Section type is required.")
        return cleaned_data
