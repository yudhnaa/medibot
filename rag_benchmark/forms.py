from django import forms

from rag_benchmark.models import BenchmarkDataset, BenchmarkSplit, DatasetSourceFormat
from rag_benchmark.services.constants import DEFAULT_RAGAS_METRICS


class BenchmarkDatasetImportForm(forms.Form):
    dataset_name = forms.CharField(max_length=120)
    dataset_version = forms.CharField(max_length=64)
    description = forms.CharField(widget=forms.Textarea, required=False)
    source_format = forms.ChoiceField(
        choices=DatasetSourceFormat.choices,
        initial=DatasetSourceFormat.JSONL,
    )
    schema_version = forms.CharField(max_length=32, initial="1.0")
    activate = forms.BooleanField(required=False, initial=True)
    jsonl_file = forms.FileField()

    def clean_jsonl_file(self):
        uploaded_file = self.cleaned_data["jsonl_file"]
        filename = str(getattr(uploaded_file, "name", "")).lower()
        if not filename.endswith(".jsonl"):
            raise forms.ValidationError("Dataset file must use .jsonl extension.")
        if getattr(uploaded_file, "size", 0) <= 0:
            raise forms.ValidationError("Dataset file is empty.")
        return uploaded_file


class BenchmarkRunAdminForm(forms.Form):
    metric_choices = [(metric, metric) for metric in DEFAULT_RAGAS_METRICS]

    dataset = forms.ModelChoiceField(
        queryset=BenchmarkDataset.objects.order_by("name", "version"),
        required=True,
        empty_label=None,
    )
    split = forms.ChoiceField(
        choices=BenchmarkSplit.choices,
        initial=BenchmarkSplit.DEV,
    )
    code_version = forms.CharField(required=False, max_length=128)
    ragas_metrics = forms.MultipleChoiceField(
        choices=metric_choices,
        required=False,
        help_text="Leave empty to use default metric set.",
    )
    disabled_ragas_metrics = forms.MultipleChoiceField(
        choices=metric_choices,
        required=False,
        help_text="Optional metrics to disable from the active set.",
    )
