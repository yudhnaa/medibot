from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models
from django.utils import timezone


class DatasetSourceFormat(models.TextChoices):
    JSONL = "jsonl", "JSONL"


class BenchmarkSplit(models.TextChoices):
    DEV = "dev", "Development"
    TEST = "test", "Test"


class ExpectedMode(models.TextChoices):
    SINGLE_DISEASE = "single-disease", "Single disease"
    MULTI_DISEASE_V2 = "multi-disease-v2", "Multi disease v2"


class ExpectedBehavior(models.TextChoices):
    ANSWER = "answer", "Answer"
    ASK_FOLLOWUP = "ask_followup", "Ask follow-up"
    ABSTAIN = "abstain", "Abstain"


class BenchmarkRunStatus(models.TextChoices):
    PENDING = "pending", "Pending"
    RUNNING = "running", "Running"
    COMPLETED = "completed", "Completed"
    FAILED = "failed", "Failed"


class BenchmarkCaseResultStatus(models.TextChoices):
    PASSED = "passed", "Passed"
    FAILED = "failed", "Failed"
    ERROR = "error", "Error"
    SKIPPED = "skipped", "Skipped"


class BenchmarkReviewStatus(models.TextChoices):
    PENDING = "pending", "Pending"
    REVIEWED = "reviewed", "Reviewed"
    NEEDS_TRIAGE = "needs_triage", "Needs triage"


class BenchmarkFailureTaxonomy(models.TextChoices):
    ANALYZER_ERROR = "analyzer_error", "Analyzer error"
    GATE_ERROR = "gate_error", "Gate error"
    RETRIEVAL_MISS = "retrieval_miss", "Retrieval miss"
    NEGATION_ERROR = "negation_error", "Negation error"
    GENERATION_UNSUPPORTED = "generation_unsupported", "Generation unsupported"
    SAFETY_ERROR = "safety_error", "Safety error"
    DATASET_ANNOTATION_ISSUE = "dataset_annotation_issue", "Dataset annotation issue"
    BENCHMARK_RUNNER_ISSUE = "benchmark_runner_issue", "Benchmark runner issue"


class BenchmarkMetricScope(models.TextChoices):
    RUN = "run", "Run"
    CASE = "case", "Case"
    SLICE = "slice", "Slice"
    RELEASE_GATE = "release_gate", "Release gate"
    JUDGE = "judge", "Judge"


class BenchmarkDataset(models.Model):
    name = models.CharField(max_length=120)
    version = models.CharField(max_length=64)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=False)
    source_format = models.CharField(
        max_length=24,
        choices=DatasetSourceFormat.choices,
        default=DatasetSourceFormat.JSONL,
    )
    schema_version = models.CharField(max_length=32, default="1.0")
    total_cases = models.PositiveIntegerField(default=0)
    split_counts = models.JSONField(default=dict, blank=True)
    composition_stats = models.JSONField(default=dict, blank=True)
    activated_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "benchmark_dataset"
        verbose_name = "Benchmark dataset"
        verbose_name_plural = "Benchmark datasets"
        constraints = [
            models.UniqueConstraint(
                fields=["name", "version"],
                name="benchmark_dataset_name_version_uniq",
            )
        ]
        indexes = [
            models.Index(fields=["name"]),
            models.Index(fields=["is_active"]),
            models.Index(fields=["-created_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.name}:{self.version}"

    def activate(self) -> None:
        BenchmarkDataset.objects.filter(name=self.name, is_active=True).exclude(
            pk=self.pk
        ).update(is_active=False, activated_at=None)
        self.is_active = True
        self.activated_at = timezone.now()
        self.save(update_fields=["is_active", "activated_at", "updated_at"])


class BenchmarkCase(models.Model):
    dataset = models.ForeignKey(
        BenchmarkDataset,
        on_delete=models.CASCADE,
        related_name="cases",
    )
    case_id = models.CharField(max_length=120)
    dataset_version = models.CharField(max_length=64)
    split = models.CharField(max_length=16, choices=BenchmarkSplit.choices)
    question = models.TextField()
    intake_payload = models.JSONField(default=dict, blank=True)
    scenario = models.CharField(max_length=80)
    expected_mode = models.CharField(max_length=32, choices=ExpectedMode.choices)
    gold_titles = models.JSONField(default=list, blank=True)
    forbidden_titles = models.JSONField(default=list, blank=True)
    must_have_sections = models.JSONField(default=list, blank=True)
    expected_behavior = models.CharField(
        max_length=24,
        choices=ExpectedBehavior.choices,
    )
    reference_answer = models.TextField(blank=True)
    notes = models.TextField(blank=True)

    gold_analysis = models.JSONField(default=dict, blank=True)
    gold_primary_title = models.CharField(max_length=255, blank=True)
    must_not_sections = models.JSONField(default=list, blank=True)
    reference_context_ids = models.JSONField(default=list, blank=True)
    requires_followup_topic = models.CharField(max_length=255, blank=True)
    risk_level = models.CharField(max_length=16, blank=True)
    annotation_metadata = models.JSONField(default=dict, blank=True)
    is_active = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "benchmark_case"
        verbose_name = "Benchmark case"
        verbose_name_plural = "Benchmark cases"
        constraints = [
            models.UniqueConstraint(
                fields=["dataset", "case_id"],
                name="benchmark_case_dataset_case_id_uniq",
            )
        ]
        indexes = [
            models.Index(fields=["split"]),
            models.Index(fields=["scenario"]),
            models.Index(fields=["expected_mode"]),
            models.Index(fields=["is_active"]),
        ]

    def __str__(self) -> str:
        return f"{self.dataset}::{self.case_id}"


class BenchmarkRun(models.Model):
    run_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    dataset = models.ForeignKey(
        BenchmarkDataset,
        on_delete=models.PROTECT,
        related_name="runs",
    )
    split = models.CharField(max_length=16, choices=BenchmarkSplit.choices)
    status = models.CharField(
        max_length=16,
        choices=BenchmarkRunStatus.choices,
        default=BenchmarkRunStatus.PENDING,
    )
    runtime_snapshot = models.JSONField(default=dict, blank=True)
    config_snapshot = models.JSONField(default=dict, blank=True)
    corpus_signature = models.CharField(max_length=255, blank=True)
    judge_configuration = models.JSONField(default=dict, blank=True)
    code_version = models.CharField(max_length=128, blank=True)

    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    total_cases = models.PositiveIntegerField(default=0)
    passed_cases = models.PositiveIntegerField(default=0)
    failed_cases = models.PositiveIntegerField(default=0)

    summary_metrics = models.JSONField(default=dict, blank=True)
    failure_slices = models.JSONField(default=dict, blank=True)
    release_gate = models.JSONField(default=dict, blank=True)
    baseline_comparison_summary = models.JSONField(default=dict, blank=True)
    error_summary = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "benchmark_run"
        verbose_name = "Benchmark run"
        verbose_name_plural = "Benchmark runs"
        indexes = [
            models.Index(fields=["dataset", "split"]),
            models.Index(fields=["status"]),
            models.Index(fields=["-created_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.dataset}::{self.run_id}"

    @property
    def primary_pass(self) -> bool:
        return bool((self.release_gate or {}).get("overall_pass", False))


class BenchmarkCaseResult(models.Model):
    run = models.ForeignKey(
        BenchmarkRun,
        on_delete=models.CASCADE,
        related_name="case_results",
    )
    case = models.ForeignKey(
        BenchmarkCase,
        on_delete=models.CASCADE,
        related_name="results",
    )
    status = models.CharField(
        max_length=16,
        choices=BenchmarkCaseResultStatus.choices,
        default=BenchmarkCaseResultStatus.FAILED,
    )

    input_payload = models.JSONField(default=dict, blank=True)
    analysis_artifact = models.JSONField(default=dict, blank=True)
    gate_artifact = models.JSONField(default=dict, blank=True)
    retrieval_artifact = models.JSONField(default=dict, blank=True)
    generation_artifact = models.JSONField(default=dict, blank=True)
    audit_metadata = models.JSONField(default=dict, blank=True)

    timing_ms = models.JSONField(default=dict, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    pass_flags = models.JSONField(default=dict, blank=True)
    source_urls = models.JSONField(default=list, blank=True)
    final_answer = models.TextField(blank=True)
    error_payload = models.JSONField(default=dict, blank=True)

    review_status = models.CharField(
        max_length=24,
        choices=BenchmarkReviewStatus.choices,
        default=BenchmarkReviewStatus.PENDING,
    )
    failure_taxonomy = models.CharField(
        max_length=48,
        choices=BenchmarkFailureTaxonomy.choices,
        blank=True,
    )
    reviewer_notes = models.TextField(blank=True)
    reviewed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    reviewed_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "benchmark_case_result"
        verbose_name = "Benchmark case result"
        verbose_name_plural = "Benchmark case results"
        constraints = [
            models.UniqueConstraint(
                fields=["run", "case"],
                name="benchmark_case_result_run_case_uniq",
            )
        ]
        indexes = [
            models.Index(fields=["run", "status"]),
            models.Index(fields=["review_status"]),
            models.Index(fields=["failure_taxonomy"]),
        ]

    def __str__(self) -> str:
        return f"{self.run.run_id}::{self.case.case_id}"

    @property
    def primary_pass(self) -> bool:
        return bool((self.pass_flags or {}).get("primary_pass", False))


class BenchmarkMetric(models.Model):
    run = models.ForeignKey(
        BenchmarkRun,
        on_delete=models.CASCADE,
        related_name="metrics",
    )
    case_result = models.ForeignKey(
        BenchmarkCaseResult,
        on_delete=models.CASCADE,
        related_name="metric_records",
        null=True,
        blank=True,
    )
    metric_name = models.CharField(max_length=120)
    metric_scope = models.CharField(max_length=24, choices=BenchmarkMetricScope.choices)
    slice_key = models.CharField(max_length=120, blank=True)
    value = models.FloatField()
    threshold = models.FloatField(null=True, blank=True)
    is_pass = models.BooleanField(null=True, blank=True)
    details = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "benchmark_metric"
        verbose_name = "Benchmark metric"
        verbose_name_plural = "Benchmark metrics"
        indexes = [
            models.Index(fields=["run", "metric_name"]),
            models.Index(fields=["metric_scope", "slice_key"]),
        ]

    def __str__(self) -> str:
        return f"{self.metric_name}={self.value}"
