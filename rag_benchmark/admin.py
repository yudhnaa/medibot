from __future__ import annotations

import json
from typing import Any, Mapping

from django.contrib import admin, messages
from django.http import HttpRequest
from django.shortcuts import redirect, render
from django.urls import path, reverse
from django.utils.functional import Promise
from django.utils.html import format_html

from rag_benchmark.forms import BenchmarkDatasetImportForm, BenchmarkRunAdminForm
from rag_benchmark.models import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkDataset,
    BenchmarkFailureTaxonomy,
    BenchmarkMetric,
    BenchmarkRun,
)
from rag_benchmark.services import (
    BenchmarkDatasetImporter,
    DatasetValidationError,
    OfflineBenchmarkRunner,
)
from rag_benchmark.services.constants import RAGAS_RELEASE_GATE_THRESHOLDS


class BenchmarkCaseInline(admin.TabularInline):
    model = BenchmarkCase
    extra = 0
    fields = (
        "case_id",
        "split",
        "scenario",
        "expected_mode",
        "expected_behavior",
        "is_active",
    )
    readonly_fields = (
        "case_id",
        "split",
        "scenario",
        "expected_mode",
        "expected_behavior",
    )
    show_change_link = True


@admin.register(BenchmarkDataset)
class BenchmarkDatasetAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "version",
        "is_active",
        "total_cases",
        "source_format",
        "schema_version",
        "run_benchmark_link",
        "created_at",
    )
    list_filter = ("is_active", "source_format", "schema_version")
    search_fields = ("name", "version", "description")
    readonly_fields = (
        "total_cases",
        "split_counts_pretty",
        "composition_stats_pretty",
        "created_at",
        "updated_at",
    )
    inlines = [BenchmarkCaseInline]
    change_list_template = "admin/rag_benchmark/benchmarkdataset/change_list.html"

    @admin.display(description="Split counts")
    def split_counts_pretty(self, obj: BenchmarkDataset) -> str:
        return self._pretty_json(obj.split_counts)

    @admin.display(description="Composition stats")
    def composition_stats_pretty(self, obj: BenchmarkDataset) -> str:
        return self._pretty_json(obj.composition_stats)

    @admin.display(description="Run")
    def run_benchmark_link(self, obj: BenchmarkDataset) -> str:
        run_url = (
            reverse("admin:rag_benchmark_benchmarkdataset_run_benchmark")
            + f"?dataset_id={obj.pk}"
        )
        return format_html('<a class="button" href="{}">Run</a>', run_url)

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                "import-jsonl/",
                self.admin_site.admin_view(self.import_jsonl_view),
                name="rag_benchmark_benchmarkdataset_import_jsonl",
            ),
            path(
                "run-benchmark/",
                self.admin_site.admin_view(self.run_benchmark_view),
                name="rag_benchmark_benchmarkdataset_run_benchmark",
            ),
        ]
        return custom_urls + urls

    def changelist_view(
        self, request: HttpRequest, extra_context: dict[str, Any] | None = None
    ):
        context = extra_context or {}
        context["import_jsonl_url"] = reverse(
            "admin:rag_benchmark_benchmarkdataset_import_jsonl"
        )
        context["run_benchmark_url"] = reverse(
            "admin:rag_benchmark_benchmarkdataset_run_benchmark"
        )
        return super().changelist_view(request, extra_context=context)

    def import_jsonl_view(self, request: HttpRequest):
        if request.method == "POST":
            form = BenchmarkDatasetImportForm(request.POST, request.FILES)
            if form.is_valid():
                importer = BenchmarkDatasetImporter()
                try:
                    dataset, report = importer.import_jsonl(
                        name=form.cleaned_data["dataset_name"],
                        version=form.cleaned_data["dataset_version"],
                        file_obj=form.cleaned_data["jsonl_file"],
                        description=form.cleaned_data["description"],
                        source_format=form.cleaned_data["source_format"],
                        schema_version=form.cleaned_data["schema_version"],
                        activate=form.cleaned_data["activate"],
                    )
                except DatasetValidationError as exc:
                    messages.error(request, str(exc))
                except Exception as exc:
                    messages.error(request, f"Import failed: {exc}")
                else:
                    messages.success(
                        request,
                        (
                            f"Imported dataset {dataset.name}:{dataset.version} "
                            f"({report.total_cases} cases)."
                        ),
                    )
                    if report.missing_scenarios:
                        messages.warning(
                            request,
                            (
                                "Missing recommended scenarios: "
                                f"{', '.join(report.missing_scenarios)}"
                            ),
                        )
                    return redirect("admin:rag_benchmark_benchmarkdataset_changelist")
        else:
            form = BenchmarkDatasetImportForm()

        context = {
            **self.admin_site.each_context(request),
            "opts": self.model._meta,
            "title": "Import Benchmark Dataset (JSONL)",
            "form": form,
        }
        return render(
            request,
            "admin/rag_benchmark/benchmarkdataset/import_jsonl.html",
            context,
        )

    def run_benchmark_view(self, request: HttpRequest):
        if request.method == "POST":
            form = BenchmarkRunAdminForm(request.POST)
            if form.is_valid():
                runner = OfflineBenchmarkRunner()
                dataset = form.cleaned_data["dataset"]
                split = form.cleaned_data["split"]
                code_version = str(form.cleaned_data.get("code_version", "")).strip()
                ragas_metrics = form.cleaned_data.get("ragas_metrics") or []
                disabled_ragas_metrics = (
                    form.cleaned_data.get("disabled_ragas_metrics") or []
                )
                try:
                    run = runner.run(
                        dataset=dataset,
                        split=split,
                        judge_configuration={
                            "enable_ragas": True,
                            "ragas_metrics": ragas_metrics,
                            "disabled_ragas_metrics": disabled_ragas_metrics,
                        },
                        code_version=code_version,
                    )
                except Exception as exc:
                    messages.error(request, f"Benchmark run failed: {exc}")
                else:
                    messages.success(
                        request,
                        (
                            f"Benchmark run created: {run.run_id} "
                            f"(dataset={dataset.name}:{dataset.version}, split={split})"
                        ),
                    )
                    run_url = reverse(
                        "admin:rag_benchmark_benchmarkrun_change",
                        args=[run.pk],
                    )
                    return redirect(run_url)
        else:
            initial = {}
            selected_dataset_id = request.GET.get("dataset_id")
            if selected_dataset_id:
                initial["dataset"] = selected_dataset_id
            form = BenchmarkRunAdminForm(initial=initial)

        context = {
            **self.admin_site.each_context(request),
            "opts": self.model._meta,
            "title": "Run Benchmark",
            "form": form,
        }
        return render(
            request,
            "admin/rag_benchmark/benchmarkdataset/run_benchmark.html",
            context,
        )

    def _pretty_json(self, value: Any) -> str:
        return format_html(
            "<pre style='white-space: pre-wrap; margin: 0'>{}</pre>",
            json.dumps(value or {}, ensure_ascii=False, indent=2),
        )


@admin.register(BenchmarkCase)
class BenchmarkCaseAdmin(admin.ModelAdmin):
    list_display = (
        "case_id",
        "dataset",
        "split",
        "scenario",
        "expected_mode",
        "expected_behavior",
        "is_active",
    )
    list_filter = (
        "split",
        "scenario",
        "expected_mode",
        "expected_behavior",
        "is_active",
    )
    search_fields = ("case_id", "question", "dataset__name", "dataset__version")
    readonly_fields = ("created_at", "updated_at")


@admin.register(BenchmarkRun)
class BenchmarkRunAdmin(admin.ModelAdmin):
    list_display = (
        "run_id",
        "dataset",
        "split",
        "status",
        "primary_pass",
        "dashboard_link",
        "total_cases",
        "passed_cases",
        "failed_cases",
        "started_at",
        "completed_at",
    )
    list_filter = ("status", "split", "dataset")
    search_fields = ("run_id", "dataset__name", "dataset__version")
    change_form_template = "admin/rag_benchmark/benchmarkrun/change_form.html"
    readonly_fields = (
        "run_id",
        "runtime_snapshot_pretty",
        "config_snapshot_pretty",
        "summary_metrics_pretty",
        "failure_slices_pretty",
        "release_gate_pretty",
        "created_at",
        "updated_at",
    )

    @admin.display(description="Dashboard")
    def dashboard_link(self, obj: BenchmarkRun) -> str:
        url = reverse("admin:rag_benchmark_benchmarkrun_dashboard", args=[obj.pk])
        return format_html('<a class="button" href="{}">View</a>', url)

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                "<path:object_id>/dashboard/",
                self.admin_site.admin_view(self.dashboard_view),
                name="rag_benchmark_benchmarkrun_dashboard",
            ),
            path(
                "<path:object_id>/case-metrics/",
                self.admin_site.admin_view(self.case_metrics_view),
                name="rag_benchmark_benchmarkrun_case_metrics",
            ),
        ]
        return custom_urls + urls

    def change_view(
        self,
        request: HttpRequest,
        object_id: str,
        form_url: str = "",
        extra_context: dict[str, Any] | None = None,
    ):
        context = extra_context or {}
        context["dashboard_url"] = reverse(
            "admin:rag_benchmark_benchmarkrun_dashboard", args=[object_id]
        )
        context["case_metrics_url"] = reverse(
            "admin:rag_benchmark_benchmarkrun_case_metrics", args=[object_id]
        )
        return super().change_view(request, object_id, form_url, extra_context=context)

    def dashboard_view(self, request: HttpRequest, object_id: str):
        run = self.get_object(request, object_id)
        if run is None:
            messages.error(request, "Benchmark run not found.")
            return redirect("admin:rag_benchmark_benchmarkrun_changelist")

        summary_metrics = (
            run.summary_metrics if isinstance(run.summary_metrics, dict) else {}
        )
        release_gate = run.release_gate if isinstance(run.release_gate, dict) else {}
        checks = release_gate.get("checks", {})
        release_checks = checks if isinstance(checks, dict) else {}
        failure_slices = (
            run.failure_slices if isinstance(run.failure_slices, dict) else {}
        )

        scenario_failures = self._to_slice_rows(failure_slices.get("scenario", {}))
        mode_failures = self._to_slice_rows(failure_slices.get("expected_mode", {}))
        confidence_failures = self._to_slice_rows(
            failure_slices.get("route_confidence_bucket", {})
        )
        question_length_failures = self._to_slice_rows(
            failure_slices.get("question_length", {})
        )

        primary_metric_keys = list(release_checks.keys())
        if not primary_metric_keys:
            primary_metric_keys = [
                key
                for key in (
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                    "primary_pass_rate",
                )
                if key in summary_metrics
            ]

        metric_cards = [
            self._build_metric_card(
                key=metric_key,
                summary_metrics=summary_metrics,
                release_checks=release_checks,
            )
            for metric_key in primary_metric_keys
        ]
        metric_cards = [card for card in metric_cards if card is not None]

        total_case_count = int(run.total_cases or 0)
        pass_count = int(run.passed_cases or 0)
        fail_count = int(run.failed_cases or 0)
        rate_denominator = max(1, total_case_count)
        pass_rate = (float(pass_count) / float(rate_denominator)) * 100.0
        fail_rate = (float(fail_count) / float(rate_denominator)) * 100.0

        top_failure_results = list(
            BenchmarkCaseResult.objects.filter(run=run, status__in=["failed", "error"])
            .select_related("case")
            .order_by("-updated_at")[:20]
        )

        taxonomy_labels = {item.value: item.label for item in BenchmarkFailureTaxonomy}
        top_failure_rows = [
            {
                "result": result,
                "reason": self._build_failure_reason(
                    result=result,
                    taxonomy_labels=taxonomy_labels,
                ),
            }
            for result in top_failure_results
        ]

        review_counts_qs = (
            BenchmarkCaseResult.objects.filter(run=run)
            .values("failure_taxonomy")
            .order_by("failure_taxonomy")
        )
        review_counts: dict[str, int] = {}
        for row in review_counts_qs:
            key = str(row.get("failure_taxonomy") or "").strip() or "unclassified"
            review_counts[key] = review_counts.get(key, 0) + 1
        review_distribution = self._to_slice_rows(review_counts)
        for row in review_distribution:
            row["label"] = taxonomy_labels.get(row["key"], row["label"])

        context = {
            **self.admin_site.each_context(request),
            "opts": self.model._meta,
            "title": f"Benchmark Dashboard · {run.run_id}",
            "run": run,
            "metric_cards": metric_cards,
            "release_checks": self._release_check_rows(release_checks),
            "scenario_failures": scenario_failures,
            "mode_failures": mode_failures,
            "confidence_failures": confidence_failures,
            "question_length_failures": question_length_failures,
            "review_distribution": review_distribution,
            "top_failure_rows": top_failure_rows,
            "total_case_count": total_case_count,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "pass_rate": round(pass_rate, 2),
            "fail_rate": round(fail_rate, 2),
            "change_url": reverse(
                "admin:rag_benchmark_benchmarkrun_change",
                args=[run.pk],
            ),
            "case_metrics_url": reverse(
                "admin:rag_benchmark_benchmarkrun_case_metrics",
                args=[run.pk],
            ),
            "case_results_url": (
                reverse("admin:rag_benchmark_benchmarkcaseresult_changelist")
                + f"?run__id__exact={run.pk}"
            ),
        }
        return render(
            request,
            "admin/rag_benchmark/benchmarkrun/dashboard.html",
            context,
        )

    def case_metrics_view(self, request: HttpRequest, object_id: str):
        run = self.get_object(request, object_id)
        if run is None:
            messages.error(request, "Benchmark run not found.")
            return redirect("admin:rag_benchmark_benchmarkrun_changelist")

        results = list(
            BenchmarkCaseResult.objects.filter(run=run)
            .select_related("case")
            .order_by("case__case_id", "-updated_at")
        )

        release_gate = run.release_gate if isinstance(run.release_gate, dict) else {}
        checks = release_gate.get("checks", {}) if isinstance(release_gate, dict) else {}
        release_checks = checks if isinstance(checks, dict) else {}
        metric_thresholds: dict[str, str] = {}
        for metric_key in (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ):
            check_payload = (
                release_checks.get(metric_key, {})
                if isinstance(release_checks.get(metric_key, {}), dict)
                else {}
            )
            operator = str(check_payload.get("operator", "")).strip()
            threshold = check_payload.get("threshold")
            if not operator or threshold is None:
                fallback_operator, fallback_threshold = RAGAS_RELEASE_GATE_THRESHOLDS.get(
                    metric_key,
                    ("", None),
                )
                operator = str(fallback_operator).strip()
                threshold = fallback_threshold
            if operator and threshold is not None:
                metric_thresholds[metric_key] = f"{operator} {threshold}"
            else:
                metric_thresholds[metric_key] = "-"

        case_rows: list[dict[str, Any]] = []
        for result in results:
            pass_flags = result.pass_flags if isinstance(result.pass_flags, dict) else {}
            metrics = result.metrics if isinstance(result.metrics, dict) else {}
            failed_checks = [
                self._humanize_metric_name(str(key).removesuffix("_pass"))
                for key, value in pass_flags.items()
                if str(key).endswith("_pass") and value is False
            ]
            case_rows.append(
                {
                    "result": result,
                    "primary_pass": bool(pass_flags.get("primary_pass", False)),
                    "faithfulness": metrics.get("faithfulness"),
                    "answer_relevancy": metrics.get("answer_relevancy"),
                    "context_precision": metrics.get("context_precision"),
                    "context_recall": metrics.get("context_recall"),
                    "failed_checks": failed_checks,
                }
            )

        total_case_count = int(run.total_cases or 0)
        pass_count = int(run.passed_cases or 0)
        fail_count = int(run.failed_cases or 0)

        context = {
            **self.admin_site.each_context(request),
            "opts": self.model._meta,
            "title": f"Case Metrics · {run.run_id}",
            "run": run,
            "case_rows": case_rows,
            "metric_thresholds": metric_thresholds,
            "total_case_count": total_case_count,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "dashboard_url": reverse(
                "admin:rag_benchmark_benchmarkrun_dashboard",
                args=[run.pk],
            ),
            "change_url": reverse(
                "admin:rag_benchmark_benchmarkrun_change",
                args=[run.pk],
            ),
            "case_results_url": (
                reverse("admin:rag_benchmark_benchmarkcaseresult_changelist")
                + f"?run__id__exact={run.pk}"
            ),
        }
        return render(
            request,
            "admin/rag_benchmark/benchmarkrun/case_metrics.html",
            context,
        )

    @admin.display(description="Runtime snapshot")
    def runtime_snapshot_pretty(self, obj: BenchmarkRun) -> str:
        return self._pretty_json(obj.runtime_snapshot)

    @admin.display(description="Config snapshot")
    def config_snapshot_pretty(self, obj: BenchmarkRun) -> str:
        return self._pretty_json(obj.config_snapshot)

    @admin.display(description="Summary metrics")
    def summary_metrics_pretty(self, obj: BenchmarkRun) -> str:
        return self._pretty_json(obj.summary_metrics)

    @admin.display(description="Failure slices")
    def failure_slices_pretty(self, obj: BenchmarkRun) -> str:
        return self._pretty_json(obj.failure_slices)

    @admin.display(description="Release gate")
    def release_gate_pretty(self, obj: BenchmarkRun) -> str:
        return self._pretty_json(obj.release_gate)

    def _pretty_json(self, value: Any) -> str:
        return format_html(
            "<pre style='white-space: pre-wrap; margin: 0'>{}</pre>",
            json.dumps(value or {}, ensure_ascii=False, indent=2),
        )

    def _build_metric_card(
        self,
        *,
        key: str,
        summary_metrics: dict[str, Any],
        release_checks: dict[str, Any],
    ) -> dict[str, Any] | None:
        raw_value = summary_metrics.get(key)
        if raw_value is None:
            return None
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None

        check = release_checks.get(key, {})
        threshold = check.get("threshold")
        operator = check.get("operator")
        is_pass = check.get("is_pass")
        percentage = min(max(value * 100.0, 0.0), 100.0)
        return {
            "key": key,
            "label": self._humanize_metric_name(key),
            "value": round(value, 4),
            "value_pct": round(percentage, 2),
            "threshold": threshold,
            "operator": operator,
            "is_pass": bool(is_pass) if is_pass is not None else None,
        }

    def _release_check_rows(
        self, release_checks: dict[str, Any]
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key, payload in release_checks.items():
            value = payload.get("value")
            threshold = payload.get("threshold")
            operator = payload.get("operator", ">=")
            rows.append(
                {
                    "key": key,
                    "label": self._humanize_metric_name(str(key)),
                    "value": (
                        round(float(value), 4)
                        if isinstance(value, (int, float))
                        else value
                    ),
                    "threshold": threshold,
                    "operator": operator,
                    "is_pass": bool(payload.get("is_pass", False)),
                }
            )
        return rows

    def _to_slice_rows(self, payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        rows: list[dict[str, Any]] = []
        max_value = max(
            [
                int(value)
                for value in payload.values()
                if isinstance(value, (int, float))
            ],
            default=0,
        )
        for key, value in payload.items():
            if not isinstance(value, (int, float)):
                continue
            normalized = (
                (float(value) / float(max_value)) * 100.0 if max_value > 0 else 0.0
            )
            rows.append(
                {
                    "key": str(key),
                    "label": self._humanize_metric_name(str(key)),
                    "value": int(value),
                    "width": round(normalized, 2),
                }
            )
        return sorted(rows, key=lambda item: item["value"], reverse=True)

    def _build_failure_reason(
        self,
        *,
        result: BenchmarkCaseResult,
        taxonomy_labels: Mapping[str, str | Promise],
    ) -> str:
        taxonomy_key = str(result.failure_taxonomy or "").strip()
        if taxonomy_key:
            label = taxonomy_labels.get(
                taxonomy_key,
                self._humanize_metric_name(taxonomy_key),
            )
            return str(label)

        error_payload = result.error_payload if isinstance(result.error_payload, dict) else {}
        error_message = str(error_payload.get("message") or "").strip()
        if error_message:
            return error_message

        pass_flags = result.pass_flags if isinstance(result.pass_flags, dict) else {}
        failed_checks = [
            self._humanize_metric_name(str(key).removesuffix("_pass"))
            for key, value in pass_flags.items()
            if str(key).endswith("_pass") and value is False
        ]
        if failed_checks:
            return ", ".join(failed_checks)

        if result.status == "error":
            return "Runtime error"
        return "Primary gate failed"

    def _humanize_metric_name(self, key: str) -> str:
        text = key.replace("_", " ").replace("@", " @ ")
        compact = " ".join(text.split())
        return compact.title()


@admin.register(BenchmarkCaseResult)
class BenchmarkCaseResultAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "run",
        "case",
        "status",
        "primary_pass",
        "review_status",
        "failure_taxonomy",
        "reviewed_by",
        "reviewed_at",
    )
    list_filter = (
        "status",
        "review_status",
        "failure_taxonomy",
        "run__dataset",
    )
    search_fields = ("case__case_id", "case__question", "reviewer_notes")
    readonly_fields = (
        "input_payload_pretty",
        "analysis_artifact_pretty",
        "gate_artifact_pretty",
        "retrieval_artifact_pretty",
        "generation_artifact_pretty",
        "timing_ms_pretty",
        "metrics_pretty",
        "pass_flags_pretty",
        "error_payload_pretty",
        "created_at",
        "updated_at",
    )
    list_editable = ("review_status", "failure_taxonomy")

    @admin.display(description="Input")
    def input_payload_pretty(self, obj: BenchmarkCaseResult) -> str:
        return self._pretty_json(obj.input_payload)

    @admin.display(description="Analysis artifact")
    def analysis_artifact_pretty(self, obj: BenchmarkCaseResult) -> str:
        return self._pretty_json(obj.analysis_artifact)

    @admin.display(description="Gate artifact")
    def gate_artifact_pretty(self, obj: BenchmarkCaseResult) -> str:
        return self._pretty_json(obj.gate_artifact)

    @admin.display(description="Retrieval artifact")
    def retrieval_artifact_pretty(self, obj: BenchmarkCaseResult) -> str:
        return self._pretty_json(obj.retrieval_artifact)

    @admin.display(description="Generation artifact")
    def generation_artifact_pretty(self, obj: BenchmarkCaseResult) -> str:
        return self._pretty_json(obj.generation_artifact)

    @admin.display(description="Timing (ms)")
    def timing_ms_pretty(self, obj: BenchmarkCaseResult) -> str:
        return self._pretty_json(obj.timing_ms)

    @admin.display(description="Metrics")
    def metrics_pretty(self, obj: BenchmarkCaseResult) -> str:
        return self._pretty_json(obj.metrics)

    @admin.display(description="Pass flags")
    def pass_flags_pretty(self, obj: BenchmarkCaseResult) -> str:
        return self._pretty_json(obj.pass_flags)

    @admin.display(description="Error payload")
    def error_payload_pretty(self, obj: BenchmarkCaseResult) -> str:
        return self._pretty_json(obj.error_payload)

    def _pretty_json(self, value: Any) -> str:
        return format_html(
            "<pre style='white-space: pre-wrap; margin: 0'>{}</pre>",
            json.dumps(value or {}, ensure_ascii=False, indent=2),
        )


@admin.register(BenchmarkMetric)
class BenchmarkMetricAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "run",
        "metric_name",
        "metric_scope",
        "slice_key",
        "value",
        "threshold",
        "is_pass",
    )
    list_filter = ("metric_scope", "metric_name", "is_pass")
    search_fields = ("metric_name", "slice_key", "run__run_id")
