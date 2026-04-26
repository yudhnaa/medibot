from __future__ import annotations

import hashlib
import json
import math
from typing import Any, cast

from django.contrib.auth import get_user_model
from django.db import transaction
from django.db.models import Count, Max
from django.utils import timezone

from chatbot.models import ChatSession, ChatbotConfig, MedicalDocument
from chatbot.services.chatbot_benchmark_service import ChatbotBenchmarkService
from rag_benchmark.models import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkCaseResultStatus,
    BenchmarkDataset,
    BenchmarkMetric,
    BenchmarkMetricScope,
    BenchmarkRun,
    BenchmarkRunStatus,
    BenchmarkSplit,
)
from rag_benchmark.services.constants import (
    BENCHMARK_RUNNER_USERNAME,
    DEFAULT_RAGAS_METRICS,
    RAGAS_JUDGE_METRICS,
    RAGAS_RELEASE_GATE_THRESHOLDS,
)
from rag_benchmark.services.ragas import RagasJudgeEvaluator
from rag_benchmark.services.scoring import RagasBenchmarkScorer
from vector_store.services.embedding_service import EmbeddingService


class OfflineBenchmarkRunner:
    """Execute benchmark cases directly against runtime services (no HTTP API)."""

    def __init__(self, *, scorer: RagasBenchmarkScorer | None = None) -> None:
        self.scorer = scorer or RagasBenchmarkScorer()

    def run(
        self,
        *,
        dataset: BenchmarkDataset,
        split: str,
        judge_configuration: dict[str, Any] | None = None,
        code_version: str = "",
    ) -> BenchmarkRun:
        split_value = split.strip().lower()
        if split_value not in {BenchmarkSplit.DEV, BenchmarkSplit.TEST}:
            raise ValueError(f"Unsupported split `{split}`")

        self._validate_dataset_before_run(dataset)
        cases = list(
            BenchmarkCase.objects.filter(
                dataset=dataset,
                split=split_value,
                is_active=True,
            ).order_by("id")
        )
        if not cases:
            raise ValueError(
                f"No active benchmark cases available for split `{split_value}`"
            )

        judge_conf = dict(judge_configuration or {})
        judge_conf["enable_ragas"] = True
        requested_metrics = self._normalize_metric_names(judge_conf.get("ragas_metrics"))
        disabled_metrics = self._normalize_metric_names(
            judge_conf.get("disabled_ragas_metrics")
        )
        effective_metrics = requested_metrics or list(DEFAULT_RAGAS_METRICS)
        if disabled_metrics:
            effective_metrics = [
                metric for metric in effective_metrics if metric not in disabled_metrics
            ]
        if not effective_metrics:
            raise ValueError("No active Ragas metrics after applying disable list")
        judge_conf["ragas_metrics"] = effective_metrics
        if disabled_metrics:
            judge_conf["disabled_ragas_metrics"] = disabled_metrics
        else:
            judge_conf.pop("disabled_ragas_metrics", None)

        run_scorer = self._build_run_scorer(active_metrics=effective_metrics)
        judge_conf["ragas_release_gate_thresholds"] = (
            self._serialize_release_thresholds(run_scorer.release_thresholds)
        )
        judge_evaluator = RagasJudgeEvaluator(metrics=effective_metrics)
        corpus_signature, corpus_payload = self._build_corpus_signature()
        runtime_snapshot = self._build_runtime_snapshot(corpus_payload=corpus_payload)
        config_snapshot = self._build_config_snapshot()

        run = BenchmarkRun.objects.create(
            dataset=dataset,
            split=split_value,
            status=BenchmarkRunStatus.RUNNING,
            runtime_snapshot=runtime_snapshot,
            config_snapshot=config_snapshot,
            corpus_signature=corpus_signature,
            judge_configuration=judge_conf,
            code_version=code_version.strip(),
            started_at=timezone.now(),
        )

        case_payloads: list[dict[str, Any]] = []
        shared_service = self._build_case_service(case=cases[0])
        try:
            pending_for_judge: list[dict[str, Any]] = []
            for case in cases:
                runtime_entry = self._execute_case_runtime(
                    case=case,
                    service=shared_service,
                )
                if runtime_entry.get("runtime_error") is not None:
                    case_payloads.append(
                        self._persist_runtime_error_case(
                            run=run,
                            case=case,
                            base_payload=runtime_entry["base_payload"],
                            error=runtime_entry["runtime_error"],
                        )
                    )
                    continue
                pending_for_judge.append(runtime_entry)

            if pending_for_judge:
                judge_results = judge_evaluator.evaluate_batch(
                    cases=[
                        cast(BenchmarkCase, item["case"]) for item in pending_for_judge
                    ],
                    runtime_outputs=[
                        cast(dict[str, Any], item["runtime_output"])
                        for item in pending_for_judge
                    ],
                )
                if len(judge_results) < len(pending_for_judge):
                    judge_results.extend(
                        [
                            {
                                "enabled": True,
                                "available": False,
                                "error": "ragas_evaluation_failed: missing_batch_result",
                            }
                            for _ in range(len(pending_for_judge) - len(judge_results))
                        ]
                    )

                for item, judge_result in zip(pending_for_judge, judge_results):
                    case_payloads.append(
                        self._persist_scored_case(
                            run=run,
                            case=cast(BenchmarkCase, item["case"]),
                            base_payload=cast(dict[str, Any], item["base_payload"]),
                            runtime_output=cast(dict[str, Any], item["runtime_output"]),
                            judge_result=judge_result,
                            scorer=run_scorer,
                        )
                    )

            aggregate = run_scorer.aggregate_run(
                run=run,
                case_result_payloads=case_payloads,
            )
            self._persist_run_aggregate(run=run, aggregate=aggregate)
        except Exception as exc:
            run.status = BenchmarkRunStatus.FAILED
            run.error_summary = str(exc)
            run.completed_at = timezone.now()
            run.save(
                update_fields=[
                    "status",
                    "error_summary",
                    "completed_at",
                    "updated_at",
                ]
            )
            raise

        return run

    def _execute_case_runtime(
        self,
        *,
        case: BenchmarkCase,
        service: ChatbotBenchmarkService,
    ) -> dict[str, Any]:
        base_payload = {
            "question": str(case.question),
            "scenario": str(case.scenario),
            "expected_mode": str(case.expected_mode),
            "expected_behavior": str(case.expected_behavior),
            "input_payload": {
                "question": str(case.question),
                "intake_payload": (
                    case.intake_payload if isinstance(case.intake_payload, dict) else {}
                ),
                "scenario": str(case.scenario),
            },
        }

        try:
            runtime_output = service.run_benchmark_case(
                question=str(case.question),
                intake_payload=(
                    case.intake_payload if isinstance(case.intake_payload, dict) else {}
                ),
            )
            return {
                "case": case,
                "base_payload": base_payload,
                "runtime_output": runtime_output,
                "runtime_error": None,
            }
        except Exception as exc:
            return {
                "case": case,
                "base_payload": base_payload,
                "runtime_output": {},
                "runtime_error": exc,
            }

    def _persist_scored_case(
        self,
        *,
        run: BenchmarkRun,
        case: BenchmarkCase,
        base_payload: dict[str, Any],
        runtime_output: dict[str, Any],
        judge_result: dict[str, Any],
        scorer: RagasBenchmarkScorer,
    ) -> dict[str, Any]:
        runtime_output["judge_output"] = judge_result
        case_score = scorer.score_case(
            case=case,
            runtime_output=runtime_output,
            judge_result=judge_result,
        )
        merged_metrics = dict(case_score.metrics)

        status = (
            BenchmarkCaseResultStatus.PASSED
            if case_score.pass_flags.get("primary_pass", False)
            else BenchmarkCaseResultStatus.FAILED
        )
        case_result = BenchmarkCaseResult.objects.create(
            run=run,
            case=case,
            status=status,
            input_payload=base_payload["input_payload"],
            analysis_artifact=runtime_output.get("analysis_output", {}),
            gate_artifact=runtime_output.get("gate_output", {}),
            retrieval_artifact=runtime_output.get("retrieval_output", {}),
            generation_artifact=runtime_output.get("generation_output", {}),
            audit_metadata=runtime_output.get("audit_metadata", {}),
            timing_ms=runtime_output.get("timings_ms", {}),
            metrics=merged_metrics,
            pass_flags=case_score.pass_flags,
            source_urls=(
                runtime_output.get("generation_output", {}).get("source_urls", [])
                if isinstance(runtime_output.get("generation_output"), dict)
                else []
            ),
            final_answer=str(
                (runtime_output.get("generation_output", {}) or {}).get(
                    "final_answer", ""
                )
            ).strip(),
            error_payload={},
        )
        self._persist_case_metric_rows(
            run=run,
            case_result=case_result,
            metrics=merged_metrics,
        )
        return {
            **base_payload,
            "metrics": merged_metrics,
            "pass_flags": case_score.pass_flags,
            "gate_artifact": runtime_output.get("gate_output", {}),
            "judge_artifact": judge_result,
        }

    def _persist_runtime_error_case(
        self,
        *,
        run: BenchmarkRun,
        case: BenchmarkCase,
        base_payload: dict[str, Any],
        error: Exception,
    ) -> dict[str, Any]:
        case_result = BenchmarkCaseResult.objects.create(
            run=run,
            case=case,
            status=BenchmarkCaseResultStatus.ERROR,
            input_payload=base_payload["input_payload"],
            analysis_artifact={},
            gate_artifact={},
            retrieval_artifact={},
            generation_artifact={},
            audit_metadata={},
            timing_ms={},
            metrics={"runtime_error_rate": 1.0, "judge_error_rate": 1.0},
            pass_flags={
                "judge_available": False,
                "judge_error": True,
                "primary_pass": False,
            },
            source_urls=[],
            final_answer="",
            error_payload={"message": str(error)},
        )
        self._persist_case_metric_rows(
            run=run,
            case_result=case_result,
            metrics={"runtime_error_rate": 1.0, "judge_error_rate": 1.0},
        )
        return {
            **base_payload,
            "metrics": {"runtime_error_rate": 1.0, "judge_error_rate": 1.0},
            "pass_flags": {
                "judge_available": False,
                "judge_error": True,
                "primary_pass": False,
            },
            "gate_artifact": {},
        }

    def _normalize_metric_names(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        seen: set[str] = set()
        names: list[str] = []
        for item in value:
            name = str(item).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
        return names

    def _build_run_scorer(self, *, active_metrics: list[str]) -> RagasBenchmarkScorer:
        base_thresholds = getattr(self.scorer, "release_thresholds", None)
        if not isinstance(base_thresholds, dict) or not base_thresholds:
            base_thresholds = dict(RAGAS_RELEASE_GATE_THRESHOLDS)
        threshold_overrides = self._load_release_threshold_overrides_from_config()
        if threshold_overrides:
            merged_thresholds = dict(base_thresholds)
            merged_thresholds.update(threshold_overrides)
            base_thresholds = merged_thresholds
        return RagasBenchmarkScorer(
            release_thresholds=dict(base_thresholds),
            active_metrics=active_metrics,
        )

    def _load_release_threshold_overrides_from_config(
        self,
    ) -> dict[str, tuple[str, float]]:
        raw = ChatbotConfig.get_config("RAGAS_RELEASE_GATE_THRESHOLDS", None)
        if not isinstance(raw, dict):
            return {}

        overrides: dict[str, tuple[str, float]] = {}
        for metric_name, default_pair in RAGAS_RELEASE_GATE_THRESHOLDS.items():
            if metric_name not in raw:
                continue
            parsed = self._parse_release_threshold_value(
                value=raw.get(metric_name),
                fallback=default_pair,
            )
            if parsed is not None:
                overrides[metric_name] = parsed
        return overrides

    def _parse_release_threshold_value(
        self,
        *,
        value: Any,
        fallback: tuple[str, float],
    ) -> tuple[str, float] | None:
        operator: str | None = None
        threshold_raw: Any = None

        if isinstance(value, (list, tuple)) and len(value) >= 2:
            operator = str(value[0]).strip()
            threshold_raw = value[1]
        elif isinstance(value, dict):
            operator = str(value.get("operator", "")).strip()
            threshold_raw = value.get("threshold")
        else:
            return None

        if operator not in {">=", "<="}:
            operator = fallback[0]

        try:
            threshold = float(threshold_raw)
        except (TypeError, ValueError):
            threshold = fallback[1]
        if not math.isfinite(threshold):
            threshold = fallback[1]
        return (operator, threshold)

    def _serialize_release_thresholds(
        self,
        thresholds: dict[str, tuple[str, float]],
    ) -> dict[str, list[Any]]:
        serialized: dict[str, list[Any]] = {}
        for metric_name, pair in (thresholds or {}).items():
            operator, threshold = pair
            serialized[str(metric_name)] = [str(operator), float(threshold)]
        return serialized

    def _build_case_service(self, *, case: BenchmarkCase) -> ChatbotBenchmarkService:
        customer = self._get_or_create_runner_customer()
        ChatSession.objects.filter(customer=customer, is_active=True).update(
            is_active=False
        )
        session = ChatSession.objects.create(
            customer=customer,
            title=f"Benchmark case {case.case_id}",
            is_active=True,
        )
        return ChatbotBenchmarkService(session)

    def _persist_run_aggregate(
        self,
        *,
        run: BenchmarkRun,
        aggregate: dict[str, Any],
    ) -> None:
        with transaction.atomic():
            run.total_cases = int(aggregate.get("total_cases", 0))
            run.passed_cases = int(aggregate.get("passed_cases", 0))
            run.failed_cases = int(aggregate.get("failed_cases", 0))
            run.summary_metrics = aggregate.get("summary_metrics", {})
            run.failure_slices = aggregate.get("failure_slices", {})
            run.release_gate = aggregate.get("release_gate", {})
            run.status = BenchmarkRunStatus.COMPLETED
            run.completed_at = timezone.now()
            run.save(
                update_fields=[
                    "total_cases",
                    "passed_cases",
                    "failed_cases",
                    "summary_metrics",
                    "failure_slices",
                    "release_gate",
                    "status",
                    "completed_at",
                    "updated_at",
                ]
            )

            metric_rows = aggregate.get("metric_rows", []) or []
            BenchmarkMetric.objects.bulk_create(
                [
                    BenchmarkMetric(
                        run=item["run"],
                        metric_name=item["metric_name"],
                        metric_scope=item["metric_scope"],
                        slice_key=item.get("slice_key", ""),
                        value=float(item.get("value", 0.0)),
                        threshold=(
                            float(item["threshold"])
                            if item.get("threshold") is not None
                            else None
                        ),
                        is_pass=item.get("is_pass"),
                        details=item.get("details", {}),
                    )
                    for item in metric_rows
                ],
                batch_size=200,
            )

    def _persist_case_metric_rows(
        self,
        *,
        run: BenchmarkRun,
        case_result: BenchmarkCaseResult,
        metrics: dict[str, Any],
    ) -> None:
        rows = []
        for metric_name, value in (metrics or {}).items():
            if isinstance(value, (int, float)):
                scope = (
                    BenchmarkMetricScope.JUDGE
                    if str(metric_name).startswith("judge_")
                    or str(metric_name) in RAGAS_JUDGE_METRICS
                    else BenchmarkMetricScope.CASE
                )
                rows.append(
                    BenchmarkMetric(
                        run=run,
                        case_result=case_result,
                        metric_name=str(metric_name),
                        metric_scope=scope,
                        value=float(value),
                        details={},
                    )
                )
        if rows:
            BenchmarkMetric.objects.bulk_create(rows, batch_size=200)

    def _validate_dataset_before_run(self, dataset: BenchmarkDataset) -> None:
        split_counts = (
            dataset.split_counts if isinstance(dataset.split_counts, dict) else {}
        )
        missing_splits = [
            split
            for split in (BenchmarkSplit.DEV, BenchmarkSplit.TEST)
            if int(split_counts.get(split, 0)) <= 0
        ]
        if missing_splits:
            joined = ", ".join(missing_splits)
            raise ValueError(
                f"Dataset composition check failed. Missing split counts for: {joined}"
            )

    def _build_runtime_snapshot(
        self, *, corpus_payload: dict[str, Any]
    ) -> dict[str, Any]:
        temperature_raw = ChatbotConfig.get_config("TEMPERATURE", 0.3)
        try:
            temperature = (
                float(str(temperature_raw).strip())
                if temperature_raw is not None
                else 0.3
            )
        except (TypeError, ValueError):
            temperature = 0.3

        return {
            "llm_provider": str(ChatbotConfig.get_config("LLM_PROVIDER", "gemini")),
            "llm_model": str(ChatbotConfig.get_config("LLM_MODEL", "gemini-2.5-flash")),
            "temperature": temperature,
            "embedding_provider": str(EmbeddingService.resolve_provider()),
            "benchmark_timestamp": timezone.now().isoformat(),
            "corpus_snapshot": corpus_payload,
        }

    def _build_config_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        keys = (
            "RAG_THRESH_C",
            "RAG_B_TOPK",
            "RAG_TITLE_TOP_M",
            "RAG_FINAL_TITLES",
            "RAG_PENALTY_ALPHA",
            "RAG_NEG_SYM_SIM_THRESH",
            "RAGAS_RELEASE_GATE_THRESHOLDS",
            "BENCHMARK_OUTPUT_LANGUAGE",
            "LLM_MODEL",
            "TEMPERATURE",
        )
        for key in keys:
            snapshot[key] = ChatbotConfig.get_config(key, None)
        return snapshot

    def _build_corpus_signature(self) -> tuple[str, dict[str, Any]]:
        index_counts = {
            row["index_type"]: int(row["total"])
            for row in MedicalDocument.objects.values("index_type").annotate(
                total=Count("id")
            )
        }
        latest = MedicalDocument.objects.aggregate(
            latest_updated=Max("updated_at"),
            latest_reembedded=Max("last_reembedded_at"),
        )
        payload = {
            "index_counts": index_counts,
            "embedding_provider": str(EmbeddingService.resolve_provider()),
            "latest_updated": (
                latest["latest_updated"].isoformat()
                if latest.get("latest_updated") is not None
                else ""
            ),
            "latest_reembedded": (
                latest["latest_reembedded"].isoformat()
                if latest.get("latest_reembedded") is not None
                else ""
            ),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return digest, payload

    def _get_or_create_runner_customer(self):
        User = get_user_model()
        customer, _ = User.objects.get_or_create(
            username=BENCHMARK_RUNNER_USERNAME,
            defaults={
                "email": "benchmark-runner@local",
                "is_active": True,
                "is_staff": True,
            },
        )
        return customer
