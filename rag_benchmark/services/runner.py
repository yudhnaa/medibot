from __future__ import annotations

import hashlib
import json
from typing import Any

from django.contrib.auth import get_user_model
from django.db import transaction
from django.db.models import Count, Max
from django.utils import timezone

from chatbot.models import ChatSession, ChatbotConfig, MedicalDocument
from chatbot.services.chatbot_service import ChatbotService
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
from rag_benchmark.services.ragas import RagasJudgeEvaluator
from rag_benchmark.services.scoring import DeterministicBenchmarkScorer
from vector_store.services.embedding_service import EmbeddingService

BENCHMARK_RUNNER_USERNAME = "__benchmark_runner__"


class OfflineBenchmarkRunner:
    """Execute benchmark cases directly against runtime services (no HTTP API)."""

    def __init__(self, *, scorer: DeterministicBenchmarkScorer | None = None) -> None:
        self.scorer = scorer or DeterministicBenchmarkScorer()

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
            dataset.cases.filter(split=split_value, is_active=True).order_by("id")
        )
        if not cases:
            raise ValueError(
                f"No active benchmark cases available for split `{split_value}`"
            )

        judge_conf = judge_configuration or {}
        judge_evaluator = RagasJudgeEvaluator(
            enabled=bool(judge_conf.get("enable_ragas", False)),
            metrics=judge_conf.get("ragas_metrics"),
        )
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
        try:
            for case in cases:
                result_payload = self._execute_case(
                    run=run,
                    case=case,
                    judge_evaluator=judge_evaluator,
                )
                case_payloads.append(result_payload)

            aggregate = self.scorer.aggregate_run(
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

    def _execute_case(
        self,
        *,
        run: BenchmarkRun,
        case: BenchmarkCase,
        judge_evaluator: RagasJudgeEvaluator,
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
            service = self._build_case_service(case=case)
            runtime_output = service.run_benchmark_case(
                question=str(case.question),
                intake_payload=(
                    case.intake_payload if isinstance(case.intake_payload, dict) else {}
                ),
            )
            case_score = self.scorer.score_case(
                case=case, runtime_output=runtime_output
            )
            judge_result = judge_evaluator.evaluate_case(
                case=case,
                runtime_output=runtime_output,
            )

            merged_metrics = dict(case_score.metrics)
            judge_scores = (judge_result or {}).get("scores", {})
            if isinstance(judge_scores, dict):
                for score_name, value in judge_scores.items():
                    if isinstance(value, (int, float)):
                        merged_metrics[f"judge_{score_name}"] = float(value)

            if judge_result:
                runtime_output["judge_output"] = judge_result

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

            payload = {
                **base_payload,
                "metrics": merged_metrics,
                "pass_flags": case_score.pass_flags,
                "gate_artifact": runtime_output.get("gate_output", {}),
            }
            return payload
        except Exception as exc:
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
                metrics={"runtime_error_rate": 1.0},
                pass_flags={"primary_pass": False},
                source_urls=[],
                final_answer="",
                error_payload={"message": str(exc)},
            )
            self._persist_case_metric_rows(
                run=run,
                case_result=case_result,
                metrics={"runtime_error_rate": 1.0},
            )
            return {
                **base_payload,
                "metrics": {"runtime_error_rate": 1.0},
                "pass_flags": {"primary_pass": False},
                "gate_artifact": {},
            }

    def _build_case_service(self, *, case: BenchmarkCase) -> ChatbotService:
        customer = self._get_or_create_runner_customer()
        ChatSession.objects.filter(customer=customer, is_active=True).update(
            is_active=False
        )
        session = ChatSession.objects.create(
            customer=customer,
            title=f"Benchmark case {case.case_id}",
            is_active=True,
        )
        service = ChatbotService(session)
        self._reset_service_intake(service=service, intake_payload=case.intake_payload)
        return service

    def _reset_service_intake(
        self,
        *,
        service: ChatbotService,
        intake_payload: dict[str, Any] | None,
    ) -> None:
        service._user_intake_db.reset_session_specific_fields()  # pyright: ignore[reportPrivateUsage]
        payload = intake_payload if isinstance(intake_payload, dict) else {}
        if not payload:
            return

        updates = {
            "disease_name": payload.get("disease_name"),
            "age": payload.get("age"),
            "sex": payload.get("sex"),
            "symptoms": payload.get("symptoms", []),
            "symptoms_negated": payload.get("symptoms_negated", []),
            "onset_days": payload.get("onset_days"),
            "pregnancy_status": payload.get("pregnancy_status"),
            "location_country": payload.get("location_country"),
            "chronic_conditions": payload.get("chronic_conditions", []),
            "allergies": payload.get("allergies", []),
            "meds": payload.get("meds", []),
        }
        safe_updates = {
            key: value for key, value in updates.items() if value is not None
        }
        if safe_updates:
            service.update_intake(**safe_updates)

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
                "Dataset composition check failed. Missing split counts for: "
                f"{joined}"
            )

    def _build_runtime_snapshot(
        self, *, corpus_payload: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "llm_provider": str(ChatbotConfig.get_config("LLM_PROVIDER", "gemini")),
            "llm_model": str(ChatbotConfig.get_config("LLM_MODEL", "gemini-2.5-flash")),
            "temperature": float(ChatbotConfig.get_config("TEMPERATURE", 0.3) or 0.3),
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
