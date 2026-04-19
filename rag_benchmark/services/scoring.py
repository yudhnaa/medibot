from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from rag_benchmark.models import BenchmarkCase, BenchmarkMetricScope, BenchmarkRun
from rag_benchmark.services.constants import RAGAS_RELEASE_GATE_THRESHOLDS


@dataclass(slots=True)
class CaseScore:
    metrics: dict[str, float]
    pass_flags: dict[str, bool]


class RagasBenchmarkScorer:
    """Aggregate and gate benchmark runs using Ragas metrics only."""

    def __init__(
        self,
        *,
        release_thresholds: dict[str, tuple[str, float]] | None = None,
        active_metrics: list[str] | None = None,
    ) -> None:
        base_thresholds = release_thresholds or dict(
            RAGAS_RELEASE_GATE_THRESHOLDS
        )
        requested = self._normalize_metric_names(active_metrics)
        if requested:
            filtered = {
                name: base_thresholds[name]
                for name in requested
                if name in base_thresholds
            }
            self.release_thresholds = filtered or dict(base_thresholds)
        else:
            self.release_thresholds = dict(base_thresholds)
        self.active_metrics = list(self.release_thresholds.keys())

    def score_case(
        self,
        *,
        case: BenchmarkCase,
        runtime_output: dict[str, Any],
        judge_result: dict[str, Any] | None,
    ) -> CaseScore:
        del case
        del runtime_output
        judge_payload = self._safe_dict(judge_result)
        judge_scores = self._safe_dict(judge_payload.get("scores", {}))
        selected_metric_names = self._safe_str_list(
            judge_payload.get("selected_metrics")
        )
        if not selected_metric_names:
            selected_metric_names = list(self.active_metrics)

        metrics: dict[str, float] = {}
        for name in selected_metric_names:
            parsed = self._coerce_float(judge_scores.get(name))
            if parsed is not None:
                metrics[name] = parsed

        judge_available = bool(judge_payload.get("available", False))
        judge_error = bool(str(judge_payload.get("error", "")).strip())
        metrics["judge_available_rate"] = 1.0 if judge_available else 0.0
        metrics["judge_error_rate"] = 1.0 if judge_error else 0.0

        metric_thresholds = {
            key: threshold
            for key, threshold in self.release_thresholds.items()
            if key in selected_metric_names
        }
        if not metric_thresholds:
            metric_thresholds = {
                key: threshold
                for key, threshold in self.release_thresholds.items()
                if key in self.active_metrics
            }

        pass_flags: dict[str, bool] = {
            "judge_available": judge_available,
            "judge_error": judge_error,
        }
        for metric_name, (operator, threshold) in metric_thresholds.items():
            value = metrics.get(metric_name)
            pass_flags[f"{metric_name}_pass"] = self._is_pass(
                value=value,
                operator=operator,
                threshold=threshold,
            )

        required_flags = [
            pass_flags[f"{metric_name}_pass"] for metric_name in metric_thresholds
        ]
        pass_flags["primary_pass"] = bool(
            judge_available
            and not judge_error
            and required_flags
            and all(required_flags)
        )
        return CaseScore(metrics=metrics, pass_flags=pass_flags)

    def aggregate_run(
        self,
        *,
        run: BenchmarkRun,
        case_result_payloads: list[dict[str, Any]],
    ) -> dict[str, Any]:
        numeric_sums: defaultdict[str, float] = defaultdict(float)
        numeric_counts: defaultdict[str, int] = defaultdict(int)
        total_cases = len(case_result_payloads)
        passed_cases = sum(
            1
            for item in case_result_payloads
            if bool((item.get("pass_flags") or {}).get("primary_pass", False))
        )
        failed_cases = max(total_cases - passed_cases, 0)

        for payload in case_result_payloads:
            metrics = payload.get("metrics", {}) or {}
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    numeric_sums[metric_name] += float(value)
                    numeric_counts[metric_name] += 1

        summary_metrics: dict[str, float] = {}
        for metric_name, metric_sum in numeric_sums.items():
            count = max(1, numeric_counts[metric_name])
            summary_metrics[metric_name] = round(metric_sum / count, 6)

        if total_cases > 0:
            summary_metrics["primary_pass_rate"] = round(passed_cases / total_cases, 6)

        failure_slices = self._build_failure_slices(case_result_payloads)
        release_gate = self._evaluate_release_gate(summary_metrics)
        metric_rows = self._build_metric_rows(
            run=run,
            summary_metrics=summary_metrics,
            failure_slices=failure_slices,
            release_gate=release_gate,
        )
        return {
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "failed_cases": failed_cases,
            "summary_metrics": summary_metrics,
            "failure_slices": failure_slices,
            "release_gate": release_gate,
            "metric_rows": metric_rows,
        }

    def _build_failure_slices(
        self,
        case_result_payloads: list[dict[str, Any]],
    ) -> dict[str, Any]:
        failures = [
            payload
            for payload in case_result_payloads
            if not bool((payload.get("pass_flags") or {}).get("primary_pass", False))
        ]
        slices: dict[str, Any] = {}
        slices["scenario"] = dict(
            Counter(str(payload.get("scenario", "")) for payload in failures)
        )
        slices["expected_mode"] = dict(
            Counter(str(payload.get("expected_mode", "")) for payload in failures)
        )
        slices["has_negation"] = {
            "true": sum(
                1
                for payload in failures
                if "negation" in str(payload.get("scenario", "")).strip().lower()
            ),
            "false": sum(
                1
                for payload in failures
                if "negation" not in str(payload.get("scenario", "")).strip().lower()
            ),
        }
        slices["has_intake_context"] = {
            "true": sum(
                1
                for payload in failures
                if bool(payload.get("input_payload", {}).get("intake_payload"))
            ),
            "false": sum(
                1
                for payload in failures
                if not bool(payload.get("input_payload", {}).get("intake_payload"))
            ),
        }
        slices["judge_error"] = {
            "true": sum(
                1
                for payload in failures
                if bool((payload.get("pass_flags") or {}).get("judge_error", False))
            ),
            "false": sum(
                1
                for payload in failures
                if not bool((payload.get("pass_flags") or {}).get("judge_error", False))
            ),
        }
        slices["question_length"] = {
            "short": 0,
            "medium": 0,
            "long": 0,
        }
        for payload in failures:
            question = str(payload.get("question", "")).strip()
            words = len(question.split())
            if words <= 8:
                slices["question_length"]["short"] += 1
            elif words <= 20:
                slices["question_length"]["medium"] += 1
            else:
                slices["question_length"]["long"] += 1

        confidence_buckets = {"C_high": 0, "C_mid": 0, "C_low": 0}
        for payload in failures:
            gate = self._safe_dict(payload.get("gate_artifact", {}))
            top_score = self._coerce_float(gate.get("top_score"))
            if top_score is None:
                top_score = 0.0
            if top_score >= 0.85:
                confidence_buckets["C_high"] += 1
            elif top_score >= 0.75:
                confidence_buckets["C_mid"] += 1
            else:
                confidence_buckets["C_low"] += 1
        slices["route_confidence_bucket"] = confidence_buckets
        return slices

    def _evaluate_release_gate(
        self, summary_metrics: dict[str, float]
    ) -> dict[str, Any]:
        checks: dict[str, Any] = {}
        all_passed = True
        for metric_name, (operator, threshold) in self.release_thresholds.items():
            value = float(summary_metrics.get(metric_name, 0.0))
            metric_pass = self._is_pass(
                value=value,
                operator=operator,
                threshold=threshold,
            )
            checks[metric_name] = {
                "value": round(value, 6),
                "operator": operator,
                "threshold": threshold,
                "is_pass": metric_pass,
            }
            if not metric_pass:
                all_passed = False
        return {"overall_pass": all_passed, "checks": checks}

    def _build_metric_rows(
        self,
        *,
        run: BenchmarkRun,
        summary_metrics: dict[str, float],
        failure_slices: dict[str, Any],
        release_gate: dict[str, Any],
    ) -> list[dict[str, Any]]:
        metric_rows: list[dict[str, Any]] = []
        for name, value in summary_metrics.items():
            metric_rows.append(
                {
                    "run": run,
                    "metric_name": name,
                    "metric_scope": BenchmarkMetricScope.RUN,
                    "slice_key": "",
                    "value": float(value),
                    "threshold": None,
                    "is_pass": None,
                    "details": {},
                }
            )

        for scenario, count in (failure_slices.get("scenario") or {}).items():
            metric_rows.append(
                {
                    "run": run,
                    "metric_name": "failure_count",
                    "metric_scope": BenchmarkMetricScope.SLICE,
                    "slice_key": f"scenario:{scenario}",
                    "value": float(count),
                    "threshold": None,
                    "is_pass": None,
                    "details": {},
                }
            )

        checks = self._safe_dict(release_gate.get("checks", {}))
        for metric_name, check in checks.items():
            metric_rows.append(
                {
                    "run": run,
                    "metric_name": metric_name,
                    "metric_scope": BenchmarkMetricScope.RELEASE_GATE,
                    "slice_key": "",
                    "value": float(check.get("value", 0.0)),
                    "threshold": float(check.get("threshold", 0.0)),
                    "is_pass": bool(check.get("is_pass", False)),
                    "details": {"operator": check.get("operator", ">=")},
                }
            )
        return metric_rows

    def _safe_dict(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        return {}

    def _safe_str_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _normalize_metric_names(self, value: Any) -> list[str]:
        names = self._safe_str_list(value)
        seen: set[str] = set()
        normalized: list[str] = []
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            normalized.append(name)
        return normalized

    def _coerce_float(self, value: Any) -> float | None:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    def _is_pass(
        self,
        *,
        value: float | None,
        operator: str,
        threshold: float,
    ) -> bool:
        if value is None:
            return False
        if operator == ">=":
            return value >= threshold
        return value <= threshold


class DeterministicBenchmarkScorer(RagasBenchmarkScorer):
    """
    Backward-compatible alias.

    The benchmark pipeline now uses Ragas metrics for scoring.
    """
