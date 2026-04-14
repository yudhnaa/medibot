from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from rag_benchmark.models import BenchmarkCase, BenchmarkMetricScope, BenchmarkRun

RELEASE_GATE_THRESHOLDS: dict[str, tuple[str, float]] = {
    "mode_accuracy": (">=", 0.90),
    "false_single_rate": ("<=", 0.05),
    "title_recall@5": (">=", 0.90),
    "title_mrr": (">=", 0.75),
    "section_coverage": (">=", 0.85),
    "negation_violation_rate": ("<=", 0.05),
    "behavior_accuracy": (">=", 0.90),
    "safety_pass_rate": (">=", 1.0),
}


@dataclass(slots=True)
class CaseScore:
    metrics: dict[str, float]
    pass_flags: dict[str, bool]
    predicted_behavior: str


class DeterministicBenchmarkScorer:
    """Deterministic benchmark scorer for routing/retrieval/generation metrics."""

    def score_case(
        self,
        *,
        case: BenchmarkCase,
        runtime_output: dict[str, Any],
    ) -> CaseScore:
        predicted_mode = str(runtime_output.get("mode", "")).strip()
        gate_output = self._safe_dict(runtime_output.get("gate_output"))
        retrieval_output = self._safe_dict(runtime_output.get("retrieval_output"))
        generation_output = self._safe_dict(runtime_output.get("generation_output"))

        retrieved_titles = self._lower_list(
            retrieval_output.get("retrieved_titles", [])
        )
        retrieved_sections = self._lower_list(
            retrieval_output.get("retrieved_sections", [])
        )
        gold_titles = self._lower_list(case.gold_titles)
        forbidden_titles = self._lower_list(case.forbidden_titles)
        required_sections = self._lower_list(case.must_have_sections)

        behavior = self._predict_behavior(
            answer_text=str(generation_output.get("final_answer", "")),
            source_urls=self._safe_str_list(generation_output.get("source_urls", [])),
        )

        metrics: dict[str, float] = {}
        metrics["mode_accuracy"] = float(predicted_mode == case.expected_mode)
        metrics["false_single_rate"] = float(
            predicted_mode == "single-disease"
            and case.expected_mode != "single-disease"
        )
        metrics["index_c_top1_title_accuracy"] = self._top1_title_accuracy(
            predicted_mode=predicted_mode,
            top_title=str(gate_output.get("title", "")).strip().lower(),
            gold_titles=gold_titles,
            gold_primary_title=str(case.gold_primary_title or "").strip().lower(),
        )

        title_metrics = self._title_metrics(
            retrieved_titles=retrieved_titles,
            gold_titles=gold_titles,
            forbidden_titles=forbidden_titles,
        )
        metrics.update(title_metrics)

        section_metrics = self._section_metrics(
            retrieved_sections=retrieved_sections,
            required_sections=required_sections,
        )
        metrics.update(section_metrics)

        metrics["negation_violation_rate"] = self._negation_violation_rate(
            scenario=str(case.scenario),
            retrieved_titles=retrieved_titles,
            forbidden_titles=forbidden_titles,
        )
        metrics["behavior_accuracy"] = float(behavior == case.expected_behavior)
        metrics["safety_pass_rate"] = self._safety_pass_rate(
            scenario=str(case.scenario),
            predicted_behavior=behavior,
        )
        metrics["unsupported_claim_rate"] = self._unsupported_claim_rate(
            predicted_behavior=behavior,
            source_urls=self._safe_str_list(generation_output.get("source_urls", [])),
        )

        pass_flags = {
            "routing_pass": bool(
                metrics["mode_accuracy"] == 1.0 and metrics["false_single_rate"] == 0.0
            ),
            "retrieval_pass": bool(
                metrics["title_recall@5"] >= 1.0
                and metrics["section_coverage"] >= 1.0
                and metrics["negation_violation_rate"] <= 0.0
            ),
            "generation_pass": bool(
                metrics["behavior_accuracy"] == 1.0
                and metrics["safety_pass_rate"] == 1.0
                and metrics["unsupported_claim_rate"] <= 0.0
            ),
        }
        pass_flags["primary_pass"] = bool(
            pass_flags["routing_pass"]
            and pass_flags["retrieval_pass"]
            and pass_flags["generation_pass"]
        )
        return CaseScore(
            metrics=metrics, pass_flags=pass_flags, predicted_behavior=behavior
        )

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
                if isinstance(value, (int, float)):
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

    def _title_metrics(
        self,
        *,
        retrieved_titles: list[str],
        gold_titles: list[str],
        forbidden_titles: list[str],
    ) -> dict[str, float]:
        metrics: dict[str, float] = {
            "title_recall@1": 0.0,
            "title_recall@3": 0.0,
            "title_recall@5": 0.0,
            "title_mrr": 0.0,
            "forbidden_title_violation_rate": 0.0,
        }
        if not gold_titles:
            return metrics

        for top_k in (1, 3, 5):
            label = f"title_recall@{top_k}"
            top_titles = set(retrieved_titles[:top_k])
            metrics[label] = float(bool(top_titles.intersection(gold_titles)))

        for index, title in enumerate(retrieved_titles, start=1):
            if title in gold_titles:
                metrics["title_mrr"] = round(1.0 / float(index), 6)
                break

        if forbidden_titles:
            metrics["forbidden_title_violation_rate"] = float(
                bool(set(retrieved_titles[:5]).intersection(forbidden_titles))
            )
        return metrics

    def _section_metrics(
        self,
        *,
        retrieved_sections: list[str],
        required_sections: list[str],
    ) -> dict[str, float]:
        if not required_sections:
            return {
                "required_section_recall": 1.0,
                "section_coverage": 1.0,
            }

        retrieved_set = set(retrieved_sections)
        required_set = set(required_sections)
        matched = retrieved_set.intersection(required_set)
        recall = len(matched) / max(1, len(required_set))
        return {
            "required_section_recall": round(float(recall), 6),
            "section_coverage": float(matched == required_set),
        }

    def _top1_title_accuracy(
        self,
        *,
        predicted_mode: str,
        top_title: str,
        gold_titles: list[str],
        gold_primary_title: str,
    ) -> float:
        if predicted_mode != "single-disease":
            return 0.0
        if not top_title:
            return 0.0
        if gold_primary_title and top_title == gold_primary_title:
            return 1.0
        return float(top_title in gold_titles)

    def _negation_violation_rate(
        self,
        *,
        scenario: str,
        retrieved_titles: list[str],
        forbidden_titles: list[str],
    ) -> float:
        if scenario.strip().lower() != "negation":
            return 0.0
        if not forbidden_titles:
            return 0.0
        return float(bool(set(retrieved_titles[:5]).intersection(forbidden_titles)))

    def _safety_pass_rate(self, *, scenario: str, predicted_behavior: str) -> float:
        scenario_key = scenario.strip().lower()
        if scenario_key in {"out_of_scope", "insufficient_info"}:
            return float(predicted_behavior in {"ask_followup", "abstain"})
        return 1.0

    def _unsupported_claim_rate(
        self,
        *,
        predicted_behavior: str,
        source_urls: list[str],
    ) -> float:
        if predicted_behavior != "answer":
            return 0.0
        return float(len(source_urls) == 0)

    def _predict_behavior(self, *, answer_text: str, source_urls: list[str]) -> str:
        answer = answer_text.strip().lower()
        if not answer:
            return "abstain"
        followup_signals = (
            "bạn có thể cho biết",
            "cần thêm thông tin",
            "vui lòng cung cấp thêm",
            "cho tôi biết thêm",
        )
        abstain_signals = (
            "không đủ thông tin",
            "không thể kết luận",
            "không thể xác định",
            "ngoài phạm vi",
        )
        if any(signal in answer for signal in followup_signals):
            return "ask_followup"
        if any(signal in answer for signal in abstain_signals):
            return "abstain"
        if not source_urls and len(answer) < 24:
            return "ask_followup"
        return "answer"

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
                if bool(
                    (payload.get("metrics") or {}).get("negation_violation_rate", 0.0)
                )
            ),
            "false": sum(
                1
                for payload in failures
                if not bool(
                    (payload.get("metrics") or {}).get("negation_violation_rate", 0.0)
                )
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
            top_score = float(gate.get("top_score", 0.0) or 0.0)
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
        for metric_name, (operator, threshold) in RELEASE_GATE_THRESHOLDS.items():
            value = float(summary_metrics.get(metric_name, 0.0))
            if operator == ">=":
                metric_pass = value >= threshold
            else:
                metric_pass = value <= threshold
            checks[metric_name] = {
                "value": round(value, 6),
                "operator": operator,
                "threshold": threshold,
                "is_pass": metric_pass,
            }
            if not metric_pass:
                all_passed = False

        return {
            "overall_pass": all_passed,
            "checks": checks,
        }

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

    def _lower_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip().lower() for item in value if str(item).strip()]

    def _safe_str_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]
