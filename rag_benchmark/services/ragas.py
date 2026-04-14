from __future__ import annotations

from typing import Any

from rag_benchmark.models import BenchmarkCase

DEFAULT_RAGAS_METRICS = (
    "faithfulness",
    "response_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)


class RagasJudgeEvaluator:
    """
    Optional Ragas-based judge layer.

    This layer is explicitly supplemental and must not replace deterministic release gates.
    """

    def __init__(self, *, enabled: bool, metrics: list[str] | None = None) -> None:
        self.enabled = enabled
        self.metric_names = metrics or list(DEFAULT_RAGAS_METRICS)

    def evaluate_case(
        self,
        *,
        case: BenchmarkCase,
        runtime_output: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}

        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_correctness,
                context_precision,
                context_recall,
                faithfulness,
                response_relevancy,
            )
        except Exception as exc:
            return {
                "enabled": True,
                "available": False,
                "error": f"ragas_unavailable: {exc}",
            }

        metric_registry = {
            "faithfulness": faithfulness,
            "response_relevancy": response_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_correctness": answer_correctness,
        }
        selected_metrics = [
            metric_registry[name]
            for name in self.metric_names
            if name in metric_registry
        ]
        if not selected_metrics:
            return {
                "enabled": True,
                "available": True,
                "scores": {},
                "warning": "No valid ragas metrics configured",
            }

        generation_output = runtime_output.get("generation_output", {}) or {}
        retrieval_output = runtime_output.get("retrieval_output", {}) or {}
        contexts = retrieval_output.get("retrieved_context_texts", []) or []
        answer_text = str(generation_output.get("final_answer", "")).strip()

        dataset = Dataset.from_dict(
            {
                "question": [str(case.question).strip()],
                "answer": [answer_text],
                "contexts": [contexts],
                "ground_truth": [str(case.reference_answer or "").strip()],
            }
        )

        try:
            result = evaluate(dataset=dataset, metrics=selected_metrics)
            score_payload = result.to_pandas().to_dict(orient="records")
        except Exception as exc:
            return {
                "enabled": True,
                "available": True,
                "error": f"ragas_evaluation_failed: {exc}",
            }

        scores: dict[str, float] = {}
        if score_payload:
            first = score_payload[0]
            for key, value in first.items():
                if isinstance(value, (int, float)):
                    scores[str(key)] = float(value)

        return {
            "enabled": True,
            "available": True,
            "scores": scores,
        }
