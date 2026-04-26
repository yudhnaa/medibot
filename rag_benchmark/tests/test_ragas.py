from __future__ import annotations

from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase

from rag_benchmark.services.ragas import RagasJudgeEvaluator


class _FakeDataFrame:
    def __init__(self, rows):
        self._rows = rows

    def to_dict(self, orient: str = "records"):
        if orient != "records":
            raise ValueError("unsupported orient")
        return self._rows


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def to_pandas(self):
        return _FakeDataFrame(self._rows)


class _FakeMetric:
    def __init__(self, name: str):
        self.name = name
        self.strictness = 1


class RagasJudgeEvaluatorTests(SimpleTestCase):
    def _build_fake_modules(self):
        datasets_module = ModuleType("datasets")

        class _Dataset:
            @classmethod
            def from_dict(cls, payload):
                return payload

        datasets_module.Dataset = _Dataset

        ragas_module = ModuleType("ragas")
        ragas_module.aevaluate = object()

        metrics_module = ModuleType("ragas.metrics")
        metrics_module.faithfulness = _FakeMetric("faithfulness")
        metrics_module.answer_relevancy = _FakeMetric("answer_relevancy")
        metrics_module.context_precision = _FakeMetric("context_precision")
        metrics_module.context_recall = _FakeMetric("context_recall")

        run_config_module = ModuleType("ragas.run_config")

        class _RunConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        run_config_module.RunConfig = _RunConfig

        return {
            "datasets": datasets_module,
            "ragas": ragas_module,
            "ragas.metrics": metrics_module,
            "ragas.run_config": run_config_module,
        }

    def test_evaluate_batch_falls_back_single_case_when_metric_missing(self):
        evaluator = RagasJudgeEvaluator(
            metrics=["answer_relevancy", "context_recall"]
        )
        fake_modules = self._build_fake_modules()
        run_calls: list[dict] = []

        def _run_side_effect(**kwargs):
            run_calls.append(kwargs)
            if len(run_calls) == 1:
                return _FakeResult(
                    [
                        {"answer_relevancy": 0.812345},
                        {
                            "answer_relevancy": 0.765432,
                            "context_recall": 0.887766,
                        },
                    ]
                )
            return _FakeResult(
                [{"answer_relevancy": 0.812345, "context_recall": 0.923456}]
            )

        cases = [
            SimpleNamespace(question="What causes COVID-19?", reference_answer="R1"),
            SimpleNamespace(question="Is fever a symptom?", reference_answer="R2"),
        ]
        runtime_outputs = [
            {
                "generation_output": {"final_answer": "A1"},
                "retrieval_output": {"retrieved_context_texts": ["ctx one", "ctx two"]},
            },
            {
                "generation_output": {"final_answer": "A2"},
                "retrieval_output": {"retrieved_context_texts": ["ctx alpha"]},
            },
        ]

        with (
            patch.dict("sys.modules", fake_modules, clear=False),
            patch.object(
                evaluator,
                "_get_int_config",
                side_effect=lambda **kwargs: kwargs["default"],
            ),
            patch.object(
                evaluator,
                "_create_ragas_models",
                return_value=("llm", "embeddings"),
            ),
            patch.object(
                evaluator,
                "_run_ragas_evaluate",
                side_effect=_run_side_effect,
            ),
        ):
            results = evaluator.evaluate_batch(
                cases=cases,
                runtime_outputs=runtime_outputs,
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(run_calls[0]["batch_size"], 2)
        self.assertEqual(run_calls[1]["batch_size"], 1)
        self.assertEqual(results[0]["scores"]["answer_relevancy"], 0.812345)
        self.assertEqual(results[0]["scores"]["context_recall"], 0.923456)
        self.assertEqual(results[1]["scores"]["answer_relevancy"], 0.765432)
        self.assertEqual(results[1]["scores"]["context_recall"], 0.887766)

    def test_evaluate_batch_uses_generation_and_retrieval_context_sources(self):
        evaluator = RagasJudgeEvaluator(metrics=["faithfulness"])
        fake_modules = self._build_fake_modules()
        run_calls: list[dict] = []

        def _run_side_effect(**kwargs):
            run_calls.append(kwargs)
            return _FakeResult([{"faithfulness": 0.9}])

        cases = [SimpleNamespace(question="Is diabetes a risk factor?", reference_answer="R1")]
        runtime_outputs = [
            {
                "generation_output": {
                    "final_answer": "A1",
                    "context_snapshot": "Context snapshot from generation.",
                },
                "retrieval_output": {
                    "summaries": {
                        "coronavirus disease (covid-19)": "Summary support text."
                    },
                    "retrieved_context_texts": ["Risk evidence text."],
                },
            }
        ]

        with (
            patch.dict("sys.modules", fake_modules, clear=False),
            patch.object(
                evaluator,
                "_get_int_config",
                side_effect=lambda **kwargs: kwargs["default"],
            ),
            patch.object(
                evaluator,
                "_create_ragas_models",
                return_value=("llm", "embeddings"),
            ),
            patch.object(
                evaluator,
                "_run_ragas_evaluate",
                side_effect=_run_side_effect,
            ),
        ):
            results = evaluator.evaluate_batch(
                cases=cases,
                runtime_outputs=runtime_outputs,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(run_calls[0]["batch_size"], 1)
        retrieved_contexts = run_calls[0]["dataset"]["retrieved_contexts"][0]
        self.assertIn("Summary support text.", retrieved_contexts)
        self.assertIn("Risk evidence text.", retrieved_contexts)
        self.assertIn("Context snapshot from generation.", retrieved_contexts)
        self.assertEqual(retrieved_contexts[0], "Risk evidence text.")
        self.assertEqual(retrieved_contexts[-1], "Context snapshot from generation.")

    def test_evaluate_batch_prioritizes_retrieved_items_and_dedups_contexts(self):
        evaluator = RagasJudgeEvaluator(metrics=["faithfulness"])
        fake_modules = self._build_fake_modules()
        run_calls: list[dict] = []

        def _run_side_effect(**kwargs):
            run_calls.append(kwargs)
            return _FakeResult([{"faithfulness": 0.9}])

        cases = [SimpleNamespace(question="What are risk factors for COVID-19?", reference_answer="R1")]
        runtime_outputs = [
            {
                "generation_output": {
                    "final_answer": "A1",
                    "context_snapshot": "Broad context snapshot.",
                },
                "retrieval_output": {
                    "rerank": {
                        "intent": {
                            "target_sections": ["risk"],
                        }
                    },
                    "retrieved_items": [
                        {
                            "title": "coronavirus disease (covid-19)",
                            "section": "risk",
                            "content_preview": "Older people and diabetes are risk factors.",
                        },
                        {
                            "title": "coronavirus disease (covid-19)",
                            "section": "general",
                            "content_preview": "General overview text.",
                        },
                    ],
                    "retrieved_context_texts": [
                        "Older people and diabetes are risk factors.",
                        "General overview text.",
                    ],
                },
            }
        ]

        with (
            patch.dict("sys.modules", fake_modules, clear=False),
            patch.object(
                evaluator,
                "_get_int_config",
                side_effect=lambda **kwargs: kwargs["default"],
            ),
            patch.object(
                evaluator,
                "_create_ragas_models",
                return_value=("llm", "embeddings"),
            ),
            patch.object(
                evaluator,
                "_run_ragas_evaluate",
                side_effect=_run_side_effect,
            ),
        ):
            results = evaluator.evaluate_batch(
                cases=cases,
                runtime_outputs=runtime_outputs,
            )

        self.assertEqual(len(results), 1)
        retrieved_contexts = run_calls[0]["dataset"]["retrieved_contexts"][0]
        self.assertEqual(
            retrieved_contexts[0],
            "Older people and diabetes are risk factors.",
        )
        self.assertEqual(retrieved_contexts.count("Older people and diabetes are risk factors."), 1)
        self.assertEqual(retrieved_contexts[-1], "Broad context snapshot.")

    def test_evaluate_batch_uses_fallback_sources_when_structured_sources_missing(self):
        evaluator = RagasJudgeEvaluator(metrics=["faithfulness"])
        fake_modules = self._build_fake_modules()
        run_calls: list[dict] = []

        def _run_side_effect(**kwargs):
            run_calls.append(kwargs)
            return _FakeResult([{"faithfulness": 0.9}])

        cases = [SimpleNamespace(question="Is diabetes a risk factor?", reference_answer="R1")]
        runtime_outputs = [
            {
                "generation_output": {
                    "final_answer": "A1",
                    "context_snapshot": "Legacy snapshot.",
                },
                "retrieval_output": {
                    "summaries": {
                        "covid": "Legacy summary.",
                    },
                    "retrieved_context_texts": ["Legacy evidence."],
                    "retrieved_items": "invalid-shape",
                },
            }
        ]

        with (
            patch.dict("sys.modules", fake_modules, clear=False),
            patch.object(
                evaluator,
                "_get_int_config",
                side_effect=lambda **kwargs: kwargs["default"],
            ),
            patch.object(
                evaluator,
                "_create_ragas_models",
                return_value=("llm", "embeddings"),
            ),
            patch.object(
                evaluator,
                "_run_ragas_evaluate",
                side_effect=_run_side_effect,
            ),
        ):
            results = evaluator.evaluate_batch(
                cases=cases,
                runtime_outputs=runtime_outputs,
            )

        self.assertEqual(len(results), 1)
        retrieved_contexts = run_calls[0]["dataset"]["retrieved_contexts"][0]
        self.assertEqual(retrieved_contexts[0], "Legacy evidence.")
        self.assertIn("Legacy summary.", retrieved_contexts)
        self.assertEqual(retrieved_contexts[-1], "Legacy snapshot.")

    def test_evaluate_batch_falls_back_to_legacy_order_when_collector_raises(self):
        evaluator = RagasJudgeEvaluator(metrics=["faithfulness"])
        fake_modules = self._build_fake_modules()
        run_calls: list[dict] = []

        def _run_side_effect(**kwargs):
            run_calls.append(kwargs)
            return _FakeResult([{"faithfulness": 0.9}])

        cases = [SimpleNamespace(question="Is diabetes a risk factor?", reference_answer="R1")]
        runtime_outputs = [
            {
                "generation_output": {
                    "final_answer": "A1",
                    "context_snapshot": "Legacy snapshot.",
                },
                "retrieval_output": {
                    "summaries": {"covid": "Legacy summary."},
                    "retrieved_context_texts": ["Legacy evidence."],
                    "retrieved_items": [
                        {
                            "title": "coronavirus disease (covid-19)",
                            "section": "risk",
                            "content_preview": "Risk preview text.",
                        }
                    ],
                },
            }
        ]

        with (
            patch.dict("sys.modules", fake_modules, clear=False),
            patch.object(
                evaluator,
                "_get_int_config",
                side_effect=lambda **kwargs: kwargs["default"],
            ),
            patch.object(
                evaluator,
                "_create_ragas_models",
                return_value=("llm", "embeddings"),
            ),
            patch.object(
                evaluator,
                "_build_retrieved_item_candidates",
                side_effect=RuntimeError("collector boom"),
            ),
            patch.object(
                evaluator,
                "_run_ragas_evaluate",
                side_effect=_run_side_effect,
            ),
        ):
            results = evaluator.evaluate_batch(
                cases=cases,
                runtime_outputs=runtime_outputs,
            )

        self.assertEqual(len(results), 1)
        retrieved_contexts = run_calls[0]["dataset"]["retrieved_contexts"][0]
        self.assertEqual(retrieved_contexts[0], "Legacy snapshot.")
        self.assertEqual(retrieved_contexts[1], "Legacy summary.")
        self.assertEqual(retrieved_contexts[2], "Legacy evidence.")
        self.assertEqual(len(retrieved_contexts), 3)
