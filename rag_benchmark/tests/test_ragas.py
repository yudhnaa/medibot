from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from rag_benchmark.models import BenchmarkCase
from rag_benchmark.services.ragas import RagasJudgeEvaluator


def _build_case(question: str, reference_answer: str) -> BenchmarkCase:
    return cast(
        BenchmarkCase,
        SimpleNamespace(question=question, reference_answer=reference_answer),
    )


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
        datasets_module = cast(Any, ModuleType("datasets"))

        class _Dataset:
            @classmethod
            def from_dict(cls, payload):
                return payload

        datasets_module.Dataset = _Dataset

        ragas_module = cast(Any, ModuleType("ragas"))
        ragas_module.aevaluate = object()

        metrics_module = cast(Any, ModuleType("ragas.metrics"))
        metrics_module.faithfulness = _FakeMetric("faithfulness")
        metrics_module.answer_relevancy = _FakeMetric("answer_relevancy")
        metrics_module.context_precision = _FakeMetric("context_precision")
        metrics_module.context_recall = _FakeMetric("context_recall")

        run_config_module = cast(Any, ModuleType("ragas.run_config"))

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
        evaluator = RagasJudgeEvaluator(metrics=["answer_relevancy", "context_recall"])
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
            _build_case("What causes COVID-19?", "R1"),
            _build_case("Is fever a symptom?", "R2"),
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

        cases = [_build_case("Is diabetes a risk factor?", "R1")]
        runtime_outputs = [
            {
                "generation_output": {
                    "final_answer": "A1",
                    "context_snapshot": "Context snapshot from generation.",
                },
                "retrieval_output": {
                    "summaries": {
                        "coronavirus disease (covid-19)": (
                            "coronavirus disease (covid-19)\n"
                            "general overview text.\n"
                            "Yếu tố nguy cơ: Summary support text."
                        )
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
        self.assertIn(
            "coronavirus disease (covid-19): Yếu tố nguy cơ: Summary support text.",
            retrieved_contexts,
        )
        self.assertNotIn("general overview text.", retrieved_contexts)
        self.assertIn("Risk evidence text.", retrieved_contexts)

    @patch("rag_benchmark.services.ragas.IS_RAGAS_FORMAT_SECTION_CONTEXT_ON", True)
    def test_evaluate_batch_prioritizes_retrieved_items_and_dedups_contexts(self):
        evaluator = RagasJudgeEvaluator(metrics=["faithfulness"])
        fake_modules = self._build_fake_modules()
        run_calls: list[dict] = []

        def _run_side_effect(**kwargs):
            run_calls.append(kwargs)
            return _FakeResult([{"faithfulness": 0.9}])

        cases = [_build_case("What are risk factors for COVID-19?", "R1")]
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
            "Risk factors include Older people and diabetes are risk factors.",
        )
        self.assertEqual(
            retrieved_contexts.count(
                "Risk factors include Older people and diabetes are risk factors."
            ),
            1,
        )
        self.assertEqual(
            retrieved_contexts.count("Older people and diabetes are risk factors."),
            0,
        )
        self.assertNotIn("Broad context snapshot.", retrieved_contexts)

    def test_build_judge_contexts_prioritizes_target_section_coverage(self):
        evaluator = RagasJudgeEvaluator(metrics=["faithfulness"])
        contexts = evaluator._build_judge_contexts(
            retrieval_output={
                "rerank": {
                    "intent": {
                        "target_sections": ["symptom", "aetiologies"],
                    }
                },
                "summaries": {
                    "coronavirus disease (covid-19)": (
                        "covid-19 is an infectious disease caused by the sars-cov-2 virus. "
                        "\nTriệu chứng: fever, cough, and tiredness."
                        "\nNguyên nhân: sars-cov-2 virus."
                    )
                },
                "retrieved_items": [
                    {
                        "title": "coronavirus disease (covid-19)",
                        "section": "symptom",
                        "content_preview": "fever cough tiredness",
                    },
                    {
                        "title": "coronavirus disease (covid-19)",
                        "section": "aetiologies",
                        "content_preview": "sars-cov-2 virus",
                    },
                ],
                "retrieved_context_texts": [
                    "fever cough tiredness",
                    "sars-cov-2 virus",
                ],
            },
            generation_output={"context_snapshot": "Broad context snapshot."},
            question_text="What are the main symptoms of COVID-19 and what causes this infectious disease?",
            default_top_k=4,
            char_limit=700,
        )

        self.assertGreaterEqual(len(contexts), 2)
        lowered = "\n".join(contexts).lower()
        self.assertIn("fever cough tiredness", lowered)
        self.assertIn("sars-cov-2 virus", lowered)
        self.assertIn("triệu chứng", lowered)
        self.assertIn("nguyên nhân", lowered)
        self.assertFalse(
            any(
                text.lower().startswith("coronavirus disease (covid-19) ")
                for text in contexts
            )
        )

    @patch("rag_benchmark.services.ragas.IS_RAGAS_FORMAT_SECTION_CONTEXT_ON", True)
    def test_build_judge_contexts_drops_redundant_general_summary_for_risk_queries(
        self,
    ):
        evaluator = RagasJudgeEvaluator(metrics=["faithfulness"])
        contexts = evaluator._build_judge_contexts(
            retrieval_output={
                "rerank": {
                    "intent": {
                        "target_sections": ["risk"],
                    }
                },
                "summaries": {
                    "coronavirus disease (covid-19)": (
                        "covid-19 is an infectious disease caused by the sars-cov-2 virus. "
                        "most people experience mild to moderate respiratory illness."
                    )
                },
                "retrieved_items": [
                    {
                        "title": "coronavirus disease (covid-19)",
                        "section": "risk",
                        "content_preview": (
                            "older people cardiovascular disease diabetes "
                            "chronic respiratory disease cancer"
                        ),
                    },
                    {
                        "title": "coronavirus disease (covid-19)",
                        "section": "general",
                        "content_preview": (
                            "covid-19 is an infectious disease caused by the "
                            "sars-cov-2 virus"
                        ),
                    },
                ],
                "retrieved_context_texts": [
                    "older people cardiovascular disease diabetes chronic respiratory disease cancer"
                ],
            },
            generation_output={"context_snapshot": "Broad context snapshot."},
            question_text="What are the risk factors for COVID-19, and is cancer one of them?",
            default_top_k=4,
            char_limit=700,
        )

        self.assertTrue(contexts)
        lowered = "\n".join(contexts).lower()
        self.assertIn("risk factors include", lowered)
        self.assertFalse(
            any(
                text.lower().startswith("coronavirus disease (covid-19) ")
                for text in contexts
            )
        )
        self.assertTrue(
            any(
                "covid-19 is an infectious disease caused by" in text.lower()
                for text in contexts
            )
        )
        self.assertLessEqual(len(contexts), 3)

    def test_detect_context_sections_maps_general_covid_context_to_risk_when_targeted(
        self,
    ):
        evaluator = RagasJudgeEvaluator(metrics=["faithfulness"])
        sections = evaluator._detect_context_sections(
            text=(
                "covid-19 is an infectious disease caused by the sars-cov-2 virus. "
                "most people experience mild to moderate respiratory illness."
            ),
            target_sections={"risk"},
        )

        self.assertEqual(sections, {"risk"})

    def test_detect_context_sections_keeps_aetiology_signal_for_symptom_cause_target(
        self,
    ):
        evaluator = RagasJudgeEvaluator(metrics=["faithfulness"])
        sections = evaluator._detect_context_sections(
            text=(
                "covid-19 is an infectious disease caused by the sars-cov-2 virus. "
                "most people experience mild to moderate respiratory illness."
            ),
            target_sections={"symptom", "aetiologies"},
        )

        self.assertIn("aetiologies", sections)
        self.assertIn("symptom", sections)

    def test_detect_context_sections_without_targets_uses_default_mapping(self):
        evaluator = RagasJudgeEvaluator(metrics=["faithfulness"])
        sections = evaluator._detect_context_sections(
            text="covid-19 is an infectious disease caused by the sars-cov-2 virus.",
            target_sections=set(),
        )

        self.assertEqual(sections, {"aetiologies", "symptom"})

    @patch("rag_benchmark.services.ragas.IS_RAGAS_FORMAT_SECTION_CONTEXT_ON", False)
    def test_format_section_context_text_can_be_disabled(self):
        evaluator = RagasJudgeEvaluator(metrics=["faithfulness"])

        self.assertEqual(
            evaluator._format_section_context_text(
                text="fever, cough, tiredness.",
                section="symptom",
            ),
            "fever, cough, tiredness.",
        )
        self.assertEqual(
            evaluator._format_section_context_text(
                text="sars-cov-2 virus",
                section="aetiologies",
            ),
            "sars-cov-2 virus",
        )

    @patch("rag_benchmark.services.ragas.IS_RAGAS_SUMMARY_CONTEXT_ON", False)
    def test_summary_context_candidates_can_be_disabled(self):
        evaluator = RagasJudgeEvaluator(metrics=["faithfulness"])

        candidates = evaluator._build_context_candidates(
            retrieval_output={
                "summaries": {"coronavirus disease (covid-19)": "summary support text"},
                "retrieved_items": [
                    {
                        "title": "coronavirus disease (covid-19)",
                        "section": "symptom",
                        "content_preview": "fever cough tiredness",
                    }
                ],
            },
            question_tokens={"fever"},
            target_sections={"symptom"},
        )

        self.assertEqual(
            [candidate.source for candidate in candidates],
            ["retrieved_items"],
        )

    @patch("rag_benchmark.services.ragas.IS_RAGAS_CONTEXT_SNAPSHOT_ON", False)
    def test_context_snapshot_fallback_can_be_disabled(self):
        evaluator = RagasJudgeEvaluator(metrics=["faithfulness"])
        contexts: list[str] = []

        evaluator._append_fallback_contexts(
            contexts=contexts,
            summaries={},
            context_snapshot="Context snapshot support text.",
        )

        self.assertEqual(contexts, [])

    def test_prepare_ragas_contexts_dedupes_prefixed_and_raw_duplicates(self):
        evaluator = RagasJudgeEvaluator(metrics=["faithfulness"])
        contexts = evaluator._prepare_ragas_contexts(
            raw_contexts=[
                "Main symptoms include fever, cough, tiredness.",
                "fever, cough, tiredness.",
                "Cause: sars-cov-2 virus.",
                "sars-cov-2 virus",
                "Prevention guidance: get vaccinated.",
                "get vaccinated",
            ],
            top_k=6,
            char_limit=700,
        )

        self.assertEqual(len(contexts), 3)
        self.assertEqual(contexts[0], "Main symptoms include fever, cough, tiredness.")
        self.assertEqual(contexts[1], "Cause: sars-cov-2 virus.")
        self.assertEqual(contexts[2], "Prevention guidance: get vaccinated.")

    def test_prepare_ragas_contexts_dedupes_long_prefix_variants(self):
        evaluator = RagasJudgeEvaluator(metrics=["faithfulness"])
        long_text = (
            "get vaccinated stay at least 1 metre apart from others wear a properly fitted "
            "mask choose open, well-ventilated spaces open a window if indoors wash hands "
            "regularly with soap and water or alcohol-based hand rub cover mouth and nose "
            "when coughing or sneezing stay home and self-isolate if unwell practice "
            "respiratory etiquette"
        )
        contexts = evaluator._prepare_ragas_contexts(
            raw_contexts=[
                f"Prevention guidance: {long_text}.",
                f"{long_text}",
                "Cause: sars-cov-2 virus.",
            ],
            top_k=6,
            char_limit=700,
        )

        self.assertEqual(len(contexts), 2)
        self.assertEqual(contexts[0], f"Prevention guidance: {long_text}.")
        self.assertEqual(contexts[1], "Cause: sars-cov-2 virus.")

    @patch("rag_benchmark.services.ragas.IS_RAGAS_FORMAT_SECTION_CONTEXT_ON", True)
    def test_evaluate_batch_includes_targeted_summary_without_general_overview(
        self,
    ):
        evaluator = RagasJudgeEvaluator(metrics=["faithfulness"])
        fake_modules = self._build_fake_modules()
        run_calls: list[dict] = []

        def _run_side_effect(**kwargs):
            run_calls.append(kwargs)
            return _FakeResult([{"faithfulness": 0.9}])

        cases = [_build_case("Is diabetes a risk factor?", "R1")]
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
                    "summaries": {
                        "coronavirus disease (covid-19)": (
                            "coronavirus disease (covid-19)\n"
                            "general overview text.\n"
                            "Yếu tố nguy cơ: Summary support text."
                        )
                    },
                    "retrieved_items": [
                        {
                            "title": "coronavirus disease (covid-19)",
                            "section": "risk",
                            "content_preview": "Older people and diabetes are risk factors.",
                        }
                    ],
                    "retrieved_context_texts": [],
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
            "Risk factors include Older people and diabetes are risk factors.",
        )
        self.assertIn(
            "coronavirus disease (covid-19): Yếu tố nguy cơ: Summary support text.",
            retrieved_contexts,
        )
        self.assertNotIn("general overview text.", retrieved_contexts)
        self.assertNotIn("Broad context snapshot.", retrieved_contexts)

    def test_summary_candidates_include_only_targeted_section_lines(self):
        evaluator = RagasJudgeEvaluator(metrics=["faithfulness"])

        candidates = evaluator._build_context_candidates(
            retrieval_output={
                "summaries": {
                    "coronavirus disease (covid-19)": (
                        "coronavirus disease (covid-19)\n"
                        "Triệu chứng: fever, cough, tiredness.\n"
                        "Nguyên nhân: sars-cov-2 virus."
                    )
                },
                "retrieved_items": [
                    {
                        "title": "coronavirus disease (covid-19)",
                        "section": "symptom",
                        "content_preview": "fever cough tiredness",
                    }
                ],
                "retrieved_context_texts": [],
            },
            question_tokens={"symptoms", "causes", "covid"},
            target_sections={"symptom", "aetiologies"},
        )

        summary_texts = [
            candidate.text for candidate in candidates if candidate.source == "summary"
        ]
        self.assertEqual(
            summary_texts,
            [
                "coronavirus disease (covid-19): Triệu chứng: fever, cough, tiredness.",
                "coronavirus disease (covid-19): Nguyên nhân: sars-cov-2 virus.",
            ],
        )

    def test_evaluate_batch_uses_fallback_sources_when_structured_sources_missing(self):
        evaluator = RagasJudgeEvaluator(metrics=["faithfulness"])
        fake_modules = self._build_fake_modules()
        run_calls: list[dict] = []

        def _run_side_effect(**kwargs):
            run_calls.append(kwargs)
            return _FakeResult([{"faithfulness": 0.9}])

        cases = [_build_case("Is diabetes a risk factor?", "R1")]
        runtime_outputs = [
            {
                "generation_output": {
                    "final_answer": "A1",
                    "context_snapshot": "Legacy snapshot.",
                },
                "retrieval_output": {
                    "summaries": {
                        "covid": "Yếu tố nguy cơ: Legacy summary.",
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
        self.assertIn("Legacy evidence.", retrieved_contexts)
        self.assertIn("covid: Yếu tố nguy cơ: Legacy summary.", retrieved_contexts)

    def test_evaluate_batch_falls_back_to_legacy_order_when_collector_raises(self):
        evaluator = RagasJudgeEvaluator(metrics=["faithfulness"])
        fake_modules = self._build_fake_modules()
        run_calls: list[dict] = []

        def _run_side_effect(**kwargs):
            run_calls.append(kwargs)
            return _FakeResult([{"faithfulness": 0.9}])

        cases = [_build_case("Is diabetes a risk factor?", "R1")]
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
        self.assertIn("Legacy summary.", retrieved_contexts)
        self.assertIn("Legacy evidence.", retrieved_contexts)
        self.assertEqual(len(retrieved_contexts), 2)

    def test_openrouter_ragas_embeddings_use_raw_string_inputs(self):
        evaluator = RagasJudgeEvaluator(metrics=["answer_relevancy"])
        embeddings_client = MagicMock()
        embeddings_client.embeddings.create.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[0.0, 2.0, 0.0]),
                SimpleNamespace(index=0, embedding=[3.0, 4.0, 0.0]),
            ]
        )

        with (
            patch.object(
                evaluator,
                "_resolve_openrouter_api_key",
                return_value="test-openrouter-key",
            ),
            patch.object(
                evaluator,
                "_resolve_openrouter_base_url",
                return_value="https://openrouter.ai/api/v1",
            ),
            patch(
                "rag_benchmark.services.ragas.ChatbotConfig.get_config",
                return_value="qwen/qwen3-embedding-8b",
            ),
            patch(
                "rag_benchmark.services.ragas.OpenAI",
                return_value=embeddings_client,
            ) as openai_cls,
        ):
            embeddings = evaluator._create_openrouter_embeddings()
            vectors = embeddings.embed_documents(["alpha", "beta"])

        openai_cls.assert_called_once_with(
            api_key="test-openrouter-key",
            base_url="https://openrouter.ai/api/v1",
        )
        embeddings_client.embeddings.create.assert_called_once_with(
            model="qwen/qwen3-embedding-8b",
            input=["alpha", "beta"],
        )
        self.assertEqual(vectors[0], [0.6, 0.8, 0.0])
        self.assertEqual(vectors[1], [0.0, 1.0, 0.0])
