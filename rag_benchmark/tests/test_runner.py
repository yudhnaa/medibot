from unittest.mock import MagicMock, patch

from django.test import TestCase

from rag_benchmark.models import BenchmarkCaseResult, BenchmarkRunStatus
from rag_benchmark.services.constants import RAGAS_RELEASE_GATE_THRESHOLDS
from rag_benchmark.services.runner import OfflineBenchmarkRunner


class OfflineBenchmarkRunnerTests(TestCase):
    @patch("rag_benchmark.services.runner.RagasJudgeEvaluator")
    @patch("rag_benchmark.services.runner.EmbeddingService.resolve_provider")
    @patch("rag_benchmark.services.runner.ChatbotBenchmarkService")
    def test_runner_executes_cases_and_persists_results(
        self,
        mock_chatbot_service: MagicMock,
        mock_resolve_provider: MagicMock,
        mock_judge_cls: MagicMock,
    ):
        mock_resolve_provider.return_value = "transformers"
        service_instance = mock_chatbot_service.return_value
        service_instance.run_benchmark_case.return_value = {
            "mode": "single-disease",
            "analysis_output": {"q_cleaned": "tôi bị sốt"},
            "gate_output": {"title": "bệnh sởi", "top_score": 0.9},
            "retrieval_output": {
                "retrieved_titles": ["bệnh sởi"],
                "retrieved_sections": ["symptom"],
                "retrieved_context_texts": ["sốt cao phát ban"],
            },
            "generation_output": {
                "final_answer": "Bạn có dấu hiệu bệnh sởi.",
                "source_urls": ["https://example.com/measles"],
            },
            "audit_metadata": {"mode": "single-disease"},
            "timings_ms": {
                "analysis_latency": 10,
                "gate_latency": 5,
                "retrieval_latency": 12,
                "generation_latency": 20,
                "total_latency": 47,
            },
        }
        judge = mock_judge_cls.return_value
        judge.evaluate_batch.return_value = [
            {
                "available": True,
                "selected_metrics": [
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                    "answer_correctness",
                ],
                "scores": {
                    "faithfulness": 0.9,
                    "answer_relevancy": 0.84,
                    "context_precision": 0.88,
                    "context_recall": 0.89,
                    "answer_correctness": 0.9,
                },
            }
        ]

        dataset = self._create_dataset_with_cases()
        runner = OfflineBenchmarkRunner()

        run = runner.run(dataset=dataset, split="test")

        self.assertEqual(run.status, BenchmarkRunStatus.COMPLETED)
        self.assertEqual(run.total_cases, 1)
        self.assertEqual(BenchmarkCaseResult.objects.filter(run=run).count(), 1)
        self.assertTrue(isinstance(run.release_gate, dict))
        self.assertIn("IS_RAGAS_FORMAT_SECTION_CONTEXT_ON", run.config_snapshot)
        self.assertIn("IS_RAGAS_SUMMARY_CONTEXT_ON", run.config_snapshot)
        self.assertIn("IS_RAGAS_CONTEXT_SNAPSHOT_ON", run.config_snapshot)

    def _create_dataset_with_cases(self):
        dataset = self._create_dataset()
        dataset.cases.create(
            case_id="dev-1",
            dataset_version="v1",
            split="dev",
            question="test dev",
            intake_payload={},
            scenario="single_clear",
            expected_mode="single-disease",
            gold_titles=["bệnh sởi"],
            forbidden_titles=[],
            must_have_sections=["symptom"],
            expected_behavior="answer",
            reference_answer="",
            notes="",
        )
        dataset.cases.create(
            case_id="test-1",
            dataset_version="v1",
            split="test",
            question="test case",
            intake_payload={},
            scenario="single_clear",
            expected_mode="single-disease",
            gold_titles=["bệnh sởi"],
            forbidden_titles=[],
            must_have_sections=["symptom"],
            expected_behavior="answer",
            reference_answer="",
            notes="",
        )
        return dataset

    def _create_dataset(self):
        from rag_benchmark.models import BenchmarkDataset

        return BenchmarkDataset.objects.create(
            name="core",
            version="v1",
            total_cases=2,
            split_counts={"dev": 1, "test": 1},
            composition_stats={},
        )

    @patch("rag_benchmark.services.runner.ChatbotConfig.get_config")
    def test_build_run_scorer_applies_threshold_overrides_from_db_config(
        self,
        mock_get_config: MagicMock,
    ) -> None:
        config = {
            "RAGAS_RELEASE_GATE_THRESHOLDS": {
                "answer_relevancy": [">=", 0.82],
                "context_precision": {"operator": ">=", "threshold": 0.74},
            }
        }
        mock_get_config.side_effect = lambda key, default=None: config.get(key, default)

        runner = OfflineBenchmarkRunner()
        scorer = runner._build_run_scorer(
            active_metrics=[
                "answer_relevancy",
                "context_precision",
                "faithfulness",
            ]
        )

        self.assertEqual(
            scorer.release_thresholds["answer_relevancy"],
            (">=", 0.82),
        )
        self.assertEqual(
            scorer.release_thresholds["context_precision"],
            (">=", 0.74),
        )
        self.assertEqual(
            scorer.release_thresholds["faithfulness"],
            RAGAS_RELEASE_GATE_THRESHOLDS["faithfulness"],
        )
