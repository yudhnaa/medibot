from unittest.mock import MagicMock, patch

from django.test import TestCase

from rag_benchmark.models import BenchmarkCaseResult, BenchmarkRunStatus
from rag_benchmark.services.runner import OfflineBenchmarkRunner


class OfflineBenchmarkRunnerTests(TestCase):
    @patch("rag_benchmark.services.runner.EmbeddingService.resolve_provider")
    @patch("rag_benchmark.services.runner.ChatbotService")
    def test_runner_executes_cases_and_persists_results(
        self,
        mock_chatbot_service: MagicMock,
        mock_resolve_provider: MagicMock,
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

        dataset = self._create_dataset_with_cases()
        runner = OfflineBenchmarkRunner()

        run = runner.run(dataset=dataset, split="test")

        self.assertEqual(run.status, BenchmarkRunStatus.COMPLETED)
        self.assertEqual(run.total_cases, 1)
        self.assertEqual(BenchmarkCaseResult.objects.filter(run=run).count(), 1)
        self.assertTrue(isinstance(run.release_gate, dict))

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
