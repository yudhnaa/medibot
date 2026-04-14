from django.test import SimpleTestCase

from rag_benchmark.models import BenchmarkCase
from rag_benchmark.services.scoring import DeterministicBenchmarkScorer


class DeterministicScoringTests(SimpleTestCase):
    def test_score_case_computes_primary_metrics(self):
        case = BenchmarkCase(
            case_id="case-1",
            dataset_version="v1",
            split="test",
            question="Tôi bị sốt và phát ban",
            intake_payload={},
            scenario="single_clear",
            expected_mode="single-disease",
            gold_titles=["bệnh sởi"],
            forbidden_titles=[],
            must_have_sections=["symptom"],
            expected_behavior="answer",
            reference_answer="",
            notes="",
            gold_primary_title="bệnh sởi",
        )
        scorer = DeterministicBenchmarkScorer()
        runtime_output = {
            "mode": "single-disease",
            "gate_output": {"title": "bệnh sởi", "top_score": 0.92},
            "retrieval_output": {
                "retrieved_titles": ["bệnh sởi"],
                "retrieved_sections": ["symptom", "general"],
            },
            "generation_output": {
                "final_answer": "Bạn có triệu chứng phù hợp bệnh sởi.",
                "source_urls": ["https://example.com/measles"],
            },
        }

        score = scorer.score_case(case=case, runtime_output=runtime_output)

        self.assertEqual(score.metrics["mode_accuracy"], 1.0)
        self.assertEqual(score.metrics["title_recall@5"], 1.0)
        self.assertEqual(score.metrics["behavior_accuracy"], 1.0)
        self.assertTrue(score.pass_flags["primary_pass"])
