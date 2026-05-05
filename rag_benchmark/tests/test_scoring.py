from django.test import SimpleTestCase

from rag_benchmark.models import BenchmarkCase
from rag_benchmark.services.scoring import RagasBenchmarkScorer


class RagasScoringTests(SimpleTestCase):
    def test_score_case_computes_primary_flags_from_ragas_metrics(self):
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
        scorer = RagasBenchmarkScorer()
        runtime_output = {
            "gate_output": {"title": "bệnh sởi", "top_score": 0.92},
        }
        judge_result = {
            "available": True,
            "selected_metrics": [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "answer_correctness",
            ],
            "scores": {
                "faithfulness": 0.91,
                "answer_relevancy": 0.84,
                "context_precision": 0.88,
                "context_recall": 0.86,
                "answer_correctness": 0.9,
            },
        }

        score = scorer.score_case(
            case=case,
            runtime_output=runtime_output,
            judge_result=judge_result,
        )

        self.assertEqual(score.metrics["faithfulness"], 0.91)
        self.assertEqual(score.metrics["answer_relevancy"], 0.84)
        self.assertTrue(score.pass_flags["primary_pass"])

    def test_score_case_respects_active_metrics_subset(self):
        case = BenchmarkCase(
            case_id="case-2",
            dataset_version="v1",
            split="test",
            question="Is cancer a risk factor for COVID-19?",
            intake_payload={},
            scenario="single_clear",
            expected_mode="single-disease",
            gold_titles=["covid-19"],
            forbidden_titles=[],
            must_have_sections=["risk"],
            expected_behavior="answer",
            reference_answer="",
            notes="",
            gold_primary_title="covid-19",
        )
        scorer = RagasBenchmarkScorer(
            active_metrics=["faithfulness", "answer_relevancy"]
        )
        judge_result = {
            "available": True,
            "selected_metrics": [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "answer_correctness",
            ],
            "scores": {
                "faithfulness": 0.95,
                "answer_relevancy": 0.81,
                "context_precision": 0.1,
                "context_recall": 1.0,
                "answer_correctness": 0.2,
            },
        }
        score = scorer.score_case(
            case=case,
            runtime_output={},
            judge_result=judge_result,
        )
        self.assertTrue(score.pass_flags["primary_pass"])
        self.assertNotIn("context_precision_pass", score.pass_flags)
        self.assertNotIn("answer_correctness_pass", score.pass_flags)
