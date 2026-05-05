from django.test import TestCase

from rag_benchmark.models import (
    BenchmarkCaseResult,
    BenchmarkCaseResultStatus,
    BenchmarkDataset,
    BenchmarkRun,
)


class BenchmarkModelTests(TestCase):
    def test_activate_dataset_deactivates_previous_version(self):
        first = BenchmarkDataset.objects.create(
            name="core",
            version="v1",
            is_active=True,
            total_cases=2,
            split_counts={"dev": 1, "test": 1},
        )
        second = BenchmarkDataset.objects.create(
            name="core",
            version="v2",
            is_active=False,
            total_cases=2,
            split_counts={"dev": 1, "test": 1},
        )

        second.activate()
        first.refresh_from_db()
        second.refresh_from_db()

        self.assertFalse(first.is_active)
        self.assertTrue(second.is_active)
        self.assertIsNotNone(second.activated_at)

    def test_case_result_primary_pass_property(self):
        dataset = BenchmarkDataset.objects.create(
            name="core",
            version="v1",
            total_cases=2,
            split_counts={"dev": 1, "test": 1},
        )
        case = dataset.cases.create(
            case_id="case-1",
            dataset_version="v1",
            split="test",
            question="Tôi bị sốt",
            intake_payload={},
            scenario="single_clear",
            expected_mode="single-disease",
            gold_titles=["bệnh sởi"],
            forbidden_titles=[],
            must_have_sections=[],
            expected_behavior="answer",
            reference_answer="",
            notes="",
        )
        run = BenchmarkRun.objects.create(dataset=dataset, split="test")
        result = BenchmarkCaseResult.objects.create(
            run=run,
            case=case,
            status=BenchmarkCaseResultStatus.PASSED,
            pass_flags={"primary_pass": True},
        )
        self.assertTrue(result.primary_pass)
