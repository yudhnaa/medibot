from typing import Any, cast
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from rag_benchmark.models import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkDataset,
    BenchmarkRun,
)


class BenchmarkAdminImportTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.superuser = cast(Any, User.objects).create_superuser(
            username="admin_benchmark",
            email="admin_benchmark@example.com",
            password="test-password-123",
        )
        self.client.force_login(self.superuser)

    def test_import_jsonl_view_creates_dataset(self):
        url = reverse("admin:rag_benchmark_benchmarkdataset_import_jsonl")
        payload = (
            '{"case_id":"dev-1","dataset_version":"v1","split":"dev",'
            '"question":"Tôi bị sốt","intake_payload":{},"scenario":"single_clear",'
            '"expected_mode":"single-disease","gold_titles":["bệnh sởi"],'
            '"forbidden_titles":[],"must_have_sections":["symptom"],'
            '"expected_behavior":"answer","reference_answer":"","notes":""}\n'
            '{"case_id":"test-1","dataset_version":"v1","split":"test",'
            '"question":"Tôi không đau bụng","intake_payload":{},"scenario":"negation",'
            '"expected_mode":"multi-disease-v2","gold_titles":["rubella"],'
            '"forbidden_titles":["viêm ruột thừa"],"must_have_sections":[],'
            '"expected_behavior":"answer","reference_answer":"","notes":""}\n'
        )
        uploaded = SimpleUploadedFile(
            "benchmark.jsonl",
            payload.encode("utf-8"),
            content_type="application/json",
        )

        response = self.client.post(
            url,
            data={
                "dataset_name": "core",
                "dataset_version": "v1",
                "description": "from admin",
                "source_format": "jsonl",
                "schema_version": "1.0",
                "activate": "on",
                "jsonl_file": uploaded,
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(BenchmarkDataset.objects.count(), 1)
        dataset = BenchmarkDataset.objects.first()
        self.assertIsNotNone(dataset)
        if dataset is not None:
            self.assertEqual(dataset.total_cases, 2)

    @patch("rag_benchmark.admin.OfflineBenchmarkRunner")
    def test_run_benchmark_view_runs_selected_dataset(
        self,
        mock_runner_cls: MagicMock,
    ):
        dataset = BenchmarkDataset.objects.create(
            name="core",
            version="v1",
            total_cases=2,
            split_counts={"dev": 1, "test": 1},
            composition_stats={},
        )
        run = BenchmarkRun.objects.create(
            dataset=dataset,
            split="dev",
            status="completed",
        )
        runner = mock_runner_cls.return_value
        runner.run.return_value = run

        url = reverse("admin:rag_benchmark_benchmarkdataset_run_benchmark")
        response = self.client.post(
            url,
            data={
                "dataset": dataset.pk,
                "split": "dev",
                "code_version": "abc123",
            },
        )

        self.assertEqual(response.status_code, 302)
        runner.run.assert_called_once()
        kwargs = runner.run.call_args.kwargs
        self.assertEqual(kwargs["dataset"].pk, dataset.pk)
        self.assertEqual(kwargs["split"], "dev")
        self.assertEqual(kwargs["judge_configuration"]["enable_ragas"], True)
        self.assertEqual(kwargs["code_version"], "abc123")


class BenchmarkAdminDashboardTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.superuser = cast(Any, User.objects).create_superuser(
            username="admin_dashboard",
            email="admin_dashboard@example.com",
            password="test-password-123",
        )
        self.client.force_login(self.superuser)

        self.dataset = BenchmarkDataset.objects.create(
            name="core",
            version="v1",
            total_cases=3,
            split_counts={"dev": 3},
            composition_stats={},
        )
        self.benchmark_run = BenchmarkRun.objects.create(
            dataset=self.dataset,
            split="dev",
            status="completed",
            total_cases=3,
            passed_cases=2,
            failed_cases=1,
            summary_metrics={
                "mode_accuracy": 0.85,
                "false_single_rate": 0.08,
                "title_recall@5": 0.92,
            },
            failure_slices={
                "scenario": {"single_clear": 1},
                "expected_mode": {"single-disease": 1},
                "route_confidence_bucket": {"low": 1},
                "question_length": {"short": 1},
            },
            release_gate={
                "overall_pass": False,
                "checks": {
                    "mode_accuracy": {
                        "operator": ">=",
                        "value": 0.85,
                        "threshold": 0.9,
                        "is_pass": False,
                    },
                    "title_recall@5": {
                        "operator": ">=",
                        "value": 0.92,
                        "threshold": 0.9,
                        "is_pass": True,
                    },
                },
            },
        )
        self.case = BenchmarkCase.objects.create(
            dataset=self.dataset,
            case_id="dev-1",
            dataset_version="v1",
            split="dev",
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
        BenchmarkCaseResult.objects.create(
            run=self.benchmark_run,
            case=self.case,
            status="failed",
            pass_flags={"primary_pass": False},
            failure_taxonomy="retrieval_miss",
        )

    def test_dashboard_view_renders_visualized_sections(self):
        url = reverse(
            "admin:rag_benchmark_benchmarkrun_dashboard",
            args=[self.benchmark_run.pk],
        )
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Primary Metrics")
        self.assertContains(response, "Release Gate Checks")
        self.assertContains(response, "Pass / Fail")
        self.assertContains(response, "2 pass · 1 fail / 3 total")
        self.assertContains(response, "Recent Failed Cases")
        self.assertContains(response, "Scenario Failures")
        self.assertContains(response, "Mode Failures")
        self.assertContains(response, "Confidence Buckets")
        self.assertContains(response, "Question Length")
        self.assertContains(response, "Review Taxonomy")
        self.assertContains(response, "Retrieval miss")

    def test_change_form_includes_dashboard_link(self):
        change_url = reverse(
            "admin:rag_benchmark_benchmarkrun_change", args=[self.benchmark_run.pk]
        )
        dashboard_url = reverse(
            "admin:rag_benchmark_benchmarkrun_dashboard",
            args=[self.benchmark_run.pk],
        )
        case_metrics_url = reverse(
            "admin:rag_benchmark_benchmarkrun_case_metrics",
            args=[self.benchmark_run.pk],
        )
        response = self.client.get(change_url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, dashboard_url)
        self.assertContains(response, case_metrics_url)

    def test_case_metrics_view_renders_case_level_metrics(self):
        result = BenchmarkCaseResult.objects.get(
            run=self.benchmark_run,
            case=self.case,
        )
        result.metrics = {
            "faithfulness": 1.0,
            "answer_relevancy": 0.91,
            "context_precision": 0.75,
            "context_recall": 0.88,
        }
        result.pass_flags = {
            "primary_pass": False,
            "faithfulness_pass": True,
            "answer_relevancy_pass": True,
            "context_precision_pass": False,
            "context_recall_pass": True,
        }
        result.save(update_fields=["metrics", "pass_flags", "updated_at"])

        url = reverse(
            "admin:rag_benchmark_benchmarkrun_case_metrics",
            args=[self.benchmark_run.pk],
        )
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Metric Thresholds")
        self.assertContains(response, "&gt;= 0.8")
        self.assertContains(response, "&gt;= 0.7")
        self.assertContains(response, "&gt;= 0.6")
        self.assertContains(response, "Per-case Metrics")
        self.assertContains(response, self.case.case_id)
        self.assertContains(response, "0.75")
        self.assertContains(response, "Context Precision")

    def test_case_metrics_view_redirects_to_login_for_anonymous_user(self):
        self.client.logout()
        url = reverse(
            "admin:rag_benchmark_benchmarkrun_case_metrics",
            args=[self.benchmark_run.pk],
        )
        response = self.client.get(url)
        redirect_url = response.headers.get("Location", "")

        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin/login/", redirect_url)
        self.assertIn(url, redirect_url)
