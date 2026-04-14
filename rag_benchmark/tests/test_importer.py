import io

from django.test import TestCase

from rag_benchmark.services import BenchmarkDatasetImporter, DatasetValidationError


class BenchmarkDatasetImporterTests(TestCase):
    def test_import_jsonl_creates_dataset_and_cases(self):
        payload = "\n".join(
            [
                (
                    '{"case_id":"dev-1","dataset_version":"v1","split":"dev",'
                    '"question":"Tôi bị sốt","intake_payload":{},"scenario":"single_clear",'
                    '"expected_mode":"single-disease","gold_titles":["bệnh sởi"],'
                    '"forbidden_titles":[],"must_have_sections":["symptom"],'
                    '"expected_behavior":"answer","reference_answer":"","notes":""}'
                ),
                (
                    '{"case_id":"test-1","dataset_version":"v1","split":"test",'
                    '"question":"Tôi không đau bụng","intake_payload":{"age":24},'
                    '"scenario":"negation","expected_mode":"multi-disease-v2",'
                    '"gold_titles":["rubella"],"forbidden_titles":["viêm ruột thừa"],'
                    '"must_have_sections":[],"expected_behavior":"answer",'
                    '"reference_answer":"","notes":""}'
                ),
            ]
        )
        importer = BenchmarkDatasetImporter()

        dataset, report = importer.import_jsonl(
            name="core",
            version="v1",
            file_obj=io.BytesIO(payload.encode("utf-8")),
            description="test dataset",
            activate=True,
        )

        self.assertEqual(dataset.total_cases, 2)
        self.assertEqual(report.total_cases, 2)
        self.assertEqual(dataset.cases.count(), 2)
        self.assertTrue(dataset.is_active)
        self.assertEqual(dataset.split_counts.get("dev"), 1)
        self.assertEqual(dataset.split_counts.get("test"), 1)

    def test_import_jsonl_raises_on_missing_required_fields(self):
        invalid_payload = (
            '{"case_id":"dev-1","dataset_version":"v1","split":"dev","question":"x"}'
        )
        importer = BenchmarkDatasetImporter()
        with self.assertRaises(DatasetValidationError):
            importer.import_jsonl(
                name="core",
                version="v1",
                file_obj=io.BytesIO(invalid_payload.encode("utf-8")),
            )
