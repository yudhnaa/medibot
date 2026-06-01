from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from chatbot.tasks import process_csv_upload


class ProcessCsvUploadTaskTests(SimpleTestCase):
    @patch("chatbot.tasks.VectorStoreManager")
    @patch("chatbot.tasks.EmbeddingService.resolve_provider")
    def test_process_csv_upload_returns_reports_and_cleans_temp_file(
        self,
        mock_resolve_provider,
        mock_manager_cls,
    ):
        mock_resolve_provider.return_value = "transformers"
        manager = mock_manager_cls.return_value
        manager.process_csv_to_documents_with_report.side_effect = [
            {
                "documents": [{"title": "Sởi"}],
                "report": {"total_rows": 1, "success_rows": 1, "failed_rows": 0},
            },
            {
                "documents": [],
                "report": {"total_rows": 1, "success_rows": 0, "failed_rows": 1},
            },
        ]
        manager.add_documents.return_value = [SimpleNamespace(pk=1)]

        with (
            patch("chatbot.tasks.os.path.exists", return_value=True),
            patch("chatbot.tasks.os.remove") as mock_remove,
        ):
            result = cast(Any, process_csv_upload).run(
                file_path="/tmp/upload.csv",
                collections=["medical_documents_disease", "medical_documents_chunks"],
                embedding_provider=None,
                source="admin_upload",
                user_id=7,
                job_id=None,
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["count"], 1)
        self.assertEqual(
            result["reports"]["medical_documents_disease"]["success_rows"], 1
        )
        self.assertEqual(
            result["reports"]["medical_documents_chunks"]["failed_rows"], 1
        )
        mock_remove.assert_called_once_with("/tmp/upload.csv")

    @patch("chatbot.tasks.os.remove")
    @patch("chatbot.tasks.os.path.exists", return_value=True)
    def test_retry_keeps_temp_file_when_retries_remain(
        self,
        mock_exists,
        mock_remove,
    ):
        task = SimpleNamespace(
            request=SimpleNamespace(retries=0),
            max_retries=3,
            retry=MagicMock(side_effect=RuntimeError("retry")),
        )

        retry_or_return_csv_error = cast(Any, process_csv_upload).run.__globals__[
            "_retry_or_return_csv_error"
        ]

        with self.assertRaises(RuntimeError):
            retry_or_return_csv_error(
                task,
                FileNotFoundError("missing"),
                file_path="/tmp/upload.csv",
            )

        mock_exists.assert_not_called()
        mock_remove.assert_not_called()

    @patch("chatbot.tasks.os.remove")
    @patch("chatbot.tasks.os.path.exists", return_value=True)
    def test_retry_cleans_temp_file_after_final_failure(
        self,
        mock_exists,
        mock_remove,
    ):
        task = SimpleNamespace(request=SimpleNamespace(retries=3), max_retries=3)

        result = cast(Any, process_csv_upload).run.__globals__[
            "_retry_or_return_csv_error"
        ](
            task,
            FileNotFoundError("missing"),
            file_path="/tmp/upload.csv",
        )

        self.assertEqual(result["status"], "error")
        mock_exists.assert_called_once_with("/tmp/upload.csv")
        mock_remove.assert_called_once_with("/tmp/upload.csv")
