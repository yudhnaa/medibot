import logging

from django.test import SimpleTestCase

from utils.logger import SQLDebugSanitizerFilter, get_logging_config


class LoggingConfigTests(SimpleTestCase):
    """Tests for shared logging configuration."""

    def test_prod_logging_config_uses_plain_stdout_defaults(self) -> None:
        """Prod config should use non-colored stdout and suppress noisy clients."""
        config = get_logging_config(
            log_level="INFO",
            enable_colors=False,
            enable_django_debug=False,
            enable_sql_debug=False,
        )

        self.assertEqual(config["handlers"]["console"]["formatter"], "verbose")
        self.assertEqual(config["loggers"]["httpx"]["level"], "WARNING")
        self.assertEqual(config["loggers"]["httpcore"]["level"], "WARNING")
        self.assertEqual(config["loggers"]["django.db.backends"]["level"], "WARNING")
        self.assertFalse(config["loggers"]["chatbot"]["propagate"])
        self.assertFalse(config["loggers"]["vector_store"]["propagate"])

    def test_sql_debug_logging_can_be_enabled_independently(self) -> None:
        """SQL logging toggle should be independent from generic Django debug logging."""
        config = get_logging_config(
            log_level="DEBUG",
            enable_colors=True,
            enable_django_debug=False,
            enable_sql_debug=True,
        )

        self.assertEqual(config["loggers"]["django"]["level"], "WARNING")
        self.assertEqual(config["loggers"]["django.db.backends"]["level"], "DEBUG")


class SQLDebugSanitizerFilterTests(SimpleTestCase):
    """Tests for human-readable SQL debug sanitization."""

    def test_filter_replaces_embedding_and_base64_payloads(self) -> None:
        """Large unreadable payloads should become concise placeholders."""
        record = logging.LogRecord(
            name="django.db.backends",
            level=logging.DEBUG,
            pathname=__file__,
            lineno=1,
            msg="(%.3f) %s; args=%s; alias=%s",
            args=(
                0.123,
                "INSERT INTO demo_table (message, embedding, image) VALUES (%s, %s, %s)",
                (
                    "Tôi bị đau đầu và sốt 2 ngày",
                    [0.1] * 1024,
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB" * 20,
                ),
                "default",
            ),
            exc_info=None,
        )
        record.duration = 0.123
        record.sql = (
            "INSERT INTO demo_table (message, embedding, image) VALUES (%s, %s, %s)"
        )
        record.params = (
            "Tôi bị đau đầu và sốt 2 ngày",
            [0.1] * 1024,
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB" * 20,
        )
        record.alias = "default"

        SQLDebugSanitizerFilter().filter(record)
        message = record.getMessage()

        self.assertIn("Tôi bị đau đầu", message)
        self.assertIn("<embedding vector dims=1024>", message)
        self.assertIn("<base64 payload len=", message)
        self.assertNotIn("[0.1, 0.1, 0.1", message)

    def test_filter_keeps_human_text_preview(self) -> None:
        """Long human text should stay readable as a preview instead of raw dump."""
        long_text = "Bệnh nhân đau đầu, ho, sốt cao. " * 20
        record = logging.LogRecord(
            name="django.db.backends",
            level=logging.DEBUG,
            pathname=__file__,
            lineno=1,
            msg="(%.3f) %s; args=%s; alias=%s",
            args=(0.045, "UPDATE notes SET content=%s", (long_text,), "default"),
            exc_info=None,
        )
        record.duration = 0.045
        record.sql = "UPDATE notes SET content=%s"
        record.params = (long_text,)
        record.alias = "default"

        SQLDebugSanitizerFilter().filter(record)
        message = record.getMessage()

        self.assertIn("Bệnh nhân đau đầu, ho, sốt cao.", message)
        self.assertIn("[len=", message)
