"""
Logger utilities with color support for Django projects.

This module provides a custom color formatter and utility functions
for getting configured loggers following Django's logging best practices.

Usage:
    from utils.logger import get_logger

    logger = get_logger(__name__)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

Django Settings Configuration (add to settings/base.py):
    from utils.logger import get_logging_config
    LOGGING = get_logging_config(log_level="DEBUG")  # or "INFO", "WARNING", etc.
"""

import itertools
import logging
import numbers
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from typing_extensions import override

# ANSI color codes for terminal output
COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[1;41m",  # Bold white on red background
    "RESET": "\033[0m",  # Reset to default
}

# Detailed format with timestamp, level, module, and message
VERBOSE_FORMAT = "[{asctime}] [{levelname:^8}] [{name}:{funcName}:{lineno}] {message}"

# Simple format with just level and message
SIMPLE_FORMAT = "[{levelname:^8}] {message}"

# Date format
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class SQLDebugSanitizerFilter(logging.Filter):
    """
    Sanitize SQL debug params so logs stay readable for humans.

    This keeps useful SQL text and short textual params visible, while replacing
    embeddings, base64 blobs, and oversized binary-ish payloads with concise
    placeholders.
    """

    BASE64_LENGTH_THRESHOLD = 256
    EMBEDDING_DIM_THRESHOLD = 64
    MAX_TEXT_PREVIEW = 160
    MAX_SEQUENCE_ITEMS = 8
    _BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")
    _SQL_WHITESPACE_RE = re.compile(r"\s+")

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        """Rewrite django.db.backends records with sanitized params."""
        if not record.name.startswith("django.db.backends"):
            return True

        sql = self._normalize_sql(getattr(record, "sql", None))
        params = self._sanitize_value(getattr(record, "params", None))
        alias = getattr(record, "alias", None)
        duration = getattr(record, "duration", None)

        if sql is not None:
            record.sql = sql
        if hasattr(record, "params"):
            record.params = params

        if isinstance(record.args, tuple) and len(record.args) == 4:
            record.args = (
                duration if duration is not None else record.args[0],
                sql if sql is not None else record.args[1],
                params,
                alias if alias is not None else record.args[3],
            )
        else:
            record.msg = "(%.3f) %s; args=%s; alias=%s"
            record.args = (
                float(duration or 0.0),
                sql or "",
                params,
                alias or "default",
            )

        return True

    def _normalize_sql(self, sql: Any) -> Any:
        """Collapse whitespace in SQL statements for easier scanning."""
        if not isinstance(sql, str):
            return sql
        return self._SQL_WHITESPACE_RE.sub(" ", sql).strip()

    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize a logged SQL parameter recursively."""
        if value is None:
            return None

        if isinstance(value, str):
            return self._sanitize_text(value)

        if isinstance(value, (bytes, bytearray, memoryview)):
            return f"<binary payload {len(value)} bytes>"

        if isinstance(value, Mapping):
            items = list(value.items())
            sanitized: dict[Any, Any] = {}
            for idx, (key, item) in enumerate(items[: self.MAX_SEQUENCE_ITEMS]):
                sanitized[key] = self._sanitize_value(item)
            if len(items) > self.MAX_SEQUENCE_ITEMS:
                sanitized["..."] = f"{len(items) - self.MAX_SEQUENCE_ITEMS} more fields"
            return sanitized

        if self._looks_like_embedding(value):
            return f"<embedding vector dims={len(value)}>"

        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray, memoryview),
        ):
            limited_items = list(itertools.islice(iter(value), self.MAX_SEQUENCE_ITEMS))
            sanitized_items = [self._sanitize_value(item) for item in limited_items]
            if hasattr(value, "__len__") and len(value) > self.MAX_SEQUENCE_ITEMS:
                sanitized_items.append(
                    f"... ({len(value) - self.MAX_SEQUENCE_ITEMS} more items)"
                )
            if isinstance(value, tuple):
                return tuple(sanitized_items)
            return sanitized_items

        return value

    def _looks_like_embedding(self, value: Any) -> bool:
        """Detect long numeric vectors such as embeddings."""
        if isinstance(value, (str, bytes, bytearray, memoryview)):
            return False
        if not isinstance(value, Sequence):
            return False
        if not hasattr(value, "__len__") or len(value) < self.EMBEDDING_DIM_THRESHOLD:
            return False

        sample = list(itertools.islice(iter(value), 8))
        if not sample:
            return False
        return all(isinstance(item, numbers.Number) for item in sample)

    def _sanitize_text(self, text: str) -> str:
        """Preserve human-readable text while hiding unreadable payloads."""
        if text.startswith("data:image/") and ";base64," in text:
            return f"<base64 image payload len={len(text)}>"

        compact = text.strip()
        if len(compact) >= self.BASE64_LENGTH_THRESHOLD and self._BASE64_RE.fullmatch(
            compact
        ):
            return f"<base64 payload len={len(text)}>"

        if len(text) <= self.MAX_TEXT_PREVIEW:
            return text

        preview = text[: self.MAX_TEXT_PREVIEW].rstrip()
        return f"{preview}... [len={len(text)}]"


class ColorFormatter(logging.Formatter):
    """
    Custom log formatter that adds ANSI color codes to log messages.

    This formatter wraps the log level name with color codes for better
    visibility in terminal output. Colors are only applied when the
    output stream supports ANSI colors.

    Attributes:
        use_colors: Whether to apply ANSI color codes to output.
    """

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: Literal["%", "{", "$"] = "{",
        use_colors: bool = True,
    ) -> None:
        """
        Initialize the ColorFormatter.

        Args:
            fmt: Log message format string.
            datefmt: Date format string.
            style: Format style ('{' for str.format, '%' for %-formatting).
            use_colors: Whether to use ANSI colors in output.
        """
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.use_colors = use_colors and self._supports_colors()

    @staticmethod
    def _supports_colors() -> bool:
        """Check if the output stream supports ANSI color codes."""
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    @override
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with optional color codes.

        Args:
            record: The log record to format.

        Returns:
            The formatted log message string.
        """
        # Create a copy to avoid modifying the original record
        record = logging.makeLogRecord(record.__dict__)

        if self.use_colors:
            level_name = record.levelname
            color = COLORS.get(level_name, COLORS["RESET"])
            reset = COLORS["RESET"]
            record.levelname = f"{color}{level_name}{reset}"

        return super().format(record)


def get_logger(name: str, level: int | str | None = None) -> logging.Logger:
    """
    Get a configured logger instance.

    This function retrieves a logger with the specified name. If a level
    is provided, it sets the logger to that level, otherwise the level
    is determined by the Django LOGGING configuration.

    Args:
        name: The name of the logger, typically __name__ from the calling module.
        level: Optional log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        A configured logging.Logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    logger = logging.getLogger(name)

    if level is not None:
        resolved_level: int
        if isinstance(level, str):
            resolved_level = getattr(logging, level.upper(), logging.DEBUG)
        else:
            resolved_level = level
        logger.setLevel(resolved_level)

    return logger


def get_logging_config(
    log_level: str = "INFO",
    enable_colors: bool = True,
    log_to_file: bool = False,
    log_file_path: str = "logs/django.log",
    enable_django_debug: bool = False,
    enable_sql_debug: bool = False,
) -> dict[str, Any]:
    """
    Generate a Django LOGGING configuration dictionary.

    This function creates a complete LOGGING configuration dictionary
    that can be used directly in Django settings. It follows Django's
    dictConfig format and includes sensible defaults.

    Args:
        log_level: The default log level for the application logger.
                   One of: DEBUG, INFO, WARNING, ERROR, CRITICAL.
        enable_colors: Whether to enable ANSI color output in console.
        log_to_file: Whether to also log messages to a file.
        log_file_path: Path to the log file (used if log_to_file is True).
        enable_django_debug: Whether to enable Django's internal debug logging.
        enable_sql_debug: Whether to enable SQL query logging via django.db.backends.

    Returns:
        A dictionary suitable for Django's LOGGING setting.

    Example:
        >>> # In settings/base.py
        >>> from utils.logger import get_logging_config
        >>> LOGGING = get_logging_config(log_level="DEBUG", enable_colors=True)
    """
    handlers: dict[str, Any] = {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "colored" if enable_colors else "verbose",
            "stream": "ext://sys.stdout",
        },
    }

    # Add file handler if requested
    if log_to_file:
        handlers["file"] = {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": log_file_path,
            "maxBytes": 10 * 1024 * 1024,  # 10 MB
            "backupCount": 5,
            "formatter": "verbose",
        }

    # Determine handlers to use
    handler_list = ["console"]
    if log_to_file:
        handler_list.append("file")

    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "verbose": {
                "format": VERBOSE_FORMAT,
                "datefmt": DATE_FORMAT,
                "style": "{",
            },
            "simple": {
                "format": SIMPLE_FORMAT,
                "style": "{",
            },
            "colored": {
                "()": ColorFormatter,
                "format": VERBOSE_FORMAT,
                "datefmt": DATE_FORMAT,
                "use_colors": enable_colors,
            },
        },
        "filters": {
            "require_debug_true": {
                "()": "django.utils.log.RequireDebugTrue",
            },
            "require_debug_false": {
                "()": "django.utils.log.RequireDebugFalse",
            },
            "sanitize_sql_debug": {
                "()": SQLDebugSanitizerFilter,
            },
        },
        "handlers": handlers,
        "root": {
            "handlers": handler_list,
            "level": "WARNING",
        },
        "loggers": {
            # Django's internal loggers
            "django": {
                "handlers": handler_list,
                "level": "INFO" if enable_django_debug else "WARNING",
                "propagate": False,
            },
            "django.request": {
                "handlers": handler_list,
                "level": "ERROR",
                "propagate": False,
            },
            "django.db.backends": {
                "handlers": handler_list,
                "level": "DEBUG" if enable_sql_debug else "WARNING",
                "filters": ["sanitize_sql_debug"],
                "propagate": False,
            },
            # Application loggers - these are the ones you typically use
            "chatbot": {
                "handlers": handler_list,
                "level": log_level,
                "propagate": False,
            },
            "nlp": {
                "handlers": handler_list,
                "level": log_level,
                "propagate": False,
            },
            "vision": {
                "handlers": handler_list,
                "level": log_level,
                "propagate": False,
            },
            "vector_store": {
                "handlers": handler_list,
                "level": log_level,
                "propagate": False,
            },
            "authentication": {
                "handlers": handler_list,
                "level": log_level,
                "propagate": False,
            },
            "django_core": {
                "handlers": handler_list,
                "level": log_level,
                "propagate": False,
            },
            "utils": {
                "handlers": handler_list,
                "level": log_level,
                "propagate": False,
            },
            "httpx": {
                "handlers": handler_list,
                "level": "WARNING",
                "propagate": False,
            },
            "httpcore": {
                "handlers": handler_list,
                "level": "WARNING",
                "propagate": False,
            },
            "huggingface_hub": {
                "handlers": handler_list,
                "level": "WARNING",
                "propagate": False,
            },
        },
    }

    return config


# Pre-configured log levels for convenience
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
