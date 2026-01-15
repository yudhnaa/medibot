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

import logging
import sys
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
                "level": "DEBUG" if enable_django_debug else "WARNING",
                "propagate": False,
            },
            # Application loggers - these are the ones you typically use
            "chatbot": {
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
