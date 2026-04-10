"""
Startup warmup helpers for chatbot-serving processes.
"""

import logging
import os
import sys
import threading

from django.conf import settings

logger = logging.getLogger(__name__)

_WARMUP_DONE = False
_WARMUP_LOCK = threading.Lock()
_SKIP_COMMANDS = {"test", "makemigrations", "migrate", "collectstatic", "shell"}


def should_warmup_chatbot_runtime() -> bool:
    """Decide whether startup warmup should run in this process."""
    if not getattr(settings, "ENABLE_STARTUP_WARMUP", False):
        return False

    argv = set(sys.argv[1:])
    if argv & _SKIP_COMMANDS:
        return False

    if "runserver" in argv and settings.DEBUG and os.environ.get("RUN_MAIN") != "true":
        return False

    return True


def warmup_chatbot_runtime(force: bool = False) -> bool:
    """Warm shared NLP and vision runtimes once per process."""
    global _WARMUP_DONE
    if _WARMUP_DONE and not force:
        return True

    with _WARMUP_LOCK:
        if _WARMUP_DONE and not force:
            return True

        from nlp.services.runtime import warmup_nlp_runtime
        from vision.services.vision_service import warmup_vision_runtime

        nlp_ready = warmup_nlp_runtime()
        vision_ready = warmup_vision_runtime(
            settings.VISION_CONFIG_PATH,
            settings.VISION_CHECKPOINT_PATH,
        )
        _WARMUP_DONE = nlp_ready and vision_ready
        logger.info(
            "Chatbot startup warmup complete (nlp=%s, vision=%s)",
            nlp_ready,
            vision_ready,
        )
        return _WARMUP_DONE
