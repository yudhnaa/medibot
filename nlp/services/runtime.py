"""
Shared NLP runtime helpers for chatbot request processing.
"""

import builtins
import threading

from nlp.services.integrator import NERNegationIntegrator
from nlp.services.medical_ner import MedicalNER
from nlp.services.text_normalizer import TextNormalizer
from utils.logger import get_logger

logger = get_logger(__name__)

_INTEGRATOR: NERNegationIntegrator | None = None
_INTEGRATOR_LOCK = threading.Lock()


def get_shared_integrator() -> NERNegationIntegrator:
    """Return a process-wide NLP integrator instance."""
    global _INTEGRATOR
    if _INTEGRATOR is None:
        with _INTEGRATOR_LOCK:
            if _INTEGRATOR is None:
                _INTEGRATOR = NERNegationIntegrator(enable_text_normalization=True)
                logger.info("Shared NLP runtime initialized")
    return _INTEGRATOR


def warmup_nlp_runtime() -> bool:
    """Warm shared NLP assets for lower first-request latency."""
    try:
        get_shared_integrator()
        return True
    except Exception as exc:
        logger.warning("NLP warmup failed: %s", exc)
        return False


def reset_nlp_runtime_cache() -> None:
    """Reset singleton NLP runtime for tests."""
    global _INTEGRATOR
    _INTEGRATOR = None
    MedicalNER.clear_cache()
    TextNormalizer._SEGMENTER = None
    TextNormalizer._SEGMENTER_INIT_TRIED = False
    cache_key = TextNormalizer._GLOBAL_CACHE_KEY
    if hasattr(builtins, cache_key):
        delattr(builtins, cache_key)
