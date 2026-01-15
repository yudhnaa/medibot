"""
Constants for NLP services.
Includes negation patterns, medical NER mappings, and text normalization settings.
"""

import os
from typing import Final

# -----------------------------------------------------------------------------
# Server Paths (Infrastructure)
# -----------------------------------------------------------------------------

_BASE_DIR: Final[str] = os.path.dirname(os.path.abspath(__file__))
_LIBS_DIR: Final[str] = os.path.join(_BASE_DIR, "libs")

VNCORENLP_PATH: Final[str] = os.path.join(_LIBS_DIR, "VNCore-NLP")
MEDICAL_NER_MODEL_PATH: Final[str] = os.path.join(
    _LIBS_DIR, "VietMed_NER", "phobert-base-v2-VietMed-NER"
)

# -----------------------------------------------------------------------------
# Negation Detector Constants
# -----------------------------------------------------------------------------

# Pre-negation patterns with token window sizes
PRE_NEG: Final[dict[str, int]] = {
    "không": 5,
    "chẳng": 4,
    "chưa": 4,
    "không phải": 6,
    "chẳng phải": 6,
    "không bao giờ": 8,
    "chưa bao giờ": 8,
    "không hề": 5,
    "chẳng hề": 5,
    "không thể": 5,
    "chẳng thể": 5,
    "không thể nào": 7,
    "hoàn toàn không": 7,
    "tuyệt đối không": 7,
    "không còn": 5,
    "chẳng còn": 5,
    "không bị": 4,
    "chẳng bị": 4,
    "không có": 5,
    "chẳng có": 5,
    "không mắc": 4,
    "chẳng mắc": 4,
    "không xuất hiện": 6,
    "không biểu hiện": 6,
    "không có dấu hiệu": 8,
    "không thấy": 5,
    "chưa thấy": 5,
    "không phát hiện": 6,
    "chưa phát hiện": 6,
    "loại trừ": 4,
    "bác bỏ": 4,
    "phủ nhận": 4,
}

# Post-negation patterns with token window sizes
POST_NEG: Final[dict[str, int]] = {
    "đâu": 3,
    "được đâu": 4,
    "gì đâu": 4,
    "hết rồi": 4,
    "khỏi rồi": 4,
    "mất rồi": 4,
    "không còn": 4,
    "không": 3,
    "chưa": 3,
}

# Pseudo-negation phrases (do not negate target)
PSEUDO_NEG: Final[list[str]] = [
    "không chỉ",
    "không những",
    "không riêng",
    "không hẳn",
    "không ít",
    "không kém",
]

# Uncertain cues with token window sizes
UNCERTAIN_PRE: Final[dict[str, int]] = {
    "chưa rõ": 5,
    "chưa xác định": 6,
    "có thể": 3,
    "nghi": 3,
    "khả năng": 3,
}

# Scope terminators
SCOPE_TERMINATORS: Final[set[str]] = {
    "nhưng",
    "tuy nhiên",
    "song",
    "trái lại",
    "mà",
    "còn",
    "ngoài ra",
    "bên cạnh đó",
    "đồng thời",
    "cũng như",
    "trong khi",
    "mặc dù",
    "dù",
    "dẫu",
}

PUNCT_TERMINATORS: Final[set[str]] = set(".!?;,:")


# -----------------------------------------------------------------------------
# Medical NER Constants
# -----------------------------------------------------------------------------

NER_LABEL_MAPPING: Final[dict[str, str]] = {
    "DISEASESYMTOM": "DISEASE",
    "DRUGCHEMICAL": "DRUG",
    "DIAGNOSTICS": "TEST",
    "TREATMENT": "TREATMENT",
    "PREVENTIVEMED": "PREVENTION",
    "ORGAN": "ANATOMY",
    "AGE": "AGE",
    "DATETIME": "DATETIME",
    "LOCATION": "LOCATION",
    "GENDER": "GENDER",
    "OCCUPATION": "OCCUPATION",
}


# -----------------------------------------------------------------------------
# Text Normalizer Constants
# -----------------------------------------------------------------------------

VNCORENLP_CACHE_KEY: Final[str] = "__VNCORENLP_SEGMENTER_SINGLETON__"

MEDICAL_PUNCTUATION_PATTERN: Final[str] = r"[^\w\s\-/.,_]"
BASIC_PUNCTUATION_PATTERN: Final[str] = r"[^\w\s_]"
