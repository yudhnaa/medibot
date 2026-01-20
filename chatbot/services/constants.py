"""
Chatbot Services Constants
"""

# NER label aliases for mapping various model outputs to standard categories
LABEL_ALIASES = {
    "SYMPTOM": {"SYMPTOM", "DISEASE", "SIGN", "FINDING"},
    "ETIOLOGY": {"ETIOLOGY", "CAUSE", "AETIOLOGY"},
    "RISK": {"RISK", "RISK_FACTOR"},
    "AGE_GROUP": {"AGE_GROUP", "AGE"},
    "PREGNANCY": {"PREGNANCY", "PREGNANT"},
    "BODY_SITE": {"BODY_SITE", "LOCATION"},
    "DURATION": {"DURATION", "TIME"},
    "SEVERITY": {"SEVERITY", "GRADE"},
}

# Synonym expansions for common medical terms (Vietnamese)
SYNONYM_MAP = {
    "phát ban": ["ban đỏ", "ban da"],
    "nôn": ["ói", "ói mửa"],
    "cmv": ["cytomegalovirus"],
    "rubella": ["sởi đức"],
}

# Section ordering and header mappings
SECTION_ORDER = [
    "general",
    "symptom",
    "aetiologies",
    "risk",
    "diagnose_and_treaty",
    "living_and_preventive",
]

SECTION_HEADERS = {
    "general": "Tổng quan",
    "symptom": "Triệu chứng",
    "aetiologies": "Nguyên nhân/Căn nguyên",
    "risk": "Yếu tố nguy cơ",
    "diagnose_and_treaty": "Chẩn đoán & Điều trị",
    "living_and_preventive": "Lối sống & Phòng ngừa",
}


# RAG thresholds (can be overridden by ChatbotConfig)
DEFAULT_RAG_THRESH_C = 0.8
DEFAULT_RAG_B_TOPK = 20
DEFAULT_RAG_MERGED_LIMIT = 40
DEFAULT_RAG_TITLE_TOP_M = 5
DEFAULT_RAG_PENALTY_ALPHA = 0.5
DEFAULT_RAG_NEG_SYM_SIM_THRESH = 0.75
DEFAULT_RAG_FINAL_TITLES = 5
DEFAULT_RAG_MERGE_WEIGHT_ENTITIES = 0.5
DEFAULT_RAG_MERGE_WEIGHT_QUERY = 0.5
