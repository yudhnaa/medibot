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

# Document retrieval and display limits
DEFAULT_SINGLE_DISEASE_DOCS_K = 200  # Number of docs to fetch for single disease mode
DEFAULT_DOCS_CACHE_SIZE = 10  # Number of docs to cache for UI display
DEFAULT_SECTION_ITEMS_LIMIT = 10  # Max items per section in context
DEFAULT_DOC_PREVIEW_LENGTH = 300  # Character limit for doc preview

# Index search defaults
DEFAULT_INDEX_C_K = 1  # Top k for Index C gate search
DEFAULT_INDEX_B_K = 100  # Default k for Index B search

# Vietnamese error and context messages
MSG_ANALYSIS_ERROR = (
    "Không thể phân tích đầy đủ truy vấn. Vui lòng cung cấp thêm thông tin."
)
MSG_NO_DOCS_FOR_TITLE = "Không tìm thấy tài liệu cho bệnh: {title}"
MSG_CONTEXT_HINT_SINGLE = "\nGỢI Ý: Tóm tắt triệu chứng, nguyên nhân, điều trị. Không suy diễn ngoài tài liệu."
MSG_CONTEXT_HINT_MULTI = (
    "\nGỢI Ý: Trình bày danh sách bệnh có tổ chức. Khuyến nghị cung cấp thêm thông tin."
)
MSG_PROCESSING_ERROR = "Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn: {error}"
MSG_STREAMING_ERROR = "Lỗi: {error}"

# Context headers
HEADER_SINGLE_DISEASE = "THÔNG TIN CHI TIẾT VỀ BỆNH: {title}"
HEADER_SINGLE_DISEASE_SUBTITLE = "(Tổng hợp từ cơ sở dữ liệu y khoa)\n"
HEADER_MULTI_DISEASE_ANALYSIS = "PHÂN TÍCH TRUY VẤN (NER & phủ định):"
HEADER_MULTI_DISEASE_CANDIDATES = "\nCÁC BỆNH CÓ KHẢ NĂNG:"
HEADER_PATIENT_INFO = "\nTHÔNG TIN BỆNH NHÂN:\n"

# X-Ray context and messages
HEADER_XRAY_ANALYSIS_START = (
    "=== KẾT QUẢ PHÂN TÍCH HÌNH ẢNH X-QUANG TỪ NGƯỜI DÙNG ===\n"
)
XRAY_RESPIRATORY_DOMAIN_CONTEXT_VI = (
    "NGỮ CẢNH MIỀN: Đây là ca tư vấn y tế hô hấp/phổi có ảnh X-quang ngực. "
    "Ưu tiên giải thích trong bối cảnh COVID-19, viêm phổi, tổn thương phổi và "
    "triệu chứng hô hấp. Nếu tài liệu truy hồi không liên quan bệnh người hoặc "
    "không thuộc ngữ cảnh hô hấp/phổi thì không dùng để kết luận.\n"
)
MSG_XRAY_INSTRUCTION = (
    "Người dùng đã gửi kèm một ảnh X-quang phổi. "
    "Hệ thống Vision AI đã phân tích ảnh này và đưa ra kết quả sau. "
    "Hãy sử dụng thông tin này cùng với triệu chứng của người dùng "
    "để đưa ra phản hồi chính xác nhất.\n\n"
)
HEADER_XRAY_PREDICTION = "- Chẩn đoán dự đoán: {pred_label}\n"
HEADER_XRAY_PROBABILITIES = "- Xác suất các bệnh lý (top 5): {probs_str}\n"
HEADER_XRAY_FINDINGS = "- Các phát hiện trên ảnh: {findings_str}\n"
HEADER_XRAY_ANALYSIS_END = "=== HẾT KẾT QUẢ PHÂN TÍCH X-QUANG ===\n\n"
