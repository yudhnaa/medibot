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

# LLM query analyzer constants
QUERY_ANALYZER_PROMPT = """\
You are a Vietnamese medical query analyzer.
Extract structured fields from a patient question for a retrieval pipeline.

Return ONLY valid JSON with these keys:
{{
  "normalized_query": "normalized vietnamese query",
  "entities_by_type": {{
    "disease": [],
    "symptom": [],
    "aetiology": [],
    "risk": [],
    "age": [],
    "gender": []
  }},
  "negated_entities": [],
  "affirmed_entities": [],
  "disease_mentions": [],
  "symptom_positive": [],
  "symptom_negative": [],
  "patient_state_extract": {{
    "age": null,
    "sex": null
  }},
  "q_cleaned": "query without negated entities",
  "q_symptom": "symptom-centric query with positive symptoms + disease mentions + patient state",
  "intent": "greeting|intake_query|medical_query|non_medical",
  "response_mode": "greeting|intake|medical|non_medical",
  "should_retrieve": true
}}

Rules:
- Use lowercase text values.
- Keep arrays deduplicated.
- Extract only concrete medical entities actually mentioned by the user.
- symptom_positive and symptom_negative must contain concrete symptom phrases
  (examples: "sốt", "ho khan", "khó thở"), not meta terms.
- Never output generic/meta placeholders in symptom arrays such as:
  "triệu chứng", "dấu hiệu", "biểu hiện", "tình trạng", "vấn đề sức khỏe".
- If user asks about symptoms in general but does not provide concrete symptoms,
  return symptom_positive=[] and symptom_negative=[].
- Set intent/response_mode/should_retrieve for routing:
  * greeting: greetings/thanks/small talk, response_mode="greeting", should_retrieve=false.
  * intake_query: user asks what saved intake/profile/basic information you know about them, including age, sex, symptoms, disease, pregnancy, onset, meds, allergies, or chronic conditions; response_mode="intake", should_retrieve=false.
  * medical_query: user asks for medical advice, diagnosis, disease/symptom explanation, treatment, prevention, or triage; response_mode="medical", should_retrieve=true.
  * non_medical: unrelated topics like weather, sports, finance, travel, or politics; response_mode="non_medical", should_retrieve=false.
- Any question asking you to recall/know/report the user's saved personal or intake data is intake_query, even if worded indirectly. These are intake_query, not medical_query: "tôi bao nhiêu tuổi?", "có thể là tôi bao nhiêu tuổi?", "bạn biết tôi bao nhiêu tuổi không?", "bạn có nhận được các thông tin cơ bản của tôi không?", "bạn có nắm các thông tin cơ bản về tôi không?", "triệu chứng của tôi là gì?", "tôi đã cung cấp thông tin gì?".
- For intake_query, do not turn known patient symptoms into symptom_positive unless the user states new symptoms in the current question.
- If unknown, return empty arrays and null scalar values.
- No markdown.

Question:
{question}

Known patient state:
age={age}, sex={sex}, symptoms={symptoms}
"""

GENERIC_SYMPTOM_TERMS = {
    "triệu chứng",
    "trieu chung",
    "symptom",
    "symptoms",
    "dấu hiệu",
    "dau hieu",
    "biểu hiện",
    "bieu hien",
    "tình trạng",
    "tinh trang",
    "vấn đề sức khỏe",
    "van de suc khoe",
    "sức khỏe",
    "suc khoe",
}

# Query routing constants
INTENT_GREETING = "greeting"
INTENT_INTAKE_QUERY = "intake_query"
INTENT_MEDICAL_QUERY = "medical_query"
INTENT_NON_MEDICAL = "non_medical"

RESPONSE_MODE_GREETING = "greeting"
RESPONSE_MODE_INTAKE = "intake"
RESPONSE_MODE_MEDICAL = "medical"
RESPONSE_MODE_NON_MEDICAL = "non_medical"

RESPONSE_ROUTES = {
    INTENT_GREETING: RESPONSE_MODE_GREETING,
    INTENT_INTAKE_QUERY: RESPONSE_MODE_INTAKE,
    INTENT_MEDICAL_QUERY: RESPONSE_MODE_MEDICAL,
    INTENT_NON_MEDICAL: RESPONSE_MODE_NON_MEDICAL,
}

FALLBACK_GREETING_TERMS = {
    "hi",
    "hello",
    "hey",
    "xin chào",
    "chào",
    "chao",
    "cảm ơn",
    "cam on",
    "thanks",
    "thank you",
}

FALLBACK_INTAKE_TERMS = {
    "intake",
    "thông tin đã lưu",
    "thong tin da luu",
    "thông tin của tôi",
    "thong tin cua toi",
    "hồ sơ của tôi",
    "ho so cua toi",
    "triệu chứng của tôi",
    "trieu chung cua toi",
    "tuổi của tôi",
    "tuoi cua toi",
    "bao nhiêu tuổi",
    "bao nhieu tuoi",
    "bao nhiều tuổi",
    "bao nhieu tuoi",
    "thông tin cơ bản",
    "thong tin co ban",
    "thông tin cơ bản về tôi",
    "thong tin co ban ve toi",
    "tôi đã cung cấp gì",
    "toi da cung cap gi",
}

FALLBACK_NON_MEDICAL_TERMS = {
    "thời tiết",
    "thoi tiet",
    "bóng đá",
    "bong da",
    "chứng khoán",
    "chung khoan",
    "đổi tiền",
    "doi tien",
}

VALID_PATIENT_SEX_VALUES = {"male", "female", "unknown"}

INTAKE_LINE_SPECS = [
    ("disease_name", "Bệnh"),
    ("symptoms", "Triệu chứng (+)"),
    ("symptoms_negated", "Triệu chứng (-)"),
    ("pregnancy_status", "Tình trạng thai kỳ"),
    ("location_country", "Quốc gia"),
    ("chronic_conditions", "Bệnh nền"),
    ("allergies", "Dị ứng"),
    ("onset_days", "Số ngày khởi phát"),
    ("meds", "Thuốc đang dùng"),
]

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
DEFAULT_RAG_TITLES_COLLECTION_THRESHOLD = 0.65
DEFAULT_RAG_CHUNKS_COLLECTION_TOPK = 20
DEFAULT_RAG_MERGED_LIMIT = 40
DEFAULT_RAG_TITLE_TOP_M = 5
DEFAULT_RAG_PENALTY_ALPHA = 0.5
DEFAULT_RAG_NEG_SYM_SIM_THRESH = 0.75
DEFAULT_RAG_NEG_EMBED_BATCH_SIZE = 32
DEFAULT_RAG_FINAL_TITLES = 5
DEFAULT_RAG_MERGE_WEIGHT_ENTITIES = 0.5
DEFAULT_RAG_MERGE_WEIGHT_QUERY = 0.5
DEFAULT_RAG_CHUNKS_COLLECTION_TOPK_FALLBACK = 20
DEFAULT_RAG_CHUNKS_COLLECTION_TOPK_ENRICH = 8

# Document retrieval and display limits
DEFAULT_SINGLE_DISEASE_DOCS_K = 200  # Number of docs to fetch for single disease mode
DEFAULT_DOCS_CACHE_SIZE = 10  # Number of docs to cache for UI display
DEFAULT_SECTION_ITEMS_LIMIT = 10  # Max items per section in context
DEFAULT_DOC_PREVIEW_LENGTH = 300  # Character limit for doc preview
DEFAULT_BENCHMARK_RERANK_PREFILTER_K = 12  # Pre-filter docs before LLM rerank
DEFAULT_BENCHMARK_RERANK_TOP_K = 3  # Final docs used for benchmark generation
DEFAULT_BENCHMARK_DOC_CHAR_LIMIT = 420  # Trim each doc to reduce benchmark token cost
DEFAULT_BENCHMARK_SINGLE_DISEASE_DOCS_K = (
    60  # Fetch fewer docs in benchmark to reduce latency/cost
)
DEFAULT_BENCHMARK_RETRIEVAL_CONTEXT_LIMIT = 3  # Cap contexts sent to Ragas per case
DEFAULT_BENCHMARK_PROMPT_CONTEXT_CHAR_LIMIT = (
    2600  # Trim assembled benchmark prompt context before generation
)
IS_SHAPE_BENCHMARK_ANSWER_ON = False
DEFAULT_BENCHMARK_SECTION_INTENT_BOOST = (
    0.18  # Boost docs whose section aligns with detected benchmark intent
)
DEFAULT_BENCHMARK_SECTION_OFF_TARGET_PENALTY = (
    0.08  # Mild penalty for off-target sections during benchmark rerank
)

# Collection search defaults
DEFAULT_TITLES_COLLECTION_K = 1  # Top k for titles collection gate search
DEFAULT_CHUNKS_COLLECTION_K = 100  # Default k for chunks collection search

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
MSG_GREETING_RESPONSE = (
    "Xin chào! Bạn có thể mô tả triệu chứng hoặc hỏi về thông tin intake đã lưu."
)
MSG_NON_MEDICAL_RESPONSE = (
    "Mình chỉ hỗ trợ câu hỏi y tế và thông tin intake đã lưu. "
    "Bạn có thể mô tả triệu chứng để mình hỗ trợ."
)
MSG_NO_INTAKE_CONTEXT = "Hiện chưa có thông tin intake đã lưu cho bạn."
MSG_INTAKE_RESPONSE_PREFIX = "## Thông tin intake hiện có của bạn\n\n"

# Context headers
HEADER_SINGLE_DISEASE = "THÔNG TIN CHI TIẾT VỀ BỆNH: {title}"
HEADER_SINGLE_DISEASE_SUBTITLE = "(Tổng hợp từ cơ sở dữ liệu y khoa)\n"
HEADER_FAQ_MATCH = "FAQ PHÙ HỢP NHẤT"
HEADER_EVIDENCE_BLOCK = "BẰNG CHỨNG THAM CHIẾU"
HEADER_MULTI_DISEASE_ANALYSIS = "PHÂN TÍCH TRUY VẤN (LLM JSON):"
HEADER_MULTI_DISEASE_CANDIDATES = "\nCÁC BỆNH CÓ KHẢ NĂNG:"
HEADER_PATIENT_INFO = "### Thông tin bệnh nhân\n"

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
