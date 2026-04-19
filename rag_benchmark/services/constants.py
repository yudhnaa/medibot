from __future__ import annotations

import re

REQUIRED_CASE_FIELDS: tuple[str, ...] = (
    "case_id",
    "dataset_version",
    "split",
    "question",
    "intake_payload",
    "scenario",
    "expected_mode",
    "gold_titles",
    "forbidden_titles",
    "must_have_sections",
    "expected_behavior",
    "reference_answer",
    "notes",
)

LIST_FIELDS: tuple[str, ...] = (
    "gold_titles",
    "forbidden_titles",
    "must_have_sections",
    "must_not_sections",
    "reference_context_ids",
)

REQUIRED_SCENARIOS: set[str] = {
    "single_clear",
    "multi_ambiguous",
    "negation",
    "demographic",
    "insufficient_info",
    "out_of_scope",
    "followup_context",
    "paraphrase",
    "noisy_query",
}

DEFAULT_RAGAS_METRICS: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)
DEFAULT_RAGAS_MAX_TOKENS = 4096
DEFAULT_RAGAS_RETRY_MAX_TOKENS = 8192
DEFAULT_RAGAS_TIMEOUT_SECONDS = 240
DEFAULT_ANSWER_RELEVANCY_STRICTNESS = 1
DEFAULT_RAGAS_MAX_WORKERS = 3
DEFAULT_RAGAS_BATCH_SIZE = 8
DEFAULT_RAGAS_CONTEXT_TOP_K = 4
DEFAULT_RAGAS_CONTEXT_CHAR_LIMIT = 700

RAGAS_RELEASE_GATE_THRESHOLDS: dict[str, tuple[str, float]] = {
    "faithfulness": (">=", 0.8),
    "answer_relevancy": (">=", 0.7),
    "context_precision": (">=", 0.6),
    "context_recall": (">=", 0.7),
}

BENCHMARK_RUNNER_USERNAME = "__benchmark_runner__"
RAGAS_JUDGE_METRICS = set(RAGAS_RELEASE_GATE_THRESHOLDS).union(
    {"judge_available_rate", "judge_error_rate"}
)

NEGATION_TOKENS: tuple[str, ...] = (
    " không ",
    " khong ",
    " not ",
    " no ",
    " without ",
)
NOISY_STYLES = {"POOR_GRAMMAR", "MISSPELLED", "NOISE"}

SECTION_PATTERN = re.compile(r"Section\[(?P<section>[^\]]+)\]:")
TITLE_PATTERN = re.compile(r"^Title:\s*(?P<title>.+?)\s*$", re.IGNORECASE)

SUPPORTED_PROVIDERS = {"gemini", "openrouter"}
