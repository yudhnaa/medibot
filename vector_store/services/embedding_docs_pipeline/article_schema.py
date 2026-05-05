"""Schema helpers for disease article ingestion and index document building."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

from chatbot.models import IndexType, SectionType

SECTION_KEYS: tuple[str, ...] = (
    "general",
    "symptom",
    "aetiologies",
    "risk",
    "diagnose_and_treaty",
    "living_and_preventive",
)

LIST_SECTION_KEYS: tuple[str, ...] = (
    "symptom",
    "aetiologies",
    "risk",
    "diagnose_and_treaty",
    "living_and_preventive",
)

REQUIRED_FIELDS: tuple[str, ...] = ("title", "general")

SECTION_TYPE_MAP: dict[str, str] = {
    "general": SectionType.GENERAL,
    "symptom": SectionType.SYMPTOM,
    "aetiologies": SectionType.AETIOLOGIES,
    "risk": SectionType.RISK,
    "diagnose_and_treaty": SectionType.DIAGNOSE_AND_TREATY,
    "living_and_preventive": SectionType.LIVING_AND_PREVENTIVE,
}

COLLECTION_NAME_BY_INDEX: dict[str, str] = {
    IndexType.C: "medical_documents_titles",
    IndexType.A: "medical_documents_disease",
    IndexType.B: "medical_documents_chunks",
}


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item.strip())
    return output


def _clean_list_values(values: list[Any]) -> list[str]:
    return _dedupe_keep_order(
        [_clean_text(item) for item in values if _clean_text(item)]
    )


def _parse_bracketed_list(text: str) -> list[str] | None:
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            return _clean_list_values(parsed)
    return None


def parse_list_items(value: Any) -> list[str]:
    """Parse list-like values from CSV/LLM payload into list[str]."""
    if value is None:
        return []
    if isinstance(value, list):
        return _clean_list_values(value)
    if not isinstance(value, str):
        return []

    text = _clean_text(value)
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        parsed_items = _parse_bracketed_list(text)
        if parsed_items is not None:
            return parsed_items

    parts = re.split(r"[;\n•·]|,\s*", text)
    return _clean_list_values(parts)


def summarize_items(items: list[str], limit: int = 3) -> list[str]:
    return items[:limit]


@dataclass
class DiseaseArticleRecord:
    """Canonical record structure for ingestion sources (CSV/URL/JSON)."""

    title: str
    general: str
    symptom: list[str] = field(default_factory=list)
    aetiologies: list[str] = field(default_factory=list)
    risk: list[str] = field(default_factory=list)
    diagnose_and_treaty: list[str] = field(default_factory=list)
    living_and_preventive: list[str] = field(default_factory=list)

    source_url: str = ""
    source_type: str = ""
    source_name: str = ""
    row_index: int | None = None
    ingestion_job_id: int | None = None
    ingestion_trace: str = ""

    @property
    def canonical_title(self) -> str:
        return _clean_text(self.title).lower()

    def to_payload(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "general": self.general,
            "symptom": list(self.symptom),
            "aetiologies": list(self.aetiologies),
            "risk": list(self.risk),
            "diagnose_and_treaty": list(self.diagnose_and_treaty),
            "living_and_preventive": list(self.living_and_preventive),
            "source_url": self.source_url,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "row_index": self.row_index,
            "ingestion_job_id": self.ingestion_job_id,
            "ingestion_trace": self.ingestion_trace,
        }

    def _base_metadata(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "canonical_title": self.canonical_title,
            "url": self.source_url,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "row_index": self.row_index,
            "ingestion_job_id": self.ingestion_job_id,
            "ingestion_trace": self.ingestion_trace,
        }

    def _build_summary_text(self) -> str:
        lines = [self.title]
        if self.general:
            lines.append(self.general)

        sections = [
            ("symptom", "Triệu chứng", self.symptom),
            ("aetiologies", "Nguyên nhân", self.aetiologies),
            ("risk", "Yếu tố nguy cơ", self.risk),
            ("diagnose_and_treaty", "Chẩn đoán và điều trị", self.diagnose_and_treaty),
            (
                "living_and_preventive",
                "Sinh hoạt và phòng ngừa",
                self.living_and_preventive,
            ),
        ]
        for section_key, section_label, items in sections:
            if not items:
                continue
            top_items = summarize_items(items, limit=3)
            lines.append(f"{section_label}: {', '.join(top_items)}")
            lines.append(f"{section_key}_top3_count={len(top_items)}")

        return "\n".join(line for line in lines if _clean_text(line))

    def build_index_documents(self, index_type: str) -> list[dict[str, Any]]:
        """Build index-ready document dicts for A/B/C from one article record."""
        base_metadata = self._base_metadata()
        collection_name = COLLECTION_NAME_BY_INDEX.get(index_type, "")

        if index_type == IndexType.C:
            metadata = dict(base_metadata)
            metadata.update(
                {
                    "collection": collection_name,
                    "section": "title",
                }
            )
            return [
                {
                    "title": self.canonical_title,
                    "content": self.canonical_title,
                    "section_type": SectionType.GENERAL,
                    "index_type": IndexType.C,
                    "source": self.source_name or self.source_type,
                    "metadata": metadata,
                }
            ]

        if index_type == IndexType.A:
            summary_content = self._build_summary_text().lower()
            metadata = dict(base_metadata)
            metadata.update(
                {
                    "collection": collection_name,
                    "section": "summary",
                }
            )
            return [
                {
                    "title": self.canonical_title,
                    "content": summary_content,
                    "section_type": SectionType.GENERAL,
                    "index_type": IndexType.A,
                    "source": self.source_name or self.source_type,
                    "metadata": metadata,
                }
            ]

        if index_type == IndexType.B:
            documents: list[dict[str, Any]] = []
            section_payloads: dict[str, str] = {
                "general": self.general,
                "symptom": "\n".join(self.symptom),
                "aetiologies": "\n".join(self.aetiologies),
                "risk": "\n".join(self.risk),
                "diagnose_and_treaty": "\n".join(self.diagnose_and_treaty),
                "living_and_preventive": "\n".join(self.living_and_preventive),
            }

            for section, raw_content in section_payloads.items():
                content = _clean_text(raw_content)
                if not content:
                    continue
                metadata = dict(base_metadata)
                metadata.update(
                    {
                        "collection": collection_name,
                        "section": section,
                    }
                )
                documents.append(
                    {
                        "title": self.canonical_title,
                        "content": content.lower(),
                        "section_type": SECTION_TYPE_MAP[section],
                        "index_type": IndexType.B,
                        "source": self.source_name or self.source_type,
                        "metadata": metadata,
                    }
                )
            return documents

        return []


def normalize_article_record(
    raw: dict[str, Any],
    *,
    source_url: str = "",
    source_type: str = "",
    source_name: str = "",
    row_index: int | None = None,
    ingestion_job_id: int | None = None,
    ingestion_trace: str = "",
) -> tuple[DiseaseArticleRecord | None, list[str]]:
    """Validate and normalize raw payload into DiseaseArticleRecord."""
    errors: list[str] = []

    title = _clean_text(raw.get("title", ""))
    general = _clean_text(raw.get("general", ""))

    if not title:
        errors.append("missing_title")
    if not general:
        errors.append("missing_general")

    if errors:
        return None, errors

    record = DiseaseArticleRecord(
        title=title,
        general=general,
        symptom=parse_list_items(raw.get("symptom")),
        aetiologies=parse_list_items(raw.get("aetiologies")),
        risk=parse_list_items(raw.get("risk")),
        diagnose_and_treaty=parse_list_items(raw.get("diagnose_and_treaty")),
        living_and_preventive=parse_list_items(raw.get("living_and_preventive")),
        source_url=_clean_text(source_url or raw.get("url", "")),
        source_type=_clean_text(source_type),
        source_name=_clean_text(source_name),
        row_index=row_index,
        ingestion_job_id=ingestion_job_id,
        ingestion_trace=_clean_text(ingestion_trace),
    )
    return record, []
