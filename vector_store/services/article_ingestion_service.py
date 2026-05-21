"""Ingestion service for disease article sources (URL crawl + LLM extraction)."""

from __future__ import annotations

import json
import logging
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from chatbot.services.gemini_manager import get_gemini_manager
from vector_store.services.embedding_docs_pipeline.article_schema import (
    normalize_article_record,
)
from vector_store.services.vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)

MAX_CRAWL_TEXT_LEN = 9999999999999

PROMPT_EXTRACT_DISEASE_ARTICLE = """\
You are a medical information extraction engine.
Extract structured disease content from the article text below.

Return ONLY valid JSON with exactly these keys:
{{
  "title": "primary disease name",
  "aliases": ["synonym1", "translated name", "abbreviation"],
  "general": "concise overview",
  "symptom": ["item1", "item2"],
  "aetiologies": ["item1", "item2"],
  "risk": ["item1", "item2"],
  "diagnose_and_treaty": ["item1", "item2"],
  "living_and_preventive": ["item1", "item2"]
}}

Rules:
- Do not include markdown.
- Keep lists concise, atomic, medically meaningful.
- Include title aliases from the article, including bilingual names, abbreviations, and common synonyms.
- If missing, return empty list for list fields.
- Ensure "title" and "general" are not empty.

Article title:
{title}

Article text:
{text}
"""


class _ArticleHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_tag = False
        self._title_chunks: list[str] = []
        self._text_chunks: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = (tag or "").lower()
        if lowered in {"script", "style", "noscript"}:
            self._skip_tag = True
        if lowered == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        lowered = (tag or "").lower()
        if lowered in {"script", "style", "noscript"}:
            self._skip_tag = False
        if lowered == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_tag:
            return
        cleaned = re.sub(r"\s+", " ", data or "").strip()
        if not cleaned:
            return
        if self._in_title:
            self._title_chunks.append(cleaned)
        self._text_chunks.append(cleaned)

    @property
    def title(self) -> str:
        return " ".join(self._title_chunks).strip()

    @property
    def text(self) -> str:
        return " ".join(self._text_chunks).strip()


class ArticleIngestionService:
    """Crawl source URL, extract structured record with LLM, and embed into medical document collections."""

    def __init__(self, embedding_provider: str | None = None) -> None:
        self.embedding_provider = embedding_provider
        self.vector_manager = VectorStoreManager(embedding_provider=embedding_provider)
        self.llm_manager = get_gemini_manager()
        self.llm = self.llm_manager.create_llm(temperature=0.0)

    def _parse_json_response(self, response_text: str) -> dict[str, Any]:
        text = str(response_text or "").strip()
        if text.startswith("```"):
            lines = [
                line for line in text.splitlines() if not line.strip().startswith("```")
            ]
            text = "\n".join(lines).strip()
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("LLM output must be a JSON object")
        return parsed

    def _crawl_url(self, url: str) -> dict[str, str]:
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Invalid URL. Only http/https URLs are supported.")

        req = Request(
            url.strip(),
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                )
            },
        )
        with urlopen(req, timeout=20) as response:  # nosec B310
            content_type = response.headers.get("Content-Type", "")
            if (
                "text/html" not in content_type
                and "application/xhtml+xml" not in content_type
            ):
                raise ValueError(
                    f"URL is not an HTML page (content-type={content_type})"
                )
            raw_html = response.read()

        html = raw_html.decode("utf-8", errors="ignore")
        parser = _ArticleHTMLParser()
        parser.feed(html)
        page_title = parser.title
        page_text = parser.text[:MAX_CRAWL_TEXT_LEN]

        if not page_text:
            raise ValueError("No extractable text found on URL")

        return {
            "title": page_title,
            "text": page_text,
        }

    def _extract_record_from_text(
        self,
        *,
        source_url: str,
        page_title: str,
        page_text: str,
        source_type: str,
        source_name: str,
        ingestion_trace: str,
        ingestion_job_id: int | None,
    ) -> dict[str, Any]:
        prompt = PROMPT_EXTRACT_DISEASE_ARTICLE.format(
            title=page_title.strip() or "unknown",
            text=page_text,
        )
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else response
        if isinstance(content, list):
            content = "\n".join(str(part) for part in content)
        payload = self._parse_json_response(str(content))

        record, errors = normalize_article_record(
            payload,
            source_url=source_url,
            source_type=source_type,
            source_name=source_name,
            row_index=None,
            ingestion_job_id=ingestion_job_id,
            ingestion_trace=ingestion_trace,
        )
        if record is None:
            raise ValueError(f"Invalid extracted payload: {errors}")
        return record.to_payload()

    def ingest_from_url(
        self,
        *,
        url: str,
        source: str = "admin_url",
        ingestion_job_id: int | None = None,
    ) -> dict[str, Any]:
        """Run URL crawl -> LLM extraction -> collection embedding."""
        crawled = self._crawl_url(url)
        return self.ingest_from_text(
            title=crawled["title"],
            text=crawled["text"],
            source_url=url,
            source=source,
            source_type="url",
            ingestion_trace="url_crawl_llm",
            ingestion_job_id=ingestion_job_id,
        )

    def ingest_from_text(
        self,
        *,
        title: str,
        text: str,
        source_url: str = "",
        source: str = "admin_paste",
        source_type: str = "paste",
        ingestion_trace: str = "paste_llm",
        ingestion_job_id: int | None = None,
    ) -> dict[str, Any]:
        """Run pasted text -> LLM extraction -> collection embedding."""
        page_title = title.strip()
        page_text = text.strip()[:MAX_CRAWL_TEXT_LEN]
        if not page_text:
            raise ValueError("No pasted text provided")

        normalized_payload = self._extract_record_from_text(
            source_url=source_url,
            page_title=page_title,
            page_text=page_text,
            source_type=source_type,
            source_name=source,
            ingestion_trace=ingestion_trace,
            ingestion_job_id=ingestion_job_id,
        )
        record, errors = normalize_article_record(
            normalized_payload,
            source_url=source_url,
            source_type=source_type,
            source_name=source,
            row_index=None,
            ingestion_job_id=ingestion_job_id,
            ingestion_trace=ingestion_trace,
        )
        if record is None:
            raise ValueError(f"Failed to normalize extracted record: {errors}")

        docs = record.build_all_collection_documents()
        created_docs = self.vector_manager.add_documents(docs)
        collection_counts: dict[str, int] = {}
        for doc in docs:
            collection_name = str(doc.get("collection_name", ""))
            collection_counts[collection_name] = (
                collection_counts.get(collection_name, 0) + 1
            )

        return {
            "title": record.canonical_title,
            "collections": collection_counts,
            "total": len(created_docs),
            "source_url": source_url,
        }
