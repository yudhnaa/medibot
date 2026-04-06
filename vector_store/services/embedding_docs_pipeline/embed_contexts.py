"""
embed_contexts.py — Multi-Index COVID-QA Embedding Pipeline.

3-stage pipeline:
  Stage 1 → Index C: Extract disease/syndrome names via LLM
  Stage 2 → Index A: Build structured summaries via LLM
  Stage 3 → Index B: Section-aware text chunking & embedding

All data is stored in the MedicalDocument table (pgvector, 768d).

Embedding is fully abstracted via EmbeddingService — swap the provider
(gemini / openrouter / tei / transformers) without touching pipeline logic:

    from vector_store.pipeline.embed_contexts import embed_contexts
    embed_contexts(provider="gemini")       # default
    embed_contexts(provider="openrouter")
    embed_contexts(provider="transformers")
    embed_contexts(provider="tei")

Static prompts and defaults live in vector_store.pipeline.constants.
Runtime values are read from ChatbotConfig (DB) after local Django bootstrap.
"""

import json
import logging
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from chatbot.models import MedicalDocument
from chatbot.models.medical_document import IndexType, SectionType
from vector_store.services.embedding_docs_pipeline.constants import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_SOURCE_TAG,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_PROVIDER,
    EMBEDDING_SLEEP_SECONDS,
    GOOGLE_API_KEY,
    LLM_MODEL,
    NUM_DOCS_TO_PROCESS,
    PROMPT_CLASSIFY_SECTIONS,
    PROMPT_EXTRACT_DISEASES,
    PROMPT_EXTRACT_SUMMARY,
    RETRY_MAX_ATTEMPTS,
    RETRY_WAIT_MAX,
    RETRY_WAIT_MIN,
    SECTION_TYPE_MAP,
    VALID_SECTIONS,
)
from vector_store.services.embedding_docs_pipeline.load_covid_qa import (
    get_unique_contexts,
    load_covid_qa_dataset,
)
from vector_store.services.embedding_service import EmbeddingService  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# =============================================================================
# LLM creation (disease extraction / summarisation)
# =============================================================================


def _create_llm() -> ChatGoogleGenerativeAI:
    """Create LLM instance for extraction tasks."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=SecretStr(GOOGLE_API_KEY),
        temperature=0.1,
    )


# =============================================================================
# Embedding via EmbeddingService (abstract — provider-agnostic)
# =============================================================================


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    reraise=True,
)
def _embed_texts_with_retry(
    service: EmbeddingService, texts: list[str]
) -> list[list[float]]:
    """
    Embed a batch of texts via EmbeddingService with retry.

    Truncation to EMBEDDING_DIMENSIONS and L2 normalisation are handled
    inside each provider (e.g. GeminiEmbeddingProvider) — no manual
    post-processing needed here.
    """
    return service.embed_documents(texts)


# =============================================================================
# Retry-wrapped LLM calls
# =============================================================================


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    reraise=True,
)
def _llm_invoke(llm: ChatGoogleGenerativeAI, prompt: str) -> str:
    """Call LLM with retry. Returns raw text response."""
    response = llm.invoke(prompt)
    return str(response.content) if hasattr(response, "content") else str(response)


def _parse_json_response(response: str) -> dict | list:
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


# =============================================================================
# Stage 1: Index C — Disease Name Extraction
# =============================================================================


def _extract_diseases_from_context(
    llm: ChatGoogleGenerativeAI, context_text: str
) -> list[str]:
    """Extract disease/syndrome/pathogen names from a single article."""
    text = context_text[:8000] if len(context_text) > 8000 else context_text
    prompt = PROMPT_EXTRACT_DISEASES.format(text=text)

    try:
        response = _llm_invoke(llm, prompt)
        diseases = _parse_json_response(response)
        if isinstance(diseases, list):
            return [str(d).strip() for d in diseases if str(d).strip()]
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Disease extraction failed: {e}")

    return []


def stage_1_index_c(
    contexts: list[dict],
    llm: ChatGoogleGenerativeAI,
    embedding_service: EmbeddingService,
) -> dict[str, set[str]]:
    """
    Stage 1: Extract disease names and populate Index C.

    Returns:
        disease_to_contexts: mapping disease_name -> set of context_ids that mention it
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: Index C — Disease Name Extraction")
    logger.info("=" * 60)

    disease_to_contexts: dict[str, set[str]] = {}

    for ctx in tqdm(contexts, desc="Stage 1: Extracting diseases"):
        context_id = ctx["context_id"]
        context_text = ctx["context"]

        diseases = _extract_diseases_from_context(llm, context_text)
        for disease in diseases:
            normalized = disease.strip().lower()
            if normalized:
                disease_to_contexts.setdefault(normalized, set()).add(context_id)

        time.sleep(EMBEDDING_SLEEP_SECONDS)

    if not disease_to_contexts:
        logger.warning("No diseases extracted. Pipeline cannot continue.")
        return {}

    logger.info(
        f"Extracted {len(disease_to_contexts)} unique diseases: "
        f"{list(disease_to_contexts.keys())}"
    )

    # Embed and store each disease name
    disease_names = list(disease_to_contexts.keys())
    embeddings = _embed_texts_with_retry(embedding_service, disease_names)

    for disease_name, embedding in zip(disease_names, embeddings):
        MedicalDocument.objects.create(
            title=disease_name,
            content=disease_name,
            embedding=embedding,
            section_type=SectionType.GENERAL,
            index_type=IndexType.C,
            source=DOCUMENT_SOURCE_TAG,
            metadata={
                "dataset": "deepset/covid_qa_deepset",
                "source_contexts": list(disease_to_contexts[disease_name]),
            },
        )

    logger.info(f"Index C: {len(disease_names)} disease entries stored")
    return disease_to_contexts


# =============================================================================
# Stage 2: Index A — Structured Summary Extraction
# =============================================================================


def _build_summary_text(summary_data: dict) -> str:
    """
    Convert LLM-extracted JSON into text format matching _build_index_a_summary.

    Format:
        disease_name
        general description sentence
        triệu chứng: item1, item2
        nguyên nhân: item1, item2
        ...
    """
    lines = []

    disease_name = summary_data.get("disease_name", "").strip()
    if disease_name:
        lines.append(disease_name)

    general = summary_data.get("general", "").strip()
    if general:
        lines.append(general)

    section_labels = [
        ("triệu chứng", summary_data.get("triệu chứng", "")),
        ("nguyên nhân", summary_data.get("nguyên nhân", "")),
        ("yếu tố nguy cơ", summary_data.get("yếu tố nguy cơ", "")),
        ("chẩn đoán và điều trị", summary_data.get("chẩn đoán và điều trị", "")),
        ("sinh hoạt và phòng ngừa", summary_data.get("sinh hoạt và phòng ngừa", "")),
    ]

    for label, value in section_labels:
        if value and value.strip():
            lines.append(f"{label}: {value.strip()}")

    return "\n".join(lines)


def stage_2_index_a(
    contexts: list[dict],
    disease_to_contexts: dict[str, set[str]],
    llm: ChatGoogleGenerativeAI,
    embedding_service: EmbeddingService,
) -> int:
    """
    Stage 2: Extract structured summaries and populate Index A.

    For each article, extract summaries for each disease mentioned in it.

    Returns:
        Number of Index A documents created
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: Index A — Structured Summary Extraction")
    logger.info("=" * 60)

    ctx_lookup = {ctx["context_id"]: ctx["context"] for ctx in contexts}
    count = 0

    for disease_name, context_ids in tqdm(
        disease_to_contexts.items(), desc="Stage 2: Building summaries"
    ):
        for context_id in context_ids:
            context_text = ctx_lookup.get(context_id, "")
            if not context_text:
                continue

            text = context_text[:8000] if len(context_text) > 8000 else context_text
            prompt = PROMPT_EXTRACT_SUMMARY.format(disease_name=disease_name, text=text)

            try:
                response = _llm_invoke(llm, prompt)
                summary_data = _parse_json_response(response)
                if not isinstance(summary_data, dict):
                    logger.warning(
                        f"Non-dict summary for {disease_name}: {type(summary_data)}"
                    )
                    continue

                summary_data["disease_name"] = disease_name

                summary_text = _build_summary_text(summary_data)
                if not summary_text.strip():
                    continue

                embedding = _embed_texts_with_retry(
                    embedding_service, [summary_text.lower()]
                )[0]

                MedicalDocument.objects.create(
                    title=disease_name,
                    content=summary_text.lower(),
                    embedding=embedding,
                    section_type=SectionType.GENERAL,
                    index_type=IndexType.A,
                    source=DOCUMENT_SOURCE_TAG,
                    metadata={
                        "context_id": context_id,
                        "dataset": "deepset/covid_qa_deepset",
                    },
                )
                count += 1

            except Exception as e:
                logger.warning(
                    f"Summary extraction failed for {disease_name} "
                    f"(context {context_id}): {e}"
                )

            time.sleep(EMBEDDING_SLEEP_SECONDS)

    logger.info(f"Index A: {count} summary entries stored")
    return count


# =============================================================================
# Stage 3: Index B — Section-Aware Chunking
# =============================================================================


def stage_3_index_b(
    contexts: list[dict],
    disease_to_contexts: dict[str, set[str]],
    llm: ChatGoogleGenerativeAI,
    embedding_service: EmbeddingService,
) -> int:
    """
    Stage 3: Classify text into sections, chunk, and populate Index B.

    Returns:
        Number of Index B documents created
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: Index B — Section-Aware Chunking")
    logger.info("=" * 60)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )

    ctx_lookup = {ctx["context_id"]: ctx["context"] for ctx in contexts}
    count = 0

    # Invert mapping: context_id -> list of diseases
    context_to_diseases: dict[str, list[str]] = {}
    for disease_name, context_ids in disease_to_contexts.items():
        for cid in context_ids:
            context_to_diseases.setdefault(cid, []).append(disease_name)

    for context_id, diseases in tqdm(
        context_to_diseases.items(), desc="Stage 3: Section chunking"
    ):
        context_text = ctx_lookup.get(context_id, "")
        if not context_text:
            continue

        text = context_text[:8000] if len(context_text) > 8000 else context_text
        prompt = PROMPT_CLASSIFY_SECTIONS.format(text=text)

        try:
            response = _llm_invoke(llm, prompt)
            sections = _parse_json_response(response)
            if not isinstance(sections, dict):
                logger.warning(f"Non-dict sections for context {context_id}")
                continue
        except Exception as e:
            logger.warning(f"Section classification failed for {context_id}: {e}")
            time.sleep(EMBEDDING_SLEEP_SECONDS)
            continue

        for section_key in VALID_SECTIONS:
            section_text = sections.get(section_key, "").strip()
            if not section_text:
                continue

            section_type = SECTION_TYPE_MAP.get(section_key, SectionType.GENERAL)
            chunks = splitter.split_text(section_text)
            if not chunks:
                continue

            try:
                chunks_lower = [c.lower() for c in chunks]
                embeddings = _embed_texts_with_retry(embedding_service, chunks_lower)
            except Exception as e:
                logger.warning(f"Embedding failed for {context_id}/{section_key}: {e}")
                continue

            for disease_name in diseases:
                for i, (chunk_text, embedding) in enumerate(
                    zip(chunks_lower, embeddings)
                ):
                    MedicalDocument.objects.create(
                        title=disease_name,
                        content=chunk_text,
                        embedding=embedding,
                        section_type=section_type,
                        index_type=IndexType.B,
                        source=DOCUMENT_SOURCE_TAG,
                        metadata={
                            "context_id": context_id,
                            "disease": disease_name,
                            "section": section_key,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "dataset": "deepset/covid_qa_deepset",
                        },
                    )
                    count += 1

            time.sleep(EMBEDDING_SLEEP_SECONDS)

    logger.info(f"Index B: {count} chunk entries stored")
    return count


# =============================================================================
# Cleanup
# =============================================================================


def clear_covid_qa_documents() -> int:
    """Remove all COVID-QA documents from MedicalDocument table."""
    result = MedicalDocument.objects.filter(source=DOCUMENT_SOURCE_TAG).delete()
    deleted_count = result[0] if isinstance(result, tuple) else result
    logger.info(f"🗑️  Deleted {deleted_count} COVID-QA documents")
    return deleted_count


# =============================================================================
# Main pipeline
# =============================================================================


def embed_contexts(
    num_docs: int | None = None,
    provider: str | None = None,
    **provider_kwargs,
) -> dict:
    """
    Main pipeline: Extract, embed, and store COVID-QA data into 3 indexes.

    Args:
        num_docs:         Number of articles to process (default: NUM_DOCS_TO_PROCESS)
        provider:         Embedding provider override — "gemini", "openrouter", "tei", "transformers".
                          Defaults to EMBEDDING_PROVIDER from ChatbotConfig.
        **provider_kwargs: Additional kwargs forwarded to EmbeddingService / provider

    Returns:
        Stats dict with counts per index

    Examples:
        embed_contexts()                           # provider from ChatbotConfig
        embed_contexts(provider="transformers")    # force local Transformers
        embed_contexts(num_docs=10, provider="tei")
    """
    n = num_docs or NUM_DOCS_TO_PROCESS
    active_provider = provider or EMBEDDING_PROVIDER  # ChatbotConfig default
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("COVID-QA Multi-Index Embedding Pipeline")
    logger.info(f"Processing {n} articles | Provider: {active_provider}")
    logger.info(
        f"Dimensions: {EMBEDDING_DIMENSIONS} | Chunk: {CHUNK_SIZE}/{CHUNK_OVERLAP}"
    )
    logger.info("=" * 60)

    # Load unique contexts
    dataset = load_covid_qa_dataset()
    unique_contexts_dict = get_unique_contexts(dataset)
    all_contexts = [
        {"context_id": k, "context": v} for k, v in unique_contexts_dict.items()
    ]
    contexts = all_contexts[:n]
    logger.info(f"Loaded {len(contexts)} contexts (of {len(all_contexts)} total)")

    # Clear existing data
    clear_covid_qa_documents()

    # Initialise models
    llm = _create_llm()
    embedding_service = EmbeddingService(provider=active_provider, **provider_kwargs)

    # Stage 1: Index C
    disease_to_contexts = stage_1_index_c(contexts, llm, embedding_service)
    if not disease_to_contexts:
        logger.error("Stage 1 failed — no diseases extracted. Aborting.")
        return {"index_c": 0, "index_a": 0, "index_b": 0}

    # Stage 2: Index A
    index_a_count = stage_2_index_a(
        contexts, disease_to_contexts, llm, embedding_service
    )

    # Stage 3: Index B
    index_b_count = stage_3_index_b(
        contexts, disease_to_contexts, llm, embedding_service
    )

    elapsed = time.time() - start_time
    stats = {
        "index_c": len(disease_to_contexts),
        "index_a": index_a_count,
        "index_b": index_b_count,
        "total": len(disease_to_contexts) + index_a_count + index_b_count,
        "elapsed_seconds": round(elapsed, 1),
    }

    logger.info("=" * 60)
    logger.info("Pipeline Complete!")
    logger.info(f"  Index C (diseases):  {stats['index_c']}")
    logger.info(f"  Index A (summaries): {stats['index_a']}")
    logger.info(f"  Index B (chunks):    {stats['index_b']}")
    logger.info(f"  Total documents:     {stats['total']}")
    logger.info(f"  Elapsed time:        {stats['elapsed_seconds']}s")
    logger.info("=" * 60)

    return stats


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="COVID-QA Multi-Index Embedding Pipeline"
    )
    parser.add_argument(
        "-n",
        "--num-docs",
        type=int,
        default=None,
        help=f"Number of articles to process (default: {NUM_DOCS_TO_PROCESS})",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["gemini", "openrouter", "tei", "transformers"],
        help=f"Embedding provider to use (default from ChatbotConfig: {EMBEDDING_PROVIDER})",
    )
    parser.add_argument(
        "--clear-only",
        action="store_true",
        help="Only clear existing COVID-QA data, don't embed",
    )
    args = parser.parse_args()

    if args.clear_only:
        clear_covid_qa_documents()
    else:
        embed_contexts(num_docs=args.num_docs, provider=args.provider)
