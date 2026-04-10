"""
load_covid_qa.py - Load and process the COVID-QA dataset from deepset (Hugging Face).

This module:
1. Loads the covid_qa_deepset dataset via HuggingFace `datasets`
2. Extracts unique contexts (articles) and deduplicates them
3. Creates a context_id → context_text mapping
4. Retrieves Q&A pairs for selected contexts

Run directly to inspect dataset statistics:
    python -m vector_store.pipeline.load_covid_qa
"""

import hashlib
import logging

from datasets import load_dataset

COVID_QA_DATASET_NAME = "deepset/covid_qa_deepset"
ARTICLE_ID_PREFIX = "covidqa"

logger = logging.getLogger(__name__)


def _generate_context_id(context_text: str) -> str:
    """
    Generate a unique context_id from article content using MD5 hash.
    Uses the first 12 characters for compactness while still ensuring uniqueness.

    Args:
        context_text: Raw article content

    Returns:
        12-character hex hash string
    """
    return hashlib.md5(context_text.encode("utf-8")).hexdigest()[:12]


def load_covid_qa_dataset():
    """
    Load the covid_qa_deepset dataset from Hugging Face.

    Returns:
        Dataset object (split 'train')
    """
    logger.info("Loading dataset '%s' from Hugging Face...", COVID_QA_DATASET_NAME)
    dataset = load_dataset(COVID_QA_DATASET_NAME, split="train")
    logger.info(f"Loaded successfully! Total samples: {len(dataset)}")
    return dataset


def get_unique_context_records(
    dataset,
    limit: int | None = None,
    start_article: int = 1,
) -> list[dict[str, str]]:
    """
    Extract unique contexts and assign deterministic article IDs.

    Supports selecting a contiguous window by 1-based article index:
    - start_article: first unique article number to include (default 1)
    - limit: maximum number of unique articles to return

    Returns:
        List of dicts with keys: article_id, context_id, context
    """
    if start_article < 1:
        raise ValueError("start_article must be >= 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1 when provided")

    seen_context_ids: set[str] = set()
    selected: list[tuple[int, str, str]] = []
    unique_ordinal = 0
    end_article = None if limit is None else start_article + limit - 1

    for sample in dataset:
        context_text = sample["context"]
        context_id = _generate_context_id(context_text)
        if context_id in seen_context_ids:
            continue

        seen_context_ids.add(context_id)
        unique_ordinal += 1

        if unique_ordinal < start_article:
            continue
        if end_article is not None and unique_ordinal > end_article:
            break

        selected.append((unique_ordinal, context_id, context_text))

    records: list[dict[str, str]] = []
    for ordinal, context_id, context_text in selected:
        width = max(3, len(str(ordinal)))
        records.append(
            {
                "article_id": f"{ARTICLE_ID_PREFIX}-{ordinal:0{width}d}",
                "context_id": context_id,
                "context": context_text,
            }
        )

    logger.info(
        "Prepared %s unique article records (start_article=%s, limit=%s)",
        len(records),
        start_article,
        limit,
    )
    return records


def get_unique_contexts(dataset) -> dict[str, str]:
    """
    Extract unique contexts (articles) from the dataset.

    Each dataset sample may share a context (article).
    This function deduplicates and returns {context_id: context_text}.

    Args:
        dataset: Dataset loaded from load_covid_qa_dataset()

    Returns:
        Dict mapping context_id → context_text (unique articles only)
    """
    records = get_unique_context_records(dataset)
    unique_contexts = {record["context_id"]: record["context"] for record in records}
    logger.info("Found %s unique articles (contexts)", len(unique_contexts))
    return unique_contexts


def get_qa_pairs_for_contexts(
    dataset,
    selected_context_ids: set[str],
) -> list[dict[str, str]]:
    """
    Retrieve all Q&A pairs belonging to the selected contexts.

    Args:
        dataset: Dataset loaded from load_covid_qa_dataset()
        selected_context_ids: Set of context_ids that were embedded

    Returns:
        List of dicts with keys: context_id, question, answer
    """
    qa_pairs: list[dict[str, str]] = []

    for sample in dataset:
        context_text = sample["context"]
        context_id = _generate_context_id(context_text)

        if context_id not in selected_context_ids:
            continue

        question = sample["question"]
        answers = sample["answers"]

        # answers structure: {"text": [...], "answer_start": [...]}
        if answers and answers["text"]:
            answer_text = answers["text"][0]
        else:
            answer_text = ""

        qa_pairs.append(
            {
                "context_id": context_id,
                "question": question,
                "answer": answer_text,
            }
        )

    logger.info(
        f"Found {len(qa_pairs)} Q&A pairs for {len(selected_context_ids)} selected contexts"
    )
    return qa_pairs


# =============================================================================
# Main — run directly to inspect dataset statistics
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("COVID-QA Dataset Loader - Statistics")
    print("=" * 60)

    ds = load_covid_qa_dataset()
    print(f"\n📊 Total samples in dataset: {len(ds)}")

    contexts = get_unique_contexts(ds)
    print(f"📄 Unique articles (contexts): {len(contexts)}")

    print("\n--- First 3 articles (200 chars) ---")
    for i, (cid, ctext) in enumerate(list(contexts.items())[:3]):
        print(f"  [{cid}] {ctext[:200]}...")

    sample_ids = set(list(contexts.keys())[:5])
    qa = get_qa_pairs_for_contexts(ds, sample_ids)
    print(f"\n❓ Q&A pairs for first 5 contexts: {len(qa)}")

    if qa:
        print("\n--- Example Q&A pair ---")
        print(f"  Q: {qa[0]['question']}")
        print(f"  A: {qa[0]['answer']}")

    print("\nDataset inspection complete!")
