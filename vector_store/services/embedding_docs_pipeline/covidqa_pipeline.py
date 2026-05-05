"""COVID-QA embedding pipeline implementation."""

from typing import Any

from vector_store.services.embedding_docs_pipeline.base import BaseEmbeddingPipeline
from vector_store.services.embedding_docs_pipeline.constants import NUM_DOCS_TO_PROCESS
from vector_store.services.embedding_docs_pipeline.load_covid_qa import (
    get_qa_pairs_for_contexts,
    get_unique_context_records,
    load_covid_qa_dataset,
)


class COVIDQAEmbeddingPipeline(BaseEmbeddingPipeline):
    """Default pipeline implementation for the COVID-QA dataset."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dataset_cache: Any = None

    def load_contexts(
        self, num_docs: int | None = None, start_article: int = 1
    ) -> list[dict]:
        dataset = load_covid_qa_dataset()
        self._dataset_cache = dataset
        n = num_docs if num_docs is not None else NUM_DOCS_TO_PROCESS
        return get_unique_context_records(
            dataset,
            limit=n,
            start_article=start_article,
        )

    def load_qa_pairs(self, contexts: list[dict]) -> list[dict[str, Any]]:
        """Load QA pairs for the selected context window."""
        dataset = self._dataset_cache or load_covid_qa_dataset()
        selected_context_ids = {
            str(context_id)
            for context_id in (
                ctx.get("context_id") for ctx in contexts if isinstance(ctx, dict)
            )
            if context_id
        }
        if not selected_context_ids:
            return []
        return get_qa_pairs_for_contexts(dataset, selected_context_ids)
