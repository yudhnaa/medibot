"""COVID-QA embedding pipeline implementation."""

from vector_store.services.embedding_docs_pipeline.base import BaseEmbeddingPipeline
from vector_store.services.embedding_docs_pipeline.constants import NUM_DOCS_TO_PROCESS
from vector_store.services.embedding_docs_pipeline.load_covid_qa import (
    get_unique_context_records,
    load_covid_qa_dataset,
)


class COVIDQAEmbeddingPipeline(BaseEmbeddingPipeline):
    """Default pipeline implementation for the COVID-QA dataset."""

    def load_contexts(
        self, num_docs: int | None = None, start_article: int = 1
    ) -> list[dict]:
        dataset = load_covid_qa_dataset()
        n = num_docs if num_docs is not None else NUM_DOCS_TO_PROCESS
        return get_unique_context_records(
            dataset,
            limit=n,
            start_article=start_article,
        )
