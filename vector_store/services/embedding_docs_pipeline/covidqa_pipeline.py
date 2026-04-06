"""COVID-QA embedding pipeline implementation."""

from vector_store.services.embedding_docs_pipeline.base import BaseEmbeddingPipeline
from vector_store.services.embedding_docs_pipeline.constants import NUM_DOCS_TO_PROCESS
from vector_store.services.embedding_docs_pipeline.load_covid_qa import (
    get_unique_contexts,
    load_covid_qa_dataset,
)


class COVIDQAEmbeddingPipeline(BaseEmbeddingPipeline):
    """Default pipeline implementation for the COVID-QA dataset."""

    def load_contexts(self, num_docs: int | None = None) -> list[dict]:
        dataset = load_covid_qa_dataset()
        unique_contexts_dict = get_unique_contexts(dataset)
        all_contexts = [
            {"context_id": k, "context": v} for k, v in unique_contexts_dict.items()
        ]
        n = num_docs or NUM_DOCS_TO_PROCESS
        return all_contexts[:n]
