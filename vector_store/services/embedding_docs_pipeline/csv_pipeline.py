"""CSV embedding pipeline implementation."""

from vector_store.services.embedding_docs_pipeline.base import BaseEmbeddingPipeline


class CSVEmbeddingPipeline(BaseEmbeddingPipeline):
    """Pipeline variant for embedding CSV records into the vector store."""

    def load_contexts(self, num_docs: int | None = None) -> list[dict]:
        raise NotImplementedError("CSVEmbeddingPipeline does not load article contexts")
