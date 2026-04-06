"""Compatibility wrapper for the class-based embedding pipelines."""

from vector_store.services.embedding_docs_pipeline.base import clear_covid_qa_documents
from vector_store.services.embedding_docs_pipeline.csv_pipeline import (
    CSVEmbeddingPipeline,
)
from vector_store.services.embedding_docs_pipeline.covidqa_pipeline import (
    COVIDQAEmbeddingPipeline,
)


def embed_contexts(
    num_docs: int | None = None,
    provider: str | None = None,
    **provider_kwargs,
) -> dict:
    """Run the default COVID-QA embedding pipeline."""
    pipeline = COVIDQAEmbeddingPipeline(provider, **provider_kwargs)
    stats = pipeline.run(num_docs=num_docs)
    return stats


def build_csv_documents(
    csv_path: str,
    source: str,
    index_type: str = "B",
    provider: str | None = None,
) -> list[dict[str, object]]:
    """Build document dictionaries from a CSV file."""
    pipeline = CSVEmbeddingPipeline(provider)
    return pipeline.build_documents_from_csv(
        csv_path=csv_path,
        source=source,
        index_type=index_type,
    )


__all__ = [
    "COVIDQAEmbeddingPipeline",
    "CSVEmbeddingPipeline",
    "build_csv_documents",
    "clear_covid_qa_documents",
    "embed_contexts",
]
