"""
Tests for VectorStoreManager
Tests document storage, search, and CSV processing with mocked embeddings.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.test import TestCase

import pandas as pd

from chatbot.models import IndexType, MedicalDocument, SectionType
from vector_store.services.embedding_docs_pipeline.covidqa_pipeline import (
    COVIDQAEmbeddingPipeline,
)
from vector_store.services.embedding_docs_pipeline.load_covid_qa import (
    get_unique_context_records,
)
from vector_store.services.vector_store_manager import VectorStoreManager


class VectorStoreManagerTestCase(TestCase):
    """Tests for VectorStoreManager with mocked embedding service."""

    def setUp(self) -> None:
        """Set up test fixtures with mocked embedding service."""
        # Mock the EmbeddingService to avoid actual API calls
        self.embedding_patcher = patch(
            "vector_store.services.vector_store_manager.EmbeddingService"
        )
        self.mock_embedding_class = self.embedding_patcher.start()

        # Configure mock embedding service
        self.mock_embedding_service = MagicMock()
        self.mock_embedding_service.embed_text.return_value = [0.1] * 768
        self.mock_embedding_service.embed_documents.return_value = [[0.1] * 768]
        self.mock_embedding_service.get_provider_name.return_value = "mock"
        self.mock_embedding_class.return_value = self.mock_embedding_service

        # Create manager with mocked embeddings
        self.manager = VectorStoreManager()

    def tearDown(self) -> None:
        """Clean up mocks."""
        self.embedding_patcher.stop()

    # -------------------------
    # Document Operations Tests
    # -------------------------

    def test_add_document_creates_medical_document(self) -> None:
        """Test that add_document creates a MedicalDocument with embedding."""
        doc = self.manager.add_document(
            content="Test symptom content",
            title="Test Disease",
            section_type=SectionType.SYMPTOM,
            index_type=IndexType.B,
            source="test_source",
            metadata={"key": "value"},
        )

        self.assertIsNotNone(doc.pk)
        self.assertEqual(doc.title, "Test Disease")
        self.assertEqual(doc.content, "Test symptom content")
        self.assertEqual(doc.section_type, SectionType.SYMPTOM)
        self.assertEqual(doc.index_type, IndexType.B)
        self.assertEqual(doc.source, "test_source")
        self.assertEqual(doc.metadata, {"key": "value"})
        self.assertIsNotNone(doc.embedding)
        self.assertEqual(len(doc.embedding), 768)

        # Verify embedding service was called
        self.mock_embedding_service.embed_text.assert_called_once_with(
            "Test symptom content"
        )

    def test_add_documents_batch(self) -> None:
        """Test that add_documents creates multiple documents in batch."""
        # Configure mock to return multiple embeddings
        self.mock_embedding_service.embed_documents.return_value = [
            [0.1] * 768,
            [0.2] * 768,
        ]

        documents = [
            {
                "content": "Content 1",
                "title": "Title 1",
                "section_type": SectionType.GENERAL,
                "index_type": IndexType.A,
                "source": "test",
            },
            {
                "content": "Content 2",
                "title": "Title 2",
                "section_type": SectionType.SYMPTOM,
                "index_type": IndexType.B,
                "source": "test",
            },
        ]

        created_docs = self.manager.add_documents(documents)

        self.assertEqual(len(created_docs), 2)
        self.assertEqual(created_docs[0].title, "Title 1")
        self.assertEqual(created_docs[1].title, "Title 2")

        # Verify batch embedding was called
        self.mock_embedding_service.embed_documents.assert_called_once_with(
            ["Content 1", "Content 2"]
        )

    # -------------------------
    # Statistics Tests
    # -------------------------

    def test_get_stats_returns_counts(self) -> None:
        """Test that get_stats returns correct document counts."""
        # Create some documents
        MedicalDocument.objects.create(
            title="A1", content="c", index_type=IndexType.A, embedding=[0.1] * 768
        )
        MedicalDocument.objects.create(
            title="B1", content="c", index_type=IndexType.B, embedding=[0.1] * 768
        )
        MedicalDocument.objects.create(
            title="B2", content="c", index_type=IndexType.B, embedding=[0.1] * 768
        )

        stats = self.manager.get_stats()

        self.assertEqual(stats["index_a_count"], 1)
        self.assertEqual(stats["index_b_count"], 2)
        self.assertEqual(stats["index_c_count"], 0)
        self.assertEqual(stats["total_documents"], 3)
        self.assertEqual(stats["embedding_provider"], "mock")

    def test_clear_index_deletes_documents(self) -> None:
        """Test that clear_index deletes documents of specific index type."""
        MedicalDocument.objects.create(
            title="A1", content="c", index_type=IndexType.A, embedding=[0.1] * 768
        )
        MedicalDocument.objects.create(
            title="B1", content="c", index_type=IndexType.B, embedding=[0.1] * 768
        )
        MedicalDocument.objects.create(
            title="B2", content="c", index_type=IndexType.B, embedding=[0.1] * 768
        )

        deleted = self.manager.clear_index(IndexType.B)

        self.assertEqual(deleted, 2)
        self.assertEqual(
            MedicalDocument.objects.filter(index_type=IndexType.B).count(), 0
        )
        self.assertEqual(
            MedicalDocument.objects.filter(index_type=IndexType.A).count(), 1
        )

    # -------------------------
    # CSV Processing Tests
    # -------------------------

    def test_safe_parse_list_with_valid_list_string(self) -> None:
        """Test parsing a valid Python list string."""
        result = self.manager._safe_parse_list("['item1', 'item2', 'item3']")
        self.assertEqual(result, ["item1", "item2", "item3"])

    def test_safe_parse_list_with_comma_separated(self) -> None:
        """Test fallback to comma-separated parsing."""
        result = self.manager._safe_parse_list("item1, item2, item3")
        self.assertEqual(result, ["item1", "item2", "item3"])

    def test_safe_parse_list_with_empty_value(self) -> None:
        """Test handling of empty values."""
        self.assertEqual(self.manager._safe_parse_list(""), [])
        self.assertEqual(self.manager._safe_parse_list(None), [])  # type: ignore[arg-type]

    def test_safe_parse_list_with_non_string(self) -> None:
        """Test handling of non-string values."""
        self.assertEqual(self.manager._safe_parse_list(123), [])  # type: ignore[arg-type]

    def test_split_sentences(self) -> None:
        """Test Vietnamese sentence splitting."""
        text = "Đây là câu một. Đây là câu hai! Câu ba?"
        sentences = self.manager._split_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertIn("Đây là câu một.", sentences[0])

    def test_matches_metadata_filters_supports_scalar_and_list(self) -> None:
        """Metadata filters should match case-insensitive scalar/list values."""
        metadata = {
            "canonical_title": "covid-19",
            "intent_hints": ["symptom", "risk_factor"],
            "primary_intent": "symptom",
        }
        self.assertTrue(
            self.manager._matches_metadata_filters(
                metadata,
                {"canonical_title": "COVID-19", "primary_intent": "symptom"},
            )
        )
        self.assertTrue(
            self.manager._matches_metadata_filters(
                metadata,
                {"intent_hints": "risk_factor"},
            )
        )
        self.assertFalse(
            self.manager._matches_metadata_filters(
                metadata,
                {"intent_hints": "diagnosis_treatment"},
            )
        )

    def test_process_csv_to_documents_index_b(self) -> None:
        """Test CSV processing for Index B (per-section documents)."""
        # Create a temporary CSV file
        csv_data = {
            "title": ["Test Disease"],
            "url": ["http://example.com"],
            "symptom": ["['fever', 'cough']"],
            "general": ["General info about disease"],
        }
        df = pd.DataFrame(csv_data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            csv_path = f.name

        try:
            documents = self.manager.process_csv_to_documents(
                csv_path=csv_path, source="test", index_type="B"
            )

            # Should have documents for symptom and general sections
            self.assertGreater(len(documents), 0)

            # Check structure
            doc = documents[0]
            self.assertIn("content", doc)
            self.assertIn("title", doc)
            self.assertIn("section_type", doc)
            self.assertIn("index_type", doc)
            self.assertEqual(doc["source"], "test")
        finally:
            Path(csv_path).unlink()

    def test_process_csv_to_documents_index_a(self) -> None:
        """Test CSV processing for Index A (disease-level summary)."""
        csv_data = {
            "title": ["Test Disease"],
            "url": ["http://example.com"],
            "general": ["General description of the disease."],
            "symptom": ["['fever', 'headache']"],
        }
        df = pd.DataFrame(csv_data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            csv_path = f.name

        try:
            documents = self.manager.process_csv_to_documents(
                csv_path=csv_path, source="test", index_type="A"
            )

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0]["index_type"], IndexType.A)
            self.assertIn("test disease", documents[0]["content"])
        finally:
            Path(csv_path).unlink()

    def test_process_csv_to_documents_index_c(self) -> None:
        """Test CSV processing for Index C (title-only)."""
        csv_data = {
            "title": ["Disease One", "Disease Two"],
            "url": ["http://a.com", "http://b.com"],
        }
        df = pd.DataFrame(csv_data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            csv_path = f.name

        try:
            documents = self.manager.process_csv_to_documents(
                csv_path=csv_path, source="test", index_type="C"
            )

            self.assertEqual(len(documents), 2)
            self.assertEqual(documents[0]["content"], "disease one")
            self.assertEqual(documents[1]["content"], "disease two")
        finally:
            Path(csv_path).unlink()

    # -------------------------
    # Document Content Tests
    # -------------------------

    def test_create_document_content_general(self) -> None:
        """Test content creation for general section."""
        row = pd.Series({"title": "Disease", "general": "General info"})
        content = self.manager.create_document_content("general", row)
        self.assertEqual(content, "General info")

    def test_create_document_content_list_section(self) -> None:
        """Test content creation for list-based section."""
        row = pd.Series({"title": "Disease", "symptom": "['fever', 'cough']"})
        content = self.manager.create_document_content("symptom", row)
        self.assertEqual(content, "fever, cough")

    def test_create_document_content_missing_section(self) -> None:
        """Test handling of missing section."""
        row = pd.Series({"title": "Disease"})
        content = self.manager.create_document_content("symptom", row)
        self.assertIsNone(content)

    def test_create_document_content_empty_section(self) -> None:
        """Test handling of empty section value."""
        row = pd.Series({"title": "Disease", "symptom": ""})
        content = self.manager.create_document_content("symptom", row)
        self.assertIsNone(content)


class VectorStoreManagerSearchTestCase(TestCase):
    """Tests for search functionality - requires database with pgvector."""

    def setUp(self) -> None:
        """Set up test fixtures with mocked embedding service."""
        self.embedding_patcher = patch(
            "vector_store.services.vector_store_manager.EmbeddingService"
        )
        self.mock_embedding_class = self.embedding_patcher.start()

        self.mock_embedding_service = MagicMock()
        self.mock_embedding_service.embed_text.return_value = [0.1] * 768
        self.mock_embedding_service.get_provider_name.return_value = "mock"
        self.mock_embedding_class.return_value = self.mock_embedding_service

        self.manager = VectorStoreManager()

        # Create test documents with embeddings
        self.doc1 = MedicalDocument.objects.create(
            title="Fever Disease",
            content="symptoms include high fever",
            section_type=SectionType.SYMPTOM,
            index_type=IndexType.B,
            embedding=[0.1] * 768,
        )
        self.doc2 = MedicalDocument.objects.create(
            title="Cold Disease",
            content="symptoms include runny nose",
            section_type=SectionType.SYMPTOM,
            index_type=IndexType.B,
            embedding=[0.2] * 768,
        )

    def tearDown(self) -> None:
        """Clean up mocks."""
        self.embedding_patcher.stop()

    def test_search_similar_returns_results(self) -> None:
        """Test that search_similar returns matching documents."""
        # Note: This test may be skipped if pgvector extension is not available
        try:
            results = self.manager.search_similar("fever symptoms", k=5)
            # Should return results (may vary based on similarity)
            self.assertIsInstance(results, list)
        except Exception as e:
            # Skip if pgvector is not available
            if "vector" in str(e).lower():
                self.skipTest("pgvector extension not available")
            raise

    def test_search_similar_with_index_filter(self) -> None:
        """Test search with index_type filter."""
        # Create a document with different index type
        MedicalDocument.objects.create(
            title="Summary",
            content="disease summary",
            index_type=IndexType.A,
            embedding=[0.15] * 768,
        )

        try:
            results = self.manager.search_similar(
                "disease", k=5, index_type=IndexType.B
            )
            # All results should be Index B
            for doc in results:
                self.assertEqual(doc.index_type, IndexType.B)
        except Exception as e:
            if "vector" in str(e).lower():
                self.skipTest("pgvector extension not available")
            raise


class COVIDQAEmbeddingPipelineTestCase(TestCase):
    """Tests for COVID-QA embedding pipeline utilities and metadata."""

    def test_get_unique_context_records_assigns_stable_article_ids(self) -> None:
        dataset_samples = [
            {"context": "Article A", "question": "q1", "answers": {"text": ["a1"]}},
            {"context": "Article A", "question": "q2", "answers": {"text": ["a2"]}},
            {"context": "Article B", "question": "q3", "answers": {"text": ["a3"]}},
        ]

        records = get_unique_context_records(dataset_samples)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["article_id"], "covidqa-001")
        self.assertEqual(records[1]["article_id"], "covidqa-002")
        self.assertIn("context_id", records[0])
        self.assertEqual(records[0]["context"], "Article A")

    def test_get_unique_context_records_supports_window_selection(self) -> None:
        dataset_samples = [
            {"context": "Article A", "question": "q1", "answers": {"text": ["a1"]}},
            {"context": "Article A", "question": "q2", "answers": {"text": ["a2"]}},
            {"context": "Article B", "question": "q3", "answers": {"text": ["a3"]}},
            {"context": "Article C", "question": "q4", "answers": {"text": ["a4"]}},
        ]

        records = get_unique_context_records(dataset_samples, limit=1, start_article=2)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["article_id"], "covidqa-002")
        self.assertEqual(records[0]["context"], "Article B")

    def test_create_llm_with_openrouter_uses_langchain_chatopenai(self) -> None:
        with patch(
            "vector_store.services.embedding_docs_pipeline.base.EmbeddingService"
        ) as mock_embedding_service, patch(
            "vector_store.services.embedding_docs_pipeline.base.ChatOpenAI"
        ) as mock_chat_openai, patch.dict(
            "os.environ",
            {"OPENROUTER_API_KEY": "test-openrouter-key"},
            clear=False,
        ):
            mock_embedding_service.return_value = MagicMock()
            mock_chat_openai.return_value = MagicMock()

            COVIDQAEmbeddingPipeline(
                embedding_provider="transformers",
                llm_provider="openrouter",
                llm_model="openai/gpt-4.1-mini",
            )

            self.assertTrue(mock_chat_openai.called)
            kwargs = mock_chat_openai.call_args.kwargs
            self.assertEqual(kwargs["model"], "openai/gpt-4.1-mini")

    def test_stage_1_index_c_stores_article_ids_in_metadata(self) -> None:
        with patch(
            "vector_store.services.embedding_docs_pipeline.base.EmbeddingService"
        ) as mock_embedding_service, patch(
            "vector_store.services.embedding_docs_pipeline.base.ChatGoogleGenerativeAI"
        ) as mock_chat_gemini, patch(
            "vector_store.services.embedding_docs_pipeline.base.time.sleep",
            return_value=None,
        ):
            mock_embedding = MagicMock()
            mock_embedding.embed_documents.return_value = [[0.1] * 768]
            mock_embedding_service.return_value = mock_embedding
            mock_chat_gemini.return_value = MagicMock()

            pipeline = COVIDQAEmbeddingPipeline(embedding_provider="transformers")
            pipeline._initialize_extraction_records(
                [
                    {
                        "article_id": "covidqa-001",
                        "context_id": "ctx-001",
                        "context": "COVID-19 causes fever",
                    }
                ]
            )
            with patch.object(
                pipeline,
                "extract_diseases_from_context",
                return_value=["COVID-19"],
            ):
                disease_to_contexts = pipeline.stage_1_index_c(
                    [
                        {
                            "article_id": "covidqa-001",
                            "context_id": "ctx-001",
                            "context": "COVID-19 causes fever",
                        }
                    ]
                )

            self.assertIn("covid-19", disease_to_contexts)
            doc = MedicalDocument.objects.get(index_type=IndexType.C)
            self.assertEqual(doc.metadata["source_article_ids"], ["covidqa-001"])

    def test_run_persists_extraction_dataset_file(self) -> None:
        with patch(
            "vector_store.services.embedding_docs_pipeline.base.EmbeddingService"
        ) as mock_embedding_service, patch(
            "vector_store.services.embedding_docs_pipeline.base.ChatGoogleGenerativeAI"
        ) as mock_chat_gemini:
            mock_embedding_service.return_value = MagicMock()
            mock_chat_gemini.return_value = MagicMock()

            pipeline = COVIDQAEmbeddingPipeline(embedding_provider="transformers")

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "extraction_dataset.json"
                pipeline.extraction_dataset_output_path = str(output_path)

                contexts = [
                    {
                        "article_id": "covidqa-001",
                        "context_id": "ctx-001",
                        "context": "Sample article context",
                    }
                ]
                with patch.object(
                    pipeline, "load_contexts", return_value=contexts
                ), patch.object(
                    pipeline, "stage_1_index_c", return_value={"covid-19": {"ctx-001"}}
                ), patch.object(
                    pipeline, "stage_2_index_a", return_value=1
                ), patch.object(
                    pipeline, "stage_3_index_b", return_value=2
                ):
                    stats = pipeline.run(num_docs=1)

                self.assertEqual(stats["articles"], 1)
                self.assertEqual(stats["extraction_dataset_path"], str(output_path))
                self.assertTrue(output_path.exists())

                payload = json.loads(output_path.read_text(encoding="utf-8"))
                self.assertEqual(payload["article_count"], 1)
                self.assertEqual(payload["articles"][0]["article_id"], "covidqa-001")

    def test_stage_2_uses_canonical_title_only(self) -> None:
        with patch(
            "vector_store.services.embedding_docs_pipeline.base.EmbeddingService"
        ) as mock_embedding_service, patch(
            "vector_store.services.embedding_docs_pipeline.base.ChatGoogleGenerativeAI"
        ) as mock_chat_gemini, patch(
            "vector_store.services.embedding_docs_pipeline.base.time.sleep",
            return_value=None,
        ):
            mock_embedding = MagicMock()
            mock_embedding.embed_documents.return_value = [[0.1] * 768]
            mock_embedding_service.return_value = mock_embedding
            mock_chat_gemini.return_value = MagicMock()

            pipeline = COVIDQAEmbeddingPipeline(embedding_provider="transformers")
            llm_payload = {
                "disease_name": "hiv-1 infection",
                "general": "summary",
                "triệu chứng": "",
                "nguyên nhân": "",
                "yếu tố nguy cơ": "",
                "chẩn đoán và điều trị": "",
                "sinh hoạt và phòng ngừa": "",
            }
            with patch.object(
                pipeline,
                "_llm_invoke",
                return_value=json.dumps(llm_payload, ensure_ascii=False),
            ):
                count = pipeline.stage_2_index_a(
                    [
                        {
                            "article_id": "covidqa-001",
                            "context_id": "ctx-001",
                            "context": "Article about HIV.",
                        }
                    ],
                    {
                        "hiv-1": {"ctx-001"},
                        "hiv-1 infection": {"ctx-001"},
                    },
                )

            self.assertEqual(count, 1)
            docs = MedicalDocument.objects.filter(index_type=IndexType.A)
            self.assertEqual(docs.count(), 1)
            doc = docs.first()
            self.assertIsNotNone(doc)
            if doc is not None:
                self.assertEqual(doc.title, "hiv-1 infection")
                self.assertEqual(
                    doc.metadata.get("title_aliases"),
                    ["hiv-1", "hiv-1 infection"],
                )

    def test_stage_3_uses_canonical_title_only(self) -> None:
        with patch(
            "vector_store.services.embedding_docs_pipeline.base.EmbeddingService"
        ) as mock_embedding_service, patch(
            "vector_store.services.embedding_docs_pipeline.base.ChatGoogleGenerativeAI"
        ) as mock_chat_gemini, patch(
            "vector_store.services.embedding_docs_pipeline.base.time.sleep",
            return_value=None,
        ):
            mock_embedding = MagicMock()
            mock_embedding.embed_documents.side_effect = [
                [[0.1] * 768],
                [[0.2] * 768],
            ]
            mock_embedding_service.return_value = mock_embedding
            mock_chat_gemini.return_value = MagicMock()

            pipeline = COVIDQAEmbeddingPipeline(embedding_provider="transformers")
            with patch.object(
                pipeline,
                "classify_sections",
                return_value={
                    "general": "General section text.",
                    "symptom": "Symptom section text.",
                    "aetiologies": "",
                    "risk": "",
                    "diagnose_and_treaty": "",
                    "living_and_preventive": "",
                },
            ):
                count = pipeline.stage_3_index_b(
                    [
                        {
                            "article_id": "covidqa-001",
                            "context_id": "ctx-001",
                            "context": "Article about HIV.",
                        }
                    ],
                    {
                        "hiv-1": {"ctx-001"},
                        "hiv-1 infection": {"ctx-001"},
                    },
                )

            self.assertEqual(count, 2)
            docs = MedicalDocument.objects.filter(index_type=IndexType.B).order_by("id")
            self.assertEqual(docs.count(), 2)
            self.assertEqual(
                set(docs.values_list("title", flat=True)),
                {"hiv-1 infection"},
            )
            first_doc = docs.first()
            self.assertIsNotNone(first_doc)
            if first_doc is not None:
                self.assertEqual(
                    first_doc.metadata.get("title_aliases"),
                    ["hiv-1", "hiv-1 infection"],
                )
