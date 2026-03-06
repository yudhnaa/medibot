"""
Tests for VectorStoreManager
Tests document storage, search, and CSV processing with mocked embeddings.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.test import TestCase

import pandas as pd

from chatbot.models import IndexType, MedicalDocument, SectionType
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
        self.mock_embedding_service.embed_text.return_value = [0.1] * 1024
        self.mock_embedding_service.embed_documents.return_value = [[0.1] * 1024]
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
        self.assertEqual(len(doc.embedding), 1024)

        # Verify embedding service was called
        self.mock_embedding_service.embed_text.assert_called_once_with(
            "Test symptom content"
        )

    def test_add_documents_batch(self) -> None:
        """Test that add_documents creates multiple documents in batch."""
        # Configure mock to return multiple embeddings
        self.mock_embedding_service.embed_documents.return_value = [
            [0.1] * 1024,
            [0.2] * 1024,
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
            title="A1", content="c", index_type=IndexType.A, embedding=[0.1] * 1024
        )
        MedicalDocument.objects.create(
            title="B1", content="c", index_type=IndexType.B, embedding=[0.1] * 1024
        )
        MedicalDocument.objects.create(
            title="B2", content="c", index_type=IndexType.B, embedding=[0.1] * 1024
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
            title="A1", content="c", index_type=IndexType.A, embedding=[0.1] * 1024
        )
        MedicalDocument.objects.create(
            title="B1", content="c", index_type=IndexType.B, embedding=[0.1] * 1024
        )
        MedicalDocument.objects.create(
            title="B2", content="c", index_type=IndexType.B, embedding=[0.1] * 1024
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
        self.mock_embedding_service.embed_text.return_value = [0.1] * 1024
        self.mock_embedding_service.get_provider_name.return_value = "mock"
        self.mock_embedding_class.return_value = self.mock_embedding_service

        self.manager = VectorStoreManager()

        # Create test documents with embeddings
        self.doc1 = MedicalDocument.objects.create(
            title="Fever Disease",
            content="symptoms include high fever",
            section_type=SectionType.SYMPTOM,
            index_type=IndexType.B,
            embedding=[0.1] * 1024,
        )
        self.doc2 = MedicalDocument.objects.create(
            title="Cold Disease",
            content="symptoms include runny nose",
            section_type=SectionType.SYMPTOM,
            index_type=IndexType.B,
            embedding=[0.2] * 1024,
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
            embedding=[0.15] * 1024,
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
