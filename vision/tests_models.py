"""
Tests for Vision app models.
"""

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase

from authentication.models import Customer
from vision.models import XRayAnalysis, XRayEmbedding


class XRayAnalysisModelTests(TestCase):
    def setUp(self):
        self.user = Customer.objects.create_user(
            username="testuser_models",
            email="testuser_models@example.com",
            password="testpass123",
        )
        self.image = SimpleUploadedFile(
            "test_xray.jpg", b"file_content", content_type="image/jpeg"
        )

    def test_create_xray_analysis(self):
        """Test creating an XRayAnalysis instance saves correctly."""
        analysis = XRayAnalysis.objects.create(
            user=self.user,
            image=self.image,
            class_probs={"COVID": 0.9, "Normal": 0.1},
            pred_label="COVID",
            findings=["bilateral involvement"],
        )

        self.assertEqual(XRayAnalysis.objects.count(), 1)
        self.assertEqual(analysis.user, self.user)
        self.assertEqual(analysis.pred_label, "COVID")
        self.assertEqual(analysis.class_probs["COVID"], 0.9)
        self.assertIn("bilateral involvement", analysis.findings)
        self.assertTrue(analysis.image.name.startswith("xrays/"))
        self.assertIsNotNone(analysis.created_at)
        self.assertEqual(
            str(analysis),
            f"Analysis for testuser_models - COVID ({analysis.created_at.strftime('%Y-%m-%d %H:%M')})",
        )


class XRayEmbeddingModelTests(TestCase):
    def test_create_xray_embedding(self):
        """Test creating an XRayEmbedding instance saves correctly."""
        embedding_vector = [0.1] * 1024
        embedding = XRayEmbedding.objects.create(
            image_id="img_123",
            label="Normal",
            embedding=embedding_vector,
            metadata={"source": "test"},
        )

        self.assertEqual(XRayEmbedding.objects.count(), 1)
        self.assertEqual(embedding.image_id, "img_123")
        self.assertEqual(embedding.label, "Normal")
        self.assertEqual(len(embedding.embedding), 1024)
        self.assertEqual(embedding.metadata["source"], "test")
        self.assertEqual(str(embedding), "Embedding for img_123 - Normal")
