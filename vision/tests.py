"""
Tests for Vision module.
Tests model builder, preprocessing, Grad-CAM, findings, and config loading.
"""

import base64
import tempfile
from io import BytesIO
from unittest.mock import MagicMock, patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase

from rest_framework.test import APIClient

import numpy as np
import torch
from PIL import Image

from authentication.models import Customer
from vision.services.vision_service import (
    clear_vision_runtime_cache,
    derive_findings,
    get_vision_runtime,
    load_image,
    overlay_heatmap,
)
from vision.utils import compute_metrics, get_device, load_config


class VisionEmbedEndpointPermissionTests(TestCase):
    def setUp(self) -> None:
        self.client = APIClient()
        self.user = Customer.objects.create_user(
            username="regular", email="regular@example.com", password="password"
        )
        self.staff = Customer.objects.create_user(
            username="staff",
            email="staff@example.com",
            password="password",
            is_staff=True,
        )

    def test_embed_denies_authenticated_non_staff(self) -> None:
        self.client.force_authenticate(user=self.user)

        response = self.client.post(
            "/api/v1/vision/embed/",
            {"batch_size": 1},
            format="json",
        )

        self.assertEqual(response.status_code, 403)

    @patch("vision.views.ingest_embeddings")
    def test_embed_allows_staff(self, mock_ingest_embeddings: MagicMock) -> None:
        mock_ingest_embeddings.return_value = {"status": "ok", "count": 1}
        self.client.force_authenticate(user=self.staff)

        response = self.client.post(
            "/api/v1/vision/embed/",
            {"batch_size": 1},
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        mock_ingest_embeddings.assert_called_once()


class GetDeviceTests(TestCase):
    """Tests for device selection utility."""

    def test_get_device_cpu(self) -> None:
        """Test explicit CPU selection."""
        device = get_device("cpu")
        self.assertEqual(device.type, "cpu")

    def test_get_device_auto(self) -> None:
        """Test auto device selection returns valid device."""
        device = get_device("auto")
        self.assertIn(device.type, ["cpu", "cuda", "mps"])

    def test_get_device_empty_string(self) -> None:
        """Test empty string defaults to auto."""
        device = get_device("")
        self.assertIn(device.type, ["cpu", "cuda", "mps"])


class LoadConfigTests(TestCase):
    """Tests for YAML config loading."""

    def test_load_config_returns_dict(self) -> None:
        """Test config loading returns a valid dict with expected keys."""
        from django.conf import settings

        cfg = load_config(settings.VISION_CONFIG_PATH)
        self.assertIsInstance(cfg, dict)
        self.assertIn("model", cfg)
        self.assertIn("data", cfg)
        self.assertIn("classes", cfg)
        self.assertIn("explain", cfg)
        self.assertIn("pgvector", cfg)

    def test_config_has_four_classes(self) -> None:
        """Test config has exactly 4 X-ray classes."""
        from django.conf import settings

        cfg = load_config(settings.VISION_CONFIG_PATH)
        self.assertEqual(len(cfg["classes"]), 4)
        self.assertIn("COVID", cfg["classes"])
        self.assertIn("Normal", cfg["classes"])

    def test_config_model_settings(self) -> None:
        """Test model config has required fields."""
        from django.conf import settings

        cfg = load_config(settings.VISION_CONFIG_PATH)
        self.assertEqual(cfg["model"]["num_classes"], 4)
        self.assertEqual(cfg["model"]["embedding_dim"], 1024)
        self.assertIn("weights", cfg["model"])


class VisionRuntimeCacheTests(TestCase):
    """Tests for process-wide vision runtime caching."""

    def tearDown(self) -> None:
        clear_vision_runtime_cache()

    @patch("vision.services.vision_service.XRayPreprocess")
    @patch("vision.services.vision_service.torch.load")
    @patch("vision.services.vision_service.build_model")
    @patch("vision.services.vision_service.get_device")
    @patch("vision.services.vision_service.load_config")
    def test_get_vision_runtime_reuses_cached_bundle(
        self,
        mock_load_config: MagicMock,
        mock_get_device: MagicMock,
        mock_build_model: MagicMock,
        mock_torch_load: MagicMock,
        mock_preprocess_cls: MagicMock,
    ) -> None:
        """Model/checkpoint/preprocess should load once per process."""
        clear_vision_runtime_cache()
        mock_load_config.return_value = {
            "model": {"device": "cpu", "num_classes": 4, "weights": "test"},
            "data": {"img_size": 224},
        }
        mock_get_device.return_value = torch.device("cpu")
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_build_model.return_value = mock_model
        mock_torch_load.return_value = {"model_state": {"weight": 1}}
        mock_preprocess = MagicMock()
        mock_preprocess_cls.return_value = mock_preprocess

        first = get_vision_runtime("/tmp/config.yaml", "/tmp/checkpoint.pt")
        second = get_vision_runtime("/tmp/config.yaml", "/tmp/checkpoint.pt")

        self.assertIs(first, second)
        mock_load_config.assert_called_once_with("/tmp/config.yaml")
        mock_build_model.assert_called_once_with(4, "test")
        mock_torch_load.assert_called_once_with(
            "/tmp/checkpoint.pt",
            map_location=torch.device("cpu"),
        )
        mock_preprocess_cls.assert_called_once_with(224, augment=False)


class DeriveFindingsTests(TestCase):
    """Tests for derive_findings function."""

    def _make_cfg(self):
        return {
            "explain": {"heatmap_threshold": 0.3},
            "findings": {"bilateral_threshold": 0.25, "diffuse_threshold": 0.45},
        }

    def test_no_findings_returns_default(self) -> None:
        """Test low activation returns default finding."""
        heatmap = np.zeros((224, 224), dtype=np.float32)
        findings = derive_findings(heatmap, self._make_cfg())
        self.assertEqual(findings, ["no dominant high-activation region"])

    def test_diffuse_involvement(self) -> None:
        """Test high activation across entire image."""
        heatmap = np.ones((224, 224), dtype=np.float32)
        findings = derive_findings(heatmap, self._make_cfg())
        self.assertIn("diffuse involvement", findings)

    def test_bilateral_involvement(self) -> None:
        """Test high activation in both left and right halves."""
        heatmap = np.ones((224, 224), dtype=np.float32) * 0.5
        findings = derive_findings(heatmap, self._make_cfg())
        self.assertIn("bilateral involvement", findings)

    def test_with_mask(self) -> None:
        """Test findings with lung mask applied."""
        heatmap = np.ones((224, 224), dtype=np.float32)
        mask = np.ones((224, 224), dtype=np.float32)
        findings = derive_findings(heatmap, self._make_cfg(), mask=mask)
        self.assertIsInstance(findings, list)
        self.assertTrue(len(findings) > 0)


class OverlayHeatmapTests(TestCase):
    """Tests for heatmap overlay function."""

    def test_overlay_output_shape(self) -> None:
        """Test overlay returns correct shape."""
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        heatmap = np.random.rand(100, 100).astype(np.float32)
        overlay = overlay_heatmap(gray, heatmap, alpha=0.5)
        self.assertEqual(overlay.shape, (100, 100, 3))

    def test_overlay_output_dtype(self) -> None:
        """Test overlay returns uint8 values."""
        gray = np.zeros((50, 50), dtype=np.uint8)
        heatmap = np.zeros((50, 50), dtype=np.float32)
        overlay = overlay_heatmap(gray, heatmap)
        self.assertEqual(overlay.dtype, np.uint8)


class LoadImageTests(TestCase):
    """Tests for image loading function."""

    @patch("vision.services.vision_service.skimage.io.imread")
    def test_load_grayscale_image(self, mock_imread: MagicMock) -> None:
        """Test loading a grayscale image converts to 3-channel."""
        mock_imread.return_value = np.zeros((100, 100), dtype=np.uint8)
        img = load_image("fake_path.png")
        self.assertEqual(img.shape, (100, 100, 3))

    @patch("vision.services.vision_service.skimage.io.imread")
    def test_load_rgb_image(self, mock_imread: MagicMock) -> None:
        """Test loading an RGB image keeps shape."""
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        img = load_image("fake_path.png")
        self.assertEqual(img.shape, (100, 100, 3))


class ComputeMetricsTests(TestCase):
    """Tests for compute_metrics utility."""

    def test_compute_metrics_accuracy(self) -> None:
        """Test accuracy computation."""
        y_true = [0, 1, 2, 3, 0, 1]
        y_pred = [0, 1, 2, 3, 0, 1]
        y_prob = np.eye(4)[[0, 1, 2, 3, 0, 1]]
        metrics = compute_metrics(y_true, y_pred, y_prob, 4)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)

    def test_compute_metrics_partial(self) -> None:
        """Test metrics with some wrong predictions."""
        y_true = [0, 1, 2, 3]
        y_pred = [0, 0, 2, 3]
        y_prob = np.eye(4)[[0, 0, 2, 3]]
        metrics = compute_metrics(y_true, y_pred, y_prob, 4)
        self.assertLess(metrics["accuracy"], 1.0)
        self.assertIn("confusion_matrix", metrics)


class BuildModelTests(TestCase):
    """Tests for model builder with mocked torchxrayvision."""

    @patch("vision.ml_models.xrv_densenet.xrv")
    def test_build_model_replaces_classifier(self, mock_xrv: MagicMock) -> None:
        """Test build_model replaces classifier head with correct output size."""
        mock_model = MagicMock()
        mock_model.classifier = torch.nn.Linear(1024, 18)
        mock_model.classifier.in_features = 1024
        mock_xrv.models.DenseNet.return_value = mock_model

        from vision.ml_models.xrv_densenet import build_model

        model = build_model(num_classes=4, weights="test")

        # Verify classifier was replaced
        self.assertIsInstance(model.classifier, torch.nn.Linear)
        self.assertEqual(model.classifier.out_features, 4)
        self.assertIsNone(model.op_threshs)


class GradCAMTests(TestCase):
    """Tests for Grad-CAM implementation."""

    def test_gradcam_output_shape(self) -> None:
        """Test GradCAM produces correct output shape."""
        from vision.explain.gradcam import GradCAM

        # Create a simple model with Conv2d
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, padding=1),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(8, 4),
        )

        cam = GradCAM(model)
        x = torch.randn(1, 1, 32, 32)
        result = cam(x, class_idx=0)

        self.assertEqual(result.shape, (1, 32, 32))
        self.assertGreaterEqual(result.min().item(), 0.0)
        self.assertLessEqual(result.max().item(), 1.0 + 1e-6)


# =============================================================================
# Phase 2: API endpoint tests
# =============================================================================


def _create_test_image() -> SimpleUploadedFile:
    """Create a minimal valid JPEG in memory."""
    buf = BytesIO()
    Image.new("L", (64, 64), color=128).save(buf, format="JPEG")
    buf.seek(0)
    return SimpleUploadedFile("xray.jpg", buf.read(), content_type="image/jpeg")


def _get_authed_client() -> APIClient:
    """Return an APIClient authenticated with a test user."""
    from authentication.models import Customer

    user = Customer.objects.create_user(  # type: ignore[attr-defined]
        username="testuser",
        email="testuser@example.com",
        password="testpass123",
    )
    client = APIClient()
    client.force_authenticate(user=user)
    return client


class AnalyzeViewTests(TestCase):
    """Tests for POST /api/v1/vision/analyze/."""

    def setUp(self) -> None:
        self.client = _get_authed_client()
        self.url = "/api/v1/vision/analyze/"

    @patch("vision.views.analyze_xray")
    def test_analyze_success(self, mock_analyze: MagicMock) -> None:
        """Test successful X-ray analysis returns classification results."""
        mock_analyze.return_value = {
            "class_probs": {
                "COVID": 0.8,
                "Normal": 0.1,
                "Viral Pneumonia": 0.05,
                "Lung Opacity": 0.05,
            },
            "pred_label": "COVID",
            "findings": ["bilateral involvement"],
            "heatmap_path": None,
            "mask_path": None,
            "embedding": [0.1] * 1024,
        }
        image = _create_test_image()
        response = self.client.post(self.url, {"image": image}, format="multipart")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        data = json_data["data"]
        self.assertEqual(data["pred_label"], "COVID")
        self.assertIn("bilateral involvement", data["findings"])
        self.assertEqual(len(data["embedding"]), 1024)
        mock_analyze.assert_called_once()

    @patch("vision.views.analyze_xray")
    def test_analyze_persists_heatmap_base64(self, mock_analyze: MagicMock) -> None:
        """Returned heatmap should also be stored on the analysis record."""
        from vision.models import XRayAnalysis

        heatmap_bytes = b"fake-heatmap-png"
        with tempfile.NamedTemporaryFile(suffix=".png") as heatmap_file:
            heatmap_file.write(heatmap_bytes)
            heatmap_file.flush()
            mock_analyze.return_value = {
                "class_probs": {
                    "COVID": 0.8,
                    "Normal": 0.1,
                    "Viral Pneumonia": 0.05,
                    "Lung Opacity": 0.05,
                },
                "pred_label": "COVID",
                "findings": ["bilateral involvement"],
                "heatmap_path": heatmap_file.name,
                "mask_path": None,
                "embedding": [0.1] * 1024,
            }

            image = _create_test_image()
            response = self.client.post(self.url, {"image": image}, format="multipart")

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        expected_heatmap = base64.b64encode(heatmap_bytes).decode("utf-8")
        analysis = XRayAnalysis.objects.get(pk=data["id"])

        self.assertEqual(data["heatmap_base64"], expected_heatmap)
        self.assertEqual(analysis.heatmap_base64, expected_heatmap)

    def test_analyze_missing_image_returns_400(self) -> None:
        """Test missing image file returns 400."""
        response = self.client.post(self.url, {}, format="multipart")
        self.assertEqual(response.status_code, 400)

    def test_analyze_unauthenticated_returns_401(self) -> None:
        """Test unauthenticated request returns 401."""
        client = APIClient()
        image = _create_test_image()
        response = client.post(self.url, {"image": image}, format="multipart")
        self.assertEqual(response.status_code, 401)


class EmbedViewTests(TestCase):
    """Tests for POST /api/v1/vision/embed/."""

    def setUp(self) -> None:
        self.client = _get_authed_client()
        self.url = "/api/v1/vision/embed/"

    @patch("vision.views.ingest_embeddings")
    def test_embed_success(self, mock_ingest: MagicMock) -> None:
        """Test successful batch embedding returns count."""
        mock_ingest.return_value = {"status": "ok", "count": 42}
        response = self.client.post(self.url, {"batch_size": 16}, format="json")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        data = json_data["data"]
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["count"], 42)
        mock_ingest.assert_called_once()

    def test_embed_unauthenticated_returns_401(self) -> None:
        """Test unauthenticated request returns 401."""
        client = APIClient()
        response = client.post(self.url, {"batch_size": 16}, format="json")
        self.assertEqual(response.status_code, 401)

    @patch("vision.views.ingest_embeddings")
    def test_embed_default_batch_size(self, mock_ingest: MagicMock) -> None:
        """Test empty body uses default batch_size=32."""
        mock_ingest.return_value = {"status": "ok", "count": 100}
        response = self.client.post(self.url, {}, format="json")
        self.assertEqual(response.status_code, 200)
        mock_ingest.assert_called_once()
        _, kwargs = mock_ingest.call_args
        self.assertEqual(kwargs["batch_size"], 32)
