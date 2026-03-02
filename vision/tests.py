"""
Tests for Vision module.
Tests model builder, preprocessing, Grad-CAM, findings, and config loading.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import torch
from django.test import TestCase

from vision.utils import load_config, get_device, compute_metrics
from vision.services.vision_service import derive_findings, load_image, overlay_heatmap


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
