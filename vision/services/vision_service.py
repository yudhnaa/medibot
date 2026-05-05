import logging
import os
import threading
from typing import Any, Optional

import cv2
import numpy as np
import skimage.io
import torch

from vision.datasets.build_dataset import XRayPreprocess
from vision.explain.gradcam import GradCAM
from vision.explain.lung_mask import LungMasker, resize_mask
from vision.ml_models.xrv_densenet import build_model, extract_features
from vision.utils import get_device, load_config

logger = logging.getLogger(__name__)

_VISION_RUNTIME: dict[str, Any] | None = None
_VISION_RUNTIME_SIGNATURE: tuple[str, str | None] | None = None
_VISION_RUNTIME_LOCK = threading.Lock()


def load_image(path):
    img = skimage.io.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    return img


def overlay_heatmap(gray, heatmap, alpha=0.5):
    heatmap_uint8 = (255 * heatmap).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    overlay = cv2.addWeighted(gray_rgb, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def derive_findings(heatmap, cfg, mask=None):
    findings = []
    h, w = heatmap.shape

    threshold = cfg["explain"]["heatmap_threshold"]
    bilateral_threshold = cfg["findings"]["bilateral_threshold"]
    diffuse_threshold = cfg["findings"]["diffuse_threshold"]

    hot = (heatmap > threshold).astype(np.float32)
    if mask is not None:
        lung = (mask > 0.5).astype(np.float32)
        lung_area = lung.sum()
        if lung_area > 0:
            hot_ratio = (hot * lung).sum() / lung_area
        else:
            hot_ratio = hot.mean()
    else:
        hot_ratio = hot.mean()
    if hot_ratio >= diffuse_threshold:
        findings.append("diffuse involvement")

    if mask is not None:
        lung = (mask > 0.5).astype(np.float32)
        left_area = lung[:, : w // 2].sum()
        right_area = lung[:, w // 2 :].sum()
        left = (hot[:, : w // 2] * lung[:, : w // 2]).sum() / max(left_area, 1.0)
        right = (hot[:, w // 2 :] * lung[:, w // 2 :]).sum() / max(right_area, 1.0)
    else:
        left = hot[:, : w // 2].mean()
        right = hot[:, w // 2 :].mean()
    if left >= bilateral_threshold and right >= bilateral_threshold:
        findings.append("bilateral involvement")

    if not findings:
        findings.append("no dominant high-activation region")

    return findings


def _build_vision_runtime(
    config_path: str,
    checkpoint_path: Optional[str] = None,
) -> dict[str, Any]:
    """Load the shared vision inference runtime once per process."""
    cfg = load_config(config_path)
    device = get_device(cfg["model"]["device"])

    model = build_model(cfg["model"]["num_classes"], cfg["model"]["weights"]).to(device)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state_dict)
    model.eval()

    preprocess = XRayPreprocess(cfg["data"]["img_size"], augment=False)
    logger.info(
        "Vision runtime initialized (device=%s, checkpoint=%s)",
        device,
        bool(checkpoint_path),
    )
    return {
        "cfg": cfg,
        "device": device,
        "model": model,
        "preprocess": preprocess,
    }


def get_vision_runtime(
    config_path: str,
    checkpoint_path: Optional[str] = None,
) -> dict[str, Any]:
    """Return the shared vision inference runtime."""
    global _VISION_RUNTIME, _VISION_RUNTIME_SIGNATURE

    signature = (config_path, checkpoint_path)
    if _VISION_RUNTIME is not None and _VISION_RUNTIME_SIGNATURE == signature:
        return _VISION_RUNTIME

    with _VISION_RUNTIME_LOCK:
        if _VISION_RUNTIME is None or _VISION_RUNTIME_SIGNATURE != signature:
            _VISION_RUNTIME = _build_vision_runtime(config_path, checkpoint_path)
            _VISION_RUNTIME_SIGNATURE = signature

    return _VISION_RUNTIME


def warmup_vision_runtime(
    config_path: str,
    checkpoint_path: Optional[str] = None,
) -> bool:
    """Warm the shared vision runtime for lower first-request latency."""
    try:
        get_vision_runtime(config_path, checkpoint_path)
        return True
    except Exception as exc:
        logger.warning("Vision warmup failed: %s", exc)
        return False


def clear_vision_runtime_cache() -> None:
    """Clear the shared vision runtime for tests."""
    global _VISION_RUNTIME, _VISION_RUNTIME_SIGNATURE
    with _VISION_RUNTIME_LOCK:
        _VISION_RUNTIME = None
        _VISION_RUNTIME_SIGNATURE = None


def analyze_xray(
    config_path: str, image_path: str, checkpoint_path: Optional[str] = None
):
    """
    Analyze a chest X-ray image.

    Returns dict with: class_probs, pred_label, findings, heatmap_path, mask_path, embedding
    """
    runtime = get_vision_runtime(config_path, checkpoint_path)
    cfg = runtime["cfg"]
    device = runtime["device"]
    model = runtime["model"]
    pre = runtime["preprocess"]

    raw = load_image(image_path)
    img = pre(raw).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())

    cam = GradCAM(model)(img, pred_idx)
    cam_np = cam.squeeze(0).cpu().numpy()

    mask_path = None
    mask = None
    if cfg.get("segmentation", {}).get("use_lung_mask", False):
        try:
            masker = LungMasker(
                device=device,
                img_size=cfg["segmentation"]["img_size"],
                threshold=cfg["segmentation"]["lung_threshold"],
            )
            mask_t = masker(raw)
            mask = resize_mask(mask_t, cam_np.shape)
            cam_np = cam_np * mask

            os.makedirs(cfg["outputs"]["masks"], exist_ok=True)
            mask_path = os.path.join(cfg["outputs"]["masks"], "lung_mask.png")
            cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
        except Exception as exc:
            logger.warning("lung_mask_failed: %s", exc)

    gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (cam_np.shape[1], cam_np.shape[0]))
    overlay = overlay_heatmap(gray, cam_np, alpha=cfg["explain"]["cam_alpha"])

    os.makedirs(cfg["outputs"]["explain"], exist_ok=True)
    heatmap_path = os.path.join(cfg["outputs"]["explain"], "gradcam.png")
    cv2.imwrite(heatmap_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    with torch.no_grad():
        features = extract_features(model, img).squeeze(0).cpu().numpy()
    findings = derive_findings(
        cam_np,
        cfg,
        mask=mask if cfg.get("segmentation", {}).get("use_lung_mask", False) else None,
    )

    return {
        "class_probs": {
            cfg["classes"][i]: float(p) for i, p in enumerate(probs.tolist())
        },
        "pred_label": cfg["classes"][pred_idx],
        "findings": findings,
        "heatmap_path": heatmap_path,
        "mask_path": mask_path,
        "embedding": features.tolist(),
    }
