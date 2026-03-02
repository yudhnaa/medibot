import logging
import os
from typing import Optional

import cv2
import numpy as np
import torch
import skimage.io

from vision.datasets.build_dataset import XRayPreprocess
from vision.ml_models.xrv_densenet import build_model, extract_features
from vision.utils import load_config, get_device
from vision.explain.gradcam import GradCAM
from vision.explain.lung_mask import LungMasker, resize_mask

logger = logging.getLogger(__name__)


def load_image(path):
    img = skimage.io.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    return img


def overlay_heatmap(gray, heatmap, alpha=0.5):
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
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


def analyze_xray(
    config_path: str, image_path: str, checkpoint_path: Optional[str] = None
):
    """
    Analyze a chest X-ray image.

    Returns dict with: class_probs, pred_label, findings, heatmap_path, mask_path, embedding
    """
    cfg = load_config(config_path)
    device = get_device(cfg["model"]["device"])

    model = build_model(cfg["model"]["num_classes"], cfg["model"]["weights"]).to(device)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    model.eval()

    pre = XRayPreprocess(cfg["data"]["img_size"], augment=False)
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
