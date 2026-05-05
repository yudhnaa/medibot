import json
import logging
from typing import Optional

import torch
from psycopg2.extras import execute_values
from torch.utils.data import DataLoader

from vision.datasets.build_dataset import XRayPreprocess
from vision.datasets.kaggle_loader import KaggleCovidDataset
from vision.ml_models.xrv_densenet import build_model, extract_features
from vision.services.retrieval_pipeline import ensure_table, get_conn
from vision.utils import get_device, load_config

logger = logging.getLogger(__name__)


def ingest_embeddings(
    config_path: str, checkpoint_path: Optional[str] = None, batch_size: int = 32
):
    """Batch ingest image embeddings into pgvector."""
    cfg = load_config(config_path)

    transform = XRayPreprocess(cfg["data"]["img_size"], augment=False)
    dataset = KaggleCovidDataset(
        cfg["data"]["root"], cfg["classes"], transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    device = get_device(cfg["model"]["device"])
    model = build_model(cfg["model"]["num_classes"], cfg["model"]["weights"]).to(device)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    model.eval()

    conn = get_conn(cfg)
    ensure_table(conn, cfg["pgvector"]["table"], cfg["model"]["embedding_dim"])

    count = 0
    with conn.cursor() as cur:
        for images, labels, paths in loader:
            images = images.to(device)
            with torch.no_grad():
                feats = extract_features(model, images).cpu().numpy()
            rows = []
            for idx in range(len(paths)):
                label_idx = int(labels[idx])
                label = cfg["classes"][label_idx]
                metadata = {"source": "kaggle", "label_idx": label_idx}
                rows.append(
                    (paths[idx], label, feats[idx].tolist(), json.dumps(metadata))
                )

            execute_values(
                cur,
                f"INSERT INTO {cfg['pgvector']['table']} (image_id, label, embedding, metadata) VALUES %s",
                rows,
            )
            count += len(rows)
        conn.commit()

    logger.info("Ingested %d embeddings", count)
    return {"status": "ok", "count": count}
