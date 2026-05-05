import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(preference: str) -> torch.device:
    pref = (preference or "auto").lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device("cpu")


def compute_class_weights(counts):
    counts = np.array(counts, dtype=np.float32)
    weights = counts.sum() / (len(counts) * counts)
    return weights


def compute_metrics(y_true, y_pred, y_prob, num_classes):
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))

    try:
        auc = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro",
        )
        metrics["macro_auc_ovr"] = float(auc)
    except Exception:
        metrics["macro_auc_ovr"] = None

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics
