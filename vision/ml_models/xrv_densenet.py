import torch
import torch.nn as nn
import torchxrayvision as xrv


def build_model(num_classes: int, weights: str):
    model = xrv.models.DenseNet(weights=weights)

    if not hasattr(model, "classifier"):
        raise RuntimeError("Expected DenseNet model to have .classifier")

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    # Disable op_threshs since they are calibrated for the original 18 classes
    # and will cause a tensor size mismatch with a different number of output classes
    model.op_threshs = None

    return model


def extract_features(model, x):
    # Use the DenseNet feature extractor for embeddings
    feats = model.features(x)
    feats = nn.functional.relu(feats, inplace=False)
    feats = nn.functional.adaptive_avg_pool2d(feats, (1, 1))
    feats = torch.flatten(feats, 1)
    return feats
