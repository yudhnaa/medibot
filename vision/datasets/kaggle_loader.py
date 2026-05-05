import os
from typing import List, Tuple

import numpy as np
import skimage.io
from torch.utils.data import Dataset


class KaggleCovidDataset(Dataset):
    def __init__(self, root: str, classes: List[str], transform=None):
        self.root = root
        self.classes = classes
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        exts = (".png", ".jpg", ".jpeg")
        for idx, cls in enumerate(classes):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for dirpath, _, filenames in os.walk(cls_dir):
                if os.path.basename(dirpath).lower() == "masks":
                    continue
                for name in filenames:
                    if name.lower().endswith(exts):
                        self.samples.append((os.path.join(dirpath, name), idx))

        if not self.samples:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = skimage.io.imread(path)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        if self.transform is not None:
            img = self.transform(img)

        return img, label, path

    def class_counts(self) -> List[int]:
        counts = [0] * len(self.classes)
        for _, label in self.samples:
            counts[label] += 1
        return counts
