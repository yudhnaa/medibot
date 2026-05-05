from dataclasses import dataclass
from typing import List, Tuple

import torch
import torchvision.transforms as T
import torchxrayvision as xrv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from vision.datasets.kaggle_loader import KaggleCovidDataset


@dataclass
class DataConfig:
    root: str
    img_size: int
    batch_size: int
    num_workers: int
    augment: bool
    split_train: float
    split_val: float
    split_test: float
    seed: int
    classes: List[str]


class XRayPreprocess:
    def __init__(self, img_size: int, augment: bool = False):
        self.img_size = img_size
        self.augment = augment
        self.center_crop = xrv.datasets.XRayCenterCrop()
        self.resizer = xrv.datasets.XRayResizer(img_size)
        self.aug = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=5),
            ]
        )

    def __call__(self, img):
        # Convert 8-bit image to [-1024, 1024] range and single channel
        img = xrv.datasets.normalize(img, 255)
        img = img.mean(2)[None, ...]
        img = self.center_crop(img)
        img = self.resizer(img)
        img = torch.from_numpy(img)

        if self.augment:
            img = self.aug(img)

        return img


def build_splits(
    dataset: KaggleCovidDataset, cfg: DataConfig
) -> Tuple[List[int], List[int], List[int]]:
    labels = [label for _, label in dataset.samples]
    indices = list(range(len(labels)))

    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices,
        labels,
        test_size=(1.0 - cfg.split_train),
        random_state=cfg.seed,
        stratify=labels,
    )

    val_size = cfg.split_val / (cfg.split_val + cfg.split_test)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_size),
        random_state=cfg.seed,
        stratify=y_temp,
    )

    return train_idx, val_idx, test_idx


def build_dataloaders(cfg: DataConfig):
    transform = XRayPreprocess(cfg.img_size, augment=cfg.augment)
    dataset = KaggleCovidDataset(cfg.root, cfg.classes, transform=transform)

    train_idx, val_idx, test_idx = build_splits(dataset, cfg)

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
