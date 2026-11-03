"""
src/data.py
Dataset definition and stain-robust augmentations for RBC morphology classification.
"""

from typing import Callable, Optional, Tuple
import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(img_size: int = 224, is_train: bool = True) -> A.Compose:
    """
    Augmentations tailored for blood smear microscopy:
    - Full rotation & reflection invariance (RBCs are orientation-free)
    - Wright-Giemsa stain jitter tolerance
    """
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=180,
                border_mode=cv2.BORDER_REFLECT,
                p=0.7
            ),
            A.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.2,
                hue=0.05,
                p=0.6
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])


class RBCDataset(Dataset):
    """Dataset for single-cell RBC crops from CSV metadata."""
    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_paths = self.df["image_path"].values
        self.labels = self.df["label"].values

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at path: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    train_ds = RBCDataset(train_df, transform=get_transforms(img_size, is_train=True))
    val_ds = RBCDataset(val_df, transform=get_transforms(img_size, is_train=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    return train_loader, val_loader
