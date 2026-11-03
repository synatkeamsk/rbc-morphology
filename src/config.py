"""
src/config.py
Centralized configuration for SmearScope RBC morphology classification.
"""

from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class Config:
    # Morphological erythrocyte classes (customizable to your dataset)
    classes: List[str] = field(default_factory=lambda: [
        "normal",        # Discocyte
        "drepanocyte",   # Sickle cell
        "dacrocyte",     # Teardrop cell
        "schistocyte",   # Fragmented cell
        "echinocyte",    # Burr cell
        "elliptocyte",   # Ovalocyte
        "spherocyte",    # Spherocyte
        "codocyte"       # Target cell
    ])

    # Dataset paths
    raw_data_dir: str = "data/raw"
    train_csv: str = "data/splits/train.csv"
    val_csv: str = "data/splits/val.csv"
    test_csv: str = "data/splits/test.csv"
    checkpoint_dir: str = "checkpoints"

    # Image & Training specs
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4

    # Backbone architecture
    backbone: str = "convnext_nano"
    pretrained: bool = True
    dropout_rate: float = 0.3

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    @property
    def num_classes(self) -> int:
        return len(self.classes)


cfg = Config()
