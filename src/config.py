"""
src/config.py
Centralized configuration for SmearScope RBC morphology classification.
"""

from dataclasses import dataclass, field
from typing import List

try:
    import torch
    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEFAULT_DEVICE = "cpu"


@dataclass
class Config:
    # 12 Morphological classes matching Chula-RBC-12 dataset
    classes: List[str] = field(default_factory=lambda: [
        "burr_cell",
        "elliptocyte",
        "hypochromia",
        "macrocyte",
        "microcyte",
        "normal",
        "ovalocyte",
        "schistocyte",
        "spherocyte",
        "stomatocyte",
        "target_cell",
        "teardrop"
    ])

    # Dataset paths
    raw_data_dir: str = "data/raw/chula_rbc_12"
    train_csv: str = "data/splits/train.csv"
    val_csv: str = "data/splits/val.csv"
    checkpoint_dir: str = "checkpoints"

    # Image & Training specs
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    epochs: int = 25
    lr: float = 3e-4
    weight_decay: float = 1e-4

    # Backbone architecture
    backbone: str = "convnext_nano"
    pretrained: bool = True
    dropout_rate: float = 0.3

    # Device
    device: str = DEFAULT_DEVICE
    seed: int = 42

    @property
    def num_classes(self) -> int:
        return len(self.classes)


cfg = Config()
