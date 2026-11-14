"""
src/prepare_chula_rbc.py
Processes the Chula-RBC-12 dataset with LEAKAGE-FREE SLIDE-LEVEL SPLITTING.
Guarantees that cells from the same blood smear slide never appear in both train and val.
"""

import os
import glob
import json
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

CLASS_NAMES = {
    0: "normal",
    1: "macrocyte",
    2: "microcyte",
    3: "spherocyte",
    4: "target_cell",
    5: "stomatocyte",
    6: "ovalocyte",
    7: "teardrop",
    8: "burr_cell",
    9: "schistocyte",
    11: "hypochromia",
    12: "elliptocyte"
}


def crop_cell(image: np.ndarray, x: int, y: int, crop_size: int = 96) -> np.ndarray:
    h, w = image.shape[:2]
    half = crop_size // 2

    x1, x2 = x - half, x + half
    y1, y2 = y - half, y + half

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        padded = cv2.copyMakeBorder(
            image, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
            borderType=cv2.BORDER_REFLECT
        )
        return padded[y1 + pad_top : y2 + pad_top, x1 + pad_left : x2 + pad_left]

    return image[y1:y2, x1:x2]


def process_dataset(
    raw_dir: str = "data/raw/chula_rbc_12",
    output_crop_dir: str = "data/processed/crops",
    splits_dir: str = "data/splits",
    crop_size: int = 96,
    seed: int = 42
):
    dataset_dir = os.path.join(raw_dir, "Dataset")
    label_dir = os.path.join(raw_dir, "Label")

    if not os.path.exists(dataset_dir) or not os.path.exists(label_dir):
        raise FileNotFoundError(f"Missing {raw_dir}/Dataset or Label directory.")

    os.makedirs(output_crop_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(dataset_dir, "*.jpg")))
    print(f"Found {len(image_files)} full blood smear images...")

    records = []
    cell_counter = 0

    for img_path in tqdm(image_files, desc="Cropping single RBCs"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(label_dir, f"{base_name}.txt")

        if not os.path.exists(lbl_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        with open(lbl_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            try:
                x = int(float(parts[0]))
                y = int(float(parts[1]))
                class_id = int(parts[2])
            except ValueError:
                continue

            if class_id not in CLASS_NAMES:
                continue

            cell_patch = crop_cell(image, x, y, crop_size=crop_size)
            if cell_patch.size == 0 or cell_patch.shape[0] < 10 or cell_patch.shape[1] < 10:
                continue

            class_name = CLASS_NAMES[class_id]
            crop_filename = f"{class_name}_{cell_counter:06d}.png"
            crop_filepath = os.path.join(output_crop_dir, crop_filename)

            cv2.imwrite(crop_filepath, cell_patch)

            records.append({
                "image_path": crop_filepath,
                "slide_id": base_name,  # Tracks parent smear image to prevent leakage!
                "original_label": class_id,
                "class_name": class_name
            })
            cell_counter += 1

    df = pd.DataFrame(records)
    print(f"\nExtracted {len(df)} total crops across {df['class_name'].nunique()} classes.")

    # 12 Classes sorted alphabetically
    unique_class_names = sorted(df["class_name"].unique())
    class_to_idx = {name: idx for idx, name in enumerate(unique_class_names)}
    df["label"] = df["class_name"].map(class_to_idx)

    # Save 12-class JSON mapping
    mapping_file = os.path.join(splits_dir, "class_mapping.json")
    with open(mapping_file, "w") as f:
        json.dump(class_to_idx, f, indent=2)

    # LEAKAGE-FREE SPLIT: StratifiedGroupKFold on slide_id
    # Guarantees that slides in train NEVER appear in validation
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(df, df["label"], groups=df["slide_id"]))

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    # Verify zero slide overlap
    train_slides = set(train_df["slide_id"].unique())
    val_slides = set(val_df["slide_id"].unique())
    assert len(train_slides.intersection(val_slides)) == 0, "Data Leakage Detected!"

    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "val.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"\n[*] LEAKAGE-FREE SPLIT COMPLETE:")
    print(f"    Train: {len(train_df)} cells across {len(train_slides)} distinct smear slides")
    print(f"    Val:   {len(val_df)} cells across {len(val_slides)} distinct smear slides")
    print(f"    Slide overlap: ZERO (Clean clinical separation)")


if __name__ == "__main__":
    process_dataset()