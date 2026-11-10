"""
src/prepare_chula_rbc.py
Processes the Chula-RBC-12 dataset:
1. Parses coordinates (x, y) and class labels from Label/*.txt.
2. Crops single RBC patches (96x96) with reflective boundary padding.
3. Saves cropped images to data/processed/crops/.
4. Remaps labels to continuous integers (0 to N-1) for PyTorch.
5. Performs stratified train/validation split and generates data/splits/train.csv and val.csv.
"""

import os
import glob
import json
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Original Chula-RBC-12 class mapping (10 is 'uncategorized' and excluded)
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
    """
    Crops a square region of size (crop_size, crop_size) centered at (x, y).
    Applies reflective boundary padding if the crop extends beyond image borders.
    """
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
            image,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=cv2.BORDER_REFLECT
        )
        return padded[y1 + pad_top : y2 + pad_top, x1 + pad_left : x2 + pad_left]

    return image[y1:y2, x1:x2]


def process_dataset(
    raw_dir: str = "data/raw/chula_rbc_12",
    output_crop_dir: str = "data/processed/crops",
    splits_dir: str = "data/splits",
    crop_size: int = 96,
    val_ratio: float = 0.2,
    seed: int = 42
):
    dataset_dir = os.path.join(raw_dir, "Dataset")
    label_dir = os.path.join(raw_dir, "Label")

    if not os.path.exists(dataset_dir) or not os.path.exists(label_dir):
        raise FileNotFoundError(
            f"Dataset or Label folder not found inside '{raw_dir}'. "
            f"Please verify that {raw_dir}/Dataset and {raw_dir}/Label exist."
        )

    os.makedirs(output_crop_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(dataset_dir, "*.jpg")))
    print(f"Found {len(image_files)} full blood smear images in {dataset_dir}...")

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

            # Skip uncategorized (class 10) or invalid class IDs
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
                "original_label": class_id,
                "class_name": class_name
            })
            cell_counter += 1

    df = pd.DataFrame(records)
    print(f"\nSuccessfully extracted {len(df)} single RBC crops!")
    print("\nClass distribution:")
    print(df["class_name"].value_counts())

    # Remap distinct class names to contiguous integers: 0, 1, 2, ..., N-1
    unique_class_names = sorted(df["class_name"].unique())
    class_to_idx = {name: idx for idx, name in enumerate(unique_class_names)}
    df["label"] = df["class_name"].map(class_to_idx)

    # Save class mapping dictionary to JSON for inference reference
    mapping_file = os.path.join(splits_dir, "class_mapping.json")
    with open(mapping_file, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"\n[*] Class mapping saved to: {mapping_file}")

    # Stratified split into train (80%) and validation (20%)
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        stratify=df["label"],
        random_state=seed
    )

    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "val.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"[*] Training split saved ({len(train_df)} cells): {train_path}")
    print(f"[*] Validation split saved ({len(val_df)} cells): {val_path}")


if __name__ == "__main__":
    process_dataset()
