"""
src/generate_lab_poster.py
Generates a high-resolution, print-ready clinical reference poster (PNG & PDF)
showcasing all 12 RBC morphological variants from the Chula-RBC dataset.
"""

import os
import glob
import textwrap
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# 12 Erythrocyte Variants with Clinical Criteria & Associated Pathologies
VARIANT_METADATA = {
    "normal": {
        "id": 0,
        "title": "Normal (Discocyte)",
        "badge": "BASELINE",
        "badge_color": "#10b981",  # Green
        "desc": "Biconcave disc, 6.0-8.0 um. Central pallor occupies ~1/3 of diameter with smooth, uniform border.",
        "causes": "Normal healthy erythrocyte homeostasis."
    },
    "schistocyte": {
        "id": 9,
        "title": "Schistocyte (Fragment)",
        "badge": "CRITICAL ALERT",
        "badge_color": "#ef4444",  # Red
        "desc": "Fragmented, jagged cell with sharp angles/horns; helmet or triangular shape lacking central pallor.",
        "causes": "TTP, HUS, DIC, Mechanical valves, Severe burns (TMA emergency)."
    },
    "teardrop": {
        "id": 7,
        "title": "Teardrop (Dacrocyte)",
        "badge": "PATHOLOGY",
        "badge_color": "#f59e0b",  # Orange
        "desc": "Pear- or teardrop-shaped erythrocyte with a single elongated, tapered tail formed during splenic passage.",
        "causes": "Primary Myelofibrosis, Bone marrow infiltration, Thalassemia."
    },
    "spherocyte": {
        "id": 3,
        "title": "Spherocyte",
        "badge": "PATHOLOGY",
        "badge_color": "#f59e0b",
        "desc": "Small, dense, spherical RBC lacking central pallor; decreased membrane area-to-volume ratio.",
        "causes": "Hereditary Spherocytosis, AIHA, Post-transfusion."
    },
    "target_cell": {
        "id": 4,
        "title": "Target Cell (Codocyte)",
        "badge": "PATHOLOGY",
        "badge_color": "#f59e0b",
        "desc": "Concentric 'bullseye' appearance with central hemoglobin, clear ring, and outer rim of hemoglobin.",
        "causes": "Hemoglobinopathies (HbC, HbS), Thalassemia, Liver disease."
    },
    "burr_cell": {
        "id": 8,
        "title": "Burr Cell (Echinocyte)",
        "badge": "MORPHOLOGY",
        "badge_color": "#38bdf8",  # Sky Blue
        "desc": "Serrate cell with 10-30 short, blunt, evenly spaced spicules projecting from the membrane.",
        "causes": "Uremia, Chronic Renal Failure, Liver disease, EDTA drying."
    },
    "ovalocyte": {
        "id": 6,
        "title": "Ovalocyte",
        "badge": "MORPHOLOGY",
        "badge_color": "#38bdf8",
        "desc": "Oval to egg-shaped erythrocyte with rounded poles; length < 2x width, central pallor preserved.",
        "causes": "Megaloblastic anemia (B12/Folate), MDS, Iron deficiency."
    },
    "elliptocyte": {
        "id": 12,
        "title": "Elliptocyte (Pencil Cell)",
        "badge": "MORPHOLOGY",
        "badge_color": "#38bdf8",
        "desc": "Elongated, slender, cigar-shaped cell with parallel sides; length > 2x width with rounded ends.",
        "causes": "Hereditary Elliptocytosis, Severe Iron Deficiency Anemia."
    },
    "stomatocyte": {
        "id": 5,
        "title": "Stomatocyte",
        "badge": "MORPHOLOGY",
        "badge_color": "#38bdf8",
        "desc": "Uniconcave cell with slit-like or mouth-shaped central pallor; loss of biconcave symmetry.",
        "causes": "Hereditary Stomatocytosis, Alcoholism, Drying artifact."
    },
    "macrocyte": {
        "id": 1,
        "title": "Macrocyte",
        "badge": "SIZE VARIANT",
        "badge_color": "#a855f7",  # Purple
        "desc": "Abnormally large erythrocyte (diameter > 8.5 um; MCV > 100 fL) with normal biconcave geometry.",
        "causes": "Vitamin B12 / Folate deficiency, Alcohol abuse, Liver disease."
    },
    "microcyte": {
        "id": 2,
        "title": "Microcyte",
        "badge": "SIZE VARIANT",
        "badge_color": "#a855f7",
        "desc": "Abnormally small erythrocyte (diameter < 6.0 um; MCV < 80 fL) with preserved biconcavity.",
        "causes": "Iron Deficiency Anemia, Thalassemia Minor, Chronic disease."
    },
    "hypochromia": {
        "id": 11,
        "title": "Hypochromia",
        "badge": "STAIN VARIANT",
        "badge_color": "#a855f7",
        "desc": "Thin rim of hemoglobin with central pallor exceeding 1/3 of cell diameter (MCHC < 32 g/dL).",
        "causes": "Chronic Iron Deficiency, Thalassemia, Lead poisoning."
    }
}


def crop_with_padding(image, x, y, patch_size=150):
    h, w = image.shape[:2]
    half = patch_size // 2

    x1, x2 = x - half, x + half
    y1, y2 = y - half, y + half

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        padded = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT
        )
        return padded[y1 + pad_top : y2 + pad_top, x1 + pad_left : x2 + pad_left]

    return image[y1:y2, x1:x2]


def find_variant_samples(raw_dir="data/raw/chula_rbc_12", patch_size=150):
    dataset_dir = os.path.join(raw_dir, "Dataset")
    label_dir = os.path.join(raw_dir, "Label")

    found_samples = {}
    label_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
    id_to_key = {meta["id"]: key for key, meta in VARIANT_METADATA.items()}

    for lbl_path in label_files:
        base_name = os.path.splitext(os.path.basename(lbl_path))[0]
        img_path = os.path.join(dataset_dir, f"{base_name}.jpg")
        if not os.path.exists(img_path):
            continue

        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    x, y, cls_id = int(float(parts[0])), int(float(parts[1])), int(parts[2])
                except ValueError:
                    continue

                if cls_id in id_to_key and id_to_key[cls_id] not in found_samples:
                    variant_key = id_to_key[cls_id]

                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    patch = crop_with_padding(img, x, y, patch_size=patch_size)

                    center = (patch.shape[1] // 2, patch.shape[0] // 2)
                    cv2.circle(patch, center, 28, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(patch, center, 27, (15, 23, 42), 1, cv2.LINE_AA)

                    found_samples[variant_key] = patch

        if len(found_samples) == len(VARIANT_METADATA):
            break

    return found_samples


def build_laboratory_poster(samples, output_png="docs/assets/SmearScope_Laboratory_Poster.png", output_pdf="docs/assets/SmearScope_Laboratory_Poster.pdf"):
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)

    # Portrait 18 x 26 inches at 300 DPI
    fig = plt.figure(figsize=(18, 26), facecolor="#0f172a")

    # Main Grid: Header (8%), 4x3 Grid (86%), Footer (6%)
    gs_main = GridSpec(nrows=3, ncols=1, height_ratios=[0.08, 0.86, 0.06], figure=fig)
    gs_main.update(left=0.04, right=0.96, top=0.96, bottom=0.03, hspace=0.04)

    # 1. Header
    ax_header = fig.add_subplot(gs_main[0, 0])
    ax_header.set_facecolor("#1e293b")
    ax_header.axis("off")
    ax_header.text(0.5, 0.68, "SMEARSCOPE • CLINICAL ERYTHROCYTE MORPHOLOGY GUIDE",
                   ha="center", va="center", color="#38bdf8", fontsize=24, fontweight="heavy")
    ax_header.text(0.5, 0.32, "Peripheral Blood Smear (PBS) Poikilocytosis & Anisocytosis Quick Reference Chart",
                   ha="center", va="center", color="#f8fafc", fontsize=15, fontweight="medium")
    ax_header.text(0.5, 0.08, "1000x Oil-Immersion Light Microscopy • ICSH Morphological Standards • Chula-RBC-12",
                   ha="center", va="center", color="#94a3b8", fontsize=11, fontstyle="italic")

    # 2. 4x3 Grid nested inside gs_main[1, 0]
    gs_grid = gs_main[1, 0].subgridspec(nrows=4, ncols=3, wspace=0.12, hspace=0.20)
    variant_keys = list(VARIANT_METADATA.keys())

    for idx, key in enumerate(variant_keys):
        row = idx // 3
        col = idx % 3

        meta = VARIANT_METADATA[key]
        patch_img = samples.get(key, np.zeros((150, 150, 3), dtype=np.uint8))

        # Nested inner grid: Image on left (42%), Text card on right (58%)
        inner_gs = gs_grid[row, col].subgridspec(nrows=1, ncols=2, width_ratios=[0.42, 0.58], wspace=0.08)

        # A. Image
        ax_img = fig.add_subplot(inner_gs[0, 0])
        ax_img.imshow(patch_img)
        ax_img.axis("off")
        rect = patches.Rectangle((0, 0), patch_img.shape[1]-1, patch_img.shape[0]-1,
                                 linewidth=2, edgecolor="#475569", facecolor="none")
        ax_img.add_patch(rect)

        # B. Text Card
        ax_txt = fig.add_subplot(inner_gs[0, 1])
        ax_txt.set_facecolor("#1e293b")
        ax_txt.axis("off")

        # Format with clean text wrapping
        desc_wrapped = textwrap.fill(meta['desc'], width=28)
        causes_wrapped = textwrap.fill(meta['causes'], width=28)

        ax_txt.text(0.04, 0.92, f"[{meta['badge']}]", color=meta['badge_color'],
                    fontsize=10.5, fontweight="bold")
        ax_txt.text(0.04, 0.77, meta['title'], color="#ffffff",
                    fontsize=13.5, fontweight="bold")
        ax_txt.text(0.04, 0.45, f"Morphology:\n{desc_wrapped}", color="#cbd5e1",
                    fontsize=9.5, va="center")
        ax_txt.text(0.04, 0.12, f"Conditions:\n{causes_wrapped}", color="#93c5fd",
                    fontsize=9, va="center", fontstyle="italic")

    # 3. Footer
    ax_footer = fig.add_subplot(gs_main[2, 0])
    ax_footer.set_facecolor("#1e293b")
    ax_footer.axis("off")
    ax_footer.text(0.03, 0.5, "CLINICAL RULE: Examine morphology strictly in the 'monolayer' zone where erythrocytes barely touch without overlapping.",
                   va="center", color="#f8fafc", fontsize=11, fontweight="bold")
    ax_footer.text(0.97, 0.5, "SmearScope Deep Learning Hematology Decision Support Pipeline",
                   ha="right", va="center", color="#64748b", fontsize=10, fontstyle="italic")

    print(f"Rendering high-resolution printable PNG: {output_png}...")
    plt.savefig(output_png, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")

    print(f"Rendering vector print-ready PDF: {output_pdf}...")
    plt.savefig(output_pdf, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

    print("\n[*] Laboratory Reference Poster generated successfully in both PNG and PDF format!")


if __name__ == "__main__":
    raw_path = "data/raw/chula_rbc_12"
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found. Please ensure raw images are present.")
    else:
        print("Extracting representative clinical views with context rings...")
        samples = find_variant_samples(raw_path)
        build_laboratory_poster(samples)