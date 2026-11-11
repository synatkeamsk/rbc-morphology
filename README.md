# SmearScope: Red Blood Cell Morphology Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/synatkeamsk/rbc-morphology/blob/main/notebooks/colab_train.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular, production-ready deep learning framework designed for automated classification of red blood cell (RBC) morphology from peripheral blood smear (PBS) microscopy images.

---

## 🎯 Clinical Motivation & Project Goal

Evaluation of erythrocyte morphology (poikilocytosis and anisocytosis) in peripheral blood smears is essential for diagnosing hemolytic anemias, hemoglobinopathies, nutritional deficiencies, and hematologic malignancies. 

Manual smear microscopy review is labor-intensive and subject to inter-observer variability. **SmearScope** automates single-cell morphological classification to support clinical laboratories, triage abnormal smears, and flag critical variants (e.g., schistocytes and dacrocytes).

---

## 🔬 Supported Erythrocyte Variants (11 Classes)

SmearScope classifies single-cell erythrocyte patches into **11 distinct morphological categories** based on the [Chula-RBC-12 Dataset](https://github.com/Chula-PIC-Lab/Chula-RBC-12-Dataset):

| Morphological Class | Clinical Description | Associated Conditions |
| :--- | :--- | :--- |
| **Normal (Discocyte)** | Biconcave disc with central pallor (~1/3 diameter) | Healthy baseline |
| **Schistocyte** | Fragmented, jagged erythrocyte | Thrombotic Microangiopathy (TTP/HUS), DIC |
| **Teardrop (Dacrocyte)** | Tapered, pear- or drop-shaped cell | Primary Myelofibrosis, Marrow Infiltration |
| **Burr Cell (Echinocyte)** | Crenated cell with short, uniform membrane projections | Uremia, Chronic Kidney Disease, Artifact |
| **Spherocyte** | Dense, spherical cell lacking central pallor | Hereditary Spherocytosis, Autoimmune Hemolytic Anemia |
| **Target Cell** | "Bullseye" pattern with central hemoglobin distribution | Hemoglobinopathies (Thalassemia, HbC), Liver Disease |
| **Ovalocyte** | Egg-shaped / oval erythrocyte | Megaloblastic Anemia, Myelodysplasia |
| **Elliptocyte** | Elongated, rod-like erythrocyte | Hereditary Elliptocytosis, Severe Iron Deficiency |
| **Stomatocyte** | Slit-like, "mouth-shaped" central pallor | Hereditary Stomatocytosis, Alcoholism |
| **Microcyte** | Abnormally small red blood cell (< 6 µm) | Iron Deficiency Anemia, Thalassemia Minor |
| **Macrocyte** | Abnormally large red blood cell (> 8 µm) | Vitamin B12 / Folate Deficiency, Liver Disease |
| **Hypochromia** | Pale cell with central pallor > 1/3 diameter | Chronic Iron Deficiency Anemia |

---

## 📁 Repository Structure

```text
rbc-morphology/
├── data/
│   ├── raw/                  # Downloaded raw full-field blood smears
│   ├── processed/crops/      # Extracted single-cell RBC patches (96x96 px)
│   └── splits/               # Stratified train.csv, val.csv, and class_mapping.json
├── notebooks/
│   └── colab_train.ipynb     # Google Colab GPU training runner
├── src/
│   ├── __init__.py
│   ├── config.py             # Centralized configuration & class taxonomies
│   ├── data.py               # Stain-invariant Albumentations & PyTorch DataLoader
│   ├── model.py              # ConvNeXt-Nano / EfficientNet transfer learning architecture
│   ├── prepare_chula_rbc.py  # Coordinate-based patch cropping & dataset splitting
│   ├── train.py              # Mixed-precision (AMP) training loop with Focal Loss
│   └── infer.py              # Single/batch inference & ONNX export pipeline
├── app/                      # Interactive inference dashboard (Roadmap)
├── docs/                     # Dataset notes & clinical validation criteria
├── requirements.txt
└── README.md