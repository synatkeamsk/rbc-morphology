"""
app/streamlit_app.py
SmearScope Lab Testing Interface (Full Blood Smear Auto-Crop + ONNX Classification).
Research Use Only - Not for standalone clinical diagnosis.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import cfg
from app.cropper import detect_and_crop_rbcs

DEFAULT_ONNX = ROOT / "model_training/v05_leakage_free/smearscope_deploy.onnx"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

st.set_page_config(page_title="SmearScope Lab Tester", layout="wide")


@st.cache_resource
def load_onnx_session(model_path: str):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


def preprocess_batch(crops_bgr: list[np.ndarray], img_size: int = 224) -> np.ndarray:
    batch = []
    for bgr in crops_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        norm = (resized.astype(np.float32) / 255.0 - MEAN) / STD
        batch.append(np.transpose(norm, (2, 0, 1)))
    return np.stack(batch, axis=0).astype(np.float32)


def softmax_batch(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_l = np.exp(shifted)
    return exp_l / np.sum(exp_l, axis=1, keepdims=True)


st.title("SmearScope — RBC Morphology Lab Tester")
st.warning(
    "Research Use Only (Validation Testing). Do not use for standalone clinical diagnosis.",
    icon="⚠️",
)

with st.sidebar:
    st.header("Model & Cropper Settings")
    model_path = st.text_input("ONNX Model Path", value=str(DEFAULT_ONNX))
    input_mode = st.radio(
        "Input Type",
        ["Full Blood Smear (Auto-Crop RBCs)", "Already Cropped Single RBC(s)"],
    )
    conf_thresh = st.slider("Uncertain Confidence Threshold", 0.30, 0.95, 0.65, 0.05)

    if input_mode == "Full Blood Smear (Auto-Crop RBCs)":
        stain_match = st.checkbox("Match Chula Stain Colors (LAB)", value=True)
        crop_mode = st.radio("Crop Scale", ["Auto-Calibrate to Smear RBC Size", "Fixed 96×96 px (Chula Scale)"])
        custom_crop = None if crop_mode.startswith("Auto") else 96
        crop_scale_factor = st.slider("Crop Multiplier (× RBC Diameter)", 1.0, 2.8, 1.8, 0.05)
        rim_sensitivity = st.slider("Pale Rim Sensitivity", 0.35, 1.00, 0.65, 0.05)
        min_contrast = st.slider("Min Cell Contrast vs Plasma", 8, 40, 16, 2)
        split_touching = st.checkbox("Split Touching RBC Clumps", value=True)
        filter_wbc = st.checkbox("Filter Dark Purple WBC Nuclei", value=True)
        min_area = st.slider("Min Candidate Area (px²)", 100, 1000, 250, 50)
        show_mask = st.checkbox("Show Segmentation Mask (Debug)", value=False)

if not Path(model_path).exists():
    st.error(f"ONNX model not found at `{model_path}`. Run `scripts/export_self_contained_onnx.py` first.")
    st.stop()

session, input_name = load_onnx_session(model_path)
classes = sorted(cfg.classes)

uploaded_files = st.file_uploader(
    "Upload peripheral blood smear image(s)" if input_mode.startswith("Full") else "Upload cropped RBC image(s)",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    accept_multiple_files=(not input_mode.startswith("Full")),
)

if not uploaded_files:
    st.info("Upload an image to begin testing.")
    st.stop()

if input_mode.startswith("Full"):
    file_bytes = np.asarray(bytearray(uploaded_files.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Could not decode uploaded image.")
        st.stop()

    with st.spinner("Detecting, stain-matching, cropping, and classifying RBCs..."):
        cells, used_crop, seg_mask, view_bgr = detect_and_crop_rbcs(
            bgr,
            crop_size=custom_crop,
            auto_scale_crop=(custom_crop is None),
            crop_scale_factor=crop_scale_factor,
            stain_match=stain_match,
            split_touching=split_touching,
            rim_sensitivity=rim_sensitivity,
            min_contrast=min_contrast,
            filter_wbc=filter_wbc,
            min_area_px=min_area,
        )

        if not cells:
            st.warning("No RBCs passed filtering. Try lowering Min Cell Contrast or Pale Rim Sensitivity.")
            if show_mask:
                st.image(seg_mask, caption="Segmentation Mask", width="stretch")
            st.stop()

        batch_x = preprocess_batch([c.crop_bgr for c in cells], img_size=cfg.img_size)
        logits = session.run(None, {input_name: batch_x})[0]
        probs = softmax_batch(logits)

    rows = []
    annotated = view_bgr.copy()
    for cell, p in zip(cells, probs):
        idx = int(np.argmax(p))
        conf = float(p[idx])
        pred_cls = classes[idx]
        flagged = conf < conf_thresh

        color = (0, 200, 0) if pred_cls == "normal" and not flagged else (0, 165, 255) if flagged else (0, 0, 230)
        x, y, bw, bh = cell.bbox
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(
            annotated,
            f"#{cell.cell_id}:{pred_cls[:4]} {conf:.2f}",
            (x, max(14, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

        row = {
            "cell_id": cell.cell_id,
            "cx": cell.cx,
            "cy": cell.cy,
            "prediction": pred_cls,
            "status": "Review (Low Conf)" if flagged else "Confident",
            "confidence": round(conf, 4),
        }
        for c_name, c_prob in zip(classes, p):
            row[f"prob_{c_name}"] = round(float(c_prob), 4)
        rows.append(row)

    df = pd.DataFrame(rows)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader(f"Annotated Smear ({len(cells)} RBCs detected | crop={used_crop}×{used_crop}px)")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
        if show_mask:
            st.image(seg_mask, caption="Binary Segmentation Mask", width="stretch")
    with col2:
        st.subheader("Morphology Differential")
        summary = (
            df["prediction"]
            .value_counts()
            .rename_axis("morphology")
            .reset_index(name="count")
        )
        summary["percent"] = (summary["count"] / len(df) * 100).round(2)
        st.dataframe(summary, width="stretch", hide_index=True)
        st.bar_chart(summary.set_index("morphology")["count"])

        st.download_button(
            "Download Cell-Level CSV Report",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="smearscope_report.csv",
            mime="text/csv",
        )

    st.subheader("Inspect Individual Cropped RBCs")
    filter_options = ["ALL"] + sorted(df["prediction"].unique().tolist())
    selected = st.selectbox("Filter crops by predicted class", filter_options)

    filtered_indices = df.index if selected == "ALL" else df[df["prediction"] == selected].index
    cols = st.columns(6)
    for grid_i, cell_i in enumerate(list(filtered_indices)[:60]):
        c = cells[cell_i]
        r = df.iloc[cell_i]
        with cols[grid_i % 6]:
            st.image(
                cv2.cvtColor(c.crop_bgr, cv2.COLOR_BGR2RGB),
                caption=f"#{c.cell_id} {r['prediction']} ({r['confidence']*100:.1f}%)",
                width="stretch",
            )

else:
    crops = []
    names = []
    for uf in uploaded_files:
        file_bytes = np.asarray(bytearray(uf.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is not None:
            crops.append(img)
            names.append(uf.name)

    if crops:
        batch_x = preprocess_batch(crops, img_size=cfg.img_size)
        logits = session.run(None, {input_name: batch_x})[0]
        probs = softmax_batch(logits)
        rows = []
        for name, p in zip(names, probs):
            idx = int(np.argmax(p))
            rows.append({
                "file": name,
                "prediction": classes[idx],
                "confidence": round(float(p[idx]), 4),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
