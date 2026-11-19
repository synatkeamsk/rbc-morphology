"""
app/cropper.py
Automatic RBC detection, Chula LAB stain matching, phone-eyepiece conversion,
and scale-preserving center cropping.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

# Measured from Chula-RBC-12 training slides
CHULA_BG_LAB = np.array([157.15, 122.00, 129.30], dtype=np.float32)
CHULA_FG_LAB = np.array([102.00, 140.35, 120.40], dtype=np.float32)


@dataclass
class DetectedCell:
    cell_id: int
    cx: int
    cy: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area: float
    crop_bgr: np.ndarray


def match_chula_stain(bgr: np.ndarray) -> np.ndarray:
    """
    Map source slide's plasma (80th pct L) and RBCs (8th pct L) in LAB space
    onto Chula-RBC-12's exact stain color & brightness distribution.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    bg_mask = L >= np.percentile(L, 80)
    fg_mask = L <= np.percentile(L, 8)

    src_bg = np.median(lab[bg_mask], axis=0)
    src_fg = np.median(lab[fg_mask], axis=0)

    l_denom = max(10.0, float(src_bg[0] - src_fg[0]))
    l_scale = float(np.clip((CHULA_BG_LAB[0] - CHULA_FG_LAB[0]) / l_denom, 0.5, 1.05))
    lab[:, :, 0] = (lab[:, :, 0] - src_bg[0]) * l_scale + CHULA_BG_LAB[0]

    a_denom = float(src_fg[1] - src_bg[1])
    a_scale = float(np.clip((CHULA_FG_LAB[1] - CHULA_BG_LAB[1]) / a_denom, 0.5, 2.0)) if a_denom > 2.0 else 1.0
    lab[:, :, 1] = (lab[:, :, 1]) + (a_scale - 1.0) * (lab[:, :, 1] - src_bg[1]) + (CHULA_BG_LAB[1] - src_bg[1])

    b_denom = float(src_fg[2] - src_bg[2])
    b_scale = float(np.clip((CHULA_FG_LAB[2] - CHULA_BG_LAB[2]) / b_denom, 0.5, 2.0)) if b_denom < -2.0 else 1.0
    lab[:, :, 2] = (lab[:, :, 2] - src_bg[2]) * b_scale + CHULA_BG_LAB[2]

    lab_u8 = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_u8, cv2.COLOR_LAB2BGR)


def _crop_eyepiece_fov(bgr: np.ndarray, margin_ratio: float = 0.88) -> np.ndarray:
    """Detect circular bright microscope field and return clean inscribed square."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)

    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k9)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k15)

    num_lbls, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    if num_lbls <= 1:
        return bgr

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_lbl = int(np.argmax(areas)) + 1
    fov_area = float(stats[best_lbl, cv2.CC_STAT_AREA])

    if fov_area >= 0.92 * h * w:
        return bgr

    fov_mask = (labels == best_lbl).astype(np.uint8) * 255
    dist = cv2.distanceTransform(fov_mask, cv2.DIST_L2, 5)
    cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
    radius = float(dist[cy, cx])

    half = max(32, int((radius * margin_ratio) / np.sqrt(2.0)))
    x1, x2 = max(0, int(cx) - half), min(w, int(cx) + half)
    y1, y2 = max(0, int(cy) - half), min(h, int(cy) + half)
    return bgr[y1:y2, x1:x2]


def _flatten_illumination(bgr: np.ndarray) -> np.ndarray:
    """Remove radial eyepiece hotspot/vignetting per BGR channel."""
    h, w = bgr.shape[:2]
    scale = 200.0 / max(h, w)
    sw, sh = max(32, int(round(w * scale))), max(32, int(round(h * scale)))
    small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    k = min(25, (min(sh, sw) // 2) | 1)
    k = max(5, k)
    bg_small = cv2.medianBlur(small, k)
    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    target_bg = np.median(bg.reshape(-1, 3), axis=0)
    flat = (bgr.astype(np.float32) / np.maximum(bg, 10.0)) * target_bg
    return np.clip(flat, 0, 255).astype(np.uint8)


def _estimate_plasma_diff(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine green-channel darkness + LAB-A pinkness relative to local plasma.
    Returns (combined_signal_uint8, pinkness_diff_uint8).
    """
    h, w = bgr.shape[:2]
    sw, sh = max(16, w // 4), max(16, h // 4)
    k_bg = min(31, (min(sh, sw) // 2) | 1)
    k_bg = max(3, k_bg)

    green = bgr[:, :, 1]
    small_g = cv2.resize(green, (sw, sh), interpolation=cv2.INTER_AREA)
    bg_g = cv2.resize(cv2.medianBlur(small_g, k_bg), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    diff_g = np.clip(bg_g - green.astype(np.float32), 0, 255)

    lab_a = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 1]
    small_a = cv2.resize(lab_a, (sw, sh), interpolation=cv2.INTER_AREA)
    bg_a = cv2.resize(cv2.medianBlur(small_a, k_bg), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    diff_a = np.clip(lab_a.astype(np.float32) - bg_a, 0, 255)

    signal = np.clip(0.65 * diff_g + 0.85 * diff_a, 0, 255).astype(np.uint8)
    return signal, diff_a.astype(np.uint8)


def _estimate_single_cell_stats(filled: np.ndarray, raw_comps: list, h: int, w: int) -> Tuple[float, float]:
    """Robustly estimate single-RBC area & diameter even when many cells are in clumps."""
    dist_all = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    dist_s = cv2.GaussianBlur(dist_all, (3, 3), 0)
    peaks = (dist_s == cv2.dilate(dist_s, np.ones((11, 11), np.uint8))) & (dist_s >= 8.0)
    peak_vals = dist_s[peaks]

    peak_area = None
    if len(peak_vals) >= 3:
        med_r = float(np.median(peak_vals))
        peak_area = float(np.pi * (med_r * 1.05) ** 2)

    compact_areas = [
        c[5]
        for c in raw_comps
        if c[5] <= 0.02 * h * w and (max(c[3], c[4]) / max(1, min(c[3], c[4]))) <= 1.65
    ]
    comp_area = None
    if compact_areas:
        arr = np.array(compact_areas, dtype=np.float32)
        p25 = float(np.percentile(arr, 25))
        single_pool = [a for a in arr if 0.60 * p25 <= a <= 1.75 * p25]
        comp_area = float(np.median(single_pool)) if single_pool else p25

    if peak_area is not None and comp_area is not None:
        median_area = min(comp_area, 1.25 * peak_area)
    elif peak_area is not None:
        median_area = peak_area
    elif comp_area is not None:
        median_area = comp_area
    else:
        median_area = float(np.median([c[5] for c in raw_comps]))

    median_diam = float(2.0 * np.sqrt(median_area / np.pi))
    return median_area, median_diam


def convert_phone_eyepiece_to_chula(
    bgr: np.ndarray,
    fov_margin: float = 0.88,
    target_cell_diam: float = 42.0,
    apply_stain_match: bool = True,
) -> Tuple[np.ndarray, float]:
    fov = _crop_eyepiece_fov(bgr, margin_ratio=fov_margin)
    fh, fw = fov.shape[:2]
    work_scale = 960.0 / max(fh, fw)
    work_bgr = cv2.resize(
        fov,
        (max(64, int(round(fw * work_scale))), max(64, int(round(fh * work_scale)))),
        interpolation=cv2.INTER_AREA if work_scale < 1.0 else cv2.INTER_CUBIC,
    )

    flat = _flatten_illumination(work_bgr)

    diff, _ = _estimate_plasma_diff(flat)
    diff_blur = cv2.GaussianBlur(diff, (3, 3), 0)
    otsu_val, _ = cv2.threshold(np.minimum(diff_blur, 65), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh = cv2.threshold(diff_blur, max(7.0, float(otsu_val) * 0.65), 255, cv2.THRESH_BINARY)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sealed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k3, iterations=1)
    contours, _ = cv2.findContours(sealed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(flat.shape[:2], dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) >= 100:
            cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)

    num_lbls, _, stats, centroids = cv2.connectedComponentsWithStats(filled)
    wh, ww = flat.shape[:2]
    raw_comps = []
    for i in range(1, num_lbls):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area >= 100:
            raw_comps.append((
                i,
                int(stats[i, cv2.CC_STAT_LEFT]),
                int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]),
                int(stats[i, cv2.CC_STAT_HEIGHT]),
                area,
                int(centroids[i, 0]),
                int(centroids[i, 1]),
            ))

    est_diam = target_cell_diam
    if raw_comps:
        _, est_diam = _estimate_single_cell_stats(filled, raw_comps, wh, ww)

    chula_scale = float(np.clip(target_cell_diam / max(10.0, est_diam), 0.35, 3.0))
    new_w, new_h = max(128, int(round(ww * chula_scale))), max(128, int(round(wh * chula_scale)))
    interp = cv2.INTER_AREA if chula_scale < 1.0 else cv2.INTER_CUBIC
    chula_bgr = cv2.resize(flat, (new_w, new_h), interpolation=interp)

    if apply_stain_match:
        chula_bgr = match_chula_stain(chula_bgr)

    return chula_bgr, est_diam


def crop_center_patch(image: np.ndarray, cx: int, cy: int, crop_size: int = 96) -> np.ndarray:
    """Matches src/prepare_chula_rbc.py crop_cell() with BORDER_REFLECT padding."""
    h, w = image.shape[:2]
    half = crop_size // 2
    x1, x2 = cx - half, cx + half
    y1, y2 = cy - half, cy + half

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        padded = cv2.copyMakeBorder(
            image,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=cv2.BORDER_REFLECT,
        )
        return padded[y1 + pad_top : y2 + pad_top, x1 + pad_left : x2 + pad_left]

    return image[y1:y2, x1:x2]


def _split_clump_roi(
    roi_mask: np.ndarray,
    x_off: int,
    y_off: int,
    median_diam: float,
    min_area: float,
):
    rh, rw = roi_mask.shape
    dist = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
    dist_smooth = cv2.GaussianBlur(dist, (3, 3), 0)

    k = max(7, int(round(0.38 * median_diam)) | 1)
    min_peak_val = max(4.0, 0.18 * median_diam)
    local_max = (dist_smooth == cv2.dilate(dist_smooth, np.ones((k, k), np.uint8))) & (
        dist_smooth >= min_peak_val
    )
    num_seeds, seed_labels = cv2.connectedComponents(local_max.astype(np.uint8))

    if num_seeds <= 2:
        ys, xs = np.where(roi_mask > 0)
        if len(xs) == 0:
            return []
        return [(roi_mask > 0, x_off, y_off, rw, rh, float(len(xs)), x_off + int(xs.mean()), y_off + int(ys.mean()), x_off, y_off)]

    inv_dist = 255 - cv2.normalize(dist_smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    inv_dist_bgr = cv2.cvtColor(inv_dist, cv2.COLOR_GRAY2BGR)

    kernel3 = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(roi_mask, kernel3, iterations=1)
    markers = np.zeros((rh, rw), dtype=np.int32)
    markers[sure_bg == 0] = 1
    markers[seed_labels > 0] = seed_labels[seed_labels > 0] + 1
    cv2.watershed(inv_dist_bgr, markers)

    parts = []
    for lbl in range(2, num_seeds + 1):
        sub_mask = (markers == lbl) & (roi_mask > 0)
        area = float(np.count_nonzero(sub_mask))
        if area < min_area:
            continue
        ys, xs = np.where(sub_mask)
        bx0, bx1 = int(xs.min()), int(xs.max())
        by0, by1 = int(ys.min()), int(ys.max())
        parts.append(
            (
                sub_mask,
                x_off + bx0,
                y_off + by0,
                bx1 - bx0 + 1,
                by1 - by0 + 1,
                area,
                x_off + int(xs.mean()),
                y_off + int(ys.mean()),
                x_off,
                y_off,
            )
        )
    return parts


def detect_and_crop_rbcs(
    bgr: np.ndarray,
    crop_size: Optional[int] = None,
    auto_scale_crop: bool = True,
    crop_scale_factor: float = 2.25,
    stain_match: bool = True,
    split_touching: bool = True,
    rim_sensitivity: float = 0.65,
    min_contrast: int = 10,
    filter_wbc: bool = True,
    min_area_px: int = 250,
) -> Tuple[List[DetectedCell], int, np.ndarray, np.ndarray]:
    clf_bgr = match_chula_stain(bgr) if stain_match else bgr
    h, w = bgr.shape[:2]

    diff, diff_a = _estimate_plasma_diff(bgr)
    diff_blur = cv2.GaussianBlur(diff, (3, 3), 0)
    otsu_val, _ = cv2.threshold(np.minimum(diff_blur, 65), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_val = max(7.0, float(otsu_val) * rim_sensitivity)
    _, thresh = cv2.threshold(diff_blur, thresh_val, 255, cv2.THRESH_BINARY)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sealed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k3, iterations=1)

    contours, _ = cv2.findContours(sealed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area_px:
            cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)

    filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, k3, iterations=1)

    num_lbls, labels, stats, centroids = cv2.connectedComponentsWithStats(filled)
    raw_comps = []
    for lbl in range(1, num_lbls):
        area = float(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        x0 = int(stats[lbl, cv2.CC_STAT_LEFT])
        y0 = int(stats[lbl, cv2.CC_STAT_TOP])
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        cx, cy = int(centroids[lbl, 0]), int(centroids[lbl, 1])
        raw_comps.append((lbl, x0, y0, bw, bh, area, cx, cy))

    if not raw_comps:
        return [], crop_size or 96, filled, clf_bgr

    median_area, median_diam = _estimate_single_cell_stats(filled, raw_comps, h, w)

    if crop_size is None:
        if auto_scale_crop:
            used_crop = max(48, int(round(median_diam * crop_scale_factor)))
            used_crop += used_crop % 2
        else:
            used_crop = 96
    else:
        used_crop = crop_size

    candidates = []
    for lbl, x0, y0, bw, bh, area, cx, cy in raw_comps:
        if split_touching and area > 1.30 * median_area:
            pad = 2
            rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
            rx1, ry1 = min(w, x0 + bw + pad), min(h, y0 + bh + pad)
            roi_mask = ((labels[ry0:ry1, rx0:rx1] == lbl).astype(np.uint8)) * 255
            parts = _split_clump_roi(
                roi_mask, rx0, ry0, median_diam=median_diam, min_area=0.35 * median_area
            )
            candidates.extend(parts)
        else:
            roi_mask = labels[y0 : y0 + bh, x0 : x0 + bw] == lbl
            candidates.append((roi_mask, x0, y0, bw, bh, area, cx, cy, x0, y0))

    pre_cells = []
    for local_mask, x0, y0, bw, bh, area, cx, cy, m_xoff, m_yoff in candidates:
        if x0 <= 2 or y0 <= 2 or (x0 + bw) >= (w - 2) or (y0 + bh) >= (h - 2):
            continue
        if area < 0.40 * median_area or area > 2.3 * median_area:
            continue
        if (max(bw, bh) / max(1, min(bw, bh))) > 2.6:
            continue

        mh, mw = local_mask.shape
        diff_roi = diff[m_yoff : m_yoff + mh, m_xoff : m_xoff + mw]
        if float(diff_roi[local_mask].mean()) < min_contrast:
            continue

        bgr_roi = bgr[m_yoff : m_yoff + mh, m_xoff : m_xoff + mw]
        mean_bgr = bgr_roi[local_mask].mean(axis=0)
        mean_gray = float(mean_bgr.mean())
        b_minus_r = float(mean_bgr[0] - mean_bgr[2])

        diff_a_roi = diff_a[m_yoff : m_yoff + mh, m_xoff : m_xoff + mw]
        if mean_gray < 80.0 and float(diff_a_roi[local_mask].mean()) < 4.0:
            continue

        pre_cells.append((x0, y0, bw, bh, area, cx, cy, mean_gray, b_minus_r))

    if not pre_cells:
        return [], used_crop, filled, clf_bgr

    med_gray = float(np.median([c[7] for c in pre_cells]))
    med_br = float(np.median([c[8] for c in pre_cells]))

    cells: List[DetectedCell] = []
    cell_idx = 0
    for x0, y0, bw, bh, area, cx, cy, mean_gray, b_minus_r in pre_cells:
        if filter_wbc and (mean_gray < med_gray - 22.0) and ((b_minus_r - med_br) > 10.0):
            continue

        patch = crop_center_patch(clf_bgr, cx, cy, crop_size=used_crop)
        cells.append(
            DetectedCell(
                cell_id=cell_idx,
                cx=cx,
                cy=cy,
                bbox=(x0, y0, bw, bh),
                area=area,
                crop_bgr=patch,
            )
        )
        cell_idx += 1

    return cells, used_crop, filled, clf_bgr
