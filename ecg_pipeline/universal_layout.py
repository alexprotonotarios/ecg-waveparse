"""Evergreen layout-agnostic ECG detector & universal per-lead neural digitizer.

Works on ANY standard clinical 12-lead ECG layout without hardcoded 3x4 geometry:
  - standard_3x4, standard_3x4_with_r1, standard_3x4_with_r3
  - standard_6x2, cabrera_6x2 (automatically inverts -aVR -> aVR)
  - standard_12x1, cabrera_12x1

Architecture (Hybrid Whole-Page Frame -> Per-Lead Parallel Decode -> Canonical Reassembly):
  1. Whole-page physical scale (`px_per_mm_x`, `px_per_mm_y`) via 2D grid FFT / autocorrelation.
  2. Local lead-label OCR (`ecg_pipeline.lead_label_identity.read_local_text`: Apple Vision / RapidOCR)
     locating every printed lead token (`I..V6`, `-aVR`) and clustering rows/columns automatically.
  3. Per-panel baseline & span estimation `[x_start, x_end, y_baseline]` for every detected panel.
  4. Universal per-lead chunking into 2.5 s (62.5 mm) x ±3.2 mV (64.0 mm) 1024x1024 crops fed into
     `CoatVggHeatmapDecoder` with a lead-agnostic blank-grid template (zero lead-name bias).
  5. Canonical reassembly into `{lead: uv_array}` + untouched-source overlay coordinates + clinical QA.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ecg_pipeline.cleanroom.io_utils import (
    ALL_12_LEADS,
    DigitizationOutput,
    _subpixel_peak_1d,
)

# Normalized lead-label mapping (handles OCR casing/punctuation variants and Cabrera -aVR)
_LEAD_REGEX = re.compile(
    r"^(?P<neg>[-−–])?\s*(?P<tok>aVR|aVL|aVF|AVR|AVL|AVF|V[1-6]|III|II|I)$",
    re.IGNORECASE,
)


def _normalize_lead_token(raw: str) -> tuple[str, bool] | None:
    s = raw.strip().replace("|", "I").replace("l", "I") if raw.strip() in ("|", "||", "|||", "l", "ll", "lll") else raw.strip()
    m = _LEAD_REGEX.match(s)
    if not m:
        return None
    neg = bool(m.group("neg"))
    tok = m.group("tok").upper()
    canon_map = {
        "I": "I",
        "II": "II",
        "III": "III",
        "AVR": "aVR",
        "AVL": "aVL",
        "AVF": "aVF",
        "V1": "V1",
        "V2": "V2",
        "V3": "V3",
        "V4": "V4",
        "V5": "V5",
        "V6": "V6",
    }
    canon = canon_map.get(tok)
    if canon is None:
        return None
    # Only -aVR is standard Cabrera; reject spurious dashes on other leads unless aVR
    if neg and canon != "aVR":
        neg = False
    return canon, neg


@dataclass
class DetectedPanel:
    lead: str
    inverted: bool  # True for Cabrera -aVR
    row_idx: int
    col_idx: int
    is_rhythm: bool
    x_start_px: float
    x_end_px: float
    y_baseline_px: float
    duration_s: float
    label_box: tuple[float, float, float, float]


@dataclass
class UniversalLayoutDetection:
    layout_name: str
    px_per_mm_x: float
    px_per_mm_y: float
    n_rows: int
    n_cols: int
    n_rhythm_rows: int
    is_cabrera: bool
    panels: list[DetectedPanel]
    metadata: dict[str, Any] = field(default_factory=dict)


def estimate_grid_px_per_mm(rgb: np.ndarray) -> tuple[float, float]:
    """Estimate physical 1-mm grid spacing (px_per_mm_x, px_per_mm_y) via 1D FFT on grid-enhanced signal.

    Works on both pink/red/blue/green grids (chrominance channel) and monochrome grids (grayscale).
    Searches for the 5-mm major grid fundamental (most reliable across DPIs) and refines with the
    1-mm harmonic when visible.
    """
    h, w = rgb.shape[:2]
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    chroma = np.abs(r - g) + np.abs(r - b)
    if float(np.mean(chroma)) > 4.0:
        # Colour grid: green/blue deficit relative to red isolates pink/red rulings cleanly
        grid_img = np.clip(r - 0.5 * (g + b), 0.0, 255.0)
        # Also support blue/green grids where red is lower than green/blue
        if float(np.percentile(grid_img, 95)) < 8.0:
            grid_img = np.abs(r - g) + np.abs(g - b)
    else:
        grid_img = 255.0 - cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Crop central 80% to exclude outer scanner border margins
    y0, y1 = int(0.10 * h), int(0.90 * h)
    x0, x1 = int(0.08 * w), int(0.92 * w)
    roi = grid_img[y0:y1, x0:x1]

    def _axis_px_per_mm(profile: np.ndarray, total_len_px: int, is_x: bool) -> float:
        p = profile - np.mean(profile)
        k = max(5, int(round(len(p) / 25)) | 1)
        trend = cv2.GaussianBlur(p.reshape(1, -1), (k, 1), 0).ravel()
        p = p - trend
        win = np.hanning(len(p))
        spec = np.abs(np.fft.rfft(p * win))
        freqs = np.fft.rfftfreq(len(p), d=1.0)  # cycles per pixel

        # Standard ECG sheet width is ~230..300 mm (46..60 major squares -> P_5mm in [W/72, W/32]).
        # Height varies by layout (160..320 mm -> P_5mm in [H/75, H/25]).
        p5_min = max(14.0, total_len_px / (72.0 if is_x else 80.0))
        p5_max = min(140.0, total_len_px / (32.0 if is_x else 24.0))
        valid = (freqs >= 1.0 / p5_max) & (freqs <= 1.0 / p5_min)
        if not valid.any():
            return total_len_px / 250.0
        idxs = np.flatnonzero(valid)
        best_score = -1.0
        best_f5 = float(freqs[idxs[0]])
        for idx in idxs:
            f5 = float(freqs[idx])
            s1 = float(spec[idx])
            i5 = int(round(idx * 5))
            s5 = float(spec[i5]) if i5 < len(spec) else 0.0
            score = s1 + 0.6 * s5
            if score > best_score:
                best_score = score
                best_f5 = f5

        p5 = 1.0 / best_f5
        px_mm = p5 / 5.0
        f1_nom = 5.0 * best_f5
        if f1_nom < 0.48:
            c_idx = int(round(f1_nom * len(p)))
            lo, hi = max(1, c_idx - 4), min(len(spec) - 2, c_idx + 5)
            if hi > lo:
                sub_idx = lo + int(np.argmax(spec[lo:hi]))
                if spec[sub_idx] > 0.25 * spec[int(round(best_f5 * len(p)))]:
                    y_l, y_c, y_r = spec[sub_idx - 1], spec[sub_idx], spec[sub_idx + 1]
                    denom = (y_l - 2 * y_c + y_r)
                    delta = 0.5 * (y_l - y_r) / denom if abs(denom) > 1e-9 else 0.0
                    f1_ref = (sub_idx + float(np.clip(delta, -0.5, 0.5))) / len(p)
                    if f1_ref > 0:
                        px_mm = 1.0 / f1_ref
        return float(px_mm)

    px_mm_x = _axis_px_per_mm(roi.mean(axis=0), w, is_x=True)
    px_mm_y = _axis_px_per_mm(roi.mean(axis=1), h, is_x=False)
    if abs(px_mm_x - px_mm_y) / max(px_mm_x, px_mm_y) > 0.08:
        px_mm_y = px_mm_x
    return px_mm_x, px_mm_y


def _red_channel_enhanced_rgb(rgb: np.ndarray, px_mm: float) -> np.ndarray:
    """Suppress pink/red ECG grid via local illumination normalization, keeping dark/slate-gray ink crisp."""
    r = rgb[:, :, 0].astype(np.float32)
    sigma = max(12.0, 3.5 * px_mm)
    local_bg = cv2.GaussianBlur(r, (0, 0), sigmaX=sigma, sigmaY=sigma)
    diff = np.clip(local_bg - r, 0.0, 255.0)
    # Boost contrast so slate-gray hospital ink (diff ~ 40..80) becomes sharp dark text on pure white
    gray = np.clip(255.0 - diff * 2.6, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _isolate_text_only(rgb: np.ndarray, px_mm: float) -> np.ndarray:
    """Erase red/pink ECG grid and long connected ECG waveforms using local adaptive contrast."""
    r = rgb[:, :, 0].astype(np.float32)
    sigma = max(12.0, 3.5 * px_mm)
    local_bg = cv2.GaussianBlur(r, (0, 0), sigmaX=sigma, sigmaY=sigma)
    ink = ((local_bg - r) > np.maximum(22.0, 0.14 * local_bg)).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    clean = np.full_like(rgb, 255)
    min_h = max(4, int(round(0.8 * px_mm)))
    max_h = int(round(9.5 * px_mm))
    max_w = int(round(16.0 * px_mm))
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if min_h <= bh <= max_h and 1 <= bw <= max_w and area >= 5:
            mask = labels[y : y + bh, x : x + bw] == i
            clean[y : y + bh, x : x + bw][mask] = 0
    return clean


def _detect_lead_labels_ocr(rgb: np.ndarray, px_mm: float) -> list[dict[str, Any]]:
    """Detect every printed lead label box via BOTH text-isolated and grid-suppressed OCR passes."""
    from ecg_pipeline.lead_label_identity import _rapidocr_ocr_runner, read_local_text

    h, w = rgb.shape[:2]
    text_only = _isolate_text_only(rgb, px_mm)
    enhanced = _red_channel_enhanced_rgb(rgb, px_mm)
    bgr_variants = [
        cv2.cvtColor(text_only, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
    ]

    all_obs = []
    for bgr_im in bgr_variants:
        for runner in (_rapidocr_ocr_runner, read_local_text):
            try:
                out = runner(bgr_im)
                all_obs.extend(out.get("observations") or [])
            except Exception:
                continue

    hits: list[dict[str, Any]] = []
    for obs in all_obs:
        try:
            x, y, bw, bh = float(obs["x"]), float(obs["y"]), float(obs["width"]), float(obs["height"])
        except Exception:
            continue
        left = x * w
        right = (x + bw) * w
        top = max(0.0, (1.0 - y - bh) * h)
        bottom = min(float(h), (1.0 - y) * h)
        if top < 0.04 * h:
            continue
        for cand in obs.get("candidates") or []:
            txt = str(cand.get("text") or "").strip()
            conf = float(cand.get("confidence") or 0.0)
            if conf < 0.30:
                continue
            parsed = _normalize_lead_token(txt)
            if parsed is None and " " in txt:
                parsed = _normalize_lead_token(txt.split()[0])
            if parsed is not None:
                lead, inverted = parsed
                cx, cy = 0.5 * (left + right), 0.5 * (top + bottom)
                # Deduplicate overlapping detections from RapidOCR & Vision
                dup = False
                for prev in hits:
                    if abs(prev["cx"] - cx) < 4.0 * px_mm and abs(prev["cy"] - cy) < 4.0 * px_mm:
                        if conf > prev["confidence"]:
                            prev.update(lead=lead, inverted=inverted, confidence=conf)
                        dup = True
                        break
                if not dup:
                    hits.append(
                        {
                            "lead": lead,
                            "inverted": inverted,
                            "confidence": conf,
                            "left": left,
                            "top": top,
                            "right": right,
                            "bottom": bottom,
                            "cx": cx,
                            "cy": cy,
                        }
                    )
                break
    return hits


def _cluster_1d(values: list[float], tol: float) -> list[list[int]]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    clusters: list[list[int]] = []
    for idx in order:
        v = values[idx]
        if not clusters or abs(v - float(np.mean([values[j] for j in clusters[-1]]))) > tol:
            clusters.append([idx])
        else:
            clusters[-1].append(idx)
    return clusters


def _estimate_panel_baseline_y(
    rgb: np.ndarray,
    x0: float,
    x1: float,
    y_hint: float,
    px_per_mm_y: float,
    row_step_px: float,
) -> float:
    """Locate the exact horizontal ECG trace baseline near a lead label via dark-ink row projection."""
    h, w = rgb.shape[:2]
    # Search within ±0.45 * row_step_px around label center (label sits either just above or just below baseline)
    search_rad = min(int(round(0.42 * row_step_px)), int(round(18.0 * px_per_mm_y)))
    ya = max(0, int(round(y_hint - search_rad)))
    yb = min(h, int(round(y_hint + search_rad)))
    xa = max(0, int(round(x0 + 3.0 * px_per_mm_y)))  # skip label text itself on far left
    xb = min(w, int(round(x1 - 2.0 * px_per_mm_y)))
    if yb - ya < 10 or xb - xa < 20:
        return float(y_hint)

    # Red channel suppresses pink/red grid while preserving black ECG ink
    r_ch = rgb[ya:yb, xa:xb, 0].astype(np.float32)
    bg = float(np.percentile(r_ch, 90))
    ink = (bg - r_ch) > max(35.0, 0.25 * bg)
    # Mask out solid horizontal page-border / rule lines ONLY on wide full-width strips (>=120 mm)
    # where no real ECG trace stays on a single pixel row for >85% of 10 seconds (avoids ever
    # masking low-amplitude flat short panels such as 2.5s aVL!).
    if ink.shape[1] >= int(round(120.0 * px_per_mm_y)):
        row_fill = ink.mean(axis=1)
        border_rows = row_fill > 0.85
        if border_rows.any():
            border_dil = cv2.dilate(border_rows.astype(np.uint8).reshape(-1, 1), np.ones((7, 1), np.uint8)).ravel() > 0
            ink[border_dil, :] = False
    # Per-column dark-ink y-centroid, then take the median across all columns (robust to QRS spikes!)
    ys = []
    for c in range(0, ink.shape[1], 2):
        rows = np.flatnonzero(ink[:, c])
        if 1 <= rows.size <= int(round(6.0 * px_per_mm_y)):
            ys.append(float(np.median(rows)))
    if len(ys) >= 15:
        return float(ya + np.median(ys))
    return float(y_hint)


_CANONICAL_LAYOUT_GRIDS: dict[str, tuple[list[list[str]], bool]] = {
    "standard_3x4": ([["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]], False),
    "standard_6x2": ([["I", "V1"], ["II", "V2"], ["III", "V3"], ["aVR", "V4"], ["aVL", "V5"], ["aVF", "V6"]], False),
    "cabrera_6x2": ([["aVL", "V1"], ["I", "V2"], ["-aVR", "V3"], ["II", "V4"], ["aVF", "V5"], ["III", "V6"]], True),
    "standard_12x1": ([[l] for l in ALL_12_LEADS], False),
    "cabrera_12x1": ([[l] for l in ["aVL", "I", "-aVR", "II", "aVF", "III", "V1", "V2", "V3", "V4", "V5", "V6"]], True),
}


def detect_universal_layout(
    rgb: np.ndarray,
    *,
    paper_speed_mm_s: float = 25.0,
) -> UniversalLayoutDetection:
    """Detect arbitrary ECG layout (3x4+1r, 3x4+3r, 6x2, 12x1, Cabrera) and every lead panel."""
    px_mm_x, px_mm_y = estimate_grid_px_per_mm(rgb)
    hits = _detect_lead_labels_ocr(rgb, px_mm_x)
    if len(hits) < 4:
        raise RuntimeError(f"Universal layout detector found too few lead labels ({len(hits)})")

    # Robust RANSAC/Hough canonical grid matching over ALL OCR hits.
    # Immune to spurious header/footer text, missed columns (e.g. thin 'I/II/III'), or missed rows!
    best_match = None
    row_mm_bounds = {3: (18.0, 58.0), 6: (15.0, 44.0), 12: (9.0, 28.0)}
    for cand_name, (grid_spec, cand_cabrera) in _CANONICAL_LAYOUT_GRIDS.items():
        R, C = len(grid_spec), len(grid_spec[0])
        col_mm_nom = 250.0 / C
        col_step_nom = col_mm_nom * px_mm_x
        r_min_px = row_mm_bounds[R][0] * px_mm_y
        r_max_px = row_mm_bounds[R][1] * px_mm_y

        lead_to_rc: dict[str, tuple[int, int]] = {}
        for r_i in range(R):
            for c_i in range(C):
                tok = grid_spec[r_i][c_i]
                lead_to_rc[tok[1:] if tok.startswith("-") else tok] = (r_i, c_i)

        cand_hits = [(hi, hits[hi], lead_to_rc[hits[hi]["lead"]]) for hi in range(len(hits)) if hits[hi]["lead"] in lead_to_rc]
        if len(cand_hits) < 3:
            continue

        # Collect candidate x-origins (quantized to ~2 mm to avoid redundant scoring)
        x_orig_cands: list[float] = []
        seen_x: set[int] = set()
        for _, ht, (_, c_i) in cand_hits:
            xo = ht["left"] - c_i * col_step_nom
            q = int(round(xo / max(1.0, 2.0 * px_mm_x)))
            if q not in seen_x:
                seen_x.add(q)
                x_orig_cands.append(xo)

        # Collect candidate (y_origin, row_step) pairs from any two hits in different canonical rows
        yr_cands: list[tuple[float, float]] = []
        seen_yr: set[tuple[int, int]] = set()
        for idx_a in range(len(cand_hits)):
            _, ha, (ra, _) = cand_hits[idx_a]
            for idx_b in range(idx_a + 1, len(cand_hits)):
                _, hb, (rb, _) = cand_hits[idx_b]
                if ra == rb:
                    continue
                rstep = (hb["cy"] - ha["cy"]) / float(rb - ra)
                if not (r_min_px <= rstep <= r_max_px):
                    continue
                yo = ha["cy"] - ra * rstep
                qy = int(round(yo / max(1.0, 2.0 * px_mm_y)))
                qr = int(round(rstep / max(1.0, 1.5 * px_mm_y)))
                if (qy, qr) not in seen_yr:
                    seen_yr.add((qy, qr))
                    yr_cands.append((yo, rstep))

        # Fallback single-row step hypothesis if only one canonical row had OCR hits
        if not yr_cands:
            def_step = (36.0 if R == 3 else (28.0 if R == 6 else 18.0)) * px_mm_y
            for _, ha, (ra, _) in cand_hits:
                yr_cands.append((ha["cy"] - ra * def_step, def_step))

        x_tol = max(18.0, 0.24 * col_step_nom)
        for xo in x_orig_cands:
            # Pre-filter hits matching x within x_tol
            x_ok = [
                (hi, ht, rc)
                for hi, ht, rc in cand_hits
                if abs(ht["left"] - (xo + rc[1] * col_step_nom)) <= x_tol
            ]
            if len(x_ok) < 3:
                continue
            for yo, rstep in yr_cands:
                y_tol = max(14.0, 0.28 * rstep)
                matched_cells: dict[tuple[int, int], tuple[float, int, dict[str, Any]]] = {}
                for hi, ht, (r_i, c_i) in x_ok:
                    err_y = abs(ht["cy"] - (yo + r_i * rstep))
                    if err_y <= y_tol:
                        err_x = abs(ht["left"] - (xo + c_i * col_step_nom))
                        err = err_x + err_y
                        prev = matched_cells.get((r_i, c_i))
                        if prev is None or err < prev[0]:
                            matched_cells[(r_i, c_i)] = (err, hi, ht)
                n_m = len(matched_cells)
                if n_m < 3:
                    continue
                has_neg_avr = any(ht["lead"] == "aVR" and ht["inverted"] for _, _, ht in matched_cells.values())
                cab_bonus = 0.5 if (cand_cabrera and has_neg_avr) else (0.0 if not cand_cabrera else -0.25)
                mean_err = float(np.mean([e for e, _, _ in matched_cells.values()]))
                score_key = (n_m + cab_bonus, -mean_err)
                if best_match is None or score_key > best_match[0]:
                    y_anchors = [(r_i, ht["cy"]) for (r_i, _), (_, _, ht) in matched_cells.items()]
                    x_anchors = [(c_i, ht["left"]) for (_, c_i), (_, _, ht) in matched_cells.items()]
                    matched_his = {hi for _, hi, _ in matched_cells.values()}
                    best_match = (
                        score_key,
                        n_m,
                        cand_name,
                        grid_spec,
                        cand_cabrera,
                        y_anchors,
                        x_anchors,
                        col_step_nom,
                        rstep,
                        yo,
                        matched_his,
                    )

    if best_match is None or best_match[1] < 3:
        raise RuntimeError(f"Could not match detected labels ({len(hits)}) to any standard ECG grid")

    (_, n_matched, base_layout, grid_spec, is_cabrera, y_anchors, x_anchors, col_step_nom, row_step_px, y_origin, matched_his) = best_match
    R, C = len(grid_spec), len(grid_spec[0])
    # Reconstruct all C column x-positions cleanly from matched x_anchors + exact FFT col_step_nom!
    x_origin_fit = float(np.median([lx - c * col_step_nom for c, lx in x_anchors]))
    col_xs = [x_origin_fit + c * col_step_nom for c in range(C)]
    # Fit exact linear row_y(r) = y_origin + r * row_step_px from matched anchors
    if len(y_anchors) >= 2:
        rs_arr = np.array([a[0] for a in y_anchors], dtype=np.float64)
        ys_arr = np.array([a[1] for a in y_anchors], dtype=np.float64)
        if np.std(rs_arr) > 1e-6:
            slope, intercept = np.polyfit(rs_arr, ys_arr, 1)
            row_step_px = float(slope)
            y_origin = float(intercept)
        else:
            y_origin = float(np.median(ys_arr) - rs_arr[0] * row_step_px)

    # Identify OCR-detected rhythm strips sitting below the main R rows near col 0
    rhythm_rc: list[list[int]] = []
    for hi, ht in enumerate(hits):
        if hi in matched_his:
            continue
        if ht["cy"] > y_origin + (R - 0.35) * row_step_px and abs(ht["left"] - x_origin_fit) <= max(25.0, 0.35 * col_step_nom):
            rhythm_rc.append([hi])

    # Panel physical duration & width
    if C == 4:
        col_step_px = float(np.median(np.diff(col_xs)))
        panel_dur_s = 2.5 * (25.0 / paper_speed_mm_s)
    elif C == 2:
        col_step_px = float(np.median(np.diff(col_xs)))
        panel_dur_s = 5.0 * (25.0 / paper_speed_mm_s)
    else:
        col_step_px = 250.0 * px_mm_x
        panel_dur_s = 10.0 * (25.0 / paper_speed_mm_s)

    expected_col_mm = panel_dur_s * paper_speed_mm_s
    # Keep the sub-pixel-accurate 2D grid FFT scale when it agrees with OCR column spacing;
    # only fall back to OCR column spacing on gridless paper where FFT disagrees substantially.
    if C in (2, 4):
        col_px_mm = col_step_px / expected_col_mm
        if abs(col_px_mm - px_mm_x) / max(px_mm_x, 1e-6) > 0.08 and col_px_mm > 2.0:
            px_mm_x = col_px_mm
            if abs(px_mm_x - px_mm_y) / max(px_mm_x, px_mm_y) > 0.08:
                px_mm_y = px_mm_x

    panel_w_px = expected_col_mm * px_mm_x
    full_w_px = 250.0 * px_mm_x

    # Refine col 0 x_start:
    # For 4-column ECGs (3x4), col 1..3 have NO calibration pulse (only clean lead starts/separators),
    # so anchor x0_col0 from col 1..3 labels/separators first to avoid snapping onto col 0's cal pulse!
    r_ch_full = rgb[:, :, 0].astype(np.float32)
    sigma_bg = max(12.0, 3.5 * px_mm_x)
    local_bg_full = cv2.GaussianBlur(r_ch_full, (0, 0), sigmaX=sigma_bg, sigmaY=sigma_bg)
    ink_full = (local_bg_full - r_ch_full) > np.maximum(22.0, 0.14 * local_bg_full)

    row_y_bases_rough = [
        _estimate_panel_baseline_y(rgb, col_xs[0], col_xs[0] + panel_w_px, y_origin + r * row_step_px, px_mm_y, row_step_px)
        for r in range(R)
    ]

    x0_col0: float | None = None
    sep_by_col: dict[int, float] = {}
    if C == 4 and len(col_xs) >= 4:
        # Look for crisp vertical separator ticks just left of col 1, 2, 3 label boxes
        col_anchors = []
        for c_i in (1, 2, 3):
            sx_lo = max(0, int(round(col_xs[c_i] - 3.5 * px_mm_x)))
            sx_hi = min(rgb.shape[1] - 1, int(round(col_xs[c_i] + 2.0 * px_mm_x)))
            if sx_hi > sx_lo + 2:
                vert_prof = np.zeros(sx_hi - sx_lo, dtype=np.float32)
                for yb in row_y_bases_rough:
                    ya = max(0, int(round(yb - 5.5 * px_mm_y)))
                    yb2 = min(rgb.shape[0], int(round(yb + 5.5 * px_mm_y)))
                    vert_prof += ink_full[ya:yb2, sx_lo:sx_hi].mean(axis=0)
                if float(vert_prof.max()) > 0.75:  # crisp vertical separator ticks across >=2 rows!
                    sep_x = float(sx_lo + int(np.argmax(vert_prof)))
                    sep_by_col[c_i] = sep_x
        # Anchor horizontal start x0_col0 to detected vertical separator ticks when >=2 columns agree
        # (prevents false separator hits on separator-free sheets while keeping exact FFT panel_w_px!).
        if len(sep_by_col) >= 2:
            offsets = [sep_x - c_i * panel_w_px for c_i, sep_x in sep_by_col.items()]
            if float(np.max(offsets) - np.min(offsets)) < 3.5 * px_mm_x:
                x0_col0 = float(np.median(offsets))
        if x0_col0 is None:
            for c_i in (1, 2, 3):
                col_anchors.append(col_xs[c_i] - 0.8 * px_mm_x - c_i * panel_w_px)
            if col_anchors:
                x0_col0 = float(np.median(col_anchors))

    if x0_col0 is None:
        x_search_lo = max(0, int(round(col_xs[0] - 4.0 * px_mm_x)))
        x_search_hi = min(rgb.shape[1] - 1, int(round(col_xs[0] + 4.0 * px_mm_x)))
        row_starts = []
        for yb in row_y_bases_rough:
            ya = max(0, int(round(yb - 5.0 * px_mm_y)))
            yb2 = min(rgb.shape[0], int(round(yb + 5.0 * px_mm_y)))
            strip = ink_full[ya:yb2, x_search_lo:x_search_hi]
            col_hit = np.flatnonzero(strip.any(axis=0))
            if col_hit.size > 0:
                row_starts.append(float(x_search_lo + int(col_hit[0])))
        if len(row_starts) >= max(2, R // 2):
            x0_col0 = float(np.median(row_starts))
        else:
            x0_col0 = col_xs[0] - 0.5 * px_mm_x

    panels: list[DetectedPanel] = []
    for r_idx in range(R):
        y_hint = y_origin + r_idx * row_step_px
        row_panels: list[DetectedPanel] = []
        for c_idx in range(C):
            spec_tok = grid_spec[r_idx][c_idx]
            inv = spec_tok.startswith("-")
            lead = spec_tok[1:] if inv else spec_tok
            x0 = x0_col0 + c_idx * panel_w_px
            x1 = x0 + panel_w_px
            y_base = _estimate_panel_baseline_y(rgb, x0, x1, y_hint, px_mm_y, row_step_px)
            row_panels.append(
                DetectedPanel(
                    lead=lead,
                    inverted=inv,
                    row_idx=r_idx,
                    col_idx=c_idx,
                    is_rhythm=False,
                    x_start_px=x0,
                    x_end_px=x1,
                    y_baseline_px=y_base,
                    duration_s=panel_dur_s,
                    label_box=(x0, y_hint - 10, x0 + 30, y_hint + 10),
                )
            )
        # All panels in the same horizontal ECG row share the same physical baseline (±3 mm tolerance);
        # clamp any outlier panel (e.g. tall QRS or faint flat lead) back to the row median baseline.
        if len(row_panels) >= 2:
            row_med_y = float(np.median([p.y_baseline_px for p in row_panels]))
            for p in row_panels:
                if abs(p.y_baseline_px - row_med_y) > 4.0 * px_mm_y:
                    p.y_baseline_px = row_med_y
        panels.extend(row_panels)

    # Collect OCR-detected rhythm strips AND recover any missed rhythm strips (e.g. thin 'II' bars!)
    # by checking horizontal trace continuity at y_origin + (R + k) * row_step_px below 3x4 grids.
    rhythm_entries: list[tuple[float, str, tuple[float, float, float, float]]] = []
    for rc in rhythm_rc:
        ht = hits[rc[0]]
        rhythm_entries.append((ht["cy"], ht["lead"], (ht["left"], ht["top"], ht["right"], ht["bottom"])))

    if R == 3 and C == 4:
        main_last_base = float(np.median([p.y_baseline_px for p in panels if p.row_idx == 2]))
        h_img, w_img = rgb.shape[:2]
        xa_r = max(0, int(round(x0_col0 + 5.0 * px_mm_x)))
        xb_r = min(w_img, int(round(x0_col0 + full_w_px - 5.0 * px_mm_x)))
        for k_r in range(3):
            y_cand = main_last_base + (k_r + 1) * row_step_px
            if y_cand > h_img - 6.0 * px_mm_y:
                break
            # Skip if already matched by an OCR rhythm label within 0.45 * row_step_px
            if any(abs(cy - y_cand) < 0.45 * row_step_px for cy, _, _ in rhythm_entries):
                continue
            ya = max(0, int(round(y_cand - 0.35 * row_step_px)))
            yb = min(h_img, int(round(y_cand + 0.35 * row_step_px)))
            if yb - ya < 12 or xb_r - xa_r < 200:
                continue
            strip = ink_full[ya:yb, xa_r:xb_r]
            col_cov = float(strip.any(axis=0).mean())
            if col_cov > 0.55:
                # Genuine full-width rhythm strip whose short label ('II') was missed by OCR!
                default_rhy_lead = "II" if (k_r == 0 or not any(l == "II" for _, l, _ in rhythm_entries)) else ("V1" if k_r == 0 else "V5")
                rhythm_entries.append((y_cand, default_rhy_lead, (x0_col0, y_cand - 10, x0_col0 + 30, y_cand + 10)))

    rhythm_entries.sort(key=lambda e: e[0])
    for rr_idx, (cy_r, r_lead, lbox) in enumerate(rhythm_entries):
        x0 = x0_col0
        x1 = x0 + full_w_px
        y_base = _estimate_panel_baseline_y(rgb, x0, x1, cy_r, px_mm_y, row_step_px)
        panels.append(
            DetectedPanel(
                lead=r_lead,
                inverted=False,
                row_idx=R + rr_idx,
                col_idx=0,
                is_rhythm=True,
                x_start_px=x0,
                x_end_px=x1,
                y_baseline_px=y_base,
                duration_s=10.0 * (25.0 / paper_speed_mm_s),
                label_box=lbox,
            )
        )

    n_rhy = len(rhythm_entries)
    lname = f"{base_layout}_with_r{n_rhy}" if n_rhy > 0 else base_layout

    return UniversalLayoutDetection(
        layout_name=lname,
        px_per_mm_x=px_mm_x,
        px_per_mm_y=px_mm_y,
        n_rows=R,
        n_cols=C,
        n_rhythm_rows=n_rhy,
        is_cabrera=is_cabrera,
        panels=panels,
        metadata={
            "pxPerMmX": round(px_mm_x, 4),
            "pxPerMmY": round(px_mm_y, 4),
            "labelsDetected": len(hits),
            "matchedGridLabels": n_matched,
            "layoutName": lname,
        },
    )


def _build_generic_blank_grid_template() -> np.ndarray:
    """Build a seamless 1024x1024 periodic 80x80 blank ECG grid base (major gridlines at 0, 80, 160...).

    Aligned per-crop via `_align_template_to_crop` so CoaT+VGG19 receives exact zero-phase-shift
    grid conditioning regardless of layout or chunk offset.
    """
    from ecg_pipeline.neural_digitizer import HEATMAP_TEMPLATES_DIR

    tpl = cv2.imread(str(HEATMAP_TEMPLATES_DIR / "V4.png"), cv2.IMREAD_GRAYSCALE)
    if tpl is None or tpl.shape != (1024, 1024):
        raise RuntimeError("Missing V4.png template for generic grid synthesis")
    vx = np.flatnonzero((tpl < 200).mean(axis=0) > 0.5)
    v_centers = [int(round(np.mean(c))) for c in np.split(vx, np.flatnonzero(np.diff(vx) > 5) + 1)]
    hy = np.flatnonzero((tpl < 200).mean(axis=1) > 0.5)
    h_centers = [int(round(np.mean(c))) for c in np.split(hy, np.flatnonzero(np.diff(hy) > 5) + 1)]
    x0_t, y0_t = v_centers[4], h_centers[2]
    tile = tpl[y0_t : y0_t + 80, x0_t : x0_t + 80]
    return np.tile(tile, (1024 // 80 + 2, 1024 // 80 + 2))[:1024, :1024].copy()


def _align_template_to_crop(crop_rgb: np.ndarray, base_grid: np.ndarray) -> np.ndarray:
    """Shift the 80x80 periodic blank grid template to match the exact major-grid phase of crop_rgb."""
    r = crop_rgb[:, :, 0].astype(np.float32)
    gb = 0.5 * (crop_rgb[:, :, 1].astype(np.float32) + crop_rgb[:, :, 2].astype(np.float32))
    g_map = np.clip(r - gb, 0.0, 255.0)
    if float(np.percentile(g_map, 95)) < 6.0:
        g_map = 255.0 - cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    px_prof = g_map.mean(axis=0)
    py_prof = g_map.mean(axis=1)
    dx = int(np.argmax([px_prof[i::80].mean() for i in range(80)]))
    dy = int(np.argmax([py_prof[i::80].mean() for i in range(80)]))
    return np.roll(np.roll(base_grid, dy, axis=0), dx, axis=1)


class UniversalPerLeadDigitizer:
    """Layout-agnostic per-lead digitizer powered by CoaT+VGG19 1024x1024 chunk decoding."""

    def __init__(self, decoder: Any | None = None, device: str = "auto") -> None:
        if decoder is None:
            from ecg_pipeline.neural_digitizer import CoatVggHeatmapDecoder

            decoder = CoatVggHeatmapDecoder(device=device, use_fp16=True, amp_dtype="bfloat16")
        self.decoder = decoder
        self.generic_template = _build_generic_blank_grid_template()

    def _decode_chunk_2p5s(
        self,
        rgb: np.ndarray,
        x0_px: float,
        y_base_px: float,
        px_per_mm_x: float,
        px_per_mm_y: float,
        gain_mm_mv: float = 10.0,
        row_step_mm: float = 36.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract one 62.5 mm (2.5 s) x 64.0 mm (±3.2 mV) chunk via Lanczos remap and decode with CoaT+VGG19."""
        H, W, L = self.decoder.H, self.decoder.W, self.decoder.L
        pad = self.decoder.pad  # 12
        # Active 1000 px spans 62.5 mm horizontally; full 1024 px spans 64.0 mm horizontally & vertically
        span_x_mm = 62.5 * (W / L)  # 64.0 mm
        span_y_mm = 64.0
        pad_x_mm = (span_x_mm - 62.5) / 2.0  # 0.75 mm

        x_left_px = x0_px - pad_x_mm * px_per_mm_x
        y_top_px = y_base_px - (span_y_mm / 2.0) * px_per_mm_y

        xs = x_left_px + (np.arange(W, dtype=np.float64) + 0.5) * (span_x_mm * px_per_mm_x / W)
        ys = y_top_px + (np.arange(H, dtype=np.float64) + 0.5) * (span_y_mm * px_per_mm_y / H)
        map_x = np.broadcast_to(xs[None, :], (H, W)).astype(np.float32)
        map_y = np.broadcast_to(ys[:, None], (H, W)).astype(np.float32)

        crop_rgb = cv2.remap(
            rgb,
            map_x,
            map_y,
            interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        aligned_tpl = _align_template_to_crop(crop_rgb, self.generic_template)
        logits = self.decoder._infer_crop_logits(crop_rgb, aligned_tpl)
        roi_logits = logits[:, pad : pad + L].copy()
        # Prevent adjacent-row jumps on dense layouts (e.g. 6x2 at 30mm or 12x1 at 22mm) by
        # staying ~6.5 mm clear of the adjacent row baseline while preserving tall R-waves.
        half_allowed_mm = min(31.5, max(12.0, row_step_mm - 6.5))
        px_per_mm_crop = H / span_y_mm  # 16 px/mm
        y_min_ok = max(0, int(round(H / 2.0 - half_allowed_mm * px_per_mm_crop)))
        y_max_ok = min(H, int(round(H / 2.0 + half_allowed_mm * px_per_mm_crop)))
        if y_min_ok > 0:
            roi_logits[:y_min_ok, :] = -1e4
        if y_max_ok < H:
            roi_logits[y_max_ok:, :] = -1e4

        roi_shifted = roi_logits - roi_logits.max(axis=0, keepdims=True)
        exp_l = np.exp(roi_shifted)
        prob = exp_l / np.maximum(exp_l.sum(axis=0, keepdims=True), 1e-12)

        y_sub_idx, _ = _subpixel_peak_1d(prob, half_window=6, power=1.0)
        # Suppress isolated 1-3 column vertical row-hop glitches via 5-sample median spike check
        y_med5 = cv2.medianBlur(y_sub_idx.astype(np.float32).reshape(1, -1), 5).ravel().astype(np.float64)
        spike = np.abs(y_sub_idx - y_med5) > (12.0 * px_per_mm_crop)  # > 1.2 mV single-column jump
        if spike.any():
            y_sub_idx = np.where(spike, y_med5, y_sub_idx)
        peak_int = np.argmax(prob, axis=0)
        offsets = np.arange(-6, 7, dtype=np.int64)[:, None]
        rows = np.clip(peak_int[None, :] + offsets, 0, H - 1)
        cols = np.broadcast_to(np.arange(L, dtype=np.int64)[None, :], rows.shape)
        local_mass = prob[rows, cols].sum(axis=0)

        # Convert continuous y coordinate to physical uV (scaled by 10 / gain_mm_mv)
        y_cont = y_sub_idx + 0.5
        sig_mv = (H / 2.0 - y_cont) / self.decoder.signal_scale_factor
        sig_uv_1000 = sig_mv * 1000.0 * (10.0 / gain_mm_mv)
        gap = local_mass < self.decoder.min_peak_mass
        sig_uv_1000[gap] = np.nan

        # Resample 1000 -> 1250 samples
        t_src = np.linspace(0.0, 1.0, L, endpoint=False)
        t_dst = np.linspace(0.0, 1.0, 1250, endpoint=False)
        if np.isfinite(sig_uv_1000).all():
            sig_1250 = np.interp(t_dst, t_src, sig_uv_1000)
            conf_1250 = np.interp(t_dst, t_src, local_mass)
        else:
            valid = np.isfinite(sig_uv_1000)
            if valid.sum() >= 2:
                sig_1250 = np.interp(t_dst, t_src[valid], sig_uv_1000[valid])
                nearest = np.clip(np.round(t_dst * L).astype(np.int64), 0, L - 1)
                sig_1250[~valid[nearest]] = np.nan
                conf_1250 = np.interp(t_dst, t_src, local_mass)
            else:
                sig_1250 = np.full(1250, np.nan, dtype=np.float64)
                conf_1250 = np.zeros(1250, dtype=np.float64)

        # Project back onto untouched original coordinates
        y_clamp = np.clip(np.round(y_sub_idx).astype(np.int64), 0, H - 1)
        col_idx = np.arange(pad, pad + L, dtype=np.int64)
        xy = np.stack([map_x[y_clamp, col_idx], map_y[y_clamp, col_idx]], axis=-1).astype(np.float64)
        xy[gap] = np.nan
        return sig_1250, conf_1250, xy

    def digitize(
        self,
        rgb: np.ndarray,
        *,
        paper_speed_mm_s: float = 25.0,
        gain_mm_mv: float = 10.0,
    ) -> tuple[UniversalLayoutDetection, DigitizationOutput]:
        t0 = time.monotonic()
        det = detect_universal_layout(rgb, paper_speed_mm_s=paper_speed_mm_s)

        eff_fs = 500.0 * (paper_speed_mm_s / 25.0)
        leads_uv = {l: np.full(5000, np.nan, dtype=np.float64) for l in ALL_12_LEADS}
        confidence = {l: np.full(5000, np.nan, dtype=np.float64) for l in ALL_12_LEADS}
        orig_xy: dict[str, np.ndarray] = {}
        short_ii_uv: np.ndarray | None = None
        rhythm_strips: dict[str, np.ndarray] = {}

        chunk_mm = 62.5
        chunk_w_px = chunk_mm * det.px_per_mm_x
        row_medians = [
            float(np.median([p.y_baseline_px for p in det.panels if p.row_idx == r]))
            for r in sorted({p.row_idx for p in det.panels})
        ]
        row_step_mm = (
            float(np.median(np.diff(row_medians))) / det.px_per_mm_y
            if len(row_medians) >= 2
            else 36.0
        )

        for p in det.panels:
            n_chunks = max(1, int(round((p.x_end_px - p.x_start_px) / chunk_w_px)))
            segs, confs, xys = [], [], []
            for k in range(n_chunks):
                s_k, c_k, xy_k = self._decode_chunk_2p5s(
                    rgb,
                    x0_px=p.x_start_px + k * chunk_w_px,
                    y_base_px=p.y_baseline_px,
                    px_per_mm_x=det.px_per_mm_x,
                    px_per_mm_y=det.px_per_mm_y,
                    gain_mm_mv=gain_mm_mv,
                    row_step_mm=row_step_mm,
                )
                segs.append(s_k)
                confs.append(c_k)
                xys.append(xy_k)
            full_sig = np.concatenate(segs)
            full_conf = np.concatenate(confs)
            full_xy = np.concatenate(xys, axis=0)

            # Invert Cabrera -aVR back into canonical aVR!
            if p.inverted:
                full_sig = -full_sig

            if np.isfinite(full_sig).any():
                full_sig = full_sig - float(np.nanmedian(full_sig))

            seg_key = f"{p.lead}_rhythm_{p.row_idx}" if p.is_rhythm else f"{p.lead}_r{p.row_idx}_c{p.col_idx}"
            orig_xy[seg_key] = full_xy

            if p.is_rhythm:
                rhythm_strips[p.lead] = full_sig[:5000]
                # Full 10s rhythm strip supersedes 2.5s panel for that lead (e.g. II, V1, V5 in 3x4+3r)
                L_use = min(5000, len(full_sig))
                leads_uv[p.lead][:L_use] = full_sig[:L_use]
                confidence[p.lead][:L_use] = full_conf[:L_use]
            else:
                s0 = p.col_idx * len(full_sig)
                s1 = min(5000, s0 + len(full_sig))
                if p.lead == "II" and any(rp.is_rhythm and rp.lead == "II" for rp in det.panels):
                    short_ii_uv = full_sig[:1250].copy()
                elif not np.isfinite(leads_uv[p.lead][s0:s1]).any():
                    leads_uv[p.lead][s0:s1] = full_sig[: s1 - s0]
                    confidence[p.lead][s0:s1] = full_conf[: s1 - s0]

        out = DigitizationOutput(
            leads_uv=leads_uv,
            short_ii_uv=short_ii_uv,
            confidence=confidence,
            orig_xy_by_segment=orig_xy,
            elapsed_seconds=time.monotonic() - t0,
            metadata={
                "decoder": "waveparse_universal_per_lead_coat_vgg19",
                "layout": det.layout_name,
                "isCabrera": det.is_cabrera,
                "panelsDecoded": len(det.panels),
                "rhythmStripLeads": sorted(rhythm_strips.keys()),
                "effectiveSampleRateHz": eff_fs,
                "pxPerMmX": round(det.px_per_mm_x, 4),
                "pxPerMmY": round(det.px_per_mm_y, 4),
            },
        )
        return det, out
