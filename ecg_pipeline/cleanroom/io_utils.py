"""100% MIT clean-room I/O, CSV/uncertainty writers, source overlay renderer, and clinical physics QA.

Zero third-party neural repositories, zero benchmark imports, zero Kaggle dependencies.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ALL_12_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


@dataclass
class DigitizationOutput:
    leads_uv: dict[str, np.ndarray]  # 12 leads, each float64 shape (5000,) in uV (NaN where absent/gap)
    short_ii_uv: np.ndarray | None  # 1250 samples from row 2 col 0 short lead II (for physics check)
    confidence: dict[str, np.ndarray]  # per-lead sample confidence in [0, 1]
    orig_xy_by_segment: dict[str, np.ndarray]  # segment_key -> float64 (N, 2) untouched-source (x, y)
    elapsed_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _subpixel_peak_1d(prob: np.ndarray, half_window: int = 5, power: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Local soft-argmax around the column-wise probability peak -> (y_subpixel, peak_prob)."""
    H, W = prob.shape
    peak_idx = np.argmax(prob, axis=0)
    peak_val = prob[peak_idx, np.arange(W)]
    offsets = np.arange(-half_window, half_window + 1, dtype=np.int64)[:, None]
    rows = np.clip(peak_idx[None, :] + offsets, 0, H - 1)
    cols = np.broadcast_to(np.arange(W, dtype=np.int64)[None, :], rows.shape)
    weights = prob[rows, cols]
    if power != 1.0:
        weights = np.power(np.maximum(weights, 1e-12), power)
    wsum = np.maximum(weights.sum(axis=0), 1e-12)
    y_sub = (rows * weights).sum(axis=0) / wsum
    return y_sub, peak_val


def write_leads_csv(path: Path, leads_uv: dict[str, np.ndarray], sample_rate_hz: float = 500.0) -> None:
    """Write canonical 12-lead ECG CSV in microvolts with empty cells for missing/gap samples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = max((len(v) for v in leads_uv.values()), default=5000)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "t_s", *ALL_12_LEADS])
        for i in range(n_samples):
            row: list[str] = [str(i), f"{i / sample_rate_hz:.6f}"]
            for ld in ALL_12_LEADS:
                arr = leads_uv.get(ld)
                if arr is not None and i < len(arr) and np.isfinite(arr[i]):
                    row.append(f"{float(arr[i]):.3f}")
                else:
                    row.append("")
            writer.writerow(row)


def write_compact_segments_csv(dig: DigitizationOutput, out_path: Path, sample_rate_hz: float = 500.0) -> None:
    """Write WaveParse compact per-lead segment CSV (each lead trimmed to its active non-NaN window)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trimmed: dict[str, np.ndarray] = {}
    max_len = 0
    for ld in ALL_12_LEADS:
        arr = dig.leads_uv.get(ld)
        if arr is None:
            trimmed[ld] = np.array([], dtype=np.float64)
            continue
        fin = np.flatnonzero(np.isfinite(arr))
        if fin.size == 0:
            trimmed[ld] = np.array([], dtype=np.float64)
        else:
            seg = arr[int(fin[0]) : int(fin[-1]) + 1]
            trimmed[ld] = seg
            max_len = max(max_len, len(seg))

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "t_s_at_500hz", *ALL_12_LEADS])
        for i in range(max_len):
            row: list[str] = [str(i), f"{i / sample_rate_hz:.6f}"]
            for ld in ALL_12_LEADS:
                seg = trimmed[ld]
                if i < len(seg) and np.isfinite(seg[i]):
                    row.append(f"{float(seg[i]):.3f}")
                else:
                    row.append("")
            writer.writerow(row)


def write_waveparse_uncertainty_csv(
    dig: DigitizationOutput,
    out_path: Path,
    sample_rate_hz: float = 500.0,
) -> dict[str, int]:
    """Write WaveParse 10-column sample-level uncertainty CSV matching the Web UI / API contract."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    uncertain_count = 0
    missing_count = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "lead",
            "lead_sample",
            "t_s_at_500hz",
            "canonical_sample",
            "value_uv",
            "review_estimate_uv",
            "status",
            "annotation_overlap",
            "candidate_count",
            "candidate_spread_uv",
        ])
        for ld in ALL_12_LEADS:
            sig = dig.leads_uv.get(ld)
            conf = dig.confidence.get(ld)
            if sig is None:
                continue
            fin = np.flatnonzero(np.isfinite(sig))
            if fin.size == 0:
                continue
            s0, s1 = int(fin[0]), int(fin[-1]) + 1
            for local_idx, canon_idx in enumerate(range(s0, s1)):
                val = sig[canon_idx]
                c_val = float(conf[canon_idx]) if (conf is not None and canon_idx < len(conf) and np.isfinite(conf[canon_idx])) else 0.0
                if not np.isfinite(val):
                    status = "missing"
                    missing_count += 1
                    val_s = ""
                elif c_val < 0.45:
                    status = "uncertain"
                    uncertain_count += 1
                    val_s = f"{float(val):.3f}"
                else:
                    status = "observed"
                    val_s = f"{float(val):.3f}"
                spread_uv = max(0.0, (1.0 - c_val) * 60.0)
                writer.writerow([
                    ld,
                    local_idx,
                    f"{local_idx / sample_rate_hz:.6f}",
                    canon_idx,
                    val_s,
                    val_s,
                    status,
                    "false",
                    1,
                    f"{spread_uv:.2f}",
                ])
    return {"uncertainSampleCount": uncertain_count, "missingSampleCount": missing_count}


def render_source_overlay(orig_rgb: np.ndarray, dig: DigitizationOutput, out_path: Path) -> None:
    """Draw digitized traces directly on top of the untouched original ECG image for visual verification."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    thick = max(2, int(round(max(h, w) / 1400.0)))
    palette = [
        (235, 115, 20),   # crisp cobalt blue (BGR)
        (45, 165, 35),    # emerald green
        (15, 95, 220),    # warm vermilion
        (190, 45, 165),   # vivid magenta
    ]
    for idx, (seg_key, xy) in enumerate(dig.orig_xy_by_segment.items()):
        if xy is None or len(xy) < 2:
            continue
        col_i = idx % len(palette)
        if "_c0" in seg_key or "_rhythm" in seg_key:
            col_i = 0
        elif "_c1" in seg_key:
            col_i = 1
        elif "_c2" in seg_key:
            col_i = 2
        elif "_c3" in seg_key:
            col_i = 3
        color = palette[col_i]
        finite = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        # Draw contiguous finite polylines only (gaps stay visibly blank!)
        runs = np.flatnonzero(np.diff(np.concatenate(([0], finite.astype(np.int8), [0]))))
        for a, b in runs.reshape(-1, 2):
            if b - a >= 2:
                pts = np.round(xy[a:b]).astype(np.int32)
                cv2.polylines(bgr, [pts], False, color, thick, cv2.LINE_AA)
    cv2.imwrite(str(out_path), bgr)


def _lag_aligned_pair(a: np.ndarray, b: np.ndarray, max_lag: int = 12) -> tuple[float | None, float | None, int | None]:
    m0 = np.isfinite(a) & np.isfinite(b)
    if m0.sum() < 200:
        return None, None, None
    best_r = -2.0
    best_rmse = None
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        sa = a[lag:] if lag >= 0 else a[: len(a) + lag]
        sb = b[: len(b) - lag] if lag >= 0 else b[-lag:]
        m = np.isfinite(sa) & np.isfinite(sb)
        if m.sum() < 200:
            continue
        xa = sa[m] - np.median(sa[m])
        xb = sb[m] - np.median(sb[m])
        denom = float(np.linalg.norm(xa) * np.linalg.norm(xb))
        if denom < 1e-6:
            continue
        r = float(np.dot(xa, xb) / denom)
        if r > best_r:
            best_r = r
            best_lag = lag
            best_rmse = float(np.sqrt(np.mean((xa - xb) ** 2)))
    if best_rmse is None:
        return None, None, None
    return round(best_rmse, 2), round(best_r, 4), best_lag


def compute_cleanroom_clinical_quality(
    dig: DigitizationOutput,
    *,
    layout: str,
    was_dewarped: bool,
    phase_pixels: int,
    sample_rate_hz: float = 500.0,
) -> dict[str, Any]:
    """Compute reference-free clinical physics self-checks and per-lead coverage/gap QA."""
    warnings: list[str] = []
    per_lead: dict[str, Any] = {}
    for ld in ALL_12_LEADS:
        sig = dig.leads_uv.get(ld)
        conf = dig.confidence.get(ld)
        if sig is None:
            per_lead[ld] = {"coverage": 0.0, "meanConfidence": 0.0, "p05Confidence": 0.0, "gapIntervalsSec": [], "lowConfidenceIntervalsSec": [], "tallAmplitudeRescued": False}
            warnings.append(f"Lead {ld} missing")
            continue
        fin = np.flatnonzero(np.isfinite(sig))
        if fin.size == 0:
            per_lead[ld] = {"coverage": 0.0, "meanConfidence": 0.0, "p05Confidence": 0.0, "gapIntervalsSec": [], "lowConfidenceIntervalsSec": [], "tallAmplitudeRescued": False}
            warnings.append(f"Lead {ld} has no observed samples")
            continue
        s0, s1 = int(fin[0]), int(fin[-1]) + 1
        seg = sig[s0:s1]
        c_seg = conf[s0:s1] if conf is not None else np.ones_like(seg)
        cov = float(np.isfinite(seg).mean())
        c_fin = c_seg[np.isfinite(c_seg)]
        mean_c = float(np.mean(c_fin)) if c_fin.size else 0.0
        p05_c = float(np.percentile(c_fin, 5)) if c_fin.size else 0.0
        gaps: list[list[float]] = []
        miss = (~np.isfinite(seg)).astype(np.int8)
        if miss.any():
            runs = np.flatnonzero(np.diff(np.concatenate(([0], miss, [0])))).reshape(-1, 2)
            for a, b in runs:
                if b - a >= 5:
                    gaps.append([round((s0 + int(a)) / sample_rate_hz, 3), round((s0 + int(b)) / sample_rate_hz, 3)])
        per_lead[ld] = {
            "coverage": round(cov, 4),
            "meanConfidence": round(mean_c, 4),
            "p05Confidence": round(p05_c, 4),
            "gapIntervalsSec": gaps,
            "lowConfidenceIntervalsSec": [],
            "tallAmplitudeRescued": False,
        }
        if cov < 0.85:
            warnings.append(f"Lead {ld} coverage is {cov * 100:.1f}%")

    # Reference-free physical ECG laws:
    # 1) Short lead II (col 0) vs simultaneous rhythm lead II [0:1250]
    short_rmse = short_corr = short_lag = None
    if dig.short_ii_uv is not None and np.isfinite(dig.leads_uv["II"][:1250]).sum() > 200:
        short_rmse, short_corr, short_lag = _lag_aligned_pair(dig.short_ii_uv, dig.leads_uv["II"][:1250])
        if short_rmse is not None and short_rmse > 120.0:
            warnings.append(f"Short vs Long Lead II disagreement ({short_rmse:.1f} uV)")

    # 2) Einthoven's law: I + III == II (where all three overlap simultaneously)
    ein_rmse = ein_corr = ein_lag = None
    m_ein = np.isfinite(dig.leads_uv["I"]) & np.isfinite(dig.leads_uv["II"]) & np.isfinite(dig.leads_uv["III"])
    if m_ein.sum() >= 250:
        idx_e = np.flatnonzero(m_ein)
        e0, e1 = int(idx_e[0]), int(idx_e[-1]) + 1
        ein_rmse, ein_corr, ein_lag = _lag_aligned_pair(
            dig.leads_uv["I"][e0:e1] + dig.leads_uv["III"][e0:e1],
            dig.leads_uv["II"][e0:e1],
        )

    # 3) Goldberger's law: aVR + aVL + aVF == 0
    gold_rms = None
    m_g = np.isfinite(dig.leads_uv["aVR"]) & np.isfinite(dig.leads_uv["aVL"]) & np.isfinite(dig.leads_uv["aVF"])
    if m_g.sum() >= 250:
        s_g = dig.leads_uv["aVR"][m_g] + dig.leads_uv["aVL"][m_g] + dig.leads_uv["aVF"][m_g]
        s_g = s_g - np.median(s_g)
        gold_rms = round(float(np.sqrt(np.mean(s_g ** 2))), 2)

    low_cov_leads = sum(1 for v in per_lead.values() if v["coverage"] < 0.70)
    disposition = "review_required" if low_cov_leads >= 3 else ("usable_with_warnings" if warnings else "usable")
    return {
        "disposition": disposition,
        "warnings": warnings,
        "layoutDetected": layout,
        "wasDewarped": was_dewarped,
        "phasePixels": phase_pixels,
        "physicsSelfChecks": {
            "shortVsLongII": {"rmseUv": short_rmse, "corr": short_corr, "shiftSamples": short_lag},
            "einthovenIPlusIIIvsII": {"rmseUv": ein_rmse, "corr": ein_corr, "shiftSamples": ein_lag},
            "goldbergerSumRmsUv": gold_rms,
        },
        "perLead": per_lead,
    }
