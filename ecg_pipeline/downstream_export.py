"""Standard downstream ECG exports (WFDB .hea/.dat, structured JSON) and any-layout paper rendering.

Designed so downstream clinical measurement and AI diagnosis tools receive exact,
unambiguous physical signals (uV, Hz, observed time windows, explicit NaN/missing
masks, calibration provenance, and clinical trust dispositions) without fake zero-fill.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ALL_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
WFDB_MISSING_INT16 = -32768


def _int16_checksum(arr: np.ndarray) -> int:
    """Standard WFDB signed 16-bit wraparound sum checksum."""
    total = int(np.sum(arr.astype(np.int64))) & 0xFFFF
    return total - 0x10000 if total >= 0x8000 else total


def write_wfdb_record(
    leads_uv: dict[str, np.ndarray],
    out_dir: Path,
    record_name: str = "digitized_ecg",
    sample_rate_hz: float = 500.0,
    leads_order: list[str] | None = None,
) -> dict[str, Path]:
    """Write a standard PhysioNet WFDB Format-16 record (.hea + .dat).

    Encoding:
      - Format 16 (little-endian signed 16-bit interleaved samples).
      - Gain = 1000 adu/mV (exact 1 uV per digital unit; range ±32.767 mV).
      - Non-finite samples (NaN gaps / unprinted windows) are encoded as the
        standard WFDB missing-sample sentinel (-32768).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    order = leads_order or ALL_12_LEADS
    n_sig = len(order)
    n_samples = max(len(leads_uv[l]) for l in order if l in leads_uv)

    digital = np.full((n_samples, n_sig), WFDB_MISSING_INT16, dtype=np.int16)
    for j, lead in enumerate(order):
        if lead not in leads_uv:
            continue
        arr = np.asarray(leads_uv[lead], dtype=np.float64)
        L = min(len(arr), n_samples)
        valid = np.isfinite(arr[:L])
        if valid.any():
            vals = np.clip(np.round(arr[:L][valid]), -32767.0, 32767.0).astype(np.int16)
            digital[:L, j][valid] = vals

    dat_name = f"{record_name}.dat"
    hea_name = f"{record_name}.hea"
    dat_path = out_dir / dat_name
    hea_path = out_dir / hea_name

    # Interleaved little-endian signed 16-bit
    dat_path.write_bytes(digital.astype("<i2").tobytes())

    fs_str = f"{int(round(sample_rate_hz))}" if abs(sample_rate_hz - round(sample_rate_hz)) < 1e-6 else f"{sample_rate_hz:g}"
    header_lines = [f"{record_name} {n_sig} {fs_str} {n_samples}"]
    for j, lead in enumerate(order):
        col = digital[:, j]
        csum = _int16_checksum(col)
        init_val = int(col[0]) if n_samples > 0 else 0
        # Format: file format gain(baseline)/units adc_res adc_zero init_val checksum block_size desc
        header_lines.append(f"{dat_name} 16 1000(0)/mV 16 0 {init_val} {csum} 0 {lead}")

    hea_path.write_text("\n".join(header_lines) + "\n", encoding="ascii")
    return {"hea": hea_path, "dat": dat_path}


def read_wfdb_record(hea_path: Path) -> tuple[dict[str, np.ndarray], float]:
    """Read a Format-16 WFDB record back into {lead: uv_array} and sample_rate_hz, verifying checksums."""
    lines = [ln.strip() for ln in hea_path.read_text(encoding="ascii").splitlines() if ln.strip() and not ln.strip().startswith("#")]
    top = lines[0].split()
    n_sig = int(top[1])
    fs = float(top[2].split("/")[0])
    n_samples = int(top[3])

    dat_file = hea_path.parent / lines[1].split()[0]
    raw = np.frombuffer(dat_file.read_bytes(), dtype="<i2").reshape(n_samples, n_sig)

    leads: dict[str, np.ndarray] = {}
    for j in range(n_sig):
        toks = lines[j + 1].split()
        gain_tok = toks[2].split("/")[0]
        if "(" in gain_tok:
            gain = float(gain_tok.split("(")[0])
            baseline = float(gain_tok.split("(")[1].rstrip(")"))
        else:
            gain = float(gain_tok)
            baseline = 0.0
        expected_csum = int(toks[6])
        desc = toks[8] if len(toks) > 8 else f"lead_{j}"
        col = raw[:, j]
        actual_csum = _int16_checksum(col)
        if actual_csum != expected_csum:
            raise ValueError(f"WFDB checksum mismatch on {desc}: got {actual_csum}, expected {expected_csum}")
        uv = np.full(n_samples, np.nan, dtype=np.float64)
        valid = col != WFDB_MISSING_INT16
        uv[valid] = (col[valid].astype(np.float64) - baseline) * (1000.0 / gain)
        leads[desc] = uv
    return leads, fs


def write_downstream_json(
    leads_uv: dict[str, np.ndarray],
    confidence: dict[str, np.ndarray] | None,
    out_path: Path,
    *,
    sample_rate_hz: float = 500.0,
    source_layout: str = "standard_3x4_with_r1",
    engine: str = "WaveParse-Neural-3x4",
    calibration: dict[str, Any] | None = None,
    clinical_quality: dict[str, Any] | None = None,
    short_ii_uv: np.ndarray | None = None,
) -> dict[str, Any]:
    """Write a self-describing JSON payload for downstream measurement & diagnostic tools."""
    leads_payload: dict[str, Any] = {}
    for lead in ALL_12_LEADS:
        if lead not in leads_uv:
            continue
        arr = np.asarray(leads_uv[lead], dtype=np.float64)
        finite_idx = np.flatnonzero(np.isfinite(arr))
        if finite_idx.size == 0:
            leads_payload[lead] = {
                "observedWindowSamples": None,
                "durationSec": 0.0,
                "coverage": 0.0,
                "valuesUv": [],
                "confidence": [],
            }
            continue
        s0 = int(finite_idx[0])
        s1 = int(finite_idx[-1]) + 1
        seg = arr[s0:s1]
        cov = float(np.isfinite(seg).mean())
        vals = [round(float(v), 2) if np.isfinite(v) else None for v in seg]
        conf_vals: list[float | None] | None = None
        if confidence is not None and lead in confidence:
            cseg = np.asarray(confidence[lead][s0:s1], dtype=np.float64)
            conf_vals = [round(float(c), 4) if np.isfinite(c) else None for c in cseg]
        leads_payload[lead] = {
            "observedWindowSamples": [s0, s1],
            "startSec": round(s0 / sample_rate_hz, 4),
            "durationSec": round((s1 - s0) / sample_rate_hz, 4),
            "coverage": round(cov, 4),
            "valuesUv": vals,
            **({"confidence": conf_vals} if conf_vals is not None else {}),
        }

    rhythm_strips: dict[str, Any] = {}
    if short_ii_uv is not None and np.isfinite(short_ii_uv).any():
        rhythm_strips["II_short"] = {
            "startSec": 0.0,
            "durationSec": round(len(short_ii_uv) / sample_rate_hz, 4),
            "valuesUv": [round(float(v), 2) if np.isfinite(v) else None for v in short_ii_uv],
        }

    disp = (clinical_quality or {}).get("disposition", "usable")
    payload = {
        "schemaVersion": "waveparse.downstream.v1",
        "engine": engine,
        "sourceLayout": source_layout,
        "sampleRateHz": float(sample_rate_hz),
        "units": "uV",
        "usableForAutomatedAnalysis": disp != "review_required",
        "calibration": calibration or {
            "paperSpeedMmPerSecond": 25.0,
            "gainMmPerMv": 10.0,
            "effectiveSampleRateHz": float(sample_rate_hz),
        },
        "clinicalQuality": clinical_quality,
        "leads": leads_payload,
        "rhythmStrips": rhythm_strips,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _extract_lead_window(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """Return the trimmed finite window of a lead signal and its start sample."""
    fin = np.flatnonzero(np.isfinite(arr))
    if fin.size == 0:
        return np.array([], dtype=np.float64), 0
    s0, s1 = int(fin[0]), int(fin[-1]) + 1
    seg = arr[s0:s1].copy()
    seg -= float(np.nanmedian(seg))
    return seg, s0


def render_any_layout_paper(
    leads_uv: dict[str, np.ndarray],
    save_path: Path,
    *,
    layout: str = "3x4_r1",
    sample_rate_hz: float = 500.0,
    paper_speed_mm_s: float = 25.0,
    gain_mm_mv: float = 10.0,
    short_ii_uv: np.ndarray | None = None,
    title: str = "",
) -> None:
    """Render digitized 12-lead ECG onto synthetic ECG paper in ANY target layout.

    Supported layouts:
      - '3x4_r1': Standard 3x4 (2.5 s panels) + 10 s rhythm strip II
      - '6x2':    6 rows x 2 columns (I/V1 .. aVF/V6)
      - '12x1':   12 stacked full-width leads
      - 'cabrera': European Cabrera order (aVL, I, -aVR, II, aVF, III, V1..V6) in 6x2
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layout_key = layout.lower().strip()
    if layout_key in ("3x4_r1", "standard_3x4_with_r1", "3x4"):
        grid_rows = [
            ["I", "aVR", "V1", "V4"],
            ["II", "aVL", "V2", "V5"],
            ["III", "aVF", "V3", "V6"],
        ]
        has_rhythm = True
        total_sec = 10.0
        col_sec = 2.5
        fig_size = (13.0, 8.5)
        row_step_mv = 4.0
    elif layout_key in ("6x2", "standard_6x2", "cabrera"):
        if layout_key == "cabrera":
            grid_rows = [
                ["aVL", "V1"],
                ["I", "V2"],
                ["-aVR", "V3"],
                ["II", "V4"],
                ["aVF", "V5"],
                ["III", "V6"],
            ]
        else:
            grid_rows = [
                ["I", "V1"],
                ["II", "V2"],
                ["III", "V3"],
                ["aVR", "V4"],
                ["aVL", "V5"],
                ["aVF", "V6"],
            ]
        has_rhythm = False
        col_sec = 5.0
        total_sec = 10.0
        fig_size = (13.0, 10.5)
        row_step_mv = 3.2
    elif layout_key in ("12x1", "standard_12x1"):
        grid_rows = [[l] for l in ALL_12_LEADS]
        has_rhythm = False
        col_sec = 10.0
        total_sec = 10.0
        fig_size = (13.0, 14.0)
        row_step_mv = 2.8
    else:
        raise ValueError(f"Unsupported render layout: {layout}")

    n_rows = len(grid_rows) + (1 if has_rhythm else 0)
    y_min = -2.0
    y_max = (n_rows - 1) * row_step_mv + 2.5

    fig, ax = plt.subplots(figsize=fig_size, dpi=150)
    ax.set_facecolor("#fff8f8")
    ax.set_xlim(0.0, total_sec)
    ax.set_ylim(y_min, y_max)

    minor_dt = 1.0 / paper_speed_mm_s
    major_dt = 5.0 / paper_speed_mm_s
    minor_dmv = 1.0 / gain_mm_mv
    major_dmv = 5.0 / gain_mm_mv

    for x in np.arange(0.0, total_sec + 1e-6, minor_dt):
        ax.axvline(x, color="#f6d6d6", lw=0.35, zorder=0)
    for y in np.arange(y_min, y_max + 1e-6, minor_dmv):
        ax.axhline(y, color="#f6d6d6", lw=0.35, zorder=0)
    for x in np.arange(0.0, total_sec + 1e-6, major_dt):
        ax.axvline(x, color="#e39a9a", lw=0.75, zorder=1)
    for y in np.arange(y_min, y_max + 1e-6, major_dmv):
        ax.axhline(y, color="#e39a9a", lw=0.75, zorder=1)

    for r_idx, row_leads in enumerate(grid_rows):
        y0 = (n_rows - 1 - r_idx) * row_step_mv
        for c_idx, label in enumerate(row_leads):
            negate = label.startswith("-")
            base_lead = label[1:] if negate else label
            if base_lead == "II" and has_rhythm and short_ii_uv is not None and np.isfinite(short_ii_uv).any():
                seg = short_ii_uv.copy()
                seg -= float(np.nanmedian(seg))
            elif base_lead in leads_uv:
                seg, _ = _extract_lead_window(leads_uv[base_lead])
            else:
                continue
            if seg.size == 0:
                continue
            if negate:
                seg = -seg
            max_pts = int(round(col_sec * sample_rate_hz))
            seg = seg[:max_pts]
            x_start = c_idx * col_sec
            t = x_start + np.arange(len(seg), dtype=np.float64) / sample_rate_hz
            ax.plot(t, y0 + seg / 1000.0, color="#111111", lw=0.9, zorder=3)
            ax.text(x_start + 0.06, y0 + 0.95, label, fontsize=9, fontweight="bold", color="#111111", zorder=4)

    if has_rhythm and "II" in leads_uv:
        y0 = 0.0
        seg, s0 = _extract_lead_window(leads_uv["II"])
        if seg.size > 0:
            t = (s0 + np.arange(len(seg), dtype=np.float64)) / sample_rate_hz
            ax.plot(t, y0 + seg / 1000.0, color="#111111", lw=0.9, zorder=3)
            ax.text(0.06, y0 + 0.95, "II (rhythm)", fontsize=9, fontweight="bold", color="#111111", zorder=4)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(
        f"{paper_speed_mm_s:g} mm/s   |   {gain_mm_mv:g} mm/mV   |   {sample_rate_hz:g} Hz   |   layout: {layout_key}",
        fontsize=9,
    )
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
