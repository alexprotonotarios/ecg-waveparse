"""Clean-room synthetic ECG training data generator using 100% open CC-BY-4.0 waveforms.

Sources: PhysioNet / CinC Challenge 2021 multi-hospital clinical 12-lead ECGs:
  - chapman_shaoxing (212 records)
  - cpsc_2018        (212 records)
  - cpsc_2018_extra  (212 records)
  - ningbo           (212 records)
Total: 848 distinct real clinical 12-lead ECGs (10,176 real leads, 500 Hz).
Explicitly excludes PTB-XL (`development20`, `known3`, `final40`, `final200`),
Emory (`Dev-40`, `Fresh-20`, `reserved96`), and Georgia so every benchmark cohort
is 100% unseen out-of-distribution validation!
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy.io

ALL_12_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
DEFAULT_OPEN_DATA_ROOT = Path(__file__).resolve().parents[2] / ".external/challenge-2021-v1.0.3/training"
OPEN_SUBDIRS = ("chapman_shaoxing", "cpsc_2018", "cpsc_2018_extra", "ningbo")


def _parse_wfdb_hea_gains(hea_path: Path) -> tuple[float, list[str], np.ndarray, np.ndarray]:
    lines = [ln.strip() for ln in hea_path.read_text(errors="ignore").splitlines() if ln.strip() and not ln.startswith("#")]
    parts0 = lines[0].split()
    n_sig = int(parts0[1])
    fs = float(parts0[2].split("/")[0])
    names: list[str] = []
    gains = np.full(n_sig, 1000.0, dtype=np.float64)
    baselines = np.zeros(n_sig, dtype=np.float64)
    for i in range(n_sig):
        p = lines[1 + i].split()
        # gain token looks like '1000/mV' or '1000(0)/mV'
        g_tok = p[2].split("/")[0]
        if "(" in g_tok:
            g_str, b_str = g_tok.split("(", 1)
            gains[i] = float(g_str) if g_str else 1000.0
            baselines[i] = float(b_str.rstrip(")"))
        else:
            gains[i] = float(g_tok) if g_tok else 1000.0
        if len(p) >= 5:
            try:
                baselines[i] = float(p[4])
            except ValueError:
                pass
        names.append(p[-1])
    return fs, names, gains, baselines


def load_open_ecg_bank(
    root: Path = DEFAULT_OPEN_DATA_ROOT,
    *,
    max_records: int | None = None,
) -> list[dict[str, np.ndarray]]:
    """Load all 848 open CC-BY-4.0 12-lead clinical ECGs in microvolts at 500 Hz."""
    bank: list[dict[str, np.ndarray]] = []
    for sub in OPEN_SUBDIRS:
        sub_dir = root / sub
        if not sub_dir.exists():
            continue
        for hea in sorted(sub_dir.rglob("*.hea")):
            mat_path = hea.with_suffix(".mat")
            if not mat_path.exists():
                continue
            try:
                fs, names, gains, baselines = _parse_wfdb_hea_gains(hea)
                raw = scipy.io.loadmat(str(mat_path))["val"].astype(np.float64)
                if raw.shape[0] != len(names) or raw.shape[1] < 2500:
                    continue
                rec: dict[str, np.ndarray] = {}
                for idx, nm in enumerate(names):
                    # Normalize standard lead names
                    nm_clean = nm.strip()
                    if nm_clean.upper() in ("AVR", "AVL", "AVF"):
                        nm_clean = "a" + nm_clean.upper()[1:]
                    if nm_clean not in ALL_12_LEADS:
                        continue
                    sig_uv = (raw[idx] - baselines[idx]) * (1000.0 / max(gains[idx], 1.0))
                    # Resample to 500 Hz if needed
                    if abs(fs - 500.0) > 1.0:
                        n_dst = int(round(len(sig_uv) * 500.0 / fs))
                        sig_uv = np.interp(
                            np.linspace(0, len(sig_uv) - 1, n_dst),
                            np.arange(len(sig_uv)),
                            sig_uv,
                        )
                    # Trim or tile to 5000 samples (10 s) and baseline-center
                    if len(sig_uv) >= 5000:
                        sig_uv = sig_uv[:5000]
                    else:
                        reps = int(math.ceil(5000 / len(sig_uv)))
                        sig_uv = np.tile(sig_uv, reps)[:5000]
                    if not np.isfinite(sig_uv).all():
                        continue
                    # Remove extreme corrupted ADC rails (> 8 mV)
                    if np.max(np.abs(sig_uv - np.median(sig_uv))) > 8000.0:
                        continue
                    rec[nm_clean] = (sig_uv - np.median(sig_uv)).astype(np.float32)
                if len(rec) == 12:
                    bank.append(rec)
                    if max_records is not None and len(bank) >= max_records:
                        return bank
            except Exception:
                continue
    return bank


# ---------------------------------------------------------------------------
# Realistic ECG paper & camera degradation utilities
# ---------------------------------------------------------------------------

def _pick_paper_palette(rng: np.random.Generator) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    """Return (bg_rgb, minor_grid_rgb, major_grid_rgb) across realistic clinical ECG papers."""
    mode = rng.choice(["pink", "red_orange", "mono_gray", "blue_teal", "green", "faint_pink", "gridless"], p=[0.36, 0.16, 0.18, 0.08, 0.06, 0.11, 0.05])
    bg_luma = int(rng.integers(236, 256))
    bg = (
        int(np.clip(bg_luma + rng.integers(-6, 5), 220, 255)),
        int(np.clip(bg_luma + rng.integers(-8, 4), 218, 255)),
        int(np.clip(bg_luma + rng.integers(-10, 4), 215, 255)),
    )
    if mode == "pink":
        maj = (int(rng.integers(205, 245)), int(rng.integers(105, 165)), int(rng.integers(110, 170)))
        mn = (int(rng.integers(230, 252)), int(rng.integers(180, 222)), int(rng.integers(185, 225)))
    elif mode == "red_orange":
        maj = (int(rng.integers(210, 248)), int(rng.integers(115, 165)), int(rng.integers(85, 135)))
        mn = (int(rng.integers(235, 252)), int(rng.integers(185, 220)), int(rng.integers(160, 205)))
    elif mode == "mono_gray":
        g_maj = int(rng.integers(110, 185))
        g_min = int(rng.integers(max(g_maj + 15, 165), 228))
        maj = (g_maj, g_maj, g_maj)
        mn = (g_min, g_min, g_min)
    elif mode == "blue_teal":
        maj = (int(rng.integers(100, 160)), int(rng.integers(155, 205)), int(rng.integers(190, 235)))
        mn = (int(rng.integers(180, 222)), int(rng.integers(208, 238)), int(rng.integers(224, 248)))
    elif mode == "green":
        maj = (int(rng.integers(115, 170)), int(rng.integers(180, 225)), int(rng.integers(125, 180)))
        mn = (int(rng.integers(190, 228)), int(rng.integers(220, 244)), int(rng.integers(195, 232)))
    elif mode == "faint_pink":
        maj = (int(rng.integers(225, 248)), int(rng.integers(175, 215)), int(rng.integers(178, 218)))
        mn = (int(rng.integers(240, 253)), int(rng.integers(215, 240)), int(rng.integers(218, 242)))
    else:  # gridless (still has faint paper texture)
        maj = bg
        mn = bg
    return bg, mn, maj


def _apply_camera_photometrics(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply realistic scanner/phone lighting gradients, blur, noise, color cast, JPEG compression."""
    out = img.astype(np.float32)
    h, w = out.shape[:2]

    # 1. Directional illumination / shadow gradient (stronger on phone photos)
    if rng.random() < 0.85:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        xx = (xx / max(w - 1, 1)) * 2.0 - 1.0
        yy = (yy / max(h - 1, 1)) * 2.0 - 1.0
        ang = rng.uniform(0, 2 * math.pi)
        grad = xx * math.cos(ang) + yy * math.sin(ang)
        strength = rng.uniform(0.03, 0.24)
        illum = 1.0 - strength * (0.5 + 0.5 * grad)
        # Optional radial vignette / local soft shadow blob
        if rng.random() < 0.5:
            cx, cy = rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)
            rad = rng.uniform(0.5, 1.5)
            blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad * rad))
            illum *= 1.0 - rng.uniform(0.05, 0.28) * blob
        out *= illum[:, :, None]

    # 2. Slight global white-balance tint & contrast/brightness
    tint = rng.uniform(0.90, 1.08, size=(1, 1, 3)).astype(np.float32)
    out = out * tint * rng.uniform(0.84, 1.06) + rng.uniform(-10.0, 10.0)

    # 3. Optional subtle screen moire / halftone pattern (15%)
    if rng.random() < 0.15:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        freq = rng.uniform(0.18, 0.55)
        ang = rng.uniform(0, math.pi)
        wave = np.sin(freq * (xx * math.cos(ang) + yy * math.sin(ang)) + rng.uniform(0, 6.28))
        out += wave[:, :, None] * rng.uniform(3.0, 11.0)

    out = np.clip(out, 0, 255).astype(np.uint8)

    # 4. Defocus / scan blur (55%)
    if rng.random() < 0.55:
        sigma = rng.uniform(0.35, 1.35)
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # 5. Sensor noise (70%)
    if rng.random() < 0.70:
        noise_std = rng.uniform(1.5, 9.5)
        out = np.clip(out.astype(np.float32) + rng.normal(0, noise_std, size=out.shape), 0, 255).astype(np.uint8)

    # 6. JPEG compression artifacts (65%)
    if rng.random() < 0.65:
        q = int(rng.integers(38, 95))
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                out = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return out


# ---------------------------------------------------------------------------
# Generator 1: Single-lead 512x1024 crop for WaveParseLeadDecoderNet (Model B)
# ---------------------------------------------------------------------------

LEAD_CROP_H = 512
LEAD_CROP_W = 1024
LEAD_ACTIVE_W = 1000
LEAD_PAD_X = 12  # (1024 - 1000) // 2
LEAD_SPAN_Y_MV = 8.0  # ±4.0 mV -> 64.0 px/mV = 6.4 px/mm vertically
LEAD_PX_PER_MM_Y = LEAD_CROP_H / 80.0  # 6.4 px/mm (80 mm = ±4.0 mV)
LEAD_PX_PER_MM_X = LEAD_ACTIVE_W / 62.5  # 16.0 px/mm (62.5 mm = 2.5 s at 25 mm/s)


def make_baseline_prior_channel(h: int = LEAD_CROP_H, w: int = LEAD_CROP_W, center_y: float | None = None) -> np.ndarray:
    """Smooth Gaussian prior centered on target lead baseline (range [0, 1])."""
    cy = (h / 2.0) if center_y is None else center_y
    ys = (np.arange(h, dtype=np.float32) + 0.5 - cy) / (0.14 * h)
    col = np.exp(-0.5 * ys * ys).astype(np.float32)
    return np.broadcast_to(col[:, None], (h, w)).copy()


def _random_trace_color(rng: np.random.Generator) -> tuple[int, int, int]:
    """Sample realistic ECG ink colors: pure black, thermal dark gray, slate blue-gray, and faint print."""
    mode = rng.choice(["black", "slate_gray", "faint_gray"], p=[0.40, 0.42, 0.18])
    if mode == "black":
        v = int(rng.integers(5, 65))
        return (v, v, v)
    if mode == "slate_gray":
        # Matches real hospital thermal/laser scans (e.g. GE MAC / Emory slate blue-gray ink)
        r = int(rng.integers(60, 125))
        g = int(np.clip(r + rng.integers(-4, 12), 55, 135))
        b = int(np.clip(r + rng.integers(4, 24), 65, 155))
        return (r, g, b)
    v = int(rng.integers(100, 158))
    return (
        int(np.clip(v + rng.integers(-8, 8), 80, 170)),
        int(np.clip(v + rng.integers(-8, 8), 80, 170)),
        int(np.clip(v + rng.integers(-8, 14), 80, 178)),
    )


def generate_lead_crop_sample(
    bank: list[dict[str, np.ndarray]],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Generate one training sample (4x512x1024 input, y_px target, visibility target) for Model B."""
    H, W = LEAD_CROP_H, LEAD_CROP_W
    px_x, px_y = LEAD_PX_PER_MM_X, LEAD_PX_PER_MM_Y

    rec = bank[int(rng.integers(0, len(bank)))]
    lead_name = ALL_12_LEADS[int(rng.integers(0, 12))]
    full_sig = rec[lead_name]

    # Pick random 2.5 s (1250 samples) slice, padded slightly to cover full 1024 px width (64.0 mm = 1280 samples)
    max_start = max(1, len(full_sig) - 1280)
    s0 = int(rng.integers(0, max_start))
    sig_1280 = full_sig[s0 : s0 + 1280].astype(np.float64).copy()
    if len(sig_1280) < 1280:
        sig_1280 = np.pad(sig_1280, (0, 1280 - len(sig_1280)), mode="edge")
    sig_1280 -= float(np.median(sig_1280))

    # Randomly vary amplitude gain (including 15% extreme tall-lead LVH/LBBB up to ±3.7 mV!) & polarity
    if rng.random() < 0.15:
        pk = max(float(np.max(np.abs(sig_1280))), 100.0)
        sig_1280 *= rng.uniform(1800.0, 3600.0) / pk
    else:
        sig_1280 *= rng.uniform(0.45, 1.75)
    if rng.random() < 0.15:
        sig_1280 = -sig_1280  # teaches Cabrera -aVR & arbitrary polarity

    # Optional slow realistic baseline wander + 15% high-frequency 50/60 Hz EMG muscle tremor
    if rng.random() < 0.40:
        t = np.linspace(0, 1, 1280)
        sig_1280 += rng.uniform(20.0, 140.0) * np.sin(2 * math.pi * rng.uniform(0.25, 1.2) * t + rng.uniform(0, 6.28))
    if rng.random() < 0.15:
        t_s = np.arange(1280, dtype=np.float64) / 500.0
        f_emg = rng.uniform(48.0, 64.0)
        sig_1280 += rng.uniform(15.0, 65.0) * np.sin(2 * math.pi * f_emg * t_s + rng.uniform(0, 6.28))

    # Clamp extreme amplitudes gently inside crop with 1.5 mm margin
    sig_1280 = np.clip(sig_1280, -3850.0, 3850.0)

    # Target baseline y with realistic baseline-estimator jitter (±3.2 mm = ±20 px)
    base_jitter_px = float(rng.uniform(-3.2, 3.2) * px_y)
    y_base = H / 2.0 + base_jitter_px

    # Resample to W=1024 columns (exact pixel-column alignment!)
    x_cols = np.arange(W, dtype=np.float64)
    sig_w = np.interp(np.linspace(0, 1279, W), np.arange(1280), sig_1280)
    y_target = y_base - (sig_w / 1000.0) * 10.0 * px_y  # 10 mm/mV * 6.4 px/mm = 64 px/mV

    # Render realistic ECG paper background & grid at arbitrary subpixel phase
    bg, mn_col, maj_col = _pick_paper_palette(rng)
    img = np.empty((H, W, 3), dtype=np.uint8)
    img[:, :] = bg

    if maj_col != bg:
        phase_x = rng.uniform(0.0, 5.0 * px_x)
        phase_y = rng.uniform(0.0, 5.0 * px_y)
        minor_thick = 1
        major_thick = int(rng.choice([1, 2], p=[0.45, 0.55]))
        # Minor 1 mm lines
        for k in range(-5, int(W / px_x) + 6):
            xk = int(round(phase_x + k * px_x))
            if 0 <= xk < W:
                cv2.line(img, (xk, 0), (xk, H - 1), mn_col, minor_thick)
        for k in range(-5, int(H / px_y) + 6):
            yk = int(round(phase_y + k * px_y))
            if 0 <= yk < H:
                cv2.line(img, (0, yk), (W - 1, yk), mn_col, minor_thick)
        # Major 5 mm lines
        for k in range(-2, int(W / (5 * px_x)) + 3):
            xk = int(round(phase_x + k * 5 * px_x))
            if 0 <= xk < W:
                cv2.line(img, (xk, 0), (xk, H - 1), maj_col, major_thick)
        for k in range(-2, int(H / (5 * px_y)) + 3):
            yk = int(round(phase_y + k * 5 * px_y))
            if 0 <= yk < H:
                cv2.line(img, (0, yk), (W - 1, yk), maj_col, major_thick)

    trace_col = _random_trace_color(rng)
    thick = int(rng.choice([1, 2, 3], p=[0.20, 0.55, 0.25]))

    # Draw realistic adjacent upper and lower ECG rows (critical so decoder never hops rows!)
    row_step_mm = float(rng.choice([20.0, 22.0, 28.0, 30.0, 35.0, 38.0]))
    for sign in (-1, +1):
        if rng.random() < 0.85:
            n_lead = ALL_12_LEADS[int(rng.integers(0, 12))]
            n_sig = rec[n_lead]
            ns0 = int(rng.integers(0, max(1, len(n_sig) - 1280)))
            n_seg = n_sig[ns0 : ns0 + 1280].astype(np.float64)
            if len(n_seg) < 1280:
                n_seg = np.pad(n_seg, (0, 1280 - len(n_seg)), mode="edge")
            n_seg = (n_seg - np.median(n_seg)) * rng.uniform(0.6, 1.8)
            n_w = np.interp(np.linspace(0, 1279, W), np.arange(1280), n_seg)
            ny_base = y_base + sign * row_step_mm * px_y + rng.uniform(-2.0, 2.0) * px_y
            ny = ny_base - (n_w / 1000.0) * 10.0 * px_y
            pts_n = np.stack([x_cols, ny], axis=-1).round().astype(np.int32)
            cv2.polylines(img, [pts_n], False, trace_col, thick, cv2.LINE_AA)

    # Draw realistic printed lead label (including close-to-baseline bold hospital labels like V4/V5/aVF!)
    if rng.random() < 0.85:
        lbl = "-aVR" if (lead_name == "aVR" and rng.random() < 0.35) else lead_name
        lx = int(rng.integers(LEAD_PAD_X, LEAD_PAD_X + int(4.5 * px_x)))
        # Real Emory/GE ECGs print bold labels just 2.5..8.5 mm below/above baseline!
        ly_off = rng.choice([-1, 1], p=[0.25, 0.75]) * rng.uniform(2.5, 9.5) * px_y
        ly = int(np.clip(y_base + ly_off, 24, H - 10))
        fscale = rng.uniform(0.65, 1.35)
        fthick = int(rng.choice([1, 2, 3], p=[0.20, 0.50, 0.30]))
        cv2.putText(img, lbl, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, fscale, trace_col, fthick, cv2.LINE_AA)

    # Thick hospital vertical lead-separator ticks & calibration-pulse edges on left/right borders!
    if rng.random() < 0.65:
        sep_thick = int(rng.choice([2, 3, 4, 5]))
        if rng.random() < 0.75:
            sx_l = int(rng.integers(4, LEAD_PAD_X + 6))
            cv2.line(img, (sx_l, int(y_base - rng.uniform(4.0, 7.5) * px_y)), (sx_l, int(y_base + rng.uniform(4.0, 7.5) * px_y)), trace_col, sep_thick)
        if rng.random() < 0.75:
            sx_r = int(rng.integers(W - LEAD_PAD_X - 6, W - 4))
            cv2.line(img, (sx_r, int(y_base - rng.uniform(4.0, 7.5) * px_y)), (sx_r, int(y_base + rng.uniform(4.0, 7.5) * px_y)), trace_col, sep_thick)
    elif rng.random() < 0.35:
        # Calibration pulse falling edge on far left (like col-0 leads I, II, III)
        cx_cal = int(rng.integers(8, LEAD_PAD_X + 5))
        cv2.line(img, (0, int(y_base - 10.0 * px_y)), (cx_cal, int(y_base - 10.0 * px_y)), trace_col, max(2, thick))
        cv2.line(img, (cx_cal, int(y_base - 10.0 * px_y)), (cx_cal, int(y_base)), trace_col, max(2, thick))

    # Optional bottom-of-page printed setting text (e.g. '10mm/mV  25mm/s') like rhythm II strip
    if rng.random() < 0.25:
        txt_y = int(np.clip(y_base + rng.uniform(12.0, 24.0) * px_y, 30, H - 12))
        txt_x = int(rng.integers(120, 620))
        cv2.putText(img, rng.choice(["10mm/mV", "25mm/s", "150Hz", "25 mm/s 10 mm/mV"]), (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(0.7, 1.15), trace_col, 2, cv2.LINE_AA)

    # Draw the TARGET waveform!
    pts = np.stack([x_cols, y_target], axis=-1).round().astype(np.int32)
    cv2.polylines(img, [pts], False, trace_col, thick, cv2.LINE_AA)

    # Optional monochrome photocopy dust / pepper speckles (like scanned thermal paper SEM-16)
    if rng.random() < 0.25:
        n_dots = int(rng.integers(15, 95))
        dx_d = rng.integers(0, W, size=n_dots)
        dy_d = rng.integers(0, H, size=n_dots)
        for xi, yi in zip(dx_d, dy_d):
            cv2.circle(img, (int(xi), int(yi)), int(rng.choice([1, 1, 2])), trace_col, -1)

    # Optional smooth residual TPS warp jitter (±1.4 px) applied identically to image & target!
    if rng.random() < 0.65:
        dy_shift = rng.uniform(-1.6, 1.6) * np.sin(2 * math.pi * rng.uniform(0.3, 1.5) * (x_cols / W) + rng.uniform(0, 6.28))
        map_x = np.broadcast_to(x_cols[None, :].astype(np.float32), (H, W))
        map_y = (np.arange(H, dtype=np.float32)[:, None] - dy_shift[None, :].astype(np.float32))
        img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        y_target = np.clip(y_target + dy_shift, 1.0, H - 2.0)

    # Optional realistic occlusion / stain / redaction -> ground-truth visibility = 0!
    visibility = np.ones(W, dtype=np.float32)
    if rng.random() < 0.22:
        occ_w = int(rng.integers(24, 160))
        occ_x0 = int(rng.integers(LEAD_PAD_X, W - LEAD_PAD_X - occ_w))
        occ_x1 = occ_x0 + occ_w
        occ_y0 = int(np.clip(np.min(y_target[occ_x0:occ_x1]) - rng.uniform(10, 45), 0, H - 1))
        occ_y1 = int(np.clip(np.max(y_target[occ_x0:occ_x1]) + rng.uniform(10, 45), 0, H - 1))
        occ_kind = rng.choice(["whiteout", "dark_block", "coffee_stain"])
        if occ_kind == "whiteout":
            img[occ_y0:occ_y1, occ_x0:occ_x1] = bg
        elif occ_kind == "dark_block":
            c = int(rng.integers(5, 40))
            img[occ_y0:occ_y1, occ_x0:occ_x1] = (c, c, c)
        else:
            img[occ_y0:occ_y1, occ_x0:occ_x1] = (int(rng.integers(90, 145)), int(rng.integers(60, 105)), int(rng.integers(35, 70)))
        visibility[occ_x0:occ_x1] = 0.0

    # Apply realistic camera/scanner photometrics
    img = _apply_camera_photometrics(img, rng)

    # Build 4-channel contiguous input (RGB [0,1] + baseline Gaussian prior centered at H/2)
    prior = make_baseline_prior_channel(H, W, center_y=H / 2.0)
    inp = np.ascontiguousarray(
        np.concatenate([img.astype(np.float32).transpose(2, 0, 1) / 255.0, prior[None, :, :]], axis=0),
        dtype=np.float32,
    )

    return {
        "input": inp,  # (4, 512, 1024) float32
        "y_target": y_target.astype(np.float32),  # (1024,) float32 in [0, 511]
        "visibility": visibility,  # (1024,) float32 in {0, 1}
    }


# ---------------------------------------------------------------------------
# Generator 2: Curved/rotated whole-page ECG for WaveParseGridRectifierNet (Model A)
# ---------------------------------------------------------------------------

RECT_IMG_SIZE = 1024
RECT_MAP_SIZE = 512  # stride 2 -> 9+ px per 5-mm box, zero 1-mm sub-grid aliasing!


def _render_random_layout_sheet(
    rec: dict[str, np.ndarray],
    rng: np.random.Generator,
    px_per_mm: float = 6.0,
) -> tuple[np.ndarray, float, float]:
    """Render a realistic hospital multi-layout ECG sheet (3x4+1r, 3x4+3r, 6x2, Cabrera, 12x1).

    Includes diverse clinical page styles (GE/Philips/Emory-like border + small top-left header +
    optional QR code + outer white printer margins, classic top banner, and separator ticks) so
    the rectifier generalizes cleanly across real hospital flatbeds and phone photos.
    Returns (sheet_rgb, width_mm, height_mm) with major 5-mm gridlines at exact integer multiples of 5 mm.
    """
    layout = str(rng.choice(["3x4_r1", "3x4_r3", "6x2", "cabrera_6x2", "12x1"], p=[0.46, 0.14, 0.16, 0.12, 0.12]))
    if layout == "3x4_r1":
        grid = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
        rhythms = ["II"]
        col_mm, row_mm = 62.5, float(rng.uniform(32.0, 39.0))
        width_mm, height_mm = 280.0, 215.0
    elif layout == "3x4_r3":
        grid = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
        rhythms = ["V1", "II", "V5"]
        col_mm, row_mm = 62.5, 28.0
        width_mm, height_mm = 280.0, 220.0
    elif layout == "6x2":
        grid = [["I", "V1"], ["II", "V2"], ["III", "V3"], ["aVR", "V4"], ["aVL", "V5"], ["aVF", "V6"]]
        rhythms = []
        col_mm, row_mm = 125.0, 28.0
        width_mm, height_mm = 280.0, 215.0
    elif layout == "cabrera_6x2":
        grid = [["aVL", "V1"], ["I", "V2"], ["-aVR", "V3"], ["II", "V4"], ["aVF", "V5"], ["III", "V6"]]
        rhythms = []
        col_mm, row_mm = 125.0, 28.0
        width_mm, height_mm = 280.0, 215.0
    else:  # 12x1
        grid = [[l] for l in ALL_12_LEADS]
        rhythms = []
        col_mm, row_mm = 250.0, 18.0
        width_mm, height_mm = 280.0, 255.0

    W = int(round(width_mm * px_per_mm))
    H = int(round(height_mm * px_per_mm))
    bg, mn_col, maj_col = _pick_paper_palette(rng)
    if maj_col == bg:
        maj_col = (225, 130, 134)
        mn_col = (242, 198, 200)
    img = np.empty((H, W, 3), dtype=np.uint8)
    img[:, :] = bg

    # Optional realistic outer white printer margin (5-12 mm) + thin black grid frame (common on GE/Emory/Philips prints!)
    has_outer_margin = rng.random() < 0.50
    mx_px = int(round(rng.uniform(5.0, 11.0) * px_per_mm)) if has_outer_margin else 0
    my_px = int(round(rng.uniform(4.0, 9.0) * px_per_mm)) if has_outer_margin else 0
    gx0, gy0, gx1, gy1 = mx_px, my_px, W - mx_px - 1, H - my_px - 1

    # Draw crisp 1-mm minor (1 px) & stronger 5-mm major (1-2 px) gridlines at EXACT integer mm coordinates
    maj_thick = int(rng.choice([1, 2], p=[0.40, 0.60]))
    for mm in range(0, int(width_mm) + 1):
        x = int(round(mm * px_per_mm))
        if gx0 <= x <= gx1:
            if mm % 5 == 0:
                cv2.line(img, (x, gy0), (x, gy1), maj_col, maj_thick)
            else:
                cv2.line(img, (x, gy0), (x, gy1), mn_col, 1)
    for mm in range(0, int(height_mm) + 1):
        y = int(round(mm * px_per_mm))
        if gy0 <= y <= gy1:
            if mm % 5 == 0:
                cv2.line(img, (gx0, y), (gx1, y), maj_col, maj_thick)
            else:
                cv2.line(img, (gx0, y), (gx1, y), mn_col, 1)

    trace_col = _random_trace_color(rng)
    thick = int(rng.choice([1, 2], p=[0.45, 0.55]))

    if has_outer_margin and rng.random() < 0.75:
        cv2.rectangle(img, (gx0, gy0), (gx1, gy1), trace_col, 1)

    # Diverse realistic clinical headers (small top-left hospital block + QR code OR classic banner!)
    hdr_style = str(rng.choice(["hospital_block", "classic_banner", "minimal"], p=[0.50, 0.35, 0.15]))
    if hdr_style == "hospital_block":
        tx0 = gx0 + int(3.0 * px_per_mm)
        ty0 = gy0 + int(5.5 * px_per_mm)
        cv2.putText(img, "ID: ECG_84F2A910C   Name:              Date:", (tx0, ty0), cv2.FONT_HERSHEY_SIMPLEX, 0.58, trace_col, 1, cv2.LINE_AA)
        cv2.putText(img, "Age: 64 yrs", (tx0, ty0 + int(4.8 * px_per_mm)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, trace_col, 1, cv2.LINE_AA)
        cv2.putText(img, "Sex: Male", (tx0, ty0 + int(9.6 * px_per_mm)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, trace_col, 1, cv2.LINE_AA)
        # Optional top-right QR/barcode square (classic hospital print orientation cue!)
        if rng.random() < 0.65:
            qsz = int(round(16.0 * px_per_mm))
            qx1 = W - int(round(4.0 * px_per_mm))
            qy0 = int(round(4.0 * px_per_mm))
            qx0 = max(0, qx1 - qsz)
            img[qy0 : qy0 + qsz, qx0:qx1] = bg
            qr_grid = (rng.random((11, 11)) < 0.52).astype(np.uint8)
            qr_up = cv2.resize(qr_grid, (qsz - 8, qsz - 8), interpolation=cv2.INTER_NEAREST)
            img[qy0 + 4 : qy0 + qsz - 4, qx0 + 4 : qx1 - 4][qr_up > 0] = trace_col
    elif hdr_style == "classic_banner":
        hdr_y = gy0 + int(round(9.0 * px_per_mm))
        cv2.putText(img, "12-LEAD ECG   25 mm/s   10 mm/mV   150Hz", (gx0 + int(8 * px_per_mm), hdr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.90, trace_col, 2, cv2.LINE_AA)
        cv2.putText(img, "HR 72 bpm   PR 160ms   QRS 92ms   QT/QTc 398/422ms", (gx0 + int(8 * px_per_mm), hdr_y + int(7.5 * px_per_mm)), cv2.FONT_HERSHEY_SIMPLEX, 0.68, trace_col, 2, cv2.LINE_AA)
        cv2.line(img, (gx0 + int(6 * px_per_mm), gy0 + int(21 * px_per_mm)), (gx1 - int(6 * px_per_mm), gy0 + int(21 * px_per_mm)), trace_col, 2)
    else:
        cv2.putText(img, "25mm/s    10mm/mV", (gx0 + int(18 * px_per_mm), gy1 - int(4 * px_per_mm)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, trace_col, 1, cv2.LINE_AA)

    # Bottom footer speed/gain markings
    cv2.putText(img, "25mm/s               10mm/mV", (gx0 + int(40 * px_per_mm), gy1 - int(4.5 * px_per_mm)), cv2.FONT_HERSHEY_SIMPLEX, 0.60, trace_col, 1, cv2.LINE_AA)

    top_mm = float(rng.uniform(34.0, 58.0)) if layout == "3x4_r1" else 30.0
    left_mm = float(rng.uniform(14.0, 20.0))

    def _draw_panel(lbl: str, x0_mm: float, y0_mm: float, dur_mm: float, draw_sep: bool = False) -> None:
        neg = lbl.startswith("-")
        base_l = lbl[1:] if neg else lbl
        sig = rec[base_l].astype(np.float64)
        n_samp = int(round((dur_mm / 25.0) * 500.0))
        seg = sig[:n_samp] if len(sig) >= n_samp else np.tile(sig, 2)[:n_samp]
        if neg:
            seg = -seg
        xs = (x0_mm + np.linspace(0, dur_mm, n_samp)) * px_per_mm
        ys = (y0_mm - (seg / 1000.0) * 10.0) * px_per_mm
        pts = np.stack([xs, ys], axis=-1).round().astype(np.int32)
        cv2.polylines(img, [pts], False, trace_col, thick, cv2.LINE_AA)
        if draw_sep:
            sx_sep = int(round(x0_mm * px_per_mm))
            y_sep = int(round(y0_mm * px_per_mm))
            cv2.line(img, (sx_sep, y_sep - int(4 * px_per_mm)), (sx_sep, y_sep + int(4 * px_per_mm)), trace_col, max(1, thick))
        cv2.putText(img, lbl, (int((x0_mm + 1.2) * px_per_mm), int((y0_mm + 5.8) * px_per_mm)), cv2.FONT_HERSHEY_SIMPLEX, 0.56, trace_col, 1, cv2.LINE_AA)

    for r_i, row in enumerate(grid):
        ym = top_mm + (r_i + 0.5) * row_mm
        if ym > height_mm - 18.0:
            break
        px0 = int(round((left_mm - 6.0) * px_per_mm))
        py0 = int(round(ym * px_per_mm))
        pw = int(round(5.0 * px_per_mm))
        ph = int(round(10.0 * px_per_mm))
        cal_pts = np.array([[px0 - 4, py0], [px0, py0], [px0, py0 - ph], [px0 + pw, py0 - ph], [px0 + pw, py0], [px0 + pw + 4, py0]], dtype=np.int32)
        cv2.polylines(img, [cal_pts], False, trace_col, thick)
        for c_i, lbl in enumerate(row):
            _draw_panel(lbl, left_mm + c_i * col_mm, ym, col_mm, draw_sep=(c_i > 0 and len(row) == 4))

    for rr_i, lbl in enumerate(rhythms):
        ym = top_mm + (len(grid) + rr_i + 0.5) * row_mm
        if ym < height_mm - 12.0:
            _draw_panel(lbl, left_mm, ym, 250.0, draw_sep=False)

    return img, width_mm, height_mm


def generate_rectifier_sample(
    bank: list[dict[str, np.ndarray]],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Generate one curved/perspective/rotated whole-page training sample for Model A."""
    S = RECT_IMG_SIZE
    M = RECT_MAP_SIZE
    rec = bank[int(rng.integers(0, len(bank)))]
    px_mm = 6.0
    sheet, w_mm, h_mm = _render_random_layout_sheet(rec, rng, px_per_mm=px_mm)
    sh, sw = sheet.shape[:2]

    # Build smooth 2D forward deformation: canonical sheet (u,v) in [0,1]x[0,1] -> output image (x,y) in [0, S)
    # Mix flatbed scans (tight margins, minimal curl), tight phone crops, and wide desk phone photos
    capture_mode = str(rng.choice(["flatbed", "tight_phone", "desk_phone"], p=[0.30, 0.35, 0.35]))
    if capture_mode == "flatbed":
        margin = rng.uniform(0.00, 0.02)
    elif capture_mode == "tight_phone":
        margin = rng.uniform(0.00, 0.05)
    else:
        margin = rng.uniform(0.05, 0.14)
    scale_x = (1.0 - 2 * margin) * S
    scale_y = scale_x * (h_mm / w_mm)
    if scale_y > (1.0 - 2 * margin) * S:
        scale_y = (1.0 - 2 * margin) * S
        scale_x = scale_y * (w_mm / h_mm)
    cx = S * 0.5 + rng.uniform(-0.03, 0.03) * S
    cy = S * 0.5 + rng.uniform(-0.03, 0.03) * S

    # Base 4 corners + perspective jitter
    src_pts = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]], dtype=np.float32)
    dst_pts = np.array([
        [cx - scale_x / 2, cy - scale_y / 2],
        [cx + scale_x / 2, cy - scale_y / 2],
        [cx + scale_x / 2, cy + scale_y / 2],
        [cx - scale_x / 2, cy + scale_y / 2],
    ], dtype=np.float32)
    # Small continuous rotation + perspective corner tilt
    max_rot = 2.5 if capture_mode == "flatbed" else 14.0
    ang = math.radians(rng.uniform(-max_rot, max_rot))
    ca, sa = math.cos(ang), math.sin(ang)
    rot_m = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
    dst_pts = ((dst_pts - np.array([[cx, cy]], dtype=np.float32)) @ rot_m.T) + np.array([[cx, cy]], dtype=np.float32)
    if capture_mode != "flatbed" and rng.random() < 0.85:
        dst_pts += rng.uniform(-0.065 * S, 0.065 * S, size=(4, 2)).astype(np.float32)

    H_s2i = cv2.getPerspectiveTransform(src_pts, dst_pts)
    H_i2s = np.linalg.inv(H_s2i)

    # Smooth low-frequency 2D page curl / wrinkle + optional smartphone barrel lens distortion
    yy_s, xx_s = np.mgrid[0:S, 0:S].astype(np.float32)
    if capture_mode != "flatbed" and rng.random() < 0.82:
        ax = rng.uniform(-20.0, 20.0)
        ay = rng.uniform(-20.0, 20.0)
        fx1, fy1 = rng.uniform(0.5, 1.8), rng.uniform(0.5, 1.8)
        px1, py1 = rng.uniform(0, 6.28), rng.uniform(0, 6.28)
        curl_x = ax * np.sin(2 * math.pi * fy1 * (yy_s / S) + py1) * np.cos(math.pi * (xx_s / S))
        curl_y = ay * np.sin(2 * math.pi * fx1 * (xx_s / S) + px1) * np.cos(math.pi * (yy_s / S))
        if rng.random() < 0.45:
            # Subtle smartphone radial lens barrel/pincushion distortion (matches real phone captures!)
            nx_r = (xx_s - S * 0.5) / (S * 0.5)
            ny_r = (yy_s - S * 0.5) / (S * 0.5)
            r2 = nx_r * nx_r + ny_r * ny_r
            k_rad = rng.uniform(-0.04, 0.04) * S * 0.5
            curl_x = curl_x + k_rad * nx_r * r2
            curl_y = curl_y + k_rad * ny_r * r2
    else:
        curl_x = np.zeros_like(xx_s)
        curl_y = np.zeros_like(yy_s)

    # Map output pixel (x, y) -> uncurled (x - curl_x, y - curl_y) -> sheet pixel (sx, sy)
    ux = xx_s - curl_x
    uy = yy_s - curl_y
    denom = H_i2s[2, 0] * ux + H_i2s[2, 1] * uy + H_i2s[2, 2]
    map_sx = ((H_i2s[0, 0] * ux + H_i2s[0, 1] * uy + H_i2s[0, 2]) / denom).astype(np.float32)
    map_sy = ((H_i2s[1, 0] * ux + H_i2s[1, 1] * uy + H_i2s[1, 2]) / denom).astype(np.float32)

    # Realistic background behind ECG sheet (wood/desk/dark/light gray/textured)
    bg_tone = rng.integers(30, 230, size=3).astype(np.uint8)
    canvas = np.empty((S, S, 3), dtype=np.uint8)
    canvas[:, :] = bg_tone
    noise_bg = rng.normal(0, rng.uniform(4.0, 18.0), size=(S // 4, S // 4, 3)).astype(np.float32)
    noise_bg = cv2.resize(noise_bg, (S, S), interpolation=cv2.INTER_LINEAR)
    canvas = np.clip(canvas.astype(np.float32) + noise_bg, 0, 255).astype(np.uint8)

    warped = cv2.remap(sheet, map_sx, map_sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    inside_s = (map_sx >= 0) & (map_sx <= sw - 1) & (map_sy >= 0) & (map_sy <= sh - 1)
    canvas[inside_s] = warped[inside_s]
    canvas = _apply_camera_photometrics(canvas, rng)

    # Exact dense continuous 2D 5-mm grid-phase & normalized sheet coordinate targets at stride-2 (M x M)!
    # 100% of foreground paper pixels carry smooth, bounded, non-zero supervision!
    sx_m = map_sx[1::2, 1::2]
    sy_m = map_sy[1::2, 1::2]
    inside_m = ((sx_m >= 0) & (sx_m <= sw - 1) & (sy_m >= 0) & (sy_m <= sh - 1)).astype(np.float32)

    p5_px = 5.0 * px_mm  # 25.0 sheet px per 5 mm
    gx_f = sx_m / p5_px
    gy_f = sy_m / p5_px
    u_norm = np.clip(sx_m / max(float(sw - 1), 1.0), 0.0, 1.0).astype(np.float32) * inside_m
    v_norm = np.clip(sy_m / max(float(sh - 1), 1.0), 0.0, 1.0).astype(np.float32) * inside_m

    # Continuous periodic 5-mm phase (cos, sin) in [-1, 1] -> unwraps analytically to exact (gx, gy)!
    cos_x = (np.cos(2.0 * math.pi * gx_f) * inside_m).astype(np.float32)
    sin_x = (np.sin(2.0 * math.pi * gx_f) * inside_m).astype(np.float32)
    cos_y = (np.cos(2.0 * math.pi * gy_f) * inside_m).astype(np.float32)
    sin_y = (np.sin(2.0 * math.pi * gy_f) * inside_m).astype(np.float32)

    # Also keep soft major-grid intersection channel (ch 7) as an auxiliary supervision signal
    dx_line = np.abs(gx_f - np.round(gx_f))
    dy_line = np.abs(gy_f - np.round(gy_f))
    grid_inter = (np.exp(-0.5 * ((dx_line ** 2 + dy_line ** 2) / (0.22 ** 2))) * inside_m).astype(np.float32)

    # Random 0 / 90 / 180 / 270 degree rotation (65% upright for crisp spatial phase learning + 35% rotated for orientation head!)
    rot_k = int(rng.choice([0, 1, 2, 3], p=[0.65, 0.12, 0.12, 0.11]))
    if rot_k > 0:
        canvas = np.ascontiguousarray(np.rot90(canvas, k=-rot_k))
        inside_m = np.ascontiguousarray(np.rot90(inside_m, k=-rot_k))
        u_norm = np.ascontiguousarray(np.rot90(u_norm, k=-rot_k))
        v_norm = np.ascontiguousarray(np.rot90(v_norm, k=-rot_k))
        cos_x = np.ascontiguousarray(np.rot90(cos_x, k=-rot_k))
        sin_x = np.ascontiguousarray(np.rot90(sin_x, k=-rot_k))
        cos_y = np.ascontiguousarray(np.rot90(cos_y, k=-rot_k))
        sin_y = np.ascontiguousarray(np.rot90(sin_y, k=-rot_k))
        grid_inter = np.ascontiguousarray(np.rot90(grid_inter, k=-rot_k))

    inp = np.ascontiguousarray(canvas.astype(np.float32).transpose(2, 0, 1) / 255.0, dtype=np.float32)
    targets = np.ascontiguousarray(
        np.stack([inside_m, u_norm, v_norm, cos_x, sin_x, cos_y, sin_y, grid_inter], axis=0),
        dtype=np.float32,
    )

    return {
        "input": inp,  # (3, 768, 768) float32
        "targets": targets,  # (8, 384, 384) float32: [mask, u, v, cos_x, sin_x, cos_y, sin_y, inter]
        "orientation_class": np.int64(rot_k),  # 0=0deg, 1=90cw, 2=180, 3=270cw (to undo: rot90(img, k=+rot_k))
    }
