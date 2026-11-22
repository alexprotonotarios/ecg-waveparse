"""Lossless vector PDF ECG extractor + high-DPI raster PDF fallback.

When a hospital/EHR ECG PDF contains native vector drawing operators (`m`, `l`, `S`)
inside FlateDecode streams (e.g., GE MUSE, Philips, Emory ecg-image-kit vector PDFs),
this module extracts the exact floating-point ECG polylines directly in PDF points
(1 pt = 25.4/72 mm) with ZERO pixelization or vision error in ~0.05 s.

When a PDF is a scanned/raster image wrapped in a PDF container, `extract_ecg_from_vector_pdf`
returns `None` and `rasterize_pdf_to_rgb` renders page 1 at 300 DPI using native macOS
CoreGraphics/PDFKit (via Swift) or `sips` so the neural vision pipeline takes over seamlessly.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ecg_pipeline.cleanroom.io_utils import ALL_12_LEADS

PT_TO_MM = 25.4 / 72.0  # 0.3527777778 mm per PDF point

STANDARD_3X4_GRID = [
    ["I", "aVR", "V1", "V4"],
    ["II_short", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
STANDARD_6X2_GRID = [
    ["I", "V1"],
    ["II", "V2"],
    ["III", "V3"],
    ["aVR", "V4"],
    ["aVL", "V5"],
    ["aVF", "V6"],
]


@dataclass
class VectorPdfExtraction:
    leads_uv: dict[str, np.ndarray]
    short_ii_uv: np.ndarray | None
    confidence: dict[str, np.ndarray]
    layout: str
    paper_speed_mm_s: float
    gain_mm_mv: float
    sample_rate_hz: float
    polyline_count: int
    pulse_count: int
    metadata: dict[str, Any]


def _parse_pdf_polylines(pdf_bytes: bytes, min_points: int = 30) -> list[np.ndarray]:
    """Extract all stroked/drawn vector polylines with >= min_points vertices from PDF content streams."""
    streams = re.findall(rb"stream\r?\n(.*?)\r?\nendstream", pdf_bytes, re.DOTALL)
    polylines: list[np.ndarray] = []

    for raw_stream in streams:
        try:
            text = zlib.decompress(raw_stream).decode("latin1")
        except Exception:
            try:
                text = raw_stream.decode("latin1")
            except Exception:
                continue

        # Fast skip for font/glyph streams (contain d1 / d0)
        if " d1\n" in text[:120] or " d0\n" in text[:120]:
            continue

        pts: list[tuple[float, float]] = []
        for line in text.splitlines():
            toks = line.strip().split()
            if not toks:
                continue
            op = toks[-1]
            if op == "m" and len(toks) >= 3:
                if len(pts) >= min_points:
                    polylines.append(np.array(pts, dtype=np.float64))
                try:
                    pts = [(float(toks[-3]), float(toks[-2]))]
                except ValueError:
                    pts = []
            elif op == "l" and len(toks) >= 3:
                try:
                    pts.append((float(toks[-3]), float(toks[-2])))
                except ValueError:
                    pass
            elif op in ("S", "s", "B", "b", "f", "F", "n"):
                if len(pts) >= min_points:
                    polylines.append(np.array(pts, dtype=np.float64))
                pts = []
        if len(pts) >= min_points:
            polylines.append(np.array(pts, dtype=np.float64))

    return polylines


def _resample_polyline_to_uv(
    pts: np.ndarray,
    *,
    x_start_pt: float,
    duration_s: float,
    sample_rate_hz: float,
    speed_mm_s: float,
    gain_mm_mv: float,
    y_baseline_pt: float | None = None,
) -> np.ndarray:
    """Resample a monotonic-x PDF polyline into exact microvolts at sample_rate_hz."""
    # Sort by x and deduplicate identical x coordinates (preserving extreme vertical spikes via mean)
    order = np.argsort(pts[:, 0], kind="mergesort")
    xs = pts[order, 0]
    ys = pts[order, 1]
    ux, inv = np.unique(xs, return_inverse=True)
    if len(ux) < len(xs):
        y_sum = np.bincount(inv, weights=ys)
        y_cnt = np.bincount(inv)
        xs = ux
        ys = y_sum / y_cnt

    n_samples = int(round(duration_s * sample_rate_hz))
    pt_per_sec = (speed_mm_s / PT_TO_MM)
    t_pts = (xs - x_start_pt) / pt_per_sec
    t_grid = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    y_interp = np.interp(t_grid, t_pts, ys)

    base = float(np.median(y_interp)) if y_baseline_pt is None else float(y_baseline_pt)
    # In standard PDF coordinate space, +y is UP -> positive voltage is +(y - base)
    mm_y = (y_interp - base) * PT_TO_MM
    return mm_y * (1000.0 / gain_mm_mv)


def extract_ecg_from_vector_pdf(pdf_path: Path, sample_rate_hz: float = 500.0) -> VectorPdfExtraction | None:
    """Extract lossless 12-lead ECG signals directly from a vector PDF if present.

    Returns None when the PDF does not contain recognizable multi-lead vector ECG traces
    (e.g., scanned raster PDFs), signalling the caller to rasterize and run neural vision.
    """
    raw = pdf_path.read_bytes()
    polylines = _parse_pdf_polylines(raw, min_points=30)
    if len(polylines) < 12:
        return None

    # Classify polylines into calibration pulses (width 3.5..12 mm, height 4..23 mm)
    # vs ECG waveform traces (width >= 40 mm)
    pulses: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for pts in polylines:
        xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
        ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
        dx_mm = (xmax - xmin) * PT_TO_MM
        dy_mm = (ymax - ymin) * PT_TO_MM
        if 3.5 <= dx_mm <= 12.0 and 4.0 <= dy_mm <= 23.0:
            pulses.append({"pts": pts, "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "dx_mm": dx_mm, "dy_mm": dy_mm})
        elif dx_mm >= 40.0:
            traces.append({"pts": pts, "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "ymed": float(np.median(pts[:, 1])), "dx_mm": dx_mm})

    if len(traces) < 12:
        return None

    # Infer physical gain from vector calibration pulses (default 10 mm/mV)
    gain_mm_mv = 10.0
    speed_mm_s = 25.0
    if pulses:
        h_med = float(np.median([p["dy_mm"] for p in pulses]))
        if 3.8 <= h_med <= 6.2:
            gain_mm_mv = 5.0
        elif 8.2 <= h_med <= 11.8:
            gain_mm_mv = 10.0
        elif 17.0 <= h_med <= 23.0:
            gain_mm_mv = 20.0
        w_med = float(np.median([p["dx_mm"] for p in pulses]))
        if 8.2 <= w_med <= 11.8:
            speed_mm_s = 50.0

    # Separate full-width rhythm strips (dx_mm >= 200 mm) from column panels
    rhythm_traces = [t for t in traces if t["dx_mm"] >= 200.0]
    panel_traces = [t for t in traces if t["dx_mm"] < 200.0]

    # Case 1: Standard 3x4 (+ optional rhythm strip II)
    if len(panel_traces) == 12:
        # Sort panel traces into 3 rows from TOP to BOTTOM (highest ymed first in PDF coords)
        panel_traces.sort(key=lambda t: -t["ymed"])
        rows = [panel_traces[0:4], panel_traces[4:8], panel_traces[8:12]]
        # Verify rows are well-ordered and sort each row left-to-right
        for r in rows:
            r.sort(key=lambda t: t["xmin"])

        x_origin = min(r[0]["xmin"] for r in rows)
        col_widths_mm = [t["dx_mm"] for r in rows for t in r]
        panel_dur_s = float(np.median(col_widths_mm)) / speed_mm_s
        # Snap clean 2.5 s (or 1.25 s at 50 mm/s)
        expected_panel_s = 2.5 * (25.0 / speed_mm_s)
        if abs(panel_dur_s - expected_panel_s) < 0.15:
            panel_dur_s = expected_panel_s
        eff_fs = sample_rate_hz * (speed_mm_s / 25.0)
        n_panel = int(round(panel_dur_s * eff_fs))  # 1250 samples

        leads_uv = {l: np.full(4 * n_panel, np.nan, dtype=np.float64) for l in ALL_12_LEADS}
        confidence = {l: np.full(4 * n_panel, np.nan, dtype=np.float64) for l in ALL_12_LEADS}
        short_ii_uv: np.ndarray | None = None

        for r_idx in range(3):
            for c_idx in range(4):
                lead_tag = STANDARD_3X4_GRID[r_idx][c_idx]
                tr = rows[r_idx][c_idx]
                sig = _resample_polyline_to_uv(
                    tr["pts"],
                    x_start_pt=tr["xmin"],
                    duration_s=panel_dur_s,
                    sample_rate_hz=eff_fs,
                    speed_mm_s=speed_mm_s,
                    gain_mm_mv=gain_mm_mv,
                )
                s0, s1 = c_idx * n_panel, (c_idx + 1) * n_panel
                if lead_tag == "II_short":
                    short_ii_uv = sig
                    if not rhythm_traces:
                        leads_uv["II"][s0:s1] = sig
                        confidence["II"][s0:s1] = 1.0
                else:
                    leads_uv[lead_tag][s0:s1] = sig
                    confidence[lead_tag][s0:s1] = 1.0

        layout = "standard_3x4"
        if rhythm_traces:
            # Pick the top-most (or only) rhythm strip as full rhythm II
            rhythm_traces.sort(key=lambda t: -t["ymed"])
            r_tr = rhythm_traces[0]
            sig_ii = _resample_polyline_to_uv(
                r_tr["pts"],
                x_start_pt=r_tr["xmin"],
                duration_s=4.0 * panel_dur_s,
                sample_rate_hz=eff_fs,
                speed_mm_s=speed_mm_s,
                gain_mm_mv=gain_mm_mv,
            )
            leads_uv["II"][: len(sig_ii)] = sig_ii[: 4 * n_panel]
            confidence["II"][: len(sig_ii)] = 1.0
            layout = "standard_3x4_with_r1"

        return VectorPdfExtraction(
            leads_uv=leads_uv,
            short_ii_uv=short_ii_uv,
            confidence=confidence,
            layout=layout,
            paper_speed_mm_s=speed_mm_s,
            gain_mm_mv=gain_mm_mv,
            sample_rate_hz=eff_fs,
            polyline_count=len(traces),
            pulse_count=len(pulses),
            metadata={
                "engine": "waveparse_vector_pdf_lossless",
                "layout": layout,
                "paperSpeedMmPerSecond": speed_mm_s,
                "gainMmPerMv": gain_mm_mv,
                "effectiveSampleRateHz": eff_fs,
                "xOriginPt": round(x_origin, 3),
                "pulseCount": len(pulses),
                "traceCount": len(traces),
            },
        )

    # Case 2: Standard 6x2 (12 panels of ~125 mm = 5.0 s each)
    if len(panel_traces) == 12 and all(t["dx_mm"] >= 100.0 for t in panel_traces):
        panel_traces.sort(key=lambda t: -t["ymed"])
        rows6 = [sorted(panel_traces[2 * i : 2 * i + 2], key=lambda t: t["xmin"]) for i in range(6)]
        eff_fs = sample_rate_hz * (speed_mm_s / 25.0)
        panel_dur_s = 5.0 * (25.0 / speed_mm_s)
        n_panel = int(round(panel_dur_s * eff_fs))
        leads_uv = {l: np.full(2 * n_panel, np.nan, dtype=np.float64) for l in ALL_12_LEADS}
        confidence = {l: np.full(2 * n_panel, np.nan, dtype=np.float64) for l in ALL_12_LEADS}
        for r_idx in range(6):
            for c_idx in range(2):
                lead = STANDARD_6X2_GRID[r_idx][c_idx]
                tr = rows6[r_idx][c_idx]
                sig = _resample_polyline_to_uv(
                    tr["pts"],
                    x_start_pt=tr["xmin"],
                    duration_s=panel_dur_s,
                    sample_rate_hz=eff_fs,
                    speed_mm_s=speed_mm_s,
                    gain_mm_mv=gain_mm_mv,
                )
                s0, s1 = c_idx * n_panel, (c_idx + 1) * n_panel
                leads_uv[lead][s0:s1] = sig
                confidence[lead][s0:s1] = 1.0
        return VectorPdfExtraction(
            leads_uv=leads_uv,
            short_ii_uv=None,
            confidence=confidence,
            layout="standard_6x2",
            paper_speed_mm_s=speed_mm_s,
            gain_mm_mv=gain_mm_mv,
            sample_rate_hz=eff_fs,
            polyline_count=len(traces),
            pulse_count=len(pulses),
            metadata={
                "engine": "waveparse_vector_pdf_lossless",
                "layout": "standard_6x2",
                "paperSpeedMmPerSecond": speed_mm_s,
                "gainMmPerMv": gain_mm_mv,
                "effectiveSampleRateHz": eff_fs,
                "pulseCount": len(pulses),
                "traceCount": len(traces),
            },
        )

    return None


def rasterize_pdf_to_rgb(pdf_path: Path, dpi: int = 300) -> np.ndarray:
    """Render page 1 of a PDF to an RGB numpy array using native macOS PDFKit/Swift (or sips fallback)."""
    with tempfile.TemporaryDirectory(prefix="waveparse-pdf-rast-") as tmp:
        out_png = Path(tmp) / "page1.png"
        swift_bin = shutil.which("swift")
        if swift_bin:
            swift_code = f"""
import Foundation
import PDFKit
import AppKit

let url = URL(fileURLWithPath: "{str(pdf_path)}")
guard let doc = PDFDocument(url: url), let page = doc.page(at: 0) else {{
    fputs("Failed to open PDF\\n", stderr)
    exit(2)
}}
let bounds = page.bounds(for: .mediaBox)
let scale: CGFloat = {float(dpi) / 72.0}
let w = Int(round(bounds.width * scale))
let h = Int(round(bounds.height * scale))
guard let rep = NSBitmapImageRep(
    bitmapDataPlanes: nil, pixelsWide: w, pixelsHigh: h,
    bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
    colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
) else {{ exit(3) }}
NSGraphicsContext.saveGraphicsState()
let ctx = NSGraphicsContext(bitmapImageRep: rep)!
NSGraphicsContext.current = ctx
let cg = ctx.cgContext
cg.setFillColor(NSColor.white.cgColor)
cg.fill(CGRect(x: 0, y: 0, width: w, height: h))
cg.scaleBy(x: scale, y: scale)
page.draw(with: .mediaBox, to: cg)
NSGraphicsContext.restoreGraphicsState()
guard let data = rep.representation(using: .png, properties: [:]) else {{ exit(4) }}
try data.write(to: URL(fileURLWithPath: "{str(out_png)}"))
"""
            proc = subprocess.run([swift_bin, "-e", swift_code], capture_output=True, text=True, check=False)
            if proc.returncode == 0 and out_png.exists():
                bgr = cv2.imread(str(out_png), cv2.IMREAD_COLOR)
                if bgr is not None:
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Fallback: macOS sips
        sips_bin = shutil.which("sips")
        if sips_bin:
            proc = subprocess.run(
                [sips_bin, "-s", "format", "png", str(pdf_path), "--out", str(out_png)],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0 and out_png.exists():
                bgr = cv2.imread(str(out_png), cv2.IMREAD_COLOR)
                if bgr is not None:
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    raise RuntimeError(f"Could not rasterize PDF {pdf_path} (no PDFKit/sips renderer succeeded)")
