from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OPEN_ECG_DIR = ROOT / "open-ecg-digitizer"
PAGE_DURATION_SECONDS = 10.0

LAYOUTS = {
    "standard_6x2": {
        "lead_layout": [
            ("I", "V1"),
            ("II", "V2"),
            ("III", "V3"),
            ("aVR", "V4"),
            ("aVL", "V5"),
            ("aVF", "V6"),
        ],
        "segment_seconds": 5.0,
        "row_step_mm": 27.0,
        "col_gap_mm": 9.0,
        "top_margin_mm": 22.0,
        "bottom_margin_mm": 16.0,
    },
    "standard_3x4": {
        "lead_layout": [
            ("I", "aVR", "V1", "V4"),
            ("II", "aVL", "V2", "V5"),
            ("III", "aVF", "V3", "V6"),
        ],
        "segment_seconds": 2.5,
        "row_step_mm": 36.0,
        "col_gap_mm": 0.0,
        "top_margin_mm": 28.0,
        "bottom_margin_mm": 20.0,
    },
    "standard_12x1": {
        "lead_layout": [
            ("I",),
            ("II",),
            ("III",),
            ("aVR",),
            ("aVL",),
            ("aVF",),
            ("V1",),
            ("V2",),
            ("V3",),
            ("V4",),
            ("V5",),
            ("V6",),
        ],
        "segment_seconds": 10.0,
        "row_step_mm": 22.0,
        "col_gap_mm": 0.0,
        "top_margin_mm": 22.0,
        "bottom_margin_mm": 16.0,
    },
    "cabrera_12x1": {
        "lead_layout": [
            ("aVL",),
            ("I",),
            ("aVR",),
            ("II",),
            ("aVF",),
            ("III",),
            ("V1",),
            ("V2",),
            ("V3",),
            ("V4",),
            ("V5",),
            ("V6",),
        ],
        "segment_seconds": 10.0,
        "row_step_mm": 22.0,
        "col_gap_mm": 0.0,
        "top_margin_mm": 22.0,
        "bottom_margin_mm": 16.0,
    },
    "standard_6x1_limb": {
        "lead_layout": [("I",), ("II",), ("III",), ("aVR",), ("aVL",), ("aVF",)],
        "segment_seconds": 10.0,
        "row_step_mm": 27.0,
        "col_gap_mm": 0.0,
        "top_margin_mm": 22.0,
        "bottom_margin_mm": 16.0,
    },
    "cabrera_6x1_limb": {
        "lead_layout": [("aVL",), ("I",), ("aVR",), ("II",), ("aVF",), ("III",)],
        "segment_seconds": 10.0,
        "row_step_mm": 27.0,
        "col_gap_mm": 0.0,
        "top_margin_mm": 22.0,
        "bottom_margin_mm": 16.0,
    },
    "precordial_6x1": {
        "lead_layout": [("V1",), ("V2",), ("V3",), ("V4",), ("V5",), ("V6",)],
        "segment_seconds": 10.0,
        "row_step_mm": 27.0,
        "col_gap_mm": 0.0,
        "top_margin_mm": 22.0,
        "bottom_margin_mm": 16.0,
    },
    "precordial_3x2": {
        "lead_layout": [("V1", "V4"), ("V2", "V5"), ("V3", "V6")],
        "segment_seconds": 5.0,
        "row_step_mm": 36.0,
        "col_gap_mm": 9.0,
        "top_margin_mm": 28.0,
        "bottom_margin_mm": 20.0,
    },
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/System/Library/Fonts/Supplemental/Courier New Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Courier New.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_ecg_grid(draw: ImageDraw.ImageDraw, width: int, height: int, px_per_mm: int) -> None:
    minor = (255, 218, 218)
    major = (255, 150, 150)
    for x in range(0, width, px_per_mm):
        color = major if (x // px_per_mm) % 5 == 0 else minor
        draw.line([(x, 0), (x, height)], fill=color, width=2 if color == major else 1)
    for y in range(0, height, px_per_mm):
        color = major if (y // px_per_mm) % 5 == 0 else minor
        draw.line([(0, y), (width, y)], fill=color, width=2 if color == major else 1)


def draw_calibration(
    draw: ImageDraw.ImageDraw,
    x_mm: float,
    baseline_y_mm: float,
    px_per_mm: int,
    color: tuple[int, int, int],
    width: int,
    paper_speed_mm_per_second: float = 25.0,
    gain_mm_per_mv: float = 10.0,
) -> None:
    x = x_mm * px_per_mm
    y0 = baseline_y_mm * px_per_mm
    y1 = (baseline_y_mm - gain_mm_per_mv) * px_per_mm
    pulse_w = paper_speed_mm_per_second * 0.2 * px_per_mm
    lead_in = 2 * px_per_mm
    pts = [(x - lead_in, y0), (x, y0), (x, y1), (x + pulse_w, y1), (x + pulse_w, y0), (x + lead_in + pulse_w, y0)]
    draw.line(pts, fill=color, width=width, joint="curve")


def draw_signal(
    draw: ImageDraw.ImageDraw,
    df: pd.DataFrame,
    lead: str,
    x0_mm: float,
    baseline_y_mm: float,
    px_per_mm: int,
    color: tuple[int, int, int],
    width: int,
    segment_seconds: float,
    paper_speed_mm_per_second: float = 25.0,
    gain_mm_per_mv: float = 10.0,
    subtract_baseline: bool = True,
) -> None:
    signal = df[lead].to_numpy(dtype=float)
    time_s = df["t_s_at_500hz"].to_numpy(dtype=float)
    finite = np.isfinite(signal) & (time_s <= segment_seconds)
    if not finite.any():
        return

    baseline_uv = float(np.nanmedian(signal)) if subtract_baseline else 0.0
    x = (x0_mm + time_s * paper_speed_mm_per_second) * px_per_mm
    y = (baseline_y_mm - ((signal - baseline_uv) / 1000.0) * gain_mm_per_mv) * px_per_mm

    # Break polylines at NaNs so dropped samples do not create long diagonal joins.
    idx = np.flatnonzero(finite)
    splits = np.where(np.diff(idx) > 1)[0] + 1
    for run in np.split(idx, splits):
        if len(run) < 2:
            continue
        points = [(float(x[i]), float(y[i])) for i in run]
        draw.line(points, fill=color, width=width, joint="curve")


def draw_uncertainty_bands(
    draw: ImageDraw.ImageDraw,
    uncertainty_df: pd.DataFrame | None,
    lead: str,
    x0_mm: float,
    baseline_y_mm: float,
    px_per_mm: int,
    segment_seconds: float,
    paper_speed_mm_per_second: float = 25.0,
) -> None:
    samples = lead_uncertainty_samples(uncertainty_df, lead, segment_seconds)
    if samples.empty:
        return

    colors = {
        "missing": (255, 220, 220),
        "uncertain_annotation": (255, 238, 207),
        "uncertain_candidate_disagreement": (224, 235, 255),
        "uncertain_annotation_and_disagreement": (239, 224, 250),
    }
    run_start = 0
    for index in range(1, len(samples) + 1):
        run_ended = index == len(samples)
        if not run_ended:
            previous = samples.iloc[index - 1]
            current = samples.iloc[index]
            run_ended = (
                current["lead_sample"] > previous["lead_sample"] + 1
                or current["status"] != previous["status"]
            )
        if not run_ended:
            continue

        run = samples.iloc[run_start:index]
        first = float(run.iloc[0]["lead_sample"])
        last = float(run.iloc[-1]["lead_sample"])
        status = str(run.iloc[0]["status"])
        x0 = int((x0_mm + (first / 500.0) * paper_speed_mm_per_second) * px_per_mm)
        x1 = max(
            x0 + 1,
            int((x0_mm + ((last + 1.0) / 500.0) * paper_speed_mm_per_second) * px_per_mm),
        )
        draw.rectangle(
            [
                (x0, int((baseline_y_mm - 15.0) * px_per_mm)),
                (x1, int((baseline_y_mm + 15.0) * px_per_mm)),
            ],
            fill=colors.get(status, colors["uncertain_annotation_and_disagreement"]),
        )
        run_start = index


def draw_review_estimate(
    draw: ImageDraw.ImageDraw,
    df: pd.DataFrame,
    uncertainty_df: pd.DataFrame | None,
    lead: str,
    x0_mm: float,
    baseline_y_mm: float,
    px_per_mm: int,
    segment_seconds: float,
    paper_speed_mm_per_second: float = 25.0,
    gain_mm_per_mv: float = 10.0,
) -> None:
    if uncertainty_df is None or "review_estimate_uv" not in uncertainty_df:
        return

    signal = df[lead].to_numpy(dtype=float)
    baseline_uv = float(np.nanmedian(signal))
    samples = lead_uncertainty_samples(uncertainty_df, lead, segment_seconds)
    samples = samples[
        samples["status"].astype(str).str.contains("annotation")
        & np.isfinite(samples["review_estimate_uv"])
    ]
    if len(samples) < 2:
        return

    indices = samples["lead_sample"].to_numpy(dtype=int)
    split_points = [
        0,
        *(np.where(np.diff(indices) > 1)[0] + 1).tolist(),
        len(samples),
    ]
    for start_index, end_index in zip(split_points, split_points[1:]):
        run = samples.iloc[start_index:end_index]
        if len(run) < 2:
            continue
        points = [
            (
                float((x0_mm + (row.lead_sample / 500.0) * paper_speed_mm_per_second) * px_per_mm),
                float(
                    (
                        baseline_y_mm
                        - ((row.review_estimate_uv - baseline_uv) / 1000.0) * gain_mm_per_mv
                    )
                    * px_per_mm
                ),
            )
            for row in run.itertuples()
        ]
        for start in range(0, len(points) - 1, 16):
            dash = points[start : min(start + 9, len(points))]
            if len(dash) >= 2:
                draw.line(dash, fill=(138, 75, 8), width=3, joint="curve")


def lead_uncertainty_samples(
    uncertainty_df: pd.DataFrame | None,
    lead: str,
    segment_seconds: float,
) -> pd.DataFrame:
    if (
        uncertainty_df is None
        or not {"lead", "lead_sample", "status"}.issubset(uncertainty_df.columns)
    ):
        return pd.DataFrame()

    samples = uncertainty_df[uncertainty_df["lead"] == lead].copy()
    samples["lead_sample"] = pd.to_numeric(samples["lead_sample"], errors="coerce")
    if "review_estimate_uv" in samples:
        samples["review_estimate_uv"] = pd.to_numeric(
            samples["review_estimate_uv"], errors="coerce"
        )
    samples = samples[
        np.isfinite(samples["lead_sample"])
        & (samples["lead_sample"] / 500.0 <= segment_seconds)
        & (samples["status"] != "observed")
    ]
    return samples.sort_values("lead_sample").reset_index(drop=True)


def render_paper(
    csv_path: Path,
    output_path: Path,
    trace_color: tuple[int, int, int] = (18, 18, 18),
    title: str = "Digitized ECG",
    layout_name: str = "standard_6x2",
    effective_sample_rate_hz: float | None = None,
    uncertainty_path: Path | None = None,
    show_uncertainty_bands: bool = False,
    paper_speed_mm_per_second: float = 25.0,
    gain_mm_per_mv: float = 10.0,
) -> None:
    df = pd.read_csv(csv_path)
    uncertainty_df = (
        pd.read_csv(uncertainty_path)
        if uncertainty_path and uncertainty_path.exists()
        else None
    )
    layout_key = normalize_layout_name(layout_name)
    layout_config = LAYOUTS[layout_key]
    lead_layout = layout_config["lead_layout"]
    has_rhythm_strip = (
        layout_key == "standard_3x4"
        and "with_r1" in layout_name.lower()
    )

    px_per_mm = 12
    segment_seconds = float(layout_config["segment_seconds"])
    segment_width_mm = segment_seconds * paper_speed_mm_per_second
    left_margin_mm = 14
    right_margin_mm = 10
    top_margin_mm = float(layout_config["top_margin_mm"])
    bottom_margin_mm = float(layout_config["bottom_margin_mm"])
    row_step_mm = float(layout_config["row_step_mm"])
    col_gap_mm = float(layout_config["col_gap_mm"])

    num_cols = max(len(row) for row in lead_layout)
    width_mm = left_margin_mm + num_cols * segment_width_mm + (num_cols - 1) * col_gap_mm + right_margin_mm
    rendered_row_count = len(lead_layout) + (1 if has_rhythm_strip else 0)
    height_mm = (
        top_margin_mm
        + row_step_mm * (rendered_row_count - 1)
        + bottom_margin_mm
    )
    width = int(width_mm * px_per_mm)
    height = int(height_mm * px_per_mm)

    img = Image.new("RGB", (width, height), (255, 253, 250))
    draw = ImageDraw.Draw(img)
    draw_ecg_grid(draw, width, height, px_per_mm)

    small_font = font(18)
    label_font = font(36, bold=True)
    header_font = font(23)
    trace_width = 3

    sample_rate_label = "500 Hz output"
    if effective_sample_rate_hz and effective_sample_rate_hz < 499.5:
        sample_rate_label += f"   ~{effective_sample_rate_hz:.0f} Hz image-limited"
    draw.text(
        (left_margin_mm * px_per_mm, 5 * px_per_mm),
        f"{gain_mm_per_mv:g} mm/mV   {paper_speed_mm_per_second:g} mm/s   {sample_rate_label}",
        fill=(30, 30, 30),
        font=header_font,
    )
    if title:
        draw.text((width - 38 * px_per_mm, 5 * px_per_mm), title, fill=(55, 55, 55), font=small_font)

    col_x = [left_margin_mm + col_idx * (segment_width_mm + col_gap_mm) for col_idx in range(num_cols)]
    for row_idx, row in enumerate(lead_layout):
        baseline_mm = top_margin_mm + row_idx * row_step_mm
        draw_calibration(
            draw,
            4.0,
            baseline_mm,
            px_per_mm,
            trace_color,
            trace_width,
            paper_speed_mm_per_second,
            gain_mm_per_mv,
        )
        for col_idx, lead in enumerate(row):
            x0_mm = col_x[col_idx]
            if show_uncertainty_bands:
                draw_uncertainty_bands(
                    draw,
                    uncertainty_df,
                    lead,
                    x0_mm + 7.0,
                    baseline_mm,
                    px_per_mm,
                    segment_seconds,
                    paper_speed_mm_per_second,
                )
            draw.text(
                (int((x0_mm + 1.0) * px_per_mm), int((baseline_mm - 14.0) * px_per_mm)),
                lead,
                fill=(20, 20, 20),
                font=label_font,
            )
            draw_signal(
                draw,
                df,
                lead,
                x0_mm + 7.0,
                baseline_mm,
                px_per_mm,
                trace_color,
                trace_width,
                segment_seconds,
                paper_speed_mm_per_second,
                gain_mm_per_mv,
            )
            draw_review_estimate(
                draw,
                df,
                uncertainty_df,
                lead,
                x0_mm + 7.0,
                baseline_mm,
                px_per_mm,
                segment_seconds,
                paper_speed_mm_per_second,
                gain_mm_per_mv,
            )

    if has_rhythm_strip:
        rhythm_baseline_mm = top_margin_mm + len(lead_layout) * row_step_mm
        draw_calibration(
            draw,
            4.0,
            rhythm_baseline_mm,
            px_per_mm,
            trace_color,
            trace_width,
            paper_speed_mm_per_second,
            gain_mm_per_mv,
        )
        rhythm_x0_mm = col_x[0]
        draw.text(
            (
                int((rhythm_x0_mm + 1.0) * px_per_mm),
                int((rhythm_baseline_mm - 14.0) * px_per_mm),
            ),
            "II rhythm",
            fill=(20, 20, 20),
            font=label_font,
        )
        draw_signal(
            draw,
            df,
            "II",
            rhythm_x0_mm + 7.0,
            rhythm_baseline_mm,
            px_per_mm,
            trace_color,
            trace_width,
            PAGE_DURATION_SECONDS,
            paper_speed_mm_per_second,
            gain_mm_per_mv,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, optimize=True)


def normalize_layout_name(layout_name: str) -> str:
    if layout_name.startswith("standard_3x4"):
        return "standard_3x4"
    if layout_name.startswith("standard_6x2"):
        return "standard_6x2"
    if layout_name.startswith("standard_12x1"):
        return "standard_12x1"
    if layout_name.startswith("cabrera_12x1"):
        return "cabrera_12x1"
    if layout_name.startswith("standard_6x1_limb"):
        return "standard_6x1_limb"
    if layout_name.startswith("cabrera_6x1_limb"):
        return "cabrera_6x1_limb"
    if layout_name.startswith("precordial_6x1"):
        return "precordial_6x1"
    if layout_name.startswith("precordial_3x2"):
        return "precordial_3x2"
    return "standard_6x2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render compact digitized ECG CSV onto synthetic ECG paper.")
    parser.add_argument("--csv", type=Path, help="Compact lead-segment CSV to render.")
    parser.add_argument("--output", type=Path, help="Output PNG path.")
    parser.add_argument(
        "--uncertainty",
        type=Path,
        default=None,
        help="Optional sample-level uncertainty CSV with review-only estimates.",
    )
    parser.add_argument(
        "--show-uncertainty-bands",
        action="store_true",
        help="Include colored QA bands. Omit for a clean paper export.",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional title printed in the output PNG. Leave blank for a clean clinical render.",
    )
    parser.add_argument(
        "--layout",
        default="standard_6x2",
        help="Detected ECG layout, for example standard_3x4, standard_6x2, or standard_12x1.",
    )
    parser.add_argument(
        "--effective-sample-rate",
        type=float,
        default=None,
        help="Estimated independent horizontal samples per second in the source raster.",
    )
    parser.add_argument(
        "--paper-speed",
        type=float,
        default=25.0,
        choices=(25.0, 50.0),
        help="Detected paper speed in mm/s; defaults to 25 when no calibration is available.",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=10.0,
        help="Detected vertical gain in mm/mV; defaults to 10 when no calibration is available.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.csv and args.output:
        render_paper(
            args.csv,
            args.output,
            title=args.title,
            layout_name=args.layout,
            effective_sample_rate_hz=args.effective_sample_rate,
            uncertainty_path=args.uncertainty,
            show_uncertainty_bands=args.show_uncertainty_bands,
            paper_speed_mm_per_second=args.paper_speed,
            gain_mm_per_mv=args.gain,
        )
        raise SystemExit(0)

    render_paper(
        OPEN_ECG_DIR / "trace_focused_segments_500hz.csv",
        OPEN_ECG_DIR / "trace_focused_paper_render.png",
        title="Open-ECG-Digitizer 1500 px",
    )
    render_paper(
        OPEN_ECG_DIR / "trace_focused_2200_segments_500hz.csv",
        OPEN_ECG_DIR / "trace_focused_2200_paper_render.png",
        title="Open-ECG-Digitizer 2200 px",
    )
    render_paper(
        OPEN_ECG_DIR / "trace_focused_2200_segments_500hz.csv",
        OPEN_ECG_DIR / "trace_focused_2200_paper_render_teal.png",
        trace_color=(0, 115, 130),
        title="Open-ECG-Digitizer 2200 px",
    )
