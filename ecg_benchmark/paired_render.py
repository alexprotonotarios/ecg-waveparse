from __future__ import annotations

import io
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .io import LEADS


LAYOUT_ROWS: dict[str, tuple[tuple[str, ...], ...]] = {
    "standard_3x4": (
        ("I", "aVR", "V1", "V4"),
        ("II", "aVL", "V2", "V5"),
        ("III", "aVF", "V3", "V6"),
    ),
    "standard_6x2": (
        ("I", "V1"),
        ("II", "V2"),
        ("III", "V3"),
        ("aVR", "V4"),
        ("aVL", "V5"),
        ("aVF", "V6"),
    ),
    "standard_12x1": tuple((lead,) for lead in LEADS),
}


def layout_column_count(layout: str) -> int:
    rows = LAYOUT_ROWS[layout]
    return max(len(row) for row in rows)


def lead_column(layout: str, lead: str) -> int:
    for row in LAYOUT_ROWS[layout]:
        if lead in row:
            return row.index(lead)
    raise ValueError(f"Lead {lead} is not part of {layout}.")


def canonical_visible_truth(
    leads: dict[str, np.ndarray],
    *,
    layout: str,
    visible_leads: Iterable[str] | None = None,
    rhythm_lead: str | None = None,
) -> dict[str, np.ndarray]:
    """Map a ten-second waveform onto only the panels printed on the page."""

    expected_samples = 5000
    visible = set(visible_leads or LEADS)
    columns = layout_column_count(layout)
    panel_samples = expected_samples // columns
    canonical: dict[str, np.ndarray] = {}
    for lead in LEADS:
        source = np.asarray(leads[lead], dtype=np.float64)
        if source.size != expected_samples:
            raise ValueError(
                f"{lead} has {source.size} samples; expected {expected_samples}."
            )
        output = np.full(expected_samples, np.nan, dtype=np.float64)
        if lead in visible:
            column = lead_column(layout, lead)
            start = column * panel_samples
            output[start : start + panel_samples] = source[
                start : start + panel_samples
            ]
        if rhythm_lead == lead and lead in visible:
            output = source.copy()
        canonical[lead] = output
    return canonical


def canonical_segment_truth(
    segments: np.ndarray,
    *,
    layout: str,
    visible_leads: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    """Map dataset-provided printed lead segments onto a ten-second canvas."""

    if segments.ndim != 2 or segments.shape[1] != len(LEADS):
        raise ValueError(f"Expected an N x 12 segment matrix, received {segments.shape}.")
    expected_samples = 5000
    columns = layout_column_count(layout)
    panel_samples = expected_samples // columns
    if segments.shape[0] != panel_samples:
        raise ValueError(
            f"{layout} requires {panel_samples} samples per lead; received "
            f"{segments.shape[0]}."
        )
    visible = set(visible_leads or LEADS)
    canonical: dict[str, np.ndarray] = {}
    for index, lead in enumerate(LEADS):
        output = np.full(expected_samples, np.nan, dtype=np.float64)
        if lead in visible:
            column = lead_column(layout, lead)
            start = column * panel_samples
            output[start : start + panel_samples] = (
                np.asarray(segments[:, index], dtype=np.float64) * 1000.0
            )
        canonical[lead] = output
    return canonical


def canonical_page_segment_truth(
    segments: np.ndarray,
    *,
    layout: str,
    visible_leads: Iterable[str],
    page: int,
    columns_per_page: int,
) -> dict[str, np.ndarray]:
    """Map page-local lead segments to a ten-second page canvas."""

    if segments.ndim != 2 or segments.shape[1] != len(LEADS):
        raise ValueError(f"Expected an N x 12 segment matrix, received {segments.shape}.")
    if columns_per_page < 1 or 5000 % columns_per_page:
        raise ValueError(f"Invalid page column count {columns_per_page}.")
    panel_samples = 5000 // columns_per_page
    if segments.shape[0] != panel_samples:
        raise ValueError(
            f"Page with {columns_per_page} column(s) requires {panel_samples} samples "
            f"per lead; received {segments.shape[0]}."
        )
    first_global_column = page * columns_per_page
    visible = set(visible_leads)
    canonical: dict[str, np.ndarray] = {}
    for index, lead in enumerate(LEADS):
        output = np.full(5000, np.nan, dtype=np.float64)
        if lead in visible:
            local_column = lead_column(layout, lead) - first_global_column
            if not 0 <= local_column < columns_per_page:
                raise ValueError(f"{lead} is outside page {page} of {layout}.")
            start = local_column * panel_samples
            output[start : start + panel_samples] = (
                np.asarray(segments[:, index], dtype=np.float64) * 1000.0
            )
        canonical[lead] = output
    return canonical


def visible_leads_for_page(
    *,
    layout: str,
    page: int,
    columns_per_page: int,
) -> tuple[str, ...]:
    first_column = page * columns_per_page
    last_column = first_column + columns_per_page
    return tuple(
        lead
        for lead in LEADS
        if first_column <= lead_column(layout, lead) < last_column
    )


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_grid(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    pixels_per_mm: float,
    *,
    palette: str,
) -> None:
    palettes = {
        "red": ((255, 224, 224), (246, 155, 155)),
        "pink": ((255, 231, 236), (239, 176, 190)),
        "grey": ((230, 230, 230), (188, 188, 188)),
        "blue": ((225, 238, 250), (164, 196, 226)),
    }
    minor, major = palettes[palette]
    step = max(1, int(round(pixels_per_mm)))
    for x in range(0, width, step):
        is_major = (x // step) % 5 == 0
        draw.line((x, 0, x, height), fill=major if is_major else minor, width=2 if is_major else 1)
    for y in range(0, height, step):
        is_major = (y // step) % 5 == 0
        draw.line((0, y, width, y), fill=major if is_major else minor, width=2 if is_major else 1)


def render_clinical_page(
    leads: dict[str, np.ndarray],
    *,
    layout: str,
    rhythm_lead: str | None,
    sample_rate_hz: float = 500.0,
    paper_speed_mm_per_second: float = 25.0,
    gain_mm_per_mv: float = 10.0,
    pixels_per_mm: float = 6.0,
    trace_width_pixels: int = 2,
    grid_palette: str = "red",
) -> Image.Image:
    rows = LAYOUT_ROWS[layout]
    columns = layout_column_count(layout)
    segment_seconds = 10.0 / columns
    segment_width_mm = segment_seconds * paper_speed_mm_per_second
    left_margin_mm = 14.0
    right_margin_mm = 10.0
    top_margin_mm = max(18.0, gain_mm_per_mv + 4.0)
    row_step_mm = 25.0 if len(rows) <= 6 else 18.0
    bottom_margin_mm = 14.0
    rhythm_gap_mm = 5.0
    rhythm_rows = 1 if rhythm_lead else 0
    height_mm = (
        top_margin_mm
        + row_step_mm * (len(rows) - 1 + rhythm_rows)
        + rhythm_gap_mm * rhythm_rows
        + bottom_margin_mm
    )
    width_mm = left_margin_mm + columns * segment_width_mm + right_margin_mm
    width = int(round(width_mm * pixels_per_mm))
    height = int(round(height_mm * pixels_per_mm))
    image = Image.new("RGB", (width, height), (255, 253, 250))
    draw = ImageDraw.Draw(image)
    _draw_grid(draw, width, height, pixels_per_mm, palette=grid_palette)
    header_font = _font(max(10, int(round(1.8 * pixels_per_mm))))
    label_font = _font(max(12, int(round(2.7 * pixels_per_mm))), bold=True)
    draw.text(
        (int(6 * pixels_per_mm), int(2 * pixels_per_mm)),
        f"{gain_mm_per_mv:g} mm/mV   {paper_speed_mm_per_second:g} mm/s",
        fill=(25, 25, 25),
        font=header_font,
    )

    panel_samples = int(round(segment_seconds * sample_rate_hz))
    for row_index, row in enumerate(rows):
        baseline_mm = top_margin_mm + row_index * row_step_mm
        pulse_x = 3.0 * pixels_per_mm
        pulse_y = baseline_mm * pixels_per_mm
        pulse_height = gain_mm_per_mv * pixels_per_mm
        pulse_width = .2 * paper_speed_mm_per_second * pixels_per_mm
        draw.line(
            (
                (pulse_x, pulse_y),
                (pulse_x, pulse_y - pulse_height),
                (pulse_x + pulse_width, pulse_y - pulse_height),
                (pulse_x + pulse_width, pulse_y),
            ),
            fill=(20, 20, 20),
            width=trace_width_pixels,
        )
        for column_index, lead in enumerate(row):
            source = np.asarray(leads[lead], dtype=np.float64)
            start = column_index * panel_samples
            segment = source[start : start + panel_samples]
            x0_mm = left_margin_mm + column_index * segment_width_mm
            draw.text(
                (int((x0_mm + 1.0) * pixels_per_mm), int((baseline_mm - 10.0) * pixels_per_mm)),
                lead,
                fill=(18, 18, 18),
                font=label_font,
            )
            x = (x0_mm + np.arange(segment.size) / sample_rate_hz * paper_speed_mm_per_second) * pixels_per_mm
            y = (baseline_mm - segment / 1000.0 * gain_mm_per_mv) * pixels_per_mm
            draw.line(
                [(float(a), float(b)) for a, b in zip(x, y)],
                fill=(18, 18, 18),
                width=trace_width_pixels,
                joint="curve",
            )

    if rhythm_lead:
        baseline_mm = top_margin_mm + len(rows) * row_step_mm + rhythm_gap_mm
        source = np.asarray(leads[rhythm_lead], dtype=np.float64)
        draw.text(
            (int((left_margin_mm + 1.0) * pixels_per_mm), int((baseline_mm - 10.0) * pixels_per_mm)),
            rhythm_lead,
            fill=(18, 18, 18),
            font=label_font,
        )
        x = (left_margin_mm + np.arange(source.size) / sample_rate_hz * paper_speed_mm_per_second) * pixels_per_mm
        y = (baseline_mm - source / 1000.0 * gain_mm_per_mv) * pixels_per_mm
        draw.line(
            [(float(a), float(b)) for a, b in zip(x, y)],
            fill=(18, 18, 18),
            width=trace_width_pixels,
            joint="curve",
        )
    return image


def apply_artifact(
    image: Image.Image,
    *,
    kind: str,
    parameters: dict[str, Any],
    seed: int,
) -> Image.Image:
    rng = np.random.default_rng(seed)
    if kind == "clean":
        return image
    if kind == "rotation":
        angle = float(parameters.get("degrees", 3.0))
        if seed % 2:
            angle *= -1
        return image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(246, 243, 238))
    if kind == "perspective":
        fraction = float(parameters.get("insetFraction", 0.035))
        x = image.width * fraction
        y = image.height * fraction
        return image.transform(
            image.size,
            Image.Transform.QUAD,
            (x, y, 0, image.height, image.width, image.height - y, image.width - x, 0),
            resample=Image.Resampling.BICUBIC,
            fillcolor=(246, 243, 238),
        )
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(float(parameters.get("radiusPixels", 1.2))))
    if kind == "low_resolution":
        target = int(parameters.get("targetWidthPixels", 900))
        height = max(1, int(round(image.height * target / image.width)))
        return image.resize((target, height), Image.Resampling.LANCZOS)
    if kind == "faded":
        output = ImageEnhance.Contrast(image).enhance(float(parameters.get("contrast", 0.55)))
        return ImageEnhance.Brightness(output).enhance(float(parameters.get("brightness", 1.08)))
    if kind == "annotation":
        output = image.copy()
        draw = ImageDraw.Draw(output)
        font = _font(max(14, image.width // 90), bold=True)
        draw.text((image.width * 0.58, image.height * 0.025), "CHECK V4", fill=(20, 60, 185), font=font)
        x = float(image.width * (0.62 + 0.08 * rng.random()))
        y0 = float(image.height * 0.24)
        y1 = float(image.height * 0.39)
        draw.line((x, y0, x, y1), fill=(20, 60, 200), width=max(3, image.width // 500))
        draw.polygon(((x - 10, y1 - 16), (x + 10, y1 - 16), (x, y1 + 4)), fill=(20, 60, 200))
        return output
    if kind == "jpeg_compression":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(parameters.get("quality", 28)), optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")
    raise ValueError(f"Unsupported benchmark artifact kind {kind}.")
