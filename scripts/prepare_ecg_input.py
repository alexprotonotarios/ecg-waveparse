from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
POLICY_THRESHOLDS = json.loads(
    (ROOT / "config/digitizer-policy.v3.json").read_text(encoding="utf-8")
)["thresholds"]
MASK_VERSION = 7
DEFAULT_MAX_WORKING_LONG_EDGE_PX = 3200
MAX_SOURCE_PIXELS = 80_000_000
MAX_SOURCE_EDGE_PIXELS = 32768
MIN_QUANTITATIVE_LONG_EDGE_PX = int(
    POLICY_THRESHOLDS["minimumQuantitativeSourceLongEdgePixels"]
)
MIN_QUANTITATIVE_SHORT_EDGE_PX = int(
    POLICY_THRESHOLDS["minimumQuantitativeSourceShortEdgePixels"]
)
MIN_QUANTITATIVE_HIGH_ASPECT_SHORT_EDGE_PX = int(
    POLICY_THRESHOLDS["minimumQuantitativeHighAspectShortEdgePixels"]
)
QUANTITATIVE_HIGH_ASPECT_RATIO_THRESHOLD = float(
    POLICY_THRESHOLDS["quantitativeHighAspectRatioThreshold"]
)
MASK_DILATION_PX = 1
FILL_RING_PX = 8
FILL_PERCENTILE = 75.0
BACKGROUND_KERNEL_SHORT_EDGE_FRACTION = 0.06
MIN_BACKGROUND_KERNEL_PX = 15
MAX_BACKGROUND_KERNEL_PX = 81
GRID_PROFILE_QUANTILE = 0.50
MIN_PREPROCESSING_SHARPNESS_LAPLACIAN_VARIANCE = 200.0
SEVERE_SOURCE_BLUR_LAPLACIAN_VARIANCE = 50.0
LOW_RESOLUTION_LONG_EDGE_PX = 1200
FOREGROUND_CONTAMINATION_TRACE_P95 = 0.35
WEAK_GRID_PROBABILITY_P95 = 0.50
TRACE_DOMINANT_SMOOTH_SOURCE_P95 = 0.90
TRACE_DOMINANT_SMOOTH_SOURCE_MAX_GRID_P95 = 0.40
MIN_PAGE_QUADRILATERAL_AREA_FRACTION = 0.45
MAX_UNCROPPED_PAGE_AREA_FRACTION = 0.96
MIN_PERSPECTIVE_STRENGTH = 0.025
MIN_DESKEW_DEGREES = 0.75
MAX_DESKEW_DEGREES = 12.0
SCREEN_CHROMA_MOIRE_THRESHOLD = 0.035
# Screen moire is a dense, page-wide chroma texture. Ordinary ECG paper also
# has strong periodic chroma edges, but they occupy less than half the pixels.
# Requiring majority coverage prevents a legitimate red/pink ECG grid from
# scheduling the screen specialist.
SCREEN_CHROMA_MOIRE_MIN_COVERAGE = 0.60
GLARE_FRACTION_THRESHOLD = 0.025


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic ECG annotation masks without modifying the source image."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument(
        "--max-working-long-edge",
        type=int,
        default=DEFAULT_MAX_WORKING_LONG_EDGE_PX,
    )
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--working-source", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--enhanced", type=Path)
    parser.add_argument("--background", type=Path)
    parser.add_argument("--trace-probability", type=Path)
    parser.add_argument("--grid-probability", type=Path)
    parser.add_argument("--exclusion-mask", type=Path)
    parser.add_argument("--geometry-corrected", type=Path)
    parser.add_argument("--artifact-preprocessed", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.max_working_long_edge < MIN_QUANTITATIVE_LONG_EDGE_PX:
        parser.error(
            f"--max-working-long-edge must be at least {MIN_QUANTITATIVE_LONG_EDGE_PX}"
        )
    if not args.inspect:
        required_outputs = (
            "prepared",
            "working_source",
            "mask",
            "enhanced",
            "background",
            "trace_probability",
            "grid_probability",
            "exclusion_mask",
            "geometry_corrected",
            "artifact_preprocessed",
            "report",
        )
        missing = [name for name in required_outputs if getattr(args, name) is None]
        if missing:
            parser.error(
                "the following outputs are required unless --inspect is used: "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )
    return args


def inspect_source(path: Path) -> dict[str, int | str | None]:
    """Read image metadata without materializing the source pixel array."""

    with Image.open(path) as source:
        validate_source_header(source)
        width, height = source.size
        return {
            "width": int(width),
            "height": int(height),
            "pixelCount": int(width * height),
            "format": source.format,
        }


def validate_source_header(source: Image.Image) -> None:
    """Bound decoding and refuse silently selecting one frame from a document."""
    width,height=source.size
    if source.format not in {"PNG","JPEG","WEBP","TIFF"}:
        raise ValueError("invalid_input: unsupported_raster_format")
    if min(width,height)<1 or max(width,height)>MAX_SOURCE_EDGE_PIXELS or width*height>MAX_SOURCE_PIXELS:
        raise ValueError("invalid_input: source_dimensions_exceed_budget")
    if getattr(source,"n_frames",1)!=1:
        raise ValueError("invalid_input: multiple_frames_require_explicit_selection")


def load_working_image(
    path: Path,
    max_long_edge: int,
) -> tuple[np.ndarray, dict[str, int | str | float | None]]:
    """Decode a bounded deterministic working image while preserving source metadata."""

    with Image.open(path) as source:
        validate_source_header(source)
        source_format = source.format
        original_width, original_height = source.size
        if max(original_width, original_height) > max_long_edge:
            source.draft("RGB", (max_long_edge, max_long_edge))
            source.thumbnail(
                (max_long_edge, max_long_edge),
                resample=Image.Resampling.LANCZOS,
                reducing_gap=3.0,
            )
        working = source.convert("RGB")
        working_width, working_height = working.size
        image = np.asarray(working).copy()

    scale_x = working_width / max(original_width, 1)
    scale_y = working_height / max(original_height, 1)
    return image, {
        "sourceWidth": int(original_width),
        "sourceHeight": int(original_height),
        "sourceFormat": source_format,
        "width": int(working_width),
        "height": int(working_height),
        "scaleX": float(scale_x),
        "scaleY": float(scale_y),
        "method": (
            "pillow-lanczos-bounded-long-edge-v1"
            if working_width != original_width or working_height != original_height
            else "source-resolution-v1"
        ),
        "maxLongEdgePixels": int(max_long_edge),
    }


def components_in_source_coordinates(
    components: list[dict[str, int | str | float]],
    *,
    scale_x: float,
    scale_y: float,
    source_width: int,
    source_height: int,
) -> list[dict[str, int | str | float]]:
    result: list[dict[str, int | str | float]] = []
    for component in components:
        result.append(
            {
                **component,
                "x0": min(
                    source_width,
                    max(0, int(round(int(component["x0"]) / scale_x))),
                ),
                "y0": min(
                    source_height,
                    max(0, int(round(int(component["y0"]) / scale_y))),
                ),
                "x1": min(
                    source_width,
                    max(0, int(round(int(component["x1"]) / scale_x))),
                ),
                "y1": min(
                    source_height,
                    max(0, int(round(int(component["y1"]) / scale_y))),
                ),
                "pixels": min(
                    source_width * source_height,
                    int(
                        round(
                            int(component["pixels"])
                            / max(scale_x * scale_y, 1e-9)
                        )
                    ),
                ),
            }
        )
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def color_candidates(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = image.astype(np.int16)
    red, green, blue = (rgb[..., index] for index in range(3))
    saturation = rgb.max(axis=2) - rgb.min(axis=2)

    blue_pixels = (
        (blue >= 100)
        & (blue - red >= 35)
        & (blue - green >= 15)
        & (saturation >= 50)
    )
    red_pixels = (
        (red >= 120)
        & (green <= 100)
        & (blue <= 100)
        & (red - green >= 35)
        & (red - blue >= 35)
        & (saturation >= 50)
    )
    return red_pixels, blue_pixels


def annotation_components(
    red_pixels: np.ndarray, blue_pixels: np.ndarray
) -> tuple[np.ndarray, list[dict[str, int | str | float]]]:
    height, width = red_pixels.shape
    accepted = np.zeros_like(red_pixels, dtype=bool)
    components: list[dict[str, int | str | float]] = []

    # Label each colour separately. Joining red and blue first can connect a blue
    # arrow to a red ECG grid at their crossings, causing the useful arrow
    # component to be rejected as a page-spanning grid.
    for dominant_color, color_pixels in (
        ("red", red_pixels),
        ("blue", blue_pixels),
    ):
        joined = ndimage.binary_dilation(color_pixels, iterations=1)
        labels, count = ndimage.label(joined)
        component_slices = ndimage.find_objects(labels, max_label=count)

        for label_id, component_slice in enumerate(component_slices, start=1):
            if component_slice is None:
                continue

            y_slice, x_slice = component_slice
            x0, x1 = int(x_slice.start), int(x_slice.stop)
            y0, y1 = int(y_slice.start), int(y_slice.stop)
            box_width = x1 - x0
            box_height = y1 - y0
            local_labels = labels[component_slice]
            local_component = local_labels == label_id
            pixel_count = int(np.count_nonzero(local_component))
            if pixel_count < 18:
                continue

            box_area = box_width * box_height
            fill_fraction = float(pixel_count / max(box_area, 1))

            # Reject page-spanning grids, thin grid fragments, and isolated
            # compression noise. Compact arrows and annotation glyphs have area
            # in both dimensions even when one stroke is narrow.
            if box_width > width * 0.15 or box_height > height * 0.20:
                continue
            if min(box_width, box_height) < 8:
                continue
            if box_area > width * height * 0.05 or fill_fraction < 0.05:
                continue

            accepted[component_slice] |= local_component
            components.append(
                {
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "pixels": pixel_count,
                    "fillFraction": fill_fraction,
                    "dominantColor": dominant_color,
                }
            )

    if accepted.any():
        accepted = ndimage.binary_dilation(accepted, iterations=MASK_DILATION_PX)
    return accepted, components


def suppress_annotations(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace annotations with a local background value, never inferred morphology."""

    prepared = image.copy()
    if not mask.any():
        return prepared

    grayscale = np.rint(image.mean(axis=2)).astype(np.uint8)
    labels, count = ndimage.label(mask)
    component_slices = ndimage.find_objects(labels, max_label=count)
    unmasked = ~mask
    fallback_values = grayscale[unmasked]
    fallback = float(
        np.percentile(fallback_values, FILL_PERCENTILE)
        if fallback_values.size
        else 255.0
    )

    for label_id, component_slice in enumerate(component_slices, start=1):
        if component_slice is None:
            continue
        y_slice, x_slice = component_slice
        y0 = max(0, int(y_slice.start) - FILL_RING_PX)
        y1 = min(mask.shape[0], int(y_slice.stop) + FILL_RING_PX)
        x0 = max(0, int(x_slice.start) - FILL_RING_PX)
        x1 = min(mask.shape[1], int(x_slice.stop) + FILL_RING_PX)
        local_slice = (slice(y0, y1), slice(x0, x1))
        component = labels[local_slice] == label_id
        ring = (
            ndimage.binary_dilation(component, iterations=FILL_RING_PX)
            & unmasked[local_slice]
        )
        ring_values = grayscale[local_slice][ring]
        fill_value = int(
            round(
                np.percentile(ring_values, FILL_PERCENTILE)
                if ring_values.size
                else fallback
            )
        )
        prepared_view = prepared[local_slice]
        prepared_view[component] = np.array(
            [fill_value, fill_value, fill_value], dtype=np.uint8
        )

    return prepared


def _odd_kernel_size(value: float) -> int:
    bounded = max(
        MIN_BACKGROUND_KERNEL_PX,
        min(MAX_BACKGROUND_KERNEL_PX, int(round(value))),
    )
    return bounded if bounded % 2 else bounded + 1


def _robust_unit_scale(
    values: np.ndarray,
    *,
    low_quantile: float = 0.05,
    high_quantile: float = 0.995,
) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low = float(np.quantile(finite, low_quantile))
    high = float(np.quantile(finite, high_quantile))
    if high - low < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values.astype(np.float32) - low) / (high - low), 0.0, 1.0)


def _profile_strength(profile: np.ndarray) -> np.ndarray:
    """Return page-spanning line evidence without assuming a fixed grid phase."""

    normalized = _robust_unit_scale(
        profile.astype(np.float32),
        low_quantile=0.35,
        high_quantile=0.995,
    )
    return cv2.GaussianBlur(normalized[:, None], (1, 3), 0.45).ravel()


def deterministic_evidence_maps(
    image: np.ndarray,
    exclusion_mask: np.ndarray,
) -> dict[str, np.ndarray | int | float]:
    """Create source-derived evidence maps without synthesizing morphology.

    Illumination is estimated on a scale larger than ECG ink. Grid evidence is
    inferred from page-spanning row/column profiles and warm chroma. Trace
    evidence retains only darkness in excess of that grid estimate, with a
    colour-aware path that also works when the grid itself is grayscale.
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB ECG image.")
    if exclusion_mask.shape != image.shape[:2]:
        raise ValueError("The exclusion mask must match the ECG image.")

    height, width = image.shape[:2]
    kernel_size = _odd_kernel_size(
        min(height, width) * BACKGROUND_KERNEL_SHORT_EDGE_FRACTION
    )
    rgb = image.astype(np.float32) / 255.0
    luminance = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[..., 0].astype(
        np.float32
    ) / 255.0

    background = cv2.morphologyEx(
        luminance,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        ),
    )
    background = cv2.GaussianBlur(
        background,
        (0, 0),
        sigmaX=max(1.0, kernel_size / 6.0),
        sigmaY=max(1.0, kernel_size / 6.0),
    )
    local_darkness = _robust_unit_scale(
        np.clip(background - luminance, 0.0, None),
        low_quantile=0.10,
        high_quantile=0.995,
    )

    red, green, blue = (rgb[..., index] for index in range(3))
    warm_chroma = np.clip(red - (green + blue) / 2.0, 0.0, None)
    warm_chroma = _robust_unit_scale(
        warm_chroma,
        low_quantile=0.40,
        high_quantile=0.995,
    )

    row_dark = np.quantile(local_darkness, GRID_PROFILE_QUANTILE, axis=1)
    column_dark = np.quantile(local_darkness, GRID_PROFILE_QUANTILE, axis=0)
    row_warm = np.quantile(warm_chroma, GRID_PROFILE_QUANTILE, axis=1)
    column_warm = np.quantile(warm_chroma, GRID_PROFILE_QUANTILE, axis=0)
    row_strength = np.maximum(
        _profile_strength(row_dark),
        _profile_strength(row_warm),
    )
    column_strength = np.maximum(
        _profile_strength(column_dark),
        _profile_strength(column_warm),
    )
    grid_geometry = np.maximum(row_strength[:, None], column_strength[None, :])
    grid_probability = np.clip(
        grid_geometry * (0.35 + 0.65 * warm_chroma),
        0.0,
        1.0,
    )

    predicted_grid_darkness = np.maximum(
        row_dark[:, None],
        column_dark[None, :],
    )
    residual_darkness = np.clip(
        (local_darkness - predicted_grid_darkness * 0.92) * 3.0,
        0.0,
        1.0,
    )
    neutral_darkness = local_darkness * (1.0 - 0.80 * warm_chroma)
    # Strong neutral ink remains evidence even when it is horizontal or
    # vertical. The nonlinear term suppresses lighter grayscale grid lines
    # without deleting genuinely dark ST/baseline or QRS strokes.
    colour_path = neutral_darkness**2.4 * (
        0.85 + 0.15 * (1.0 - grid_geometry)
    )
    trace_probability = np.maximum(residual_darkness, colour_path * 0.85)
    trace_probability = cv2.GaussianBlur(
        trace_probability,
        (3, 3),
        0.45,
    )
    trace_probability[exclusion_mask] = 0.0

    background_flattened = np.clip(
        1.0 - local_darkness * 0.88,
        0.0,
        1.0,
    )
    # The model-ready image retains a faint grid for scale/layout evidence and
    # renders the source-derived trace probability as dark ink.
    enhanced = np.clip(
        1.0 - trace_probability * 0.94 - grid_probability * 0.10,
        0.0,
        1.0,
    )
    enhanced[exclusion_mask] = 1.0

    return {
        "backgroundKernelPixels": kernel_size,
        "backgroundFlattened": np.rint(background_flattened * 255).astype(
            np.uint8
        ),
        "traceProbability": np.rint(trace_probability * 255).astype(np.uint8),
        "gridProbability": np.rint(grid_probability * 255).astype(np.uint8),
        "enhanced": np.rint(enhanced * 255).astype(np.uint8),
        "traceProbabilityP95": float(np.quantile(trace_probability, 0.95)),
        "gridProbabilityP95": float(np.quantile(grid_probability, 0.95)),
    }


def _ordered_quadrilateral(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    coordinate_sum = points.sum(axis=1)
    coordinate_difference = np.diff(points, axis=1).ravel()
    return np.asarray(
        [
            points[np.argmin(coordinate_sum)],
            points[np.argmin(coordinate_difference)],
            points[np.argmax(coordinate_sum)],
            points[np.argmax(coordinate_difference)],
        ],
        dtype=np.float32,
    )


def _consensus_horizontal_rotation(
    gray: np.ndarray,
) -> tuple[float, float, int]:
    """Return the median near-horizontal long-line angle and its MAD."""

    blurred = cv2.GaussianBlur(gray, (5, 5), 0.8)
    edges = cv2.Canny(blurred, 40, 130)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
    )
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 1800.0,
        threshold=max(30, round(gray.shape[1] * 0.10)),
        minLineLength=max(40, round(gray.shape[1] * 0.20)),
        maxLineGap=max(8, round(gray.shape[1] * 0.025)),
    )
    angles: list[float] = []
    if lines is not None:
        for x0, y0, x1, y1 in lines[:, 0]:
            angle = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
            if -MAX_DESKEW_DEGREES <= angle <= MAX_DESKEW_DEGREES:
                angles.append(angle)
    rotation = float(np.median(angles)) if len(angles) >= 4 else 0.0
    angle_mad = (
        float(np.median(np.abs(np.asarray(angles) - rotation)))
        if angles
        else float("inf")
    )
    return rotation, angle_mad, len(angles)


def _expanded_rotation_transform(
    width: int,
    height: int,
    rotation: float,
) -> tuple[np.ndarray, int, int]:
    center = (width / 2.0, height / 2.0)
    affine = cv2.getRotationMatrix2D(center, rotation, 1.0)
    cosine = abs(float(affine[0, 0]))
    sine = abs(float(affine[0, 1]))
    output_width = max(1, int(np.ceil(height * sine + width * cosine)))
    output_height = max(1, int(np.ceil(height * cosine + width * sine)))
    affine[0, 2] += output_width / 2.0 - center[0]
    affine[1, 2] += output_height / 2.0 - center[1]
    return (
        np.vstack([affine, [0.0, 0.0, 1.0]]),
        output_width,
        output_height,
    )


def detect_page_geometry(image: np.ndarray) -> dict[str, object]:
    """Find a conservative page homography or small deskew rotation.

    Geometry correction is applied only when a large, convex page boundary is
    explicit or when several long near-horizontal lines agree on the same
    rotation. The original image remains untouched and the transform is fully
    recorded for audit.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    scale = min(1.0, 1600.0 / max(height, width))
    if scale < 1.0:
        small = cv2.resize(
            gray,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = gray
    blurred = cv2.GaussianBlur(small, (5, 5), 0.8)
    edges = cv2.Canny(blurred, 40, 130)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
    )
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    page_area = float(small.shape[0] * small.shape[1])
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, perimeter * 0.02, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        area_fraction = float(cv2.contourArea(polygon) / max(page_area, 1.0))
        if not MIN_PAGE_QUADRILATERAL_AREA_FRACTION <= area_fraction <= 0.995:
            continue
        candidates.append((area_fraction, polygon.reshape(4, 2).astype(np.float32)))

    if candidates:
        for area_fraction, small_points in sorted(
            candidates,
            key=lambda item: item[0],
            reverse=True,
        ):
            points = _ordered_quadrilateral(small_points / scale)
            top_left, top_right, bottom_right, bottom_left = points
            top_width = float(np.linalg.norm(top_right - top_left))
            bottom_width = float(np.linalg.norm(bottom_right - bottom_left))
            left_height = float(np.linalg.norm(bottom_left - top_left))
            right_height = float(np.linalg.norm(bottom_right - top_right))
            target_width = max(1, round(max(top_width, bottom_width)))
            target_height = max(1, round(max(left_height, right_height)))
            perspective_strength = max(
                abs(top_width - bottom_width) / max(top_width, bottom_width, 1.0),
                abs(left_height - right_height) / max(left_height, right_height, 1.0),
                abs(float(top_left[1] - top_right[1])) / max(height, 1),
                abs(float(bottom_left[1] - bottom_right[1])) / max(height, 1),
            )
            page_crop_supported = bool(
                area_fraction <= MAX_UNCROPPED_PAGE_AREA_FRACTION
            )
            confidence = float(
                min(1.0, area_fraction / 0.80)
                * min(
                    1.0,
                    min(
                        target_width / max(width, 1),
                        target_height / max(height, 1),
                    )
                    + 0.35,
                )
            )
            if not (
                (
                    perspective_strength >= MIN_PERSPECTIVE_STRENGTH
                    or page_crop_supported
                )
                and target_width >= 320
                and target_height >= 160
            ):
                continue

            method = (
                "large-convex-page-quadrilateral-homography-v2"
                if perspective_strength >= MIN_PERSPECTIVE_STRENGTH
                else "large-convex-page-boundary-crop-v2"
            )
            destination = np.asarray(
                [
                    [0, 0],
                    [target_width - 1, 0],
                    [target_width - 1, target_height - 1],
                    [0, target_height - 1],
                ],
                dtype=np.float32,
            )
            transform = cv2.getPerspectiveTransform(points, destination)
            # A page contour can follow a clipped photograph boundary rather
            # than the internal ECG grid.  Rectifying that contour alone may
            # leave several degrees of residual row slope.  Measure the
            # rectified raster and compose one conservative deskew when many
            # long lines still agree; this is one deterministic transform,
            # fully mapped back to the untouched source coordinates.
            preview_width = max(1, round(target_width * scale))
            preview_height = max(1, round(target_height * scale))
            preview_destination = np.asarray(
                [
                    [0, 0],
                    [preview_width - 1, 0],
                    [preview_width - 1, preview_height - 1],
                    [0, preview_height - 1],
                ],
                dtype=np.float32,
            )
            preview_transform = cv2.getPerspectiveTransform(
                _ordered_quadrilateral(small_points),
                preview_destination,
            )
            preview = cv2.warpPerspective(
                small,
                preview_transform,
                (preview_width, preview_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            residual_rotation, residual_mad, residual_line_count = (
                _consensus_horizontal_rotation(preview)
            )
            residual_applied = bool(
                residual_line_count >= 4
                and MIN_DESKEW_DEGREES
                <= abs(residual_rotation)
                <= MAX_DESKEW_DEGREES
                and residual_mad <= 0.75
            )
            if residual_applied:
                residual_transform, output_width, output_height = (
                    _expanded_rotation_transform(
                        target_width,
                        target_height,
                        residual_rotation,
                    )
                )
                transform = residual_transform @ transform
                target_width = output_width
                target_height = output_height
                method = f"{method}-plus-residual-deskew-v3"
            return {
                "applied": True,
                "method": method,
                "confidence": confidence,
                "perspectiveStrength": float(perspective_strength),
                "pageAreaFraction": float(area_fraction),
                "pageBoundaryCrop": page_crop_supported,
                "rotationDegrees": (
                    float(residual_rotation) if residual_applied else 0.0
                ),
                "residualDeskewApplied": residual_applied,
                "residualDeskewLineCount": int(residual_line_count),
                "residualDeskewAngleMad": (
                    float(residual_mad) if np.isfinite(residual_mad) else None
                ),
                "sourceCorners": points.tolist(),
                "outputWidth": target_width,
                "outputHeight": target_height,
                "transform": transform.tolist(),
            }

    rotation, angle_mad, angle_count = _consensus_horizontal_rotation(small)
    if (
        MIN_DESKEW_DEGREES <= abs(rotation) <= MAX_DESKEW_DEGREES
        and angle_mad <= 0.75
    ):
        transform, output_width, output_height = _expanded_rotation_transform(
            width,
            height,
            rotation,
        )
        return {
            "applied": True,
            "method": "consensus-long-grid-line-deskew-v2",
            "confidence": float(min(0.95, angle_count / 20.0) * max(0.0, 1.0 - angle_mad / 0.75)),
            "perspectiveStrength": 0.0,
            "rotationDegrees": rotation,
            "sourceCorners": None,
            "outputWidth": output_width,
            "outputHeight": output_height,
            "transform": transform.tolist(),
        }

    return {
        "applied": False,
        "method": "no-confident-page-transform-v1",
        "confidence": 0.0,
        "perspectiveStrength": 0.0,
        "rotationDegrees": 0.0,
        "sourceCorners": None,
        "outputWidth": width,
        "outputHeight": height,
        "transform": np.eye(3, dtype=np.float32).tolist(),
    }


def apply_page_geometry(
    image: np.ndarray,
    geometry: dict[str, object],
    *,
    nearest: bool = False,
) -> np.ndarray:
    if not geometry.get("applied"):
        return image.copy()
    transform = np.asarray(geometry["transform"], dtype=np.float32)
    output_size = (
        int(geometry["outputWidth"]),
        int(geometry["outputHeight"]),
    )
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    border_value: int | tuple[int, int, int]
    border_value = 0 if image.ndim == 2 and nearest else 255
    if image.ndim == 3:
        border_value = (255, 255, 255)
    return cv2.warpPerspective(
        image,
        transform,
        output_size,
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def classify_artifacts(image: np.ndarray) -> dict[str, object]:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    luminance = lab[..., 0] / 255.0
    chroma = lab[..., 1:] - 128.0
    chroma_high_frequency = chroma - cv2.GaussianBlur(chroma, (0, 0), 1.2)
    chroma_high_frequency_norm = np.linalg.norm(
        chroma_high_frequency,
        axis=2,
    )
    chroma_moire_score = float(
        np.quantile(chroma_high_frequency_norm, 0.95) / 128.0
    )
    chroma_moire_coverage = float(
        np.mean(
            chroma_high_frequency_norm
            >= SCREEN_CHROMA_MOIRE_THRESHOLD * 128.0
        )
    )
    neutral_chroma = np.linalg.norm(chroma, axis=2) <= 5.0
    glare = (luminance >= 0.985) & neutral_chroma
    glare_fraction = float(np.mean(glare))
    screen_likely = bool(
        chroma_moire_score >= SCREEN_CHROMA_MOIRE_THRESHOLD
        and chroma_moire_coverage >= SCREEN_CHROMA_MOIRE_MIN_COVERAGE
    )
    # A clean white ECG page is not glare. A highlight is eligible only when
    # it is a bounded bright region in an otherwise meaningfully darker photo.
    median_luminance = float(np.median(luminance))
    glare_likely = bool(
        GLARE_FRACTION_THRESHOLD <= glare_fraction <= 0.35
        and median_luminance <= 0.92
    )
    reasons: list[str] = []
    if screen_likely:
        reasons.append("periodic-screen-chroma")
    if glare_likely:
        reasons.append("broad-low-contrast-glare")
    if not reasons:
        reasons.append("no-supported-photo-artifact")
    return {
        "method": "chroma-periodicity-and-low-contrast-highlight-v3",
        "screenArtifactLikely": bool(screen_likely),
        "chromaMoireScore": chroma_moire_score,
        "chromaMoireThreshold": SCREEN_CHROMA_MOIRE_THRESHOLD,
        "chromaMoireCoverage": chroma_moire_coverage,
        "chromaMoireMinimumCoverage": SCREEN_CHROMA_MOIRE_MIN_COVERAGE,
        "glareLikely": bool(glare_likely),
        "glareFraction": glare_fraction,
        "glareFractionThreshold": GLARE_FRACTION_THRESHOLD,
        "medianLuminance": median_luminance,
        "reasons": reasons,
    }


def suppress_background_moire(
    image: np.ndarray,
    trace_probability: np.ndarray,
) -> np.ndarray:
    """Suppress screen/background texture while preserving source trace pixels."""

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    luminance = lab[..., 0].astype(np.float32)
    chroma = lab[..., 1:].astype(np.float32)
    smooth_luminance = cv2.GaussianBlur(luminance, (0, 0), 0.8)
    smooth_chroma = cv2.GaussianBlur(chroma, (0, 0), 1.2)
    trace_weight = np.clip(trace_probability.astype(np.float32) / 255.0, 0.0, 1.0)
    trace_weight = cv2.GaussianBlur(trace_weight, (3, 3), 0.45)[..., None]
    filtered = np.empty_like(lab)
    filtered[..., 0] = np.rint(
        luminance * trace_weight[..., 0]
        + (0.35 * luminance + 0.65 * smooth_luminance) * (1.0 - trace_weight[..., 0])
    ).astype(np.uint8)
    filtered[..., 1:] = np.rint(
        chroma * trace_weight + smooth_chroma * (1.0 - trace_weight)
    ).astype(np.uint8)
    return cv2.cvtColor(filtered, cv2.COLOR_LAB2RGB)


def classify_adaptive_preprocessing(
    *,
    width: int,
    height: int,
    sharpness_laplacian_variance: float,
    trace_probability_p95: float,
    grid_probability_p95: float,
) -> dict[str, bool | float | int | list[str]]:
    """Conservatively decide whether preprocessing may enter selection.

    Resolution loss and foreground/grid contamination are supported use cases.
    Blur remains a veto unless the page is strongly trace-dominant and the grid
    response is weak. That exception covers antialiased or thick-ink figure
    panels whose low Laplacian variance reflects smooth rendering rather than
    missing waveform detail.
    """

    long_edge = max(width, height)
    below_sharpness_floor = (
        sharpness_laplacian_variance
        < MIN_PREPROCESSING_SHARPNESS_LAPLACIAN_VARIANCE
    )
    low_resolution = long_edge < LOW_RESOLUTION_LONG_EDGE_PX
    foreground_grid_degraded = (
        trace_probability_p95 >= FOREGROUND_CONTAMINATION_TRACE_P95
        and grid_probability_p95 <= WEAK_GRID_PROBABILITY_P95
    )
    trace_dominant_smooth_source = (
        below_sharpness_floor
        and trace_probability_p95 >= TRACE_DOMINANT_SMOOTH_SOURCE_P95
        and grid_probability_p95
        <= TRACE_DOMINANT_SMOOTH_SOURCE_MAX_GRID_P95
    )
    blur_rejected = below_sharpness_floor and not trace_dominant_smooth_source
    eligible = bool(
        not blur_rejected
        and (low_resolution or foreground_grid_degraded)
    )
    reasons: list[str] = []
    if blur_rejected:
        reasons.append("blur-rejected")
    if trace_dominant_smooth_source:
        reasons.append("trace-dominant-smooth-source")
    if low_resolution:
        reasons.append("low-resolution")
    if foreground_grid_degraded:
        reasons.append("foreground-grid-degraded")
    if not reasons:
        reasons.append("clean-or-unsupported")

    return {
        "method": "conservative-resolution-sharpness-grid-gate-v2",
        "eligible": eligible,
        "reasons": reasons,
        "longEdgePixels": long_edge,
        "sharpnessLaplacianVariance": sharpness_laplacian_variance,
        "minimumSharpnessLaplacianVariance": (
            MIN_PREPROCESSING_SHARPNESS_LAPLACIAN_VARIANCE
        ),
        "lowResolution": low_resolution,
        "lowResolutionLongEdgeThresholdPixels": (
            LOW_RESOLUTION_LONG_EDGE_PX
        ),
        "foregroundGridDegraded": foreground_grid_degraded,
        "foregroundContaminationTraceP95Threshold": (
            FOREGROUND_CONTAMINATION_TRACE_P95
        ),
        "weakGridProbabilityP95Threshold": WEAK_GRID_PROBABILITY_P95,
        "belowSharpnessFloor": below_sharpness_floor,
        "traceDominantSmoothSource": trace_dominant_smooth_source,
        "traceDominantSmoothSourceP95Threshold": (
            TRACE_DOMINANT_SMOOTH_SOURCE_P95
        ),
        "traceDominantSmoothSourceMaxGridP95": (
            TRACE_DOMINANT_SMOOTH_SOURCE_MAX_GRID_P95
        ),
        "blurRejected": blur_rejected,
    }


def classify_input_quality(
    *,
    source_width: int,
    source_height: int,
    working_scale: float,
    adaptive_preprocessing: dict[str, object],
    artifact_preprocessing: dict[str, object],
) -> dict[str, object]:
    """Gate quantitative extraction on source information, not upscaled pixels."""

    original_long_edge = max(source_width, source_height)
    original_short_edge = min(source_width, source_height)
    aspect_ratio = original_long_edge / max(original_short_edge, 1)
    minimum_short_edge = (
        MIN_QUANTITATIVE_HIGH_ASPECT_SHORT_EDGE_PX
        if aspect_ratio >= QUANTITATIVE_HIGH_ASPECT_RATIO_THRESHOLD
        else MIN_QUANTITATIVE_SHORT_EDGE_PX
    )
    reasons: list[str] = []
    if original_long_edge < MIN_QUANTITATIVE_LONG_EDGE_PX:
        reasons.append("source-resolution-below-quantitative-floor")
    if original_short_edge < minimum_short_edge:
        reasons.append("source-short-edge-below-quantitative-floor")
    sharpness = float(
        adaptive_preprocessing.get("sharpnessLaplacianVariance", 0.0) or 0.0
    )
    if (
        adaptive_preprocessing.get("blurRejected") is True
        and sharpness < SEVERE_SOURCE_BLUR_LAPLACIAN_VARIANCE
    ):
        reasons.append("source-blur-below-quantitative-floor")
    quantitative_eligible = not reasons
    if artifact_preprocessing.get("screenArtifactLikely") is True:
        reasons.append("screen-capture-artifact-requires-extra-review")
    if artifact_preprocessing.get("glareLikely") is True:
        reasons.append("glare-requires-extra-review")
    outcome = (
        "insufficient"
        if not quantitative_eligible
        else "review"
        if reasons
        else "acceptable"
    )
    if not reasons:
        reasons.append("source-quality-gate-passed")
    return {
        "method": "source-resolution-blur-artifact-gate-v2",
        "outcome": outcome,
        "quantitativeEligible": quantitative_eligible,
        "reasons": reasons,
        "originalLongEdgePixels": original_long_edge,
        "originalShortEdgePixels": original_short_edge,
        "aspectRatio": aspect_ratio,
        "minimumQuantitativeLongEdgePixels": MIN_QUANTITATIVE_LONG_EDGE_PX,
        "minimumQuantitativeShortEdgePixels": MIN_QUANTITATIVE_SHORT_EDGE_PX,
        "minimumQuantitativeHighAspectShortEdgePixels": (
            MIN_QUANTITATIVE_HIGH_ASPECT_SHORT_EDGE_PX
        ),
        "quantitativeHighAspectRatioThreshold": (
            QUANTITATIVE_HIGH_ASPECT_RATIO_THRESHOLD
        ),
        "effectiveMinimumQuantitativeShortEdgePixels": minimum_short_edge,
        "severeSourceBlurLaplacianVariance": (
            SEVERE_SOURCE_BLUR_LAPLACIAN_VARIANCE
        ),
        "workingScale": working_scale,
    }


def write_outputs(
    source_path: Path,
    prepared_path: Path,
    mask_path: Path,
    enhanced_path: Path,
    background_path: Path,
    trace_probability_path: Path,
    grid_probability_path: Path,
    exclusion_mask_path: Path,
    geometry_corrected_path: Path,
    artifact_preprocessed_path: Path,
    report_path: Path,
    max_working_long_edge: int = DEFAULT_MAX_WORKING_LONG_EDGE_PX,
    working_source_path: Path | None = None,
) -> None:
    image, working_image = load_working_image(
        source_path,
        max_working_long_edge,
    )
    source_format = working_image["sourceFormat"]
    source_width = int(working_image["sourceWidth"])
    source_height = int(working_image["sourceHeight"])
    scale_x = float(working_image["scaleX"])
    scale_y = float(working_image["scaleY"])

    red_pixels, blue_pixels = color_candidates(image)
    mask, components = annotation_components(red_pixels, blue_pixels)

    prepared = suppress_annotations(image, mask)
    evidence = deterministic_evidence_maps(prepared, mask)
    page_geometry = detect_page_geometry(image)
    geometry_corrected = apply_page_geometry(prepared, page_geometry)
    corrected_mask = apply_page_geometry(
        mask.astype(np.uint8) * 255,
        page_geometry,
        nearest=True,
    ) > 0
    corrected_artifacts = classify_artifacts(geometry_corrected)
    corrected_evidence = deterministic_evidence_maps(
        geometry_corrected,
        corrected_mask,
    )
    artifact_source = (
        suppress_background_moire(
            geometry_corrected,
            np.asarray(corrected_evidence["traceProbability"]),
        )
        if corrected_artifacts["screenArtifactLikely"]
        else geometry_corrected
    )
    artifact_evidence = deterministic_evidence_maps(
        artifact_source,
        corrected_mask,
    )
    artifact_eligible = bool(
        page_geometry["applied"]
        or corrected_artifacts["screenArtifactLikely"]
        or corrected_artifacts["glareLikely"]
    )
    prepared_luminance = cv2.cvtColor(prepared, cv2.COLOR_RGB2GRAY)
    sharpness_laplacian_variance = float(
        cv2.Laplacian(
            prepared_luminance,
            cv2.CV_64F,
        ).var()
    )
    adaptive_preprocessing = classify_adaptive_preprocessing(
        width=source_width,
        height=source_height,
        sharpness_laplacian_variance=sharpness_laplacian_variance,
        trace_probability_p95=float(evidence["traceProbabilityP95"]),
        grid_probability_p95=float(evidence["gridProbabilityP95"]),
    )

    mask_preview = np.full_like(image, 248)
    mask_preview[mask] = np.array([245, 158, 11], dtype=np.uint8)
    mask_preview[red_pixels & mask] = np.array([220, 38, 38], dtype=np.uint8)
    mask_preview[blue_pixels & mask] = np.array([37, 99, 235], dtype=np.uint8)

    for destination in (
        *([working_source_path] if working_source_path is not None else []),
        prepared_path,
        mask_path,
        enhanced_path,
        background_path,
        trace_probability_path,
        grid_probability_path,
        exclusion_mask_path,
        geometry_corrected_path,
        artifact_preprocessed_path,
        report_path,
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)

    if working_source_path is not None:
        Image.fromarray(image).save(working_source_path, format="PNG")
    Image.fromarray(prepared).save(prepared_path, format="PNG")
    Image.fromarray(mask_preview).save(mask_path, format="PNG")
    Image.fromarray(
        np.repeat(
            np.asarray(evidence["enhanced"])[..., None],
            3,
            axis=2,
        )
    ).save(enhanced_path, format="PNG")
    Image.fromarray(
        np.asarray(evidence["backgroundFlattened"])
    ).save(background_path, format="PNG")
    Image.fromarray(
        np.asarray(evidence["traceProbability"])
    ).save(trace_probability_path, format="PNG")
    Image.fromarray(
        np.asarray(evidence["gridProbability"])
    ).save(grid_probability_path, format="PNG")
    Image.fromarray(mask.astype(np.uint8) * 255).save(
        exclusion_mask_path,
        format="PNG",
    )
    Image.fromarray(geometry_corrected).save(
        geometry_corrected_path,
        format="PNG",
    )
    Image.fromarray(
        np.repeat(
            np.asarray(artifact_evidence["enhanced"])[..., None],
            3,
            axis=2,
        )
    ).save(artifact_preprocessed_path, format="PNG")

    height, width = mask.shape
    masked_pixels = int(mask.sum())
    source_masked_pixels = int(
        round(masked_pixels / max(scale_x * scale_y, 1e-9))
    )
    source_components = components_in_source_coordinates(
        components,
        scale_x=scale_x,
        scale_y=scale_y,
        source_width=source_width,
        source_height=source_height,
    )
    input_quality = classify_input_quality(
        source_width=source_width,
        source_height=source_height,
        working_scale=min(scale_x, scale_y),
        adaptive_preprocessing=adaptive_preprocessing,
        artifact_preprocessing=corrected_artifacts,
    )
    report = {
        "version": MASK_VERSION,
        "sourceSha256": sha256(source_path),
        "source": {
            "width": source_width,
            "height": source_height,
            "format": source_format,
        },
        "workingImage": {
            "width": width,
            "height": height,
            "scaleX": scale_x,
            "scaleY": scale_y,
            "method": working_image["method"],
            "maxLongEdgePixels": working_image["maxLongEdgePixels"],
        },
        "annotationMask": {
            "maskedPixels": source_masked_pixels,
            "maskedFraction": masked_pixels / mask.size,
            "components": source_components,
        },
        "preparedImage": {
            "method": "colour-separated-saturated-component-mask-with-local-background-fill",
            "maskDilationPixels": MASK_DILATION_PX,
            "fillRingPixels": FILL_RING_PX,
            "fillPercentile": FILL_PERCENTILE,
            "morphologyReconstructed": False,
        },
        "evidenceMaps": {
            "method": "background-flattened-colour-aware-periodic-grid-residual-v1",
            "backgroundKernelPixels": evidence["backgroundKernelPixels"],
            "backgroundKernelShortEdgeFraction": (
                BACKGROUND_KERNEL_SHORT_EDGE_FRACTION
            ),
            "gridProfileQuantile": GRID_PROFILE_QUANTILE,
            "traceProbabilityP95": evidence["traceProbabilityP95"],
            "gridProbabilityP95": evidence["gridProbabilityP95"],
            "excludedPixels": source_masked_pixels,
            "morphologyReconstructed": False,
        },
        "adaptivePreprocessing": adaptive_preprocessing,
        "geometryCorrection": {
            **page_geometry,
            "morphologyReconstructed": False,
        },
        "artifactPreprocessing": {
            **corrected_artifacts,
            "eligible": artifact_eligible,
            "backgroundMoireSuppressed": bool(
                corrected_artifacts["screenArtifactLikely"]
            ),
            "geometryCorrected": bool(page_geometry["applied"]),
            "traceProbabilityP95": artifact_evidence["traceProbabilityP95"],
            "gridProbabilityP95": artifact_evidence["gridProbabilityP95"],
            "morphologyReconstructed": False,
        },
        "inputQuality": input_quality,
    }
    report_path.write_text(f"{json.dumps(report, indent=2)}\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.inspect:
        print(json.dumps(inspect_source(args.input)))
        return
    write_outputs(
        args.input,
        args.prepared,
        args.mask,
        args.enhanced,
        args.background,
        args.trace_probability,
        args.grid_probability,
        args.exclusion_mask,
        args.geometry_corrected,
        args.artifact_preprocessed,
        args.report,
        args.max_working_long_edge,
        working_source_path=args.working_source,
    )


if __name__ == "__main__":
    main()
