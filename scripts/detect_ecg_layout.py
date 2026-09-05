from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecg_pipeline.lead_label_identity import attach_semantic_lead_identity


MIN_THREE_BY_FOUR_PANEL_INK_SUPPORT = 0.015


def _connected_components_with_stats(
    mask: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Return OpenCV-compatible 8-connected component statistics safely.

    OpenCV's default connected-component dispatcher selects the Bolelli
    implementation in the macOS arm64 build used by the local worker. That
    native implementation can segfault for detector masks instead of raising a
    Python exception. Guard empty windows, normalize the raster, and select the
    stable SAUF implementation explicitly so a bad native default cannot take
    down the worker process.
    """

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("Connected-component masks must be two-dimensional.")
    if binary.size == 0 or 0 in binary.shape:
        return (
            1,
            np.zeros(binary.shape, dtype=np.int32),
            np.zeros((1, 5), dtype=np.int32),
            np.zeros((1, 2), dtype=np.float64),
        )
    native_mask = np.ascontiguousarray(binary, dtype=np.uint8)
    return cv2.connectedComponentsWithStatsWithAlgorithm(
        native_mask,
        8,
        cv2.CV_32S,
        cv2.CCL_SAUF,
    )


def _trace_ink(gray: np.ndarray) -> np.ndarray:
    _, dark = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    height, width = gray.shape
    horizontal = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(9, width // 40), 1),
        ),
    )
    vertical = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(7, height // 30)),
        ),
    )
    return cv2.subtract(dark, cv2.max(horizontal, vertical))


def _row_profile(trace_ink: np.ndarray) -> np.ndarray:
    height = trace_ink.shape[0]
    profile = (trace_ink > 0).mean(axis=1).astype(np.float32)
    return cv2.GaussianBlur(
        profile[:, None],
        (1, 0),
        sigmaX=0,
        sigmaY=max(2.0, height / 100),
    ).ravel()


def _label_ink_mask(image: np.ndarray, threshold: float) -> np.ndarray:
    """Return compact dark ink without a coloured ECG-paper grid.

    On red/pink paper the grid can connect otherwise separate label glyphs
    into page-wide components.  Neutral dark ink isolates black labels and
    traces without attempting OCR.  Grayscale pages retain the historical
    threshold path because colour cannot provide independent separation.
    """

    gray = _as_gray(image)
    raw = (gray < threshold).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] < 3:
        return raw
    channels = image[..., :3].astype(np.int16)
    channel_max = channels.max(axis=2)
    channel_min = channels.min(axis=2)
    chroma = channel_max - channel_min
    chromatic_fraction = float(
        np.mean((chroma >= 30) & (channel_max >= 100))
    )
    if chromatic_fraction < 0.002:
        return raw
    neutral = ((channel_max < threshold) & (chroma <= 28)).astype(np.uint8)
    return neutral if np.count_nonzero(neutral) >= 8 else raw


def _seven_row_sequence(
    peaks: np.ndarray,
    profile: np.ndarray,
    height: int,
) -> tuple[np.ndarray | None, float]:
    if peaks.size < 7:
        return None, 0.0

    strongest = peaks[np.argsort(profile[peaks])[-12:]]
    best: np.ndarray | None = None
    best_score = float("-inf")

    for combination in itertools.combinations(np.sort(strongest).tolist(), 7):
        centers = np.asarray(combination, dtype=np.int32)
        spacing = np.diff(centers).astype(np.float64)
        median_spacing = float(np.median(spacing))
        if median_spacing <= 0:
            continue

        spacing_cv = float(np.std(spacing) / median_spacing)
        boundary_penalty = (
            max(0.0, centers[0] / height - 0.15)
            + max(0.0, 0.84 - centers[-1] / height)
        )
        strength = float(
            np.mean(profile[centers]) / max(float(profile.max()), 1e-9)
        )
        score = strength - spacing_cv * 1.8 - boundary_penalty * 4
        if score > best_score:
            best = centers
            best_score = score

    return best, best_score


def _six_plus_rhythm_row_sequence(
    peaks: np.ndarray,
    profile: np.ndarray,
    height: int,
) -> tuple[np.ndarray | None, float]:
    """Select six primary rows followed by a potentially separated rhythm row.

    Some ECG systems add extra whitespace before the long rhythm strip. Using
    the spacing variation across all seven rows made those pages look like a
    three-row layout. The first six rows remain regular, while the last gap is
    allowed to be wider but is still bounded explicitly.
    """

    if peaks.size < 7:
        return None, 0.0
    strongest = peaks[np.argsort(profile[peaks])[-14:]]
    best: np.ndarray | None = None
    best_score = float("-inf")
    for combination in itertools.combinations(np.sort(strongest).tolist(), 7):
        centers = np.asarray(combination, dtype=np.int32)
        primary_spacing = np.diff(centers[:6]).astype(np.float64)
        median_spacing = float(np.median(primary_spacing))
        if median_spacing <= 0:
            continue
        primary_spacing_cv = float(np.std(primary_spacing) / median_spacing)
        rhythm_gap_ratio = float(
            (centers[6] - centers[5]) / max(median_spacing, 1e-9)
        )
        if not 0.62 <= rhythm_gap_ratio <= 2.55:
            continue
        boundary_penalty = (
            max(0.0, centers[0] / height - 0.26)
            + max(0.0, 0.74 - centers[-1] / height)
        )
        rhythm_gap_penalty = max(0.0, rhythm_gap_ratio - 1.35) * 0.20
        strength = float(
            np.mean(profile[centers]) / max(float(profile.max()), 1e-9)
        )
        score = (
            strength
            - primary_spacing_cv * 1.8
            - rhythm_gap_penalty
            - boundary_penalty * 4.0
        )
        if score > best_score:
            best = centers
            best_score = score
    return best, best_score


def _six_row_sequence(
    peaks: np.ndarray,
    profile: np.ndarray,
    height: int,
) -> tuple[np.ndarray | None, float]:
    if peaks.size < 6:
        return None, 0.0

    strongest = peaks[np.argsort(profile[peaks])[-12:]]
    best: np.ndarray | None = None
    best_score = float("-inf")
    for combination in itertools.combinations(np.sort(strongest).tolist(), 6):
        centers = np.asarray(combination, dtype=np.int32)
        spacing = np.diff(centers).astype(np.float64)
        median_spacing = float(np.median(spacing))
        if median_spacing <= 0:
            continue
        spacing_cv = float(np.std(spacing) / median_spacing)
        boundary_penalty = (
            max(0.0, centers[0] / height - 0.20)
            + max(0.0, 0.78 - centers[-1] / height)
        )
        strength = float(
            np.mean(profile[centers]) / max(float(profile.max()), 1e-9)
        )
        score = strength - spacing_cv * 1.8 - boundary_penalty * 4
        if score > best_score:
            best = centers
            best_score = score
    return best, best_score


def _twelve_row_sequence(
    peaks: np.ndarray,
    profile: np.ndarray,
    height: int,
) -> tuple[np.ndarray | None, float]:
    """Select a page-spanning twelve-row sequence from dense ink peaks."""

    if peaks.size < 12:
        return None, 0.0
    strongest = peaks[np.argsort(profile[peaks])[-18:]]
    best: np.ndarray | None = None
    best_score = float("-inf")
    for combination in itertools.combinations(np.sort(strongest).tolist(), 12):
        centers = np.asarray(combination, dtype=np.int32)
        spacing = np.diff(centers).astype(np.float64)
        median_spacing = float(np.median(spacing))
        if median_spacing <= 0:
            continue
        spacing_cv = float(np.std(spacing) / median_spacing)
        coverage = float((centers[-1] - centers[0]) / max(height, 1))
        boundary_penalty = (
            max(0.0, centers[0] / height - 0.16)
            + max(0.0, 0.82 - centers[-1] / height)
        )
        strength = float(
            np.mean(profile[centers]) / max(float(profile.max()), 1e-9)
        )
        score = (
            strength
            - spacing_cv * 1.45
            - boundary_penalty * 4.0
            + min(coverage, 0.90) * 0.25
        )
        if score > best_score:
            best = centers
            best_score = score
    return best, best_score


def _four_row_sequence(
    peaks: np.ndarray,
    profile: np.ndarray,
    height: int,
) -> tuple[np.ndarray | None, float]:
    if peaks.size < 4:
        return None, 0.0

    strongest = peaks[np.argsort(profile[peaks])[-10:]]
    best: np.ndarray | None = None
    best_score = float("-inf")

    for combination in itertools.combinations(np.sort(strongest).tolist(), 4):
        centers = np.asarray(combination, dtype=np.int32)
        spacing = np.diff(centers).astype(np.float64)
        median_spacing = float(np.median(spacing))
        if median_spacing <= 0:
            continue

        spacing_cv = float(np.std(spacing) / median_spacing)
        boundary_penalty = (
            max(0.0, centers[0] / height - 0.20)
            + max(0.0, 0.78 - centers[-1] / height)
        )
        strength = float(np.mean(profile[centers]) / max(float(profile.max()), 1e-9))
        score = strength - spacing_cv * 2.0 - boundary_penalty * 4
        if score > best_score:
            best = centers
            best_score = score

    return best, best_score


def _three_row_sequence(
    peaks: np.ndarray,
    profile: np.ndarray,
    height: int,
) -> tuple[np.ndarray | None, float]:
    if peaks.size < 3:
        return None, 0.0

    strongest = peaks[np.argsort(profile[peaks])[-8:]]
    best: np.ndarray | None = None
    best_score = float("-inf")

    for combination in itertools.combinations(np.sort(strongest).tolist(), 3):
        centers = np.asarray(combination, dtype=np.int32)
        spacing = np.diff(centers).astype(np.float64)
        median_spacing = float(np.median(spacing))
        if median_spacing <= 0:
            continue

        spacing_cv = float(np.std(spacing) / median_spacing)
        boundary_penalty = (
            max(0.0, centers[0] / height - 0.20)
            + max(0.0, 0.72 - centers[-1] / height)
        )
        strength = float(np.mean(profile[centers]) / max(float(profile.max()), 1e-9))
        score = strength - spacing_cv * 2.0 - boundary_penalty * 4
        if score > best_score:
            best = centers
            best_score = score

    return best, best_score


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 2:
        return image
    raise ValueError("Expected a grayscale or BGR ECG raster.")


def detect_ecg_content_box(image: np.ndarray) -> dict[str, Any]:
    """Return a conservative crop around wide waveform-bearing ink bands."""

    gray = _as_gray(image)
    height, width = gray.shape
    result: dict[str, Any] = {
        "contentBox": None,
        "contentConfidence": 0.0,
    }
    if height < 120 or width < 240:
        return result

    trace_ink = _trace_ink(gray)
    connected = cv2.morphologyEx(
        trace_ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(11, width // 45), 1),
        ),
    )
    connected = cv2.dilate(
        connected,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(3, width // 180), max(3, height // 90)),
        ),
    )
    contours, _ = cv2.findContours(
        connected,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    boxes = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < width * 0.28 or box_height < 3:
            continue
        boxes.append((x, y, x + box_width, y + box_height))
    if len(boxes) < 3:
        return result

    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    horizontal_padding = max(4, round(width * 0.025))
    vertical_padding = max(4, round(height * 0.035))
    left = max(0, left - horizontal_padding)
    top = max(0, top - vertical_padding)
    right = min(width, right + horizontal_padding)
    bottom = min(height, bottom + vertical_padding)
    retained_width = right - left
    retained_height = bottom - top
    if retained_width < width * 0.45 or retained_height < height * 0.2:
        return result

    wide_band_fraction = min(1.0, len(boxes) / 6.0)
    retained_fraction = (retained_width * retained_height) / (width * height)
    confidence = wide_band_fraction * min(1.0, retained_width / (width * 0.8))
    result.update(
        {
            "contentBox": {
                "left": int(left),
                "top": int(top),
                "right": int(right),
                "bottom": int(bottom),
            },
            "contentConfidence": float(confidence),
            "contentRetainedFraction": float(retained_fraction),
        }
    )
    return result


def _six_row_panel_score(image: np.ndarray) -> float:
    """Score a crop as one six-row, full-width ECG panel."""

    gray = _as_gray(image)
    height, width = gray.shape
    if height < 120 or width < 120:
        return 0.0
    trace_ink = _trace_ink(gray)
    profile = _row_profile(trace_ink)
    dense_peaks, _ = find_peaks(
        profile,
        distance=max(8, round(height / 16.0)),
        prominence=max(0.003, float(np.ptp(profile)) * 0.025),
    )
    # A continuous 12x1 page can look like two six-row panels when it happens
    # to contain a vertical whitespace band. At the coarser six-row spacing the
    # peak finder then selects every other row. Reject the crop when the finer
    # profile reveals more than eight distinct waveform rows.
    if dense_peaks.size > 8:
        return 0.0
    peaks, _ = find_peaks(
        profile,
        distance=max(8, round(height / 9.0)),
        prominence=max(0.003, float(np.ptp(profile)) * 0.025),
    )
    centers, sequence_score = _six_row_sequence(peaks, profile, height)
    if centers is None:
        if 5 <= peaks.size <= 8:
            coverage = (peaks[-1] - peaks[0]) / max(height, 1)
            if coverage >= 0.58 and peaks[0] <= height * 0.32 and peaks[-1] >= height * 0.68:
                return float(min(0.30, coverage * 0.35))
        return 0.0
    spacing = np.diff(centers).astype(np.float64)
    median_spacing = float(np.median(spacing))
    spacing_cv = float(np.std(spacing) / max(median_spacing, 1e-9))
    coverage = (centers[-1] - centers[0]) / max(height, 1)
    if (
        spacing_cv > 0.28
        or coverage < 0.52
        or centers[0] > height * 0.30
        or centers[-1] < height * 0.70
    ):
        return 0.0
    return float(
        min(
            1.0,
            max(0.0, 1.0 - spacing_cv / 0.28)
            * min(1.0, coverage / 0.72)
            * min(1.0, max(0.0, sequence_score) / 0.55),
        )
    )


def _six_row_content_bounds(image: np.ndarray) -> tuple[int, int]:
    """Bound a six-row panel to its trace rows, retaining lead labels."""

    gray = _as_gray(image)
    height = gray.shape[0]
    profile = _row_profile(_trace_ink(gray))
    peaks, _ = find_peaks(
        profile,
        distance=max(8, round(height / 9.0)),
        prominence=max(0.003, float(np.ptp(profile)) * 0.025),
    )
    centers, _ = _six_row_sequence(peaks, profile, height)
    if centers is None:
        return 0, height
    spacing = float(np.median(np.diff(centers)))
    return (
        max(0, round(float(centers[0]) - spacing * 0.55)),
        min(height, round(float(centers[-1]) + spacing * 0.55)),
    )


def _compact_label_row_centers(
    image: np.ndarray,
    *,
    expected_trace_centers: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float]:
    """Locate six repeated compact lead labels in a panel's left margin."""

    gray = _as_gray(image)
    height, width = gray.shape
    band_width = max(24, round(width * 0.26))
    band_image = image[:, :band_width]
    band = gray[:, :band_width]
    # Retain antialiased label strokes after moderate optical blur. On colour
    # ECG paper `_label_ink_mask` rejects the red grid by chromaticity before
    # applying this luminance ceiling.
    threshold = min(225.0, max(80.0, float(np.quantile(band, 0.12))))
    mask = _label_ink_mask(band_image, threshold)
    count, _, stats, centroids = _connected_components_with_stats(mask)
    components: list[dict[str, float]] = []
    for index in range(1, count):
        _, _, component_width, component_height, area = stats[index]
        if not (
            max(3.0, height * 0.009) <= component_height <= height * 0.055
            and 1 <= component_width <= band_width * 0.30
            and area >= 4
        ):
            continue
        components.append(
            {
                "x": float(centroids[index][0]),
                "y": float(centroids[index][1]),
                "area": float(area),
                "width": float(component_width),
            }
        )
    components.sort(key=lambda item: item["y"])
    groups: list[list[dict[str, float]]] = []
    tolerance = max(4.0, height * 0.014)
    for component in components:
        if groups and abs(
            component["y"]
            - float(np.mean([item["y"] for item in groups[-1]]))
        ) <= tolerance:
            groups[-1].append(component)
        else:
            groups.append([component])
    candidates: list[dict[str, float]] = []
    for group in groups:
        # At native resolution V and its digit are normally separate glyphs.
        # Moderate blur can join them into a single, wider component; accept
        # that form only when its width is itself label-like.
        if len(group) < 2 and max(item["width"] for item in group) < max(
            8.0,
            band_width * 0.06,
        ):
            continue
        areas = np.asarray([item["area"] for item in group])
        candidates.append(
            {
                "y": float(
                    np.average(
                        [item["y"] for item in group],
                        weights=areas,
                    )
                ),
                "x": float(np.median([item["x"] for item in group])),
                "area": float(np.sum(areas)),
            }
        )
    if len(candidates) < 6:
        return None, 0.0
    candidates = sorted(candidates, key=lambda item: item["area"], reverse=True)[:12]
    best: np.ndarray | None = None
    best_score = float("-inf")
    maximum_area = max(item["area"] for item in candidates)
    expected = (
        np.asarray(expected_trace_centers, dtype=np.float64)
        if expected_trace_centers is not None
        else np.asarray([], dtype=np.float64)
    )
    expected_spacing = (
        float(np.median(np.diff(expected))) if expected.size == 6 else 0.0
    )
    for combination in itertools.combinations(
        sorted(candidates, key=lambda item: item["y"]),
        6,
    ):
        centers = np.asarray([item["y"] for item in combination])
        spacing = np.diff(centers)
        median_spacing = float(np.median(spacing))
        if median_spacing <= 0:
            continue
        spacing_cv = float(np.std(spacing) / median_spacing)
        coverage = float((centers[-1] - centers[0]) / max(height, 1))
        x_mad = float(
            np.median(
                np.abs(
                    np.asarray([item["x"] for item in combination])
                    - np.median([item["x"] for item in combination])
                )
            )
            / max(band_width, 1)
        )
        x_spread = float(
            np.quantile(
                np.abs(
                    np.asarray([item["x"] for item in combination])
                    - np.median([item["x"] for item in combination])
                ),
                0.90,
            )
            / max(band_width, 1)
        )
        strength = float(
            np.mean([item["area"] / maximum_area for item in combination])
        )
        boundary_penalty = (
            # Content cropping intentionally leaves the first label close to
            # the panel edge. Penalising the leading six percent made V1 lose
            # to an interior annotation on tightly cropped examples.
            max(0.0, 0.015 - centers[0] / height)
            + max(0.0, centers[-1] / height - 0.96)
        )
        alignment_error = 0.0
        if expected.size == 6 and expected_spacing > 0:
            translation = float(np.median(expected - centers))
            residuals = np.abs(centers + translation - expected)
            # Lead labels may sit above their waveform baselines, but all six
            # must share essentially the same offset. This rejects headers or
            # calibration markers substituted for only V1 while retaining a
            # consistently translated V1-V6 sequence.
            if (
                abs(translation) > expected_spacing * 0.75
                or float(np.max(residuals)) > expected_spacing * 0.16
            ):
                continue
            alignment_error = float(np.median(residuals) / expected_spacing)
        score = (
            strength * 1.8
            - spacing_cv * 0.8
            - x_mad * 3.0
            - x_spread * 4.0
            - boundary_penalty * 5.0
            - alignment_error * 2.0
            + min(coverage, 0.82) * 0.25
        )
        if coverage >= 0.55 and spacing_cv <= 0.42 and score > best_score:
            best = np.rint(centers).astype(np.int32)
            best_score = score
    if best is None:
        return None, 0.0
    return best, float(min(1.0, max(0.0, best_score)))


def _sequential_lead_label_row_centers(
    image: np.ndarray,
) -> tuple[np.ndarray | None, float, dict[str, Any]]:
    """Locate a standard twelve-lead sequence from its left-margin labels.

    Very large precordial complexes can force deliberately uneven vertical
    spacing and merge several waveform rows in the global ink profile.  The
    printed I/II/III/aV* and V1-V6 labels remain independent anchors.  This
    recogniser is intentionally narrow: it requires the Roman-label component
    progression in the upper block and six multi-component precordial labels
    spanning the lower block.
    """

    gray = _as_gray(image)
    height, width = gray.shape
    if height < 300 or width < 400:
        return None, 0.0, {}

    band_left = max(0, round(width * 0.015))
    band_right = min(width, round(width * 0.11))
    band = gray[:, band_left:band_right]
    # Keep the label mask below the grey paper grid.  Perspective photos can
    # put enough grid ink in this narrow band for the lower quantile to rise
    # above 180, at which point repeated grid intersections masquerade as
    # hundreds of one-pixel glyph components.
    threshold = min(170.0, max(80.0, float(np.quantile(band, 0.12))))
    mask = (band < threshold).astype(np.uint8)
    count, _, stats, centroids = _connected_components_with_stats(mask)
    components: list[dict[str, float]] = []
    for index in range(1, count):
        x, _, component_width, component_height, area = stats[index]
        page_x = float(centroids[index][0] + band_left)
        if not (
            3 <= component_height <= height * 0.03
            and 1 <= component_width <= max(5.0, width * 0.014)
            and area >= 4
            # Mild perspective makes the top label sit farther left than the
            # remaining eleven even after global deskew.  The full standard
            # component-count sequence below remains the semantic safeguard.
            and page_x >= width * 0.018
            and x + band_left + component_width <= width * 0.105
        ):
            continue
        components.append(
            {
                "x": page_x,
                "y": float(centroids[index][1]),
                "height": float(component_height),
                "width": float(component_width),
                "area": float(area),
            }
        )
    components.sort(key=lambda item: item["y"])

    groups: list[list[dict[str, float]]] = []
    tolerance = max(4.0, height * 0.0105)
    for component in components:
        if groups:
            group_center = float(
                np.average(
                    [item["y"] for item in groups[-1]],
                    weights=[item["area"] for item in groups[-1]],
                )
            )
        else:
            group_center = float("nan")
        if groups and abs(component["y"] - group_center) <= tolerance:
            groups[-1].append(component)
        else:
            groups.append([component])

    candidates: list[dict[str, float]] = []
    for group in groups:
        if (
            max(item["height"] for item in group) < height * 0.0065
            or sum(item["area"] for item in group) < 6
        ):
            continue
        maximum_component_width = max(item["width"] for item in group)
        maximum_component_height = max(item["height"] for item in group)
        minimum_component_width = min(item["width"] for item in group)
        merged_multi_glyph_like = bool(
            len(group) == 1
            and maximum_component_width >= max(7.0, width * 0.004)
            and maximum_component_width <= width * 0.010
            and maximum_component_height >= height * 0.008
            and maximum_component_width
            / max(maximum_component_height, 1.0)
            >= 0.50
        )
        # A small deskew/perspective resample can bridge two adjacent strokes
        # of the Roman ``III`` label through the paper grid.  Preserve that
        # explicit semantic anchor only for the characteristic two-component
        # shape: one compact double-stroke plus one ordinary narrow stroke.
        # The value is consumed only at the third position of a complete
        # I/II/III/aVR/aVL/aVF sequence below, so it cannot independently
        # establish lead identity.
        merged_roman_triple_like = bool(
            len(group) == 2
            and maximum_component_width
            >= max(6.0, width * 0.0035)
            and maximum_component_width <= width * 0.0075
            and minimum_component_width
            <= maximum_component_width * 0.65
            and maximum_component_height >= height * 0.008
            and 0.35
            <= maximum_component_width
            / max(maximum_component_height, 1.0)
            <= 0.85
        )
        candidates.append(
            {
                "y": float(
                    np.average(
                        [item["y"] for item in group],
                        weights=[item["area"] for item in group],
                    )
                ),
                "x": float(np.median([item["x"] for item in group])),
                "area": float(sum(item["area"] for item in group)),
                "componentCount": float(len(group)),
                "mergedMultiGlyphLike": merged_multi_glyph_like,
                "mergedRomanTripleLike": merged_roman_triple_like,
            }
        )
    if len(candidates) < 12:
        return None, 0.0, {}

    # Twelve-strip exports commonly use either two uneven six-lead blocks or
    # twelve nearly uniform rows.  In the latter format aVF can sit below 38%
    # of the page and V1 below 50%, so keep the label search wide enough for
    # both arrangements.  The component-count progression and the required
    # six-row sequences remain the semantic safeguards.
    limb_pool = [
        item for item in candidates if 0.04 <= item["y"] / height <= 0.48
    ]
    best_limb: tuple[dict[str, float], ...] | None = None
    best_limb_score = float("-inf")
    for combination in itertools.combinations(limb_pool, 6):
        ordered = tuple(sorted(combination, key=lambda item: item["y"]))
        centers = np.asarray([item["y"] for item in ordered])
        spacing = np.diff(centers)
        median_spacing = float(np.median(spacing))
        spacing_cv = float(np.std(spacing) / max(median_spacing, 1e-9))
        component_counts = [int(item["componentCount"]) for item in ordered]
        if not (
            0.03 * height <= median_spacing <= 0.09 * height
            and spacing_cv <= 0.42
            and 0.20 * height <= centers[-1] - centers[0] <= 0.44 * height
            and centers[0] <= 0.16 * height
            and centers[-1] >= 0.27 * height
            and component_counts[0] >= 1
            and component_counts[1] >= 2
            and (
                component_counts[2] >= 3
                or ordered[2]["mergedRomanTripleLike"]
            )
            and component_counts[3] >= 3
            and component_counts[4] >= 3
            and component_counts[5] >= 3
        ):
            continue
        area_strength = float(np.mean(np.log1p([item["area"] for item in ordered])))
        score = area_strength + sum(component_counts[:4]) * 0.08 - spacing_cv * 2.0
        if score > best_limb_score:
            best_limb = ordered
            best_limb_score = score
    if best_limb is None:
        return None, 0.0, {}

    limb_end = best_limb[-1]["y"]
    precordial_pool = [
        item
        for item in candidates
        if item["y"] >= limb_end + height * 0.035
        and item["y"] <= height * 0.96
        and (
            item["componentCount"] >= 2
            or item["mergedMultiGlyphLike"]
        )
    ]
    best_precordial: tuple[dict[str, float], ...] | None = None
    best_precordial_score = float("-inf")
    for combination in itertools.combinations(precordial_pool, 6):
        ordered = tuple(sorted(combination, key=lambda item: item["y"]))
        centers = np.asarray([item["y"] for item in ordered])
        spacing = np.diff(centers)
        if not (
            np.min(spacing) >= height * 0.035
            and np.max(spacing) <= height * 0.19
            and 0.35 * height <= centers[-1] - centers[0] <= 0.65 * height
            and centers[0] <= 0.58 * height
            and centers[-1] >= 0.86 * height
        ):
            continue
        x_values = np.asarray([item["x"] for item in ordered])
        x_steps = np.diff(x_values)
        backwards_x = float(np.sum(np.clip(-x_steps - 2.0, 0.0, None)))
        area_strength = float(np.mean(np.log1p([item["area"] for item in ordered])))
        component_strength = float(
            np.mean(
                [
                    max(
                        item["componentCount"],
                        2.0 if item["mergedMultiGlyphLike"] else 1.0,
                    )
                    for item in ordered
                ]
            )
        )
        score = area_strength + component_strength * 0.22 - backwards_x * 0.08
        if score > best_precordial_score:
            best_precordial = ordered
            best_precordial_score = score
    if best_precordial is None:
        return None, 0.0, {}

    selected = (*best_limb, *best_precordial)
    centers = np.rint([item["y"] for item in selected]).astype(np.int32)
    limb_spacing = np.diff(centers[:6]).astype(np.float64)
    precordial_spacing = np.diff(centers[6:]).astype(np.float64)
    confidence = min(
        0.95,
        0.55
        + max(0.0, 1.0 - float(np.std(limb_spacing) / np.median(limb_spacing)) / 0.42)
        * 0.20
        + min(1.0, float(np.median([item["componentCount"] for item in best_precordial])) / 3.0)
        * 0.20,
    )
    return (
        centers,
        float(confidence),
        {
            "passed": True,
            "order": "standard",
            "method": "roman-and-precordial-lead-label-anchors-v1",
            # Component geometry locates label-like anchors but does not OCR
            # their values. It must never establish semantic lead identity.
            "semanticIdentityConfirmed": False,
            "limbComponentCounts": [
                int(item["componentCount"]) for item in best_limb
            ],
            "limbMergedMultiGlyphRows": [
                "III"
                for index, item in enumerate(best_limb)
                if index == 2 and item["mergedRomanTripleLike"]
            ],
            "precordialComponentCounts": [
                int(item["componentCount"]) for item in best_precordial
            ],
            "precordialMergedMultiGlyphRows": [
                index + 1
                for index, item in enumerate(best_precordial)
                if item["mergedMultiGlyphLike"]
            ],
            "limbMedianSpacing": float(np.median(limb_spacing)),
            "precordialMedianSpacing": float(np.median(precordial_spacing)),
        },
    )


def detect_six_row_panel_geometry(image: np.ndarray) -> dict[str, Any]:
    """Detect six local rows, falling back to repeated lead-label geometry."""

    gray = _as_gray(image)
    height = gray.shape[0]
    trace_ink = _trace_ink(gray)
    profile = _row_profile(trace_ink)
    peaks, _ = find_peaks(
        profile,
        distance=max(8, round(height / 9.0)),
        prominence=max(0.003, float(np.ptp(profile)) * 0.025),
    )
    centers, sequence_score = _six_row_sequence(peaks, profile, height)
    waveform_confidence = 0.0
    if centers is not None:
        spacing = np.diff(centers).astype(np.float64)
        median_spacing = float(np.median(spacing))
        spacing_cv = float(np.std(spacing) / max(median_spacing, 1e-9))
        coverage = float((centers[-1] - centers[0]) / max(height, 1))
        if spacing_cv <= 0.32 and coverage >= 0.55:
            waveform_confidence = float(
                min(1.0, max(0.0, sequence_score) * (1.0 - spacing_cv / 0.32))
            )
    label_centers, label_confidence = _compact_label_row_centers(
        image,
        expected_trace_centers=(centers if waveform_confidence > 0 else None),
    )
    if label_centers is not None and label_confidence >= waveform_confidence:
        if centers is None or waveform_confidence <= 0:
            centers = label_centers
            method = "compact-lead-label-sequence-v1"
        else:
            # Labels establish row identity; waveform peaks establish the
            # extraction baselines. Do not use above-baseline text coordinates
            # as trace centers when both are available.
            method = "waveform-row-sequence-plus-compact-labels-v1"
        confidence = label_confidence
    elif centers is not None:
        method = "waveform-row-sequence-v1"
        confidence = waveform_confidence
    else:
        return {"rowCenters": [], "confidence": 0.0, "method": None}
    spacing = np.diff(centers).astype(np.float64)
    return {
        "rowCenters": centers.tolist(),
        "confidence": float(confidence),
        "method": method,
        "medianRowSpacing": float(np.median(spacing)),
        "rowSpacingCv": float(np.std(spacing) / max(float(np.median(spacing)), 1e-9)),
    }


def detect_standard_limb_label_sequence(
    image: np.ndarray,
    row_centers: list[int],
) -> dict[str, Any]:
    """Recognise the I, II, III, aVR, aVL, aVF label-width sequence.

    This is intentionally a narrow geometric recogniser rather than OCR. The
    three Roman-numeral labels must increase in width, share a right edge, and
    be followed by three wider multi-glyph augmented-lead labels.
    """

    if len(row_centers) != 6:
        return {"passed": False, "order": None, "confidence": 0.0}
    gray = _as_gray(image)
    height, width = gray.shape
    spacing = float(np.median(np.diff(row_centers)))
    band_width = max(20, round(width * 0.14))
    band_image = image[:, :band_width]
    band = gray[:, :band_width]
    threshold = min(180.0, max(70.0, float(np.quantile(band, 0.10))))
    # Blur lifts the antialiased outer strokes of otherwise dark text. Try a
    # second deterministic threshold; the neutral-ink mask still excludes the
    # chromatic ECG grid on colour paper.
    label_masks = [
        (
            candidate_threshold,
            _label_ink_mask(band_image, candidate_threshold),
        )
        for candidate_threshold in sorted({threshold, min(225.0, threshold + 40.0)})
    ]

    def hint_for_window(
        page_mask: np.ndarray,
        label_threshold: float,
        upper_fraction: float,
        lower_fraction: float,
    ) -> dict[str, Any]:
        signatures: list[dict[str, float]] = []
        for center in row_centers:
            y_start = max(0, round(center + spacing * upper_fraction))
            y_end = min(height, round(center + spacing * lower_fraction))
            mask = page_mask[y_start:y_end]
            count, _, stats, _ = _connected_components_with_stats(mask)
            components: list[tuple[int, int, int]] = []
            for index in range(1, count):
                x, _, component_width, component_height, area = stats[index]
                center_x = x + component_width / 2
                if (
                    component_height >= max(3.0, spacing * 0.04)
                    # Printed lead glyphs are much shorter than the
                    # calibration pulse.  The tighter height bound lets a
                    # blur-joined augmented label remain wide without pulling
                    # the pulse into its horizontal signature.
                    and component_height <= spacing * 0.18
                    # Mild optical blur can join the three glyphs in an
                    # augmented lead label into one component.  Keep the
                    # complete label while still excluding long waveform and
                    # grid runs from this narrow, above-baseline band.
                    and component_width <= band_width * 0.36
                    and band_width * 0.18 <= center_x <= band_width * 0.82
                    and area >= 4
                ):
                    components.append(
                        (int(x), int(x + component_width), int(area))
                    )
            if not components:
                return {"passed": False, "order": None, "confidence": 0.0}
            signatures.append(
                {
                    "left": float(min(item[0] for item in components)),
                    "right": float(max(item[1] for item in components)),
                    "width": float(
                        max(item[1] for item in components)
                        - min(item[0] for item in components)
                    ),
                    "componentCount": float(len(components)),
                }
            )

        widths = np.asarray([item["width"] for item in signatures])
        roman_growth = bool(
            widths[0] <= widths[1] * 0.70
            and widths[1] <= widths[2] * 0.82
        )
        roman_right_edges = np.asarray(
            [item["right"] for item in signatures[:3]]
        )
        right_edge_spread = float(
            np.ptp(roman_right_edges) / max(band_width, 1)
        )
        augmented_width_ratio = float(
            np.median(widths[3:]) / max(widths[2], 1.0)
        )
        augmented_complexity = float(
            np.median([item["componentCount"] for item in signatures[3:]])
        )
        blur_joined_augmented_labels = bool(
            augmented_complexity >= 1.0 and augmented_width_ratio >= 2.0
        )
        passed = bool(
            roman_growth
            # A modest horizontal drift is expected on photographed pages
            # with perspective. The width sequence and augmented-label
            # evidence remain the primary identity constraints.
            and right_edge_spread <= 0.20
            and augmented_width_ratio >= 1.25
            and (augmented_complexity >= 2.0 or blur_joined_augmented_labels)
        )
        confidence = (
            min(1.0, augmented_width_ratio / 1.6)
            * max(0.0, 1.0 - right_edge_spread / 0.20)
            if passed
            else 0.0
        )
        return {
            "passed": passed,
            "order": "standard" if passed else None,
            "confidence": float(confidence),
            "method": "roman-and-augmented-lead-label-geometry-v2",
            # Width and component-count geometry is not value recognition.
            "semanticIdentityConfirmed": False,
            "labelWidths": widths.tolist(),
            "labelLeftEdges": [item["left"] for item in signatures],
            "labelThreshold": float(label_threshold),
            "verticalWindow": [upper_fraction, lower_fraction],
        }

    hints = [
        hint_for_window(page_mask, label_threshold, *window)
        for label_threshold, page_mask in label_masks
        for window in ((-0.58, -0.08), (-0.24, 0.24))
    ]
    return max(
        hints,
        key=lambda hint: (
            bool(hint.get("passed")),
            float(hint.get("confidence", 0.0)),
        ),
    )


def detect_labeled_six_by_two_geometry(image: np.ndarray) -> dict[str, Any]:
    """Recover six-row 6 x 2 pages from repeated V-label geometry.

    The V1-V6 column provides row anchors even when deep complexes defeat the
    global row-profile detector. A reasonably matching waveform sequence or a
    separately recognised standard limb-label sequence is required.
    """

    gray = _as_gray(image)
    height, width = gray.shape
    fallback: dict[str, Any] = {
        "layoutHint": None,
        "confidence": 0.0,
        "rowCenters": [],
        "aspectRatio": width / max(height, 1),
    }
    if min(height, width) < 180:
        return fallback
    waveform = detect_six_row_panel_geometry(image)
    waveform_method = str(waveform.get("method") or "")
    waveform_independently_detected = waveform_method.startswith(
        "waveform-row-sequence"
    )
    # A label-only fallback is not independent waveform evidence. Treating
    # the same compact text sequence as both the V1-V6 anchor and the matching
    # trace sequence can turn a 3 x 4 page's twelve panel labels into a false
    # 6 x 2 layout. Label-only geometry remains usable when the separate
    # standard limb-label recogniser passes below.
    waveform_centers = np.asarray(
        waveform.get("rowCenters") or []
        if waveform_independently_detected
        else [],
        dtype=np.float64,
    )
    waveform_spacing = (
        float(np.median(np.diff(waveform_centers)))
        if waveform_centers.size == 6
        else 0.0
    )
    candidates: list[dict[str, Any]] = []
    # Search a bounded vertical strip for each possible V-label column.  The
    # older suffix crop handed `_compact_label_row_centers` a wide region; on
    # tall 6 x 2 pages its internal band then included the adjacent waveform.
    # Faded label glyphs could consequently join the trace into a row-wide
    # component and disappear from the compact-label filter.  A fourteen-percent
    # page window is wide enough for V1-V6 across the stepped search positions
    # while keeping the trace onset outside the compact-label sub-band
    # the label band.  The six-row spacing and independent limb/waveform
    # checks below remain required before the geometry is accepted.
    label_window_width = max(48, round(width * 0.14))
    for fraction in np.linspace(0.20, 0.60, 17):
        left = round(width * float(fraction))
        bounded_right = min(width, left + label_window_width)
        # Retain the historical suffix search as a second view.  It is useful
        # when blur spreads a large V glyph beyond the bounded strip, whereas
        # the bounded view prevents faded labels from merging into the trace.
        # Either view still has to pass the identical six-row spacing and
        # independent limb/waveform checks below.
        for right in (bounded_right, width):
            centers, label_confidence = _compact_label_row_centers(
                image[:, left:right],
                expected_trace_centers=(
                    waveform_centers if waveform_centers.size == 6 else None
                ),
            )
            if centers is None:
                continue
            spacing = float(np.median(np.diff(centers)))
            waveform_match = False
            waveform_error = 1.0
            resolved_centers = centers
            if waveform_centers.size == 6 and waveform_spacing > 0:
                translation = float(
                    np.median(
                        waveform_centers - centers.astype(np.float64)
                    )
                )
                differences = np.abs(
                    centers.astype(np.float64)
                    + translation
                    - waveform_centers
                )
                waveform_error = float(
                    np.median(differences) / waveform_spacing
                )
                waveform_match = bool(
                    abs(translation) <= waveform_spacing * 0.70
                    and waveform_error <= 0.14
                    and float(np.max(differences))
                    <= waveform_spacing * 0.16
                )
                if waveform_match:
                    resolved_centers = np.rint(waveform_centers).astype(
                        np.int32
                    )
            limb_hints = [
                detect_standard_limb_label_sequence(
                    image[:, round(width * offset_fraction) :],
                    centers.tolist(),
                )
                for offset_fraction in np.linspace(0.0, 0.10, 6)
            ]
            limb_hint = max(
                limb_hints,
                key=lambda hint: float(hint.get("confidence", 0.0)),
            )
            if not waveform_match and not limb_hint.get("passed"):
                continue
            confidence = min(
                0.90,
                max(
                    float(label_confidence) * 0.55
                    + float(waveform.get("confidence", 0.0)) * 0.35
                    + (0.20 if limb_hint.get("passed") else 0.0),
                    0.12,
                ),
            )
            candidates.append(
                {
                    "centers": resolved_centers,
                    "confidence": confidence,
                    "labelConfidence": float(label_confidence),
                    "waveformError": waveform_error,
                    "limbHint": limb_hint,
                    "labelSearchLeft": left,
                    "labelSearchRight": right,
                    "spacing": spacing,
                }
            )
    if not candidates:
        return fallback
    selected = max(
        candidates,
        key=lambda item: (
            item["confidence"],
            -item["waveformError"],
            item["labelConfidence"],
        ),
    )
    return {
        "layoutHint": "standard_6x2",
        "confidence": float(selected["confidence"]),
        "rowCenters": selected["centers"].tolist(),
        "medianRowSpacing": float(selected["spacing"]),
        "aspectRatio": width / max(height, 1),
        "method": "paired-standard-limb-and-v-label-geometry-v1",
        "precordialLabelValidation": {
            "passed": True,
            "method": "compact-lead-label-sequence-v1",
            # Repeated compact glyphs locate the right-panel rows but do not
            # distinguish the actual V1-V6 values or their order.
            "semanticIdentityConfirmed": False,
            "confidence": float(selected["labelConfidence"]),
            "labelSearchLeft": int(selected["labelSearchLeft"]),
            "labelSearchRight": int(selected["labelSearchRight"]),
        },
        "limbLabelValidation": selected["limbHint"],
    }


def prefer_labeled_six_by_two_geometry(
    labeled: dict[str, Any],
    rhythm: dict[str, Any],
) -> bool:
    """Prefer explicit V1-V6 row identity when it is independently strong.

    A seventh waveform-like row can make a compact 6 x 2 figure resemble a
    page with a rhythm strip. Strong repeated precordial labels are more
    specific for assigning the twelve primary leads; limb algebra remains a
    downstream publication requirement.
    """

    rhythm_centers = rhythm.get("rowCenters") or []
    explicit_rhythm_strip = bool(
        rhythm.get("layoutHint") == "standard_6x2_with_r1_ignored"
        and len(rhythm_centers) == 7
        and float(rhythm.get("confidence", 0.0)) >= 0.85
        and float(rhythm.get("bilateralInkFloor", 0.0)) >= 0.04
    )
    return bool(
        labeled.get("layoutHint") == "standard_6x2"
        and labeled.get("method")
        == "paired-standard-limb-and-v-label-geometry-v1"
        and len(labeled.get("rowCenters") or []) == 6
        and float(labeled.get("confidence", 0.0)) >= 0.60
        and (labeled.get("precordialLabelValidation") or {}).get("passed")
        and not explicit_rhythm_strip
        and (
            not rhythm.get("layoutHint")
            or float(labeled.get("confidence", 0.0))
            >= float(rhythm.get("confidence", 0.0)) - 0.35
        )
    )


def _central_whitespace_split(profile: np.ndarray) -> tuple[int, float]:
    """Locate a broad central whitespace gutter in a 1-D ink profile."""

    length = profile.size
    if length < 160:
        return 0, 0.0
    window = max(5, round(length * 0.025))
    smoothed = np.convolve(profile, np.ones(window) / window, mode="same")
    start = round(length * 0.28)
    end = round(length * 0.72)
    if end <= start:
        return 0, 0.0
    split = start + int(np.argmin(smoothed[start:end]))
    gutter = float(smoothed[split])
    left_ink = float(np.quantile(smoothed[:start], 0.65))
    right_ink = float(np.quantile(smoothed[end:], 0.65))
    flank = min(left_ink, right_ink)
    contrast = max(0.0, 1.0 - gutter / max(flank, 1e-9))
    edge_balance = min(split, length - split) / max(length * 0.5, 1)
    return split, float(contrast * edge_balance)


def detect_compound_ecg_panels(image: np.ndarray) -> dict[str, Any]:
    """Detect two independent six-lead panels separated by whitespace.

    The detector does not assign lead names. It only proposes two source
    regions. The label-aware Open-ECG identifier must validate the limb,
    Cabrera, and precordial mappings before a composite can be selected.
    """

    gray = _as_gray(image)
    height, width = gray.shape
    result: dict[str, Any] = {"compoundPanels": None}
    if height < 200 or width < 180:
        return result
    # A landscape/square 12 x 1 sheet can contain a broad blank vertical scar
    # while every row still continues across the page. Do not reinterpret it
    # as two six-row panels merely because the column profile has a gutter.
    twelve_row = detect_twelve_row_geometry(image)
    if (
        twelve_row.get("layoutHint") == "standard_12x1"
        and width / max(height, 1) >= 0.85
    ):
        return result
    trace_ink = _trace_ink(gray)
    candidates: list[dict[str, Any]] = []

    # Titles and panel letters can bridge a genuine gutter. Estimate the
    # vertical split from the six trace rows when those anchors are available.
    labeled_six = detect_labeled_six_by_two_geometry(image)
    labeled_centers = labeled_six.get("rowCenters") or []
    if len(labeled_centers) == 6:
        labeled_spacing = float(np.median(np.diff(labeled_centers)))
        profile_top = max(0, round(labeled_centers[0] - labeled_spacing * 0.55))
        profile_bottom = min(
            height,
            round(labeled_centers[-1] + labeled_spacing * 0.55),
        )
        column_profile = (trace_ink[profile_top:profile_bottom] > 0).mean(
            axis=0
        ).astype(np.float32)
        # Some publications paste two small ECG-paper panels on a white page.
        # Their title obscures the whitespace profile, but the paper itself is
        # a pair of dense rectangular grid bands. Recover those source boxes.
        paper_profile = (
            gray[profile_top:profile_bottom] < 245
        ).mean(axis=0).astype(np.float32)
        paper_mask = (paper_profile > 0.72).astype(np.uint8)[None, :]
        close_width = max(5, round(width * 0.06))
        paper_mask = cv2.morphologyEx(
            paper_mask,
            cv2.MORPH_CLOSE,
            np.ones((1, close_width), dtype=np.uint8),
        )[0]
        paper_indexes = np.flatnonzero(paper_mask)
        paper_runs = (
            [
                run
                for run in np.split(
                    paper_indexes,
                    np.flatnonzero(np.diff(paper_indexes) > 1) + 1,
                )
                if run.size >= width * 0.18
            ]
            if paper_indexes.size
            else []
        )
        if (
            len(paper_runs) == 2
            and paper_runs[1][0] - paper_runs[0][-1] >= width * 0.05
        ):
            # Labels have already supplied identity evidence. Trace only the
            # actual paper rectangles so page whitespace is not stretched
            # into invented signal time.
            first_left = int(paper_runs[0][0])
            first_right = int(paper_runs[0][-1] + 1)
            second_left = int(paper_runs[1][0])
            second_right = int(paper_runs[1][-1] + 1)
            grid_support = min(
                float(np.mean(paper_profile[paper_runs[0]])),
                float(np.mean(paper_profile[paper_runs[1]])),
            )
            confidence = min(
                0.95,
                float(labeled_six.get("confidence", 0.0)) * 0.45
                + grid_support * 0.35
                + float(
                    (
                        labeled_six.get("precordialLabelValidation") or {}
                    ).get("confidence", 0.0)
                )
                * 0.20,
            )
            candidates.append(
                {
                    "orientation": "side_by_side",
                    "first": {
                        "left": first_left,
                        "top": profile_top,
                        "right": first_right,
                        "bottom": profile_bottom,
                        "rowCenters": list(labeled_centers),
                        "rowMethod": "waveform-row-sequence-v1",
                        "leadOrderHint": labeled_six.get(
                            "limbLabelValidation"
                        ),
                    },
                    "second": {
                        "left": second_left,
                        "top": profile_top,
                        "right": second_right,
                        "bottom": profile_bottom,
                        "rowCenters": list(labeled_centers),
                        "rowMethod": "compact-lead-label-sequence-v1",
                        "leadOrderHint": {
                            "passed": False,
                            "order": None,
                            "confidence": 0.0,
                        },
                    },
                    "confidence": float(confidence),
                    "method": "separate-grid-panels-plus-label-geometry-v1",
                }
            )
    else:
        column_profile = (trace_ink > 0).mean(axis=0).astype(np.float32)
    split_x, gutter_confidence = _central_whitespace_split(column_profile)
    if gutter_confidence >= 0.35:
        # Lead names sit just inside the panel gutter; retain enough overlap
        # for the label recognizer without mixing the waveform columns.
        margin = max(8, round(width * 0.05))
        left_box = (0, 0, min(width, split_x + margin), height)
        right_box = (max(0, split_x - margin), 0, width, height)
        left_panel = image[:, left_box[0] : left_box[2]]
        right_panel = image[:, right_box[0] : right_box[2]]
        left_geometry = detect_six_row_panel_geometry(left_panel)
        right_geometry = detect_six_row_panel_geometry(right_panel)
        left_score = max(
            _six_row_panel_score(left_panel),
            float(left_geometry["confidence"]) * 0.72,
        )
        right_score = max(
            _six_row_panel_score(right_panel),
            float(right_geometry["confidence"]) * 0.72,
        )
        left_top, left_bottom = _six_row_content_bounds(left_panel)
        right_top, right_bottom = _six_row_content_bounds(right_panel)
        shared_top = max(left_top, right_top)
        shared_bottom = min(left_bottom, right_bottom)
        if shared_bottom - shared_top < height * 0.55:
            shared_top, shared_bottom = 0, height
        left_box = (left_box[0], shared_top, left_box[2], shared_bottom)
        right_box = (right_box[0], shared_top, right_box[2], shared_bottom)
        left_geometry = detect_six_row_panel_geometry(
            image[shared_top:shared_bottom, left_box[0] : left_box[2]]
        )
        right_geometry = detect_six_row_panel_geometry(
            image[shared_top:shared_bottom, right_box[0] : right_box[2]]
        )
        left_order = detect_standard_limb_label_sequence(
            image[shared_top:shared_bottom, left_box[0] : left_box[2]],
            left_geometry["rowCenters"],
        )
        right_order = detect_standard_limb_label_sequence(
            image[shared_top:shared_bottom, right_box[0] : right_box[2]],
            right_geometry["rowCenters"],
        )
        confidence = min(left_score, right_score) * gutter_confidence
        if confidence >= 0.08:
            candidates.append(
                {
                    "orientation": "side_by_side",
                    "first": {
                        "left": left_box[0],
                        "top": left_box[1],
                        "right": left_box[2],
                        "bottom": left_box[3],
                        "rowCenters": [
                            int(center + shared_top)
                            for center in left_geometry["rowCenters"]
                        ],
                        "rowMethod": left_geometry["method"],
                        "leadOrderHint": left_order,
                    },
                    "second": {
                        "left": right_box[0],
                        "top": right_box[1],
                        "right": right_box[2],
                        "bottom": right_box[3],
                        "rowCenters": [
                            int(center + shared_top)
                            for center in right_geometry["rowCenters"]
                        ],
                        "rowMethod": right_geometry["method"],
                        "leadOrderHint": right_order,
                    },
                    "confidence": float(confidence),
                }
            )

    row_profile = (trace_ink > 0).mean(axis=1).astype(np.float32)
    split_y, gutter_confidence = _central_whitespace_split(row_profile)
    full_trace_profile = _row_profile(trace_ink)
    full_dense_peaks, _ = find_peaks(
        full_trace_profile,
        distance=max(8, round(height / 16.0)),
        prominence=max(0.003, float(np.ptp(full_trace_profile)) * 0.025),
    )
    has_distinct_stacked_gutter = True
    if full_dense_peaks.size >= 10:
        crossing_index = int(np.searchsorted(full_dense_peaks, split_y))
        if 0 < crossing_index < full_dense_peaks.size:
            crossing_gap = float(
                full_dense_peaks[crossing_index]
                - full_dense_peaks[crossing_index - 1]
            )
            ordinary_gaps = np.diff(full_dense_peaks).astype(np.float64)
            median_gap = float(np.median(ordinary_gaps))
            has_distinct_stacked_gutter = crossing_gap >= median_gap * 1.45
    if gutter_confidence >= 0.35 and has_distinct_stacked_gutter:
        margin = max(2, round(height * 0.008))
        upper_box = (0, 0, width, min(height, split_y + margin))
        lower_box = (0, max(0, split_y - margin), width, height)
        upper_panel = image[upper_box[1] : upper_box[3], :]
        lower_panel = image[lower_box[1] : lower_box[3], :]
        upper_geometry = detect_six_row_panel_geometry(upper_panel)
        lower_geometry = detect_six_row_panel_geometry(lower_panel)
        upper_order = detect_standard_limb_label_sequence(
            upper_panel,
            upper_geometry["rowCenters"],
        )
        lower_order = detect_standard_limb_label_sequence(
            lower_panel,
            lower_geometry["rowCenters"],
        )
        upper_score = max(
            _six_row_panel_score(upper_panel),
            float(upper_geometry["confidence"]) * 0.72,
        )
        lower_score = max(
            _six_row_panel_score(lower_panel),
            float(lower_geometry["confidence"]) * 0.72,
        )
        confidence = min(upper_score, lower_score) * gutter_confidence
        if confidence >= 0.08:
            candidates.append(
                {
                    "orientation": "stacked",
                    "first": {
                        "left": upper_box[0],
                        "top": upper_box[1],
                        "right": upper_box[2],
                        "bottom": upper_box[3],
                        "rowCenters": [
                            int(center + upper_box[1])
                            for center in upper_geometry["rowCenters"]
                        ],
                        "rowMethod": upper_geometry["method"],
                        "leadOrderHint": upper_order,
                    },
                    "second": {
                        "left": lower_box[0],
                        "top": lower_box[1],
                        "right": lower_box[2],
                        "bottom": lower_box[3],
                        "rowCenters": [
                            int(center + lower_box[1])
                            for center in lower_geometry["rowCenters"]
                        ],
                        "rowMethod": lower_geometry["method"],
                        "leadOrderHint": lower_order,
                    },
                    "confidence": float(confidence),
                }
            )

    if candidates:
        result["compoundPanels"] = max(
            candidates,
            key=lambda candidate: candidate["confidence"],
        )
    return result


def _grid_period(profile: np.ndarray) -> tuple[float | None, float]:
    length = profile.size
    if length < 80:
        return None, 0.0
    values = profile.astype(np.float64)
    values -= cv2.GaussianBlur(
        values[:, None].astype(np.float32),
        (1, 0),
        sigmaX=0,
        sigmaY=max(8.0, length / 24),
    ).ravel()
    scale = float(np.linalg.norm(values))
    if scale <= 1e-9:
        return None, 0.0
    maximum = min(48, length // 8)
    correlations = np.asarray(
        [
            float(np.dot(values[:-lag], values[lag:]))
            / max(float(np.linalg.norm(values[:-lag]) * np.linalg.norm(values[lag:])), 1e-9)
            for lag in range(2, maximum + 1)
        ]
    )
    peaks, _ = find_peaks(correlations, prominence=0.03)
    if peaks.size == 0:
        return None, 0.0
    # Prefer the first credible repeat; major grid periods are harmonics.
    credible = [index for index in peaks if correlations[index] >= 0.18]
    if not credible:
        return None, 0.0
    index = credible[0]
    subpixel_offset = 0.0
    if 0 < index < correlations.size - 1:
        denominator = (
            correlations[index - 1]
            - 2.0 * correlations[index]
            + correlations[index + 1]
        )
        if abs(float(denominator)) > 1e-9:
            subpixel_offset = float(
                np.clip(
                    0.5
                    * (correlations[index - 1] - correlations[index + 1])
                    / denominator,
                    -0.5,
                    0.5,
                )
            )
    return (
        float(index + 2 + subpixel_offset),
        float(min(1.0, correlations[index])),
    )


def _chromatic_excess_planes(image: np.ndarray) -> list[np.ndarray]:
    if image.ndim != 3 or image.shape[2] < 3:
        return []
    blue, green, red = cv2.split(image[:, :, :3].astype(np.float32))
    return [
        np.clip(primary - (secondary_a + secondary_b) * 0.5, 0.0, 255.0)
        / 255.0
        for primary, secondary_a, secondary_b in (
            (red, green, blue),
            (green, red, blue),
            (blue, red, green),
        )
    ]


def detect_ecg_calibration(image: np.ndarray) -> dict[str, Any]:
    """Detect grid spacing and a rectangular 1 mV calibration pulse."""

    gray = _as_gray(image)
    height, width = gray.shape
    darkness = 1.0 - gray.astype(np.float32) / 255.0
    grayscale_periods = (
        *_grid_period(darkness.mean(axis=0)),
        *_grid_period(darkness.mean(axis=1)),
    )
    period_candidates = [grayscale_periods]
    if image.ndim == 3 and image.shape[2] >= 3:
        # ECG paper grids are commonly red/pink, blue/cyan, or green. A dark
        # waveform can dominate the grayscale row profile while the grid's
        # chromatic excess still supplies independent periodic evidence on
        # both axes. Test every channel symmetrically and retain the same
        # cross-axis agreement and calibration-pulse requirements below; a
        # coloured period alone never establishes a physical scale.
        for channel_excess in _chromatic_excess_planes(image):
            period_candidates.append(
                (
                    *_grid_period(channel_excess.mean(axis=0)),
                    *_grid_period(channel_excess.mean(axis=1)),
                )
            )

    grid_candidates: list[dict[str, float]] = []
    for x_period, x_confidence, y_period, y_confidence in period_candidates:
        if x_period is None or y_period is None:
            continue
        disagreement = abs(x_period - y_period) / max(
            x_period,
            y_period,
            1e-9,
        )
        if disagreement > 0.15:
            continue
        grid_candidates.append(
            {
                "xPeriod": float(x_period),
                "yPeriod": float(y_period),
                "xConfidence": float(x_confidence),
                "yConfidence": float(y_confidence),
                "confidence": float(
                    min(x_confidence, y_confidence) * (1.0 - disagreement)
                ),
            }
        )
    selected_grid = (
        max(grid_candidates, key=lambda candidate: candidate["confidence"])
        if grid_candidates
        else None
    )
    if selected_grid:
        x_period = selected_grid["xPeriod"]
        y_period = selected_grid["yPeriod"]
        x_confidence = selected_grid["xConfidence"]
        y_confidence = selected_grid["yConfidence"]
    else:
        x_period, x_confidence, y_period, y_confidence = grayscale_periods
    # `_grid_period` finds the first credible repeat, which is a 1 mm minor
    # period when minor lines survive the raster. At 12 pixels and above the
    # same observed repeat could instead be a sparse 5 mm major grid; pixel
    # magnitude alone cannot distinguish those physical scales. Keep such a
    # period as raw evidence, but do not publish pixels/mm unless a calibration
    # pulse below independently resolves the competing hypotheses.
    standalone_grid_scale_mm = (
        1.0
        if selected_grid is not None
        and max(selected_grid["xPeriod"], selected_grid["yPeriod"]) < 12.0
        else None
    )
    result: dict[str, Any] = {
        "calibration": {
            "method": "grid-period-plus-rectangular-pulse-v1",
            "detected": False,
            "gridSpacingXPixels": x_period,
            "gridSpacingYPixels": y_period,
            "gridScaleDetected": standalone_grid_scale_mm is not None,
            "gridScaleAmbiguous": bool(
                selected_grid is not None and standalone_grid_scale_mm is None
            ),
            "gridScaleConfidence": (
                selected_grid["confidence"]
                if standalone_grid_scale_mm is not None and selected_grid
                else 0.0
            ),
            **(
                {
                    "gridScaleMm": standalone_grid_scale_mm,
                    "gridScaleMmX": standalone_grid_scale_mm,
                    "gridScaleMmY": standalone_grid_scale_mm,
                    "pixelsPerMmX": selected_grid["xPeriod"]
                    / standalone_grid_scale_mm,
                    "pixelsPerMmY": selected_grid["yPeriod"]
                    / standalone_grid_scale_mm,
                }
                if standalone_grid_scale_mm is not None and selected_grid
                else {}
            ),
            "confidence": 0.0,
        }
    }
    if x_period is None or y_period is None:
        return result

    otsu_threshold, _ = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    _, dark = cv2.threshold(
        gray,
        min(160.0, max(40.0, otsu_threshold * 0.75)),
        255,
        cv2.THRESH_BINARY_INV,
    )
    minimum_vertical = max(8, round(y_period * 1.4))
    vertical = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, minimum_vertical)),
    )
    contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: list[tuple[float, int, int, int]] = []
    for contour in contours:
        x, y, line_width, line_height = cv2.boundingRect(contour)
        if line_height < minimum_vertical or line_width > max(5, round(x_period * 0.8)):
            continue
        lines.append((x + line_width / 2, y, y + line_height, line_width))

    best: dict[str, Any] | None = None
    for first, second in itertools.combinations(lines, 2):
        x0, y0, y1, first_line_width = first
        x1, other_y0, other_y1, second_line_width = second
        if x1 < x0:
            x0, x1 = x1, x0
            y0, y1, other_y0, other_y1 = other_y0, other_y1, y0, y1
            first_line_width, second_line_width = (
                second_line_width,
                first_line_width,
            )
        pulse_width_pixels = x1 - x0
        pulse_height_pixels = (
            min(y1, other_y1)
            - max(y0, other_y0)
            - max(first_line_width, second_line_width)
        )
        if pulse_width_pixels <= 0 or pulse_height_pixels <= 0:
            continue
        top = round(max(y0, other_y0))
        x_start = max(0, round(x0))
        x_end = min(width, round(x1) + 1)
        if x_end - x_start < 3:
            continue
        horizontal_support = float(np.mean(dark[max(0, top - 1) : min(height, top + 2), x_start:x_end] > 0))
        if horizontal_support < 0.45:
            continue

        grid_scales_mm_x = (1.0, 5.0) if x_period >= 12.0 else (1.0,)
        grid_scales_mm_y = (1.0, 5.0) if y_period >= 12.0 else (1.0,)
        for grid_scale_mm_x, grid_scale_mm_y in itertools.product(
            grid_scales_mm_x,
            grid_scales_mm_y,
        ):
            pixels_per_mm_x = x_period / grid_scale_mm_x
            pixels_per_mm_y = y_period / grid_scale_mm_y
            height_mm = pulse_height_pixels / pixels_per_mm_y
            width_mm = pulse_width_pixels / pixels_per_mm_x
            gain_candidates = (5.0, 10.0, 20.0)
            gain = min(gain_candidates, key=lambda value: abs(height_mm - value))
            gain_error = abs(height_mm - gain) / gain
            speed_candidates = (25.0, 50.0)
            speed = min(speed_candidates, key=lambda value: abs(width_mm / 0.2 - value))
            speed_error = abs(width_mm / 0.2 - speed) / speed
            if gain_error > 0.35 or speed_error > 0.35:
                continue
            # Once the standard rectangular pulse is validated, its known
            # 200 ms width and 1 mV height refine scale after non-integer image
            # resizing (where a 5.3 px minor grid can alias to a 5 px period).
            pulse_pixels_per_mm_x = pulse_width_pixels / (speed * 0.2)
            pulse_pixels_per_mm_y = pulse_height_pixels / gain
            refined_disagreement = abs(
                pulse_pixels_per_mm_x - pulse_pixels_per_mm_y
            ) / max(pulse_pixels_per_mm_x, pulse_pixels_per_mm_y, 1e-9)
            if refined_disagreement <= 0.15:
                refined_pixels_per_mm_x = pulse_pixels_per_mm_x
                refined_pixels_per_mm_y = pulse_pixels_per_mm_y
            else:
                refined_pixels_per_mm_x = pixels_per_mm_x
                refined_pixels_per_mm_y = pixels_per_mm_y
            confidence = (
                min(x_confidence, y_confidence)
                * horizontal_support
                * max(0.0, 1.0 - gain_error)
                * max(0.0, 1.0 - speed_error)
            )
            candidate = {
                "method": "grid-period-plus-rectangular-pulse-v1",
                "detected": True,
                "gridScaleDetected": True,
                "gridScaleAmbiguous": False,
                "gridScaleConfidence": float(confidence),
                **(
                    {"gridScaleMm": grid_scale_mm_x}
                    if grid_scale_mm_x == grid_scale_mm_y
                    else {}
                ),
                "gridScaleMmX": grid_scale_mm_x,
                "gridScaleMmY": grid_scale_mm_y,
                "gridSpacingXPixels": x_period,
                "gridSpacingYPixels": y_period,
                "pixelsPerMmX": float(refined_pixels_per_mm_x),
                "pixelsPerMmY": float(refined_pixels_per_mm_y),
                "paperSpeedMmPerSecond": speed,
                "gainMmPerMv": gain,
                "measuredGainMmPerMv": height_mm,
                "pulseWidthMm": width_mm,
                "pulseStartX": x_start,
                "pulseEndX": x_end,
                "confidence": float(confidence),
            }
            if best is None or candidate["confidence"] > best["confidence"]:
                best = candidate

    if best is not None:
        result["calibration"] = best
    return result


def horizontal_grid_scale_by_rows(
    image: np.ndarray,
    row_centers: list[int],
    *,
    grid_scale_mm: float | None = None,
) -> list[float] | None:
    """Measure local horizontal grid scale on mildly perspective pages."""

    if image.ndim != 3 or len(row_centers) not in {6, 12}:
        return None
    spacing = float(np.median(np.diff(row_centers)))
    if len(row_centers) == 6:
        blue, green, red = cv2.split(image[:, :, :3].astype(np.float32))
        grid_evidence = (
            np.clip(red - (green + blue) * 0.5, 0.0, 255.0) / 255.0
        )
        radius = max(8, round(spacing * 0.07))
    else:
        # Grey-grid twelve-strip exports have no useful red excess. A broad
        # row-local strip suppresses the waveform while retaining the repeated
        # vertical grid. Perspective changes that period smoothly by row.
        gray = _as_gray(image)
        grid_evidence = 1.0 - gray.astype(np.float32) / 255.0
        radius = max(12, round(spacing * 0.40))
    scales: list[float] = []
    for center in row_centers:
        top = max(0, int(center) - radius)
        bottom = min(image.shape[0], int(center) + radius + 1)
        period, confidence = _grid_period(
            grid_evidence[top:bottom].mean(axis=0)
        )
        if period is None or confidence < 0.18:
            if len(row_centers) == 6:
                return None
            scales.append(float("nan"))
            continue
        resolved_scale_mm = grid_scale_mm
        if resolved_scale_mm is None and period < 12.0:
            resolved_scale_mm = 1.0
        if resolved_scale_mm not in {1.0, 5.0}:
            if len(row_centers) == 6:
                return None
            scales.append(float("nan"))
            continue
        scales.append(float(period / resolved_scale_mm))
    if len(row_centers) == 12:
        values = np.asarray(scales, dtype=np.float64)
        centers = np.asarray(row_centers, dtype=np.float64)
        valid = np.isfinite(values) & (values >= 1.0) & (values <= 40.0)
        if int(np.count_nonzero(valid)) < 8:
            return None
        coefficients = np.polyfit(centers[valid], values[valid], 1)
        predicted = np.polyval(coefficients, centers)
        residuals = np.abs(values[valid] - predicted[valid])
        median_scale = float(np.median(values[valid]))
        residual_mad = float(np.median(residuals))
        inliers = valid.copy()
        inliers[valid] = residuals <= max(
            median_scale * 0.06,
            residual_mad * 3.5,
        )
        if int(np.count_nonzero(inliers)) < 8:
            return None
        coefficients = np.polyfit(centers[inliers], values[inliers], 1)
        predicted = np.polyval(coefficients, centers)
        inlier_residual = np.abs(values[inliers] - predicted[inliers])
        if (
            np.any(predicted < 1.0)
            or np.any(predicted > 40.0)
            or float(np.max(predicted) / np.min(predicted)) > 1.25
            or float(np.median(inlier_residual) / median_scale) > 0.04
        ):
            return None
        return [float(value) for value in predicted]
    median_scale = float(np.median(scales))
    if (
        not 1.0 <= median_scale <= 40.0
        or float(np.std(scales) / max(median_scale, 1e-9)) > 0.10
    ):
        return None
    return scales


def detect_twelve_row_geometry(image: np.ndarray) -> dict[str, Any]:
    """Detect twelve full-width, sequential ECG rows without model labels.

    This intentionally recognizes geometry only. Lead order is validated by
    the row-local digitizer using the simultaneous limb-lead relationships
    before any named CSV is eligible for publication.
    """

    gray = _as_gray(image)
    height, width = gray.shape
    aspect_ratio = width / max(height, 1)
    result: dict[str, Any] = {
        "layoutHint": None,
        "confidence": 0.0,
        "rowCenters": [],
        "aspectRatio": aspect_ratio,
    }
    if height < 240 or width < 180 or not 0.55 <= aspect_ratio <= 2.2:
        return result

    label_centers, label_confidence, label_validation = (
        _sequential_lead_label_row_centers(image)
    )
    if label_centers is not None:
        spacing = np.diff(label_centers).astype(np.float64)
        median_spacing = float(np.median(spacing))
        trace_centers = label_centers.astype(np.float64).copy()
        trace_ink = _trace_ink(gray)
        trace_left = round(width * 0.105)
        trace_right = max(trace_left + 3, round(width * 0.18))
        early_trace_profile = (
            trace_ink[:, trace_left:trace_right] > 0
        ).mean(axis=1).astype(np.float32)
        early_trace_profile = cv2.GaussianBlur(
            early_trace_profile[:, None],
            (1, 0),
            sigmaX=0,
            sigmaY=1.2,
        ).ravel()
        right_trace_profile = (
            trace_ink[:, round(width * 0.92) :] > 0
        ).mean(axis=1).astype(np.float32)
        right_trace_profile = cv2.GaussianBlur(
            right_trace_profile[:, None],
            (1, 0),
            sigmaX=0,
            sigmaY=1.2,
        ).ravel()
        trace_row_support = [1.0] * 12
        # Labels may be printed on the trace baseline or near the top of each
        # strip.  Resolve the actual waveform centre independently for every
        # row; using label centroids as the first six trace centres lets tall,
        # vertically aligned QRS complexes pull the path into adjacent rows.
        for index in range(12):
            search_start = int(label_centers[index])
            search_end = (
                int(label_centers[index + 1])
                if index < 11
                else height - 1
            )
            if search_end > search_start:
                early_local = early_trace_profile[search_start:search_end]
                early_offset = int(np.argmax(early_local))
                early_support = float(early_local[early_offset])
                trace_row_support[index] = early_support
                if early_support < 0.12:
                    right_local = right_trace_profile[search_start:search_end]
                    trace_centers[index] = search_start + int(
                        np.argmax(right_local)
                    )
                else:
                    trace_centers[index] = search_start + early_offset
        result.update(
            {
                "layoutHint": "standard_12x1",
                "confidence": float(label_confidence),
                "rowCenters": label_centers.tolist(),
                "traceRowCenters": np.rint(trace_centers)
                .astype(np.int32)
                .tolist(),
                "traceRowSupport": trace_row_support,
                "medianRowSpacing": median_spacing,
                "rowSpacingCv": float(
                    np.std(spacing) / max(median_spacing, 1e-9)
                ),
                "strongRowPeakCount": 12,
                "labelSupportedRowCount": 12,
                "method": "standard-and-precordial-lead-label-anchors-v1",
                "leadLabelValidation": label_validation,
            }
        )
        return result

    trace_ink = _trace_ink(gray)
    profile = _row_profile(trace_ink)
    prominence = max(0.0015, float(np.ptp(profile)) * 0.012)
    peaks, _ = find_peaks(
        profile,
        distance=max(5, round(height / 20.0)),
        prominence=prominence,
    )
    strong_floor = max(float(profile.max()) * 0.16, 0.0025)
    strong_peaks = peaks[profile[peaks] >= strong_floor]
    centers, sequence_score = _twelve_row_sequence(
        strong_peaks,
        profile,
        height,
    )
    if centers is None:
        return result

    spacing = np.diff(centers).astype(np.float64)
    median_spacing = float(np.median(spacing))
    spacing_cv = float(np.std(spacing) / max(median_spacing, 1e-9))
    coverage = float((centers[-1] - centers[0]) / max(height, 1))
    band_radius = max(2, round(height * 0.006))
    waveform_profile = (trace_ink[:, round(width * 0.16) :] > 0).mean(axis=1)
    label_dark = 1.0 - gray[:, : max(20, round(width * 0.16))].astype(np.float32) / 255.0
    label_profile = label_dark.mean(axis=1)

    def band_support(profile_values: np.ndarray, center: int) -> float:
        start = max(0, center - band_radius)
        end = min(height, center + band_radius + 1)
        return float(np.mean(profile_values[start:end]))

    waveform_support = np.asarray(
        [band_support(waveform_profile, int(center)) for center in centers]
    )
    label_support = np.asarray(
        [band_support(label_profile, int(center)) for center in centers]
    )
    waveform_floor = float(np.quantile(waveform_support, 0.10))
    label_supported_rows = int(
        np.count_nonzero(
            label_support
            >= max(float(np.median(label_profile)) * 1.12, 0.035)
        )
    )
    geometry_valid = (
        0.045 * height <= median_spacing <= 0.11 * height
        and spacing_cv <= 0.27
        and coverage >= 0.76
        and centers[0] <= 0.18 * height
        and centers[-1] >= 0.82 * height
        and 12 <= strong_peaks.size <= 16
        and waveform_floor >= 0.008
        and label_supported_rows >= 6
        and sequence_score >= 0.05
    )
    confidence = min(
        1.0,
        max(0.0, 1.0 - spacing_cv / 0.27)
        * min(1.0, coverage / 0.88)
        * min(1.0, waveform_floor / 0.035)
        * min(1.0, label_supported_rows / 10.0),
    )
    result.update(
        {
            "confidence": confidence,
            "rowCenters": centers.tolist(),
            "medianRowSpacing": median_spacing,
            "rowSpacingCv": spacing_cv,
            "strongRowPeakCount": int(strong_peaks.size),
            "waveformInkFloor": waveform_floor,
            "labelSupportedRowCount": label_supported_rows,
            "sequenceScore": sequence_score,
        }
    )
    if geometry_valid:
        result["layoutHint"] = "standard_12x1"
    return result


def detect_six_by_two_rhythm_geometry(image: np.ndarray) -> dict[str, Any]:
    gray = _as_gray(image)

    height, width = gray.shape
    aspect_ratio = width / max(height, 1)
    result: dict[str, Any] = {
        "layoutHint": None,
        "confidence": 0.0,
        "rowCenters": [],
        "aspectRatio": aspect_ratio,
    }
    if height < 160 or width < 320 or not 1.20 <= aspect_ratio <= 2.8:
        return result

    trace_ink = _trace_ink(gray)
    row_profile = _row_profile(trace_ink)
    prominence = max(0.004, float(np.ptp(row_profile)) * 0.035)
    peaks, _ = find_peaks(
        row_profile,
        distance=max(12, round(height / 10.5)),
        prominence=prominence,
    )
    centers, sequence_score = _six_plus_rhythm_row_sequence(
        peaks,
        row_profile,
        height,
    )
    if centers is None:
        return result

    primary_spacing = np.diff(centers[:6]).astype(np.float64)
    median_spacing = float(np.median(primary_spacing))
    spacing_cv = float(
        np.std(primary_spacing) / max(median_spacing, 1e-9)
    )
    rhythm_gap_ratio = float(
        (centers[6] - centers[5]) / max(median_spacing, 1e-9)
    )
    left_profile = (trace_ink[:, : width // 2] > 0).mean(axis=1)
    right_profile = (trace_ink[:, width // 2 :] > 0).mean(axis=1)
    band_radius = max(2, round(height * 0.012))

    def band_support(profile: np.ndarray, center: int) -> float:
        start = max(0, center - band_radius)
        end = min(height, center + band_radius + 1)
        return float(np.mean(profile[start:end]))

    left_support = np.asarray(
        [band_support(left_profile, int(center)) for center in centers]
    )
    right_support = np.asarray(
        [band_support(right_profile, int(center)) for center in centers]
    )
    bilateral_floor = min(
        float(np.min(left_support[:6])),
        float(np.min(right_support[:6])),
    )

    geometry_valid = (
        0.07 * height <= median_spacing <= 0.22 * height
        and spacing_cv <= 0.24
        and 0.62 <= rhythm_gap_ratio <= 2.55
        and centers[0] <= 0.26 * height
        and centers[5] >= 0.60 * height
        and centers[6] >= 0.74 * height
        and bilateral_floor >= 0.035
        and sequence_score >= 0.08
    )
    spacing_score = max(0.0, 1.0 - spacing_cv / 0.24)
    support_score = min(1.0, bilateral_floor / 0.08)
    sequence_confidence = min(1.0, max(0.0, sequence_score) / 0.8)
    confidence = float(
        np.cbrt(spacing_score * support_score * sequence_confidence)
    )

    result.update(
        {
            "confidence": confidence,
            "rowCenters": centers.tolist(),
            "medianRowSpacing": median_spacing,
            "rowSpacingCv": spacing_cv,
            "rhythmGapRatio": rhythm_gap_ratio,
            "bilateralInkFloor": bilateral_floor,
        }
    )
    if geometry_valid:
        result["layoutHint"] = "standard_6x2_with_r1_ignored"
    return result


def detect_calibration_anchored_six_by_two_rhythm_geometry(
    image: np.ndarray,
) -> dict[str, Any]:
    """Use seven repeated left-edge calibration pulses as row anchors.

    Very large QRS complexes can merge several waveform rows in the global
    ink profile. Repeated rectangular calibration pulses are independent of
    morphology and provide transparent row geometry for these pages.
    """

    gray = _as_gray(image)
    height, width = gray.shape
    fallback: dict[str, Any] = {
        "layoutHint": None,
        "confidence": 0.0,
        "rowCenters": [],
        "aspectRatio": width / max(height, 1),
    }
    if height < 220 or width < 320 or not 1.15 <= width / height <= 2.8:
        return fallback
    band_width = max(24, round(width * 0.08))
    band = gray[:, :band_width]
    vertical = cv2.morphologyEx(
        (band < 205).astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(6, round(height * 0.018))),
        ),
    )
    contours, _ = cv2.findContours(
        vertical,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    components: list[dict[str, float]] = []
    for contour in contours:
        x, y, component_width, component_height = cv2.boundingRect(contour)
        if not (
            height * 0.025 <= component_height <= height * 0.09
            and component_width <= max(4, round(width * 0.012))
            and x <= band_width * 0.78
        ):
            continue
        components.append(
            {
                "x": float(x + component_width / 2),
                "top": float(y),
                "bottom": float(y + component_height),
            }
        )
    components.sort(key=lambda item: item["top"])
    groups: list[list[dict[str, float]]] = []
    tolerance = max(3.0, height * 0.014)
    for component in components:
        if groups and abs(
            component["top"]
            - float(np.median([item["top"] for item in groups[-1]]))
        ) <= tolerance:
            groups[-1].append(component)
        else:
            groups.append([component])
    anchors: list[dict[str, float]] = []
    for group in groups:
        distinct_x = np.unique(
            np.rint([item["x"] for item in group]).astype(np.int32)
        )
        if distinct_x.size < 2:
            continue
        anchors.append(
            {
                "center": float(np.median([item["bottom"] for item in group])),
                "strength": float(min(1.0, distinct_x.size / 3.0)),
            }
        )
    if len(anchors) < 7:
        return fallback
    anchors = sorted(anchors, key=lambda item: item["strength"], reverse=True)[:14]
    best: tuple[np.ndarray, float, float, float] | None = None
    for combination in itertools.combinations(
        sorted(anchors, key=lambda item: item["center"]),
        7,
    ):
        centers = np.rint([item["center"] for item in combination]).astype(np.int32)
        primary_spacing = np.diff(centers[:6]).astype(np.float64)
        median_spacing = float(np.median(primary_spacing))
        if median_spacing <= 0:
            continue
        spacing_cv = float(np.std(primary_spacing) / median_spacing)
        rhythm_gap_ratio = float(
            (centers[6] - centers[5]) / max(median_spacing, 1e-9)
        )
        coverage = float((centers[-1] - centers[0]) / height)
        strength = float(np.mean([item["strength"] for item in combination]))
        if not (
            spacing_cv <= 0.16
            and 0.62 <= rhythm_gap_ratio <= 2.55
            and coverage >= 0.52
        ):
            continue
        score = strength - spacing_cv - abs(rhythm_gap_ratio - 1.0) * 0.08
        if best is None or score > best[1]:
            best = (centers, score, spacing_cv, rhythm_gap_ratio)
    if best is None:
        return fallback
    centers, score, spacing_cv, rhythm_gap_ratio = best
    trace_ink = _trace_ink(gray)
    left_profile = (trace_ink[:, : width // 2] > 0).mean(axis=1)
    right_profile = (trace_ink[:, width // 2 :] > 0).mean(axis=1)
    band_radius = max(2, round(height * 0.012))

    def band_support(profile: np.ndarray, center: int) -> float:
        start = max(0, center - band_radius)
        end = min(height, center + band_radius + 1)
        return float(np.mean(profile[start:end]))

    bilateral_floor = float(
        min(
            min(
                band_support(left_profile, int(center)),
                band_support(right_profile, int(center)),
            )
            for center in centers[:6]
        )
    )
    if bilateral_floor < 0.025:
        return fallback
    confidence = min(
        0.94,
        max(0.0, 1.0 - spacing_cv / 0.16) * 0.45
        + min(1.0, bilateral_floor / 0.08) * 0.30
        + max(0.0, score) * 0.25,
    )
    return {
        "layoutHint": "standard_6x2_with_r1_ignored",
        "confidence": float(confidence),
        "rowCenters": centers.tolist(),
        "aspectRatio": width / max(height, 1),
        "medianRowSpacing": float(np.median(np.diff(centers[:6]))),
        "rowSpacingCv": float(spacing_cv),
        "rhythmGapRatio": float(rhythm_gap_ratio),
        "bilateralInkFloor": bilateral_floor,
        "method": "repeated-calibration-pulse-row-anchors-v1",
        "calibrationRowAnchorCount": 7,
    }


def detect_calibration_anchored_three_by_four_geometry(
    image: np.ndarray,
) -> dict[str, Any]:
    """Recover three primary rows from repeated left calibration pulses.

    Large or clipped QRS complexes can dominate the global row profile and
    make a plainly visible 3 x 4 page look like the wrong row count. The three
    independently repeated 1 mV calibration rectangles remain stable in that
    situation. This fallback requires paired vertical pulse strokes, regular
    spacing, landscape page geometry, and waveform support in every primary
    panel; no waveform morphology is inferred from the anchors.
    """

    gray = _as_gray(image)
    height, width = gray.shape
    aspect_ratio = width / max(height, 1)
    fallback: dict[str, Any] = {
        "layoutHint": None,
        "confidence": 0.0,
        "rowCenters": [],
        "aspectRatio": aspect_ratio,
    }
    if height < 240 or width < 480 or not 2.0 <= aspect_ratio <= 4.2:
        return fallback

    band_width = max(36, round(width * 0.085))
    band = gray[:, :band_width]
    vertical = cv2.morphologyEx(
        (band < 205).astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(12, round(height * 0.055))),
        ),
    )
    contours, _ = cv2.findContours(
        vertical,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    components: list[dict[str, float]] = []
    for contour in contours:
        x, y, component_width, component_height = cv2.boundingRect(contour)
        if not (
            height * 0.05 <= component_height <= height * 0.19
            and component_width <= max(16, round(width * 0.016))
            and x <= band_width * 0.72
        ):
            continue
        components.append(
            {
                "x": float(x + component_width / 2),
                "top": float(y),
                # Morphology and antialiasing expand a vertical stroke around
                # its centreline. Recenter its lower endpoint so the anchor is
                # the trace baseline, not the outer edge of the ink stroke.
                "bottom": float(
                    y
                    + component_height
                    - max(1.0, component_width / 2.0)
                ),
            }
        )
    components.sort(key=lambda item: item["top"])
    groups: list[list[dict[str, float]]] = []
    tolerance = max(4.0, height * 0.02)
    for component in components:
        candidates = [
            group
            for group in groups
            if abs(
                component["top"]
                - float(np.median([item["top"] for item in group]))
            )
            <= tolerance
        ]
        if candidates:
            min(
                candidates,
                key=lambda group: abs(
                    component["top"]
                    - float(np.median([item["top"] for item in group]))
                ),
            ).append(component)
        else:
            groups.append([component])

    anchors: list[dict[str, float]] = []
    for group in groups:
        distinct_x = np.unique(
            np.rint([item["x"] for item in group]).astype(np.int32)
        )
        horizontal_span = float(np.ptp(distinct_x)) if distinct_x.size else 0.0
        if not (
            distinct_x.size >= 2
            and width * 0.006 <= horizontal_span <= width * 0.04
        ):
            continue
        anchors.append(
            {
                "center": float(np.median([item["bottom"] for item in group])),
                "strength": float(min(1.0, distinct_x.size / 2.0)),
            }
        )
    if len(anchors) < 3:
        return fallback

    best: tuple[np.ndarray, float, float] | None = None
    for combination in itertools.combinations(
        sorted(anchors, key=lambda item: item["center"]),
        3,
    ):
        centers = np.asarray(
            [item["center"] for item in combination],
            dtype=np.float64,
        )
        spacing = np.diff(centers)
        median_spacing = float(np.median(spacing))
        spacing_cv = float(np.std(spacing) / max(median_spacing, 1e-9))
        if not (
            0.16 * height <= median_spacing <= 0.40 * height
            and spacing_cv <= 0.12
            and centers[0] <= 0.35 * height
            and centers[2] >= 0.50 * height
        ):
            continue
        score = float(
            np.mean([item["strength"] for item in combination]) - spacing_cv
        )
        if best is None or score > best[1]:
            best = (centers, score, spacing_cv)
    if best is None:
        return fallback

    primary_centers, anchor_score, spacing_cv = best
    median_spacing = float(np.median(np.diff(primary_centers)))
    # This fallback is deliberately limited to a plain three-row page. Rhythm
    # pages have an independently observable fourth waveform row and retain
    # the stricter rhythm-layout detector above.
    if primary_centers[2] <= 0.72 * height:
        return fallback
    neutral_ink = _label_ink_mask(image, 205)
    panel_ranges = (
        (0, width // 4),
        (width // 4, width // 2),
        (width // 2, width * 3 // 4),
        (width * 3 // 4, width),
    )
    support_radius = max(6, round(median_spacing * 0.40))
    primary_panel_floor = float(
        min(
            np.mean(
                neutral_ink[
                    max(0, int(round(center)) - support_radius) : min(
                        height,
                        int(round(center)) + support_radius + 1,
                    ),
                    x_start:x_end,
                ]
            )
            for center in primary_centers
            for x_start, x_end in panel_ranges
        )
    )
    if primary_panel_floor < 0.003:
        return fallback

    row_centers = np.rint(primary_centers).astype(np.int32).tolist()

    confidence = float(
        min(
            0.96,
            max(0.0, anchor_score) * 0.45
            + max(0.0, 1.0 - spacing_cv / 0.12) * 0.30
            + min(1.0, primary_panel_floor / 0.03) * 0.25,
        )
    )
    return {
        "layoutHint": "standard_3x4",
        "confidence": confidence,
        "rowCenters": row_centers,
        "aspectRatio": aspect_ratio,
        "medianRowSpacing": median_spacing,
        "rowSpacingCv": spacing_cv,
        "panelInkFloor": primary_panel_floor,
        "method": "repeated-calibration-pulse-three-by-four-row-anchors-v1",
        "calibrationRowAnchorCount": 3,
    }


def detect_six_by_two_geometry(image: np.ndarray) -> dict[str, Any]:
    """Detect a six-row, two-column page without requiring a landscape aspect."""

    gray = _as_gray(image)
    height, width = gray.shape
    aspect_ratio = width / max(height, 1)
    result: dict[str, Any] = {
        "layoutHint": None,
        "confidence": 0.0,
        "rowCenters": [],
        "aspectRatio": aspect_ratio,
    }
    if height < 160 or width < 160 or not 0.72 <= aspect_ratio <= 2.8:
        return result

    trace_ink = _trace_ink(gray)
    row_profile = _row_profile(trace_ink)
    prominence = max(0.004, float(np.ptp(row_profile)) * 0.035)
    peaks, _ = find_peaks(
        row_profile,
        distance=max(12, round(height / 9.0)),
        prominence=prominence,
    )
    fine_peaks, _ = find_peaks(
        row_profile,
        distance=max(6, round(height / 16)),
        prominence=prominence,
    )
    strong_row_peak_count = int(
        np.count_nonzero(
            row_profile[fine_peaks] >= max(float(row_profile.max()) * 0.35, 1e-9)
        )
    )
    centers, sequence_score = _six_row_sequence(peaks, row_profile, height)
    if centers is None:
        return result

    spacing = np.diff(centers).astype(np.float64)
    median_spacing = float(np.median(spacing))
    spacing_cv = float(np.std(spacing) / max(median_spacing, 1e-9))
    left_profile = (trace_ink[:, : width // 2] > 0).mean(axis=1)
    right_profile = (trace_ink[:, width // 2 :] > 0).mean(axis=1)
    band_radius = max(2, round(height * 0.014))

    def band_support(profile: np.ndarray, center: int) -> float:
        start = max(0, center - band_radius)
        end = min(height, center + band_radius + 1)
        return float(np.mean(profile[start:end]))

    bilateral_support = np.asarray(
        [
            min(
                band_support(left_profile, int(center)),
                band_support(right_profile, int(center)),
            )
            for center in centers
        ]
    )
    bilateral_floor = float(np.min(bilateral_support))
    geometry_valid = (
        0.11 * height <= median_spacing <= 0.22 * height
        and spacing_cv <= 0.22
        and centers[0] <= 0.23 * height
        and centers[-1] >= 0.75 * height
        and bilateral_floor >= 0.025
        and 5 <= strong_row_peak_count <= 7
        and sequence_score >= 0.25
    )
    confidence = min(
        1.0,
        max(0.0, 1.0 - spacing_cv / 0.22)
        * min(1.0, bilateral_floor / 0.075)
        * min(1.0, max(0.0, sequence_score) / 0.75),
    )
    result.update(
        {
            "confidence": confidence,
            "rowCenters": centers.tolist(),
            "medianRowSpacing": median_spacing,
            "rowSpacingCv": spacing_cv,
            "bilateralInkFloor": bilateral_floor,
            "strongRowPeakCount": strong_row_peak_count,
            "sequenceScore": sequence_score,
        }
    )
    if geometry_valid:
        result["layoutHint"] = "standard_6x2"
    return result


def detect_three_by_four_rhythm_geometry(image: np.ndarray) -> dict[str, Any]:
    gray = _as_gray(image)
    height, width = gray.shape
    aspect_ratio = width / max(height, 1)
    result: dict[str, Any] = {
        "layoutHint": None,
        "confidence": 0.0,
        "rowCenters": [],
        "aspectRatio": aspect_ratio,
    }
    if height < 320 or width < 480 or not 1.05 <= aspect_ratio <= 3.2:
        return result

    trace_ink = _trace_ink(gray)
    row_profile = _row_profile(trace_ink)
    prominence = max(0.004, float(np.ptp(row_profile)) * 0.035)
    peaks, _ = find_peaks(
        row_profile,
        distance=max(20, round(height / 5.5)),
        prominence=prominence,
    )
    fine_peaks, _ = find_peaks(
        row_profile,
        distance=max(8, round(height / 16)),
        prominence=prominence,
    )
    strong_row_peak_count = int(
        np.count_nonzero(
            row_profile[fine_peaks] >= max(float(row_profile.max()) * 0.32, 1e-9)
        )
    )
    centers, sequence_score = _four_row_sequence(peaks, row_profile, height)
    if centers is None:
        return result

    spacing = np.diff(centers).astype(np.float64)
    median_spacing = float(np.median(spacing))
    spacing_cv = float(np.std(spacing) / max(median_spacing, 1e-9))
    quarter_profiles = [
        (trace_ink[:, start:end] > 0).mean(axis=1)
        for start, end in (
            (0, width // 4),
            (width // 4, width // 2),
            (width // 2, width * 3 // 4),
            (width * 3 // 4, width),
        )
    ]
    full_profile = (trace_ink > 0).mean(axis=1)
    band_radius = max(3, round(height * 0.014))

    def band_support(profile: np.ndarray, center: int) -> float:
        start = max(0, center - band_radius)
        end = min(height, center + band_radius + 1)
        return float(np.mean(profile[start:end]))

    primary_support = np.asarray(
        [
            band_support(profile, int(center))
            for center in centers[:3]
            for profile in quarter_profiles
        ]
    )
    rhythm_support = band_support(full_profile, int(centers[3]))
    panel_floor = float(np.min(primary_support))

    geometry_valid = (
        0.18 * height <= median_spacing <= 0.32 * height
        and spacing_cv <= 0.18
        and centers[0] <= 0.20 * height
        and 0.50 * height <= centers[2] <= 0.72 * height
        and centers[3] >= 0.78 * height
        and panel_floor >= 0.022
        and rhythm_support >= 0.022
        and strong_row_peak_count == 4
        and sequence_score >= 0.30
    )
    confidence = min(
        1.0,
        max(0.0, 1.0 - spacing_cv / 0.18)
        * min(1.0, panel_floor / 0.065)
        * min(1.0, rhythm_support / 0.065)
        * min(1.0, max(0.0, sequence_score) / 0.8),
    )
    result.update(
        {
            "confidence": confidence,
            "rowCenters": centers.tolist(),
            "medianRowSpacing": median_spacing,
            "rowSpacingCv": spacing_cv,
            "panelInkFloor": panel_floor,
            "rhythmInkSupport": rhythm_support,
            "strongRowPeakCount": strong_row_peak_count,
            "leadLabelValidation": detect_standard_three_by_four_label_grid(
                image,
                centers[:3],
            ),
        }
    )
    if geometry_valid:
        result["layoutHint"] = "standard_3x4_with_r1"
    return result


def detect_inset_three_by_four_rhythm_geometry(
    image: np.ndarray,
    initial: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-evaluate a strong four-row sequence inside large report margins."""

    fallback = initial or detect_three_by_four_rhythm_geometry(image)
    if fallback.get("layoutHint"):
        return fallback
    centers = [int(value) for value in fallback.get("rowCenters") or []]
    if (
        len(centers) != 4
        or float(fallback.get("confidence", 0.0)) < 0.45
        or float(fallback.get("rowSpacingCv", 1.0)) > 0.18
    ):
        return fallback
    height = image.shape[0]
    spacing = float(np.median(np.diff(centers)))
    top = max(0, round(centers[0] - spacing * 0.55))
    bottom = min(height, round(centers[3] + spacing * 0.55))
    if bottom - top < height * 0.55 or (top == 0 and bottom == height):
        return fallback
    inset = detect_three_by_four_rhythm_geometry(image[top:bottom])
    if not inset.get("layoutHint"):
        return fallback
    return {
        **inset,
        "rowCenters": [int(value + top) for value in inset["rowCenters"]],
        "aspectRatio": image.shape[1] / max(height, 1),
        "method": "inset-three-by-four-rhythm-geometry-v1",
        "layoutRegion": {
            "left": 0,
            "top": int(top),
            "right": int(image.shape[1]),
            "bottom": int(bottom),
        },
    }


def detect_standard_three_by_four_label_grid(
    image: np.ndarray,
    row_centers: list[int] | tuple[int, ...] | np.ndarray,
    *,
    _panel_label_fractions: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Validate the explicit standard labels on a three-row, four-panel ECG."""

    centers = np.asarray(row_centers, dtype=np.float64)
    fallback: dict[str, Any] = {
        "passed": False,
        "order": None,
        "confidence": 0.0,
        "method": "standard-three-by-four-label-grid-v1",
        "semanticIdentityConfirmed": False,
        "cellCount": 0,
    }
    if centers.size != 3:
        return fallback
    height, width = image.shape[:2]
    spacing = float(np.median(np.diff(centers)))
    if spacing <= 0:
        return fallback

    if _panel_label_fractions is None:
        # Two common exporters place labels either immediately inside the
        # quarter-panel boundary or nearer the first waveform complex. Test
        # both explicit templates independently so widening one cell cannot
        # admit unrelated trace ink from the other convention.
        hypotheses = (
            ("panel-boundary", (0.030, 0.260, 0.495, 0.735)),
            ("waveform-start", (0.055, 0.283, 0.511, 0.739)),
        )
        evaluated = [
            (
                name,
                detect_standard_three_by_four_label_grid(
                    image,
                    row_centers,
                    _panel_label_fractions=fractions,
                ),
            )
            for name, fractions in hypotheses
        ]
        hypothesis, selected = max(
            evaluated,
            key=lambda item: (
                bool(item[1].get("passed")),
                float(item[1].get("confidence", 0.0)),
                int(item[1].get("cellCount", 0)),
            ),
        )
        return {**selected, "anchorHypothesis": hypothesis}

    label_thresholds = (120.0, 160.0, 200.0)
    masks = [
        (threshold, _label_ink_mask(image, threshold))
        for threshold in label_thresholds
    ]
    expected_component_counts = (
        (1, 3, 2, 2),
        (2, 3, 2, 2),
        (3, 3, 2, 2),
    )
    # Standard exports print the label just inside each quarter-page panel.
    # These anchors deliberately follow the panel boundaries rather than the
    # waveform start, which can be displaced by a QRS after homography.
    panel_label_x = [width * fraction for fraction in _panel_label_fractions]
    boundary_anchor_hypothesis = _panel_label_fractions[0] < 0.040
    left_label_search_fraction = 0.010
    cells: list[dict[str, float]] = []
    occlusion_recovered_cells: list[dict[str, float]] = []
    clipped_leading_i_label_recovered = False

    def has_tall_trace_occlusion(
        expected_x: float,
        expected_y: float,
        x_start: int,
        x_end: int,
    ) -> bool:
        """Identify a waveform excursion physically crossing a V-label cell.

        A very tall QRS at a panel boundary can join the printed ``V`` glyph
        into one component.  That is different from an absent label: the dark
        component must cross well above and below the label baseline while
        remaining narrow enough to be a trace excursion rather than text.
        """

        expanded_y_start = max(0, round(expected_y - spacing * 0.42))
        expanded_y_end = min(height, round(expected_y + spacing * 0.46))
        for threshold, mask in masks[:2]:
            region = mask[
                expanded_y_start:expanded_y_end,
                x_start:x_end,
            ]
            count, _, stats, centroids = _connected_components_with_stats(
                region
            )
            for index in range(1, count):
                x, y, component_width, component_height, area = stats[index]
                page_x = float(centroids[index][0] + x_start)
                page_top = int(y + expanded_y_start)
                page_bottom = int(page_top + component_height)
                component_density = float(
                    area / max(component_width * component_height, 1)
                )
                trace_shaped = bool(
                    component_width <= width * 0.020
                    or (
                        component_width <= width * 0.050
                        and component_height >= component_width * 0.95
                        and component_density <= 0.20
                    )
                )
                if (
                    component_height >= max(spacing * 0.34, height * 0.09)
                    and trace_shaped
                    and area >= 30
                    and abs(page_x - expected_x) <= width * 0.020
                    and (
                        (
                            page_top <= expected_y - spacing * 0.04
                            and page_bottom >= expected_y + spacing * 0.14
                        )
                        or (
                            page_top <= expected_y - spacing * 0.14
                            and page_bottom >= expected_y + spacing * 0.04
                        )
                    )
                ):
                    return True
        return False

    for row_index, center in enumerate(centers):
        expected_y = center - spacing * 0.33
        for column, expected_x in enumerate(panel_label_x):
            x_start = max(
                0,
                round(expected_x - width * left_label_search_fraction),
            )
            right_label_search_fraction = (
                0.028
                if boundary_anchor_hypothesis and column == 0
                else 0.035
            )
            x_end = min(
                width,
                round(expected_x + width * right_label_search_fraction),
            )
            y_start = max(0, round(expected_y - spacing * 0.085))
            y_end = min(height, round(expected_y + spacing * 0.085))
            if x_end <= x_start or y_end <= y_start:
                return fallback
            expected_count = expected_component_counts[row_index][column]
            maximum_label_component_height = (
                max(8.0, spacing * 0.11)
                if boundary_anchor_hypothesis
                and column == 0
                else height * 0.050
            )
            component_options: list[
                tuple[
                    float,
                    list[tuple[int, int, int, int, int, float, float]],
                ]
            ] = []
            for threshold, mask in masks:
                cell = mask[y_start:y_end, x_start:x_end]
                count, _, stats, centroids = _connected_components_with_stats(
                    cell
                )
                components: list[
                    tuple[int, int, int, int, int, float, float]
                ] = []
                for index in range(1, count):
                    x, y, component_width, component_height, area = stats[index]
                    page_x = float(centroids[index][0] + x_start)
                    page_y = float(centroids[index][1] + y_start)
                    if not (
                        max(4.0, height * 0.010)
                        <= component_height
                        <= maximum_label_component_height
                        and 1 <= component_width <= width * 0.014
                        and area >= 4
                        and page_x
                        >= expected_x
                        - max(2.0, width * left_label_search_fraction)
                        # Printed glyph centroids sit on or slightly above
                        # the expected label baseline.  A lower component is
                        # usually the adjacent waveform entering the box.
                        and expected_y - spacing * 0.075
                        <= page_y
                        <= expected_y + spacing * 0.040
                    ):
                        continue
                    components.append(
                        (
                            int(x),
                            int(y),
                            int(component_width),
                            int(component_height),
                            int(area),
                            page_x,
                            page_y,
                        )
                    )
                component_options.append((threshold, components))
            accepted_options = [
                option
                for option in component_options
                if len(option[1]) == expected_count
                or (column >= 2 and len(option[1]) == expected_count + 1)
            ]
            if not accepted_options:
                closest_threshold, closest_components = min(
                    component_options,
                    key=lambda option: abs(len(option[1]) - expected_count),
                )
                # Some exports crop or overprint only the first ``I`` label.
                # The remaining II/III progression plus all augmented and
                # precordial labels can still establish the standard grid.
                # Recover no other missing limb label and contribute no glyph
                # evidence for this placeholder.
                if (
                    row_index == 0
                    and column == 0
                    and all(not components for _, components in component_options)
                ):
                    cells.append(
                        {
                            "row": 0.0,
                            "column": 0.0,
                            "x": float(expected_x),
                            "y": float(expected_y),
                            "left": float(expected_x),
                            "width": 0.0,
                            "componentCount": 0.0,
                            "labelThreshold": float(closest_threshold),
                        }
                    )
                    clipped_leading_i_label_recovered = True
                    continue
                # Recover at most two precordial labels in one column when the
                # source itself shows a tall trace physically occluding each
                # exact cell. At least ten printed labels (including the other
                # row in that column) still establish the standard grid; limb
                # identity is never inferred this way.
                if (
                    column >= 2
                    and len(occlusion_recovered_cells) < 2
                    and (
                        not occlusion_recovered_cells
                        or all(
                            int(recovered["column"]) == column
                            for recovered in occlusion_recovered_cells
                        )
                    )
                    and has_tall_trace_occlusion(
                        expected_x,
                        expected_y,
                        x_start,
                        x_end,
                    )
                ):
                    cells.append(
                        {
                            "row": float(row_index),
                            "column": float(column),
                            "x": float(expected_x),
                            "y": float(expected_y),
                            "left": float(expected_x),
                            "width": 0.0,
                            "componentCount": float(len(closest_components)),
                            "labelThreshold": float(closest_threshold),
                        }
                    )
                    occlusion_recovered_cells.append(
                        {
                            "row": float(row_index),
                            "column": float(column),
                            "labelThreshold": float(closest_threshold),
                        }
                    )
                    continue
                return {
                    **fallback,
                    "cellCount": len(cells),
                    "failedCell": [row_index, column],
                    "observedComponentCount": len(closest_components),
                    "expectedComponentCount": expected_count,
                    "labelThreshold": closest_threshold,
                }
            selected_threshold, components = min(
                accepted_options,
                key=lambda option: (
                    len(option[1]) != expected_count,
                    option[0],
                ),
            )
            left = min(x_start + component[0] for component in components)
            right = max(
                x_start + component[0] + component[2]
                for component in components
            )
            cells.append(
                {
                    "row": float(row_index),
                    "column": float(column),
                    "x": float(np.median([component[5] for component in components])),
                    "y": float(np.median([component[6] for component in components])),
                    "left": float(left),
                    "width": float(right - left),
                    "componentCount": float(len(components)),
                    "labelThreshold": float(selected_threshold),
                }
            )

    # Place an occluded cell on the medians established by the two visible
    # labels in its column and the three visible labels in its row.  The
    # placeholder contributes no recognition evidence; it only lets the
    # eleven observed anchors participate in the usual alignment check.
    for recovered in occlusion_recovered_cells:
        row = int(recovered["row"])
        column = int(recovered["column"])
        index = row * 4 + column
        column_cells = [
            cells[other_row * 4 + column]
            for other_row in range(3)
            if other_row != row
        ]
        row_cells = [
            cells[row * 4 + other_column]
            for other_column in range(4)
            if other_column != column
        ]
        cells[index]["x"] = float(np.median([cell["x"] for cell in column_cells]))
        cells[index]["y"] = float(np.median([cell["y"] for cell in row_cells]))

    if clipped_leading_i_label_recovered:
        cells[0]["x"] = float(np.median([cells[4]["x"], cells[8]["x"]]))
        cells[0]["y"] = float(np.median([cell["y"] for cell in cells[1:4]]))
        cells[0]["width"] = float(cells[4]["width"] * 0.45)

    roman_widths = np.asarray([cells[index * 4]["width"] for index in range(3)])
    roman_growth = bool(
        roman_widths[0] <= roman_widths[1] * 0.72
        and roman_widths[1] <= roman_widths[2] * 0.90
        and roman_widths[2] <= width * 0.025
    )
    column_x_spreads = [
        float(np.ptp([cells[row * 4 + column]["x"] for row in range(3)]))
        for column in range(4)
    ]
    column_label_left_edges = [
        float(
            np.median(
                [
                    cells[row * 4 + column]["left"]
                    for row in range(3)
                    if cells[row * 4 + column]["width"] > 0
                ]
            )
        )
        for column in range(4)
    ]
    row_y_spreads = [
        float(np.ptp([cells[row * 4 + column]["y"] for column in range(4)]))
        for row in range(3)
    ]
    normalized_column_x_spread = max(
        column_x_spreads[0] / max(width * 0.025, 1.0),
        *(spread / max(width * 0.015, 1.0) for spread in column_x_spreads[1:]),
    )
    aligned = bool(
        # Residual row-wise perspective can shear printed labels by roughly
        # two glyph widths even after the grid itself is rectified. Roman
        # I/II/III also grow horizontally, so their component centroids get a
        # slightly wider allowance than the three corroborating columns.
        normalized_column_x_spread <= 1.0
        and max(row_y_spreads) <= spacing * 0.06
    )
    passed = bool(roman_growth and aligned)
    confidence = (
        min(
            1.0,
            max(0.0, 1.0 - normalized_column_x_spread)
            * 0.45
            + max(0.0, 1.0 - max(row_y_spreads) / max(spacing * 0.06, 1.0))
            * 0.35
            + 0.20,
        )
        if passed
        else 0.0
    )
    if occlusion_recovered_cells:
        confidence *= 0.82
    if clipped_leading_i_label_recovered:
        confidence *= 0.82
    return {
        "passed": passed,
        "order": "standard" if passed else None,
        "confidence": float(confidence),
        "method": "standard-three-by-four-label-grid-v1",
        # This detector verifies grid placement and glyph topology only. It
        # deliberately cannot authorize semantic lead names until a separate
        # value-aware recognizer confirms every printed label and order.
        "semanticIdentityConfirmed": False,
        "cellCount": len(cells),
        "romanWidths": roman_widths.tolist(),
        "columnXSpreads": column_x_spreads,
        # A printed lead label begins just inside its panel's time origin.
        # Expose the independently repeated left glyph edge so the native
        # vectorizer can map time without rebasing the first visible waveform
        # ink to zero.  These coordinates are metadata only unless all four
        # columns also agree with the measured grid duration.
        "columnLabelLeftEdgesX": column_label_left_edges,
        "rowYSpreads": row_y_spreads,
        "labelThresholds": [cell["labelThreshold"] for cell in cells],
        "occlusionRecoveredCells": occlusion_recovered_cells,
        "clippedLeadingILabelRecovered": clipped_leading_i_label_recovered,
    }


def detect_three_by_four_geometry(image: np.ndarray) -> dict[str, Any]:
    """Detect a plain 3 x 4 page when a rhythm strip is not present.

    Lead names can be clipped at a page edge, so this detector deliberately
    uses only transparent page geometry: three evenly spaced trace rows with
    ink support in every quarter-width panel. The stricter strong-row count
    prevents a dense 6 x 2 or 12 x 1 page from being forced into this layout.
    """

    gray = _as_gray(image)
    height, width = gray.shape
    aspect_ratio = width / max(height, 1)
    result: dict[str, Any] = {
        "layoutHint": None,
        "confidence": 0.0,
        "rowCenters": [],
        "aspectRatio": aspect_ratio,
    }
    if height < 240 or width < 480 or not 1.45 <= aspect_ratio <= 4.0:
        return result

    trace_ink = _trace_ink(gray)
    row_profile = _row_profile(trace_ink)
    prominence = max(0.004, float(np.ptp(row_profile)) * 0.035)
    peaks, _ = find_peaks(
        row_profile,
        distance=max(20, round(height / 4.5)),
        prominence=prominence,
    )
    fine_peaks, _ = find_peaks(
        row_profile,
        distance=max(8, round(height / 8)),
        prominence=prominence,
    )
    strong_row_peak_count = int(
        np.count_nonzero(
            row_profile[fine_peaks]
            >= max(float(row_profile.max()) * 0.55, 1e-9)
        )
    )
    centers, sequence_score = _three_row_sequence(peaks, row_profile, height)
    if centers is None:
        return result

    spacing = np.diff(centers).astype(np.float64)
    median_spacing = float(np.median(spacing))
    spacing_cv = float(np.std(spacing) / max(median_spacing, 1e-9))
    quarter_profiles = [
        (trace_ink[:, start:end] > 0).mean(axis=1)
        for start, end in (
            (0, width // 4),
            (width // 4, width // 2),
            (width // 2, width * 3 // 4),
            (width * 3 // 4, width),
        )
    ]
    band_radius = max(3, round(height * 0.014))

    def band_support(profile: np.ndarray, center: int) -> float:
        start = max(0, center - band_radius)
        end = min(height, center + band_radius + 1)
        return float(np.mean(profile[start:end]))

    primary_support = np.asarray(
        [
            band_support(profile, int(center))
            for center in centers
            for profile in quarter_profiles
        ]
    )
    panel_floor = float(np.min(primary_support))

    geometry_valid = (
        0.24 * height <= median_spacing <= 0.45 * height
        and spacing_cv <= 0.20
        and centers[0] <= 0.25 * height
        and 0.35 * height <= centers[1] <= 0.62 * height
        and 0.72 * height <= centers[2] <= 0.95 * height
        and panel_floor >= MIN_THREE_BY_FOUR_PANEL_INK_SUPPORT
        and strong_row_peak_count == 3
        and sequence_score >= 0.30
    )
    confidence = min(
        1.0,
        max(0.0, 1.0 - spacing_cv / 0.20)
        * min(1.0, panel_floor / 0.025)
        * min(1.0, max(0.0, sequence_score) / 0.8),
    )
    result.update(
        {
            "confidence": confidence,
            "rowCenters": centers.tolist(),
            "medianRowSpacing": median_spacing,
            "rowSpacingCv": spacing_cv,
            "panelInkFloor": panel_floor,
            "strongRowPeakCount": strong_row_peak_count,
            "leadLabelValidation": detect_standard_three_by_four_label_grid(
                image,
                centers,
            ),
        }
    )
    if geometry_valid:
        result["layoutHint"] = "standard_3x4"
    return result


def detect_ecg_layout_geometry(
    image: np.ndarray,
    *,
    recognise_semantic_labels: bool = False,
) -> dict[str, Any]:
    twelve_row = detect_twelve_row_geometry(image)
    if twelve_row.get("layoutHint"):
        result = twelve_row
    else:
        six_by_two = detect_six_by_two_rhythm_geometry(image)
        labeled_six_by_two = detect_labeled_six_by_two_geometry(image)
        calibration_six_by_two = (
            detect_calibration_anchored_six_by_two_rhythm_geometry(image)
        )
        if prefer_labeled_six_by_two_geometry(
            labeled_six_by_two,
            six_by_two,
        ):
            result = labeled_six_by_two
        elif (
            calibration_six_by_two.get("layoutHint")
            and float(calibration_six_by_two.get("confidence", 0.0))
            >= float(six_by_two.get("confidence", 0.0)) - 0.02
        ):
            # Prefer morphology-independent calibration anchors when their
            # confidence matches the global waveform-row detector.
            result = calibration_six_by_two
        elif six_by_two.get("layoutHint"):
            # V-label geometry establishes lead identity but cannot erase an
            # independently detected seventh full-width waveform row. Retain
            # the label evidence on the rhythm-aware layout so downstream
            # panel specialists can validate the two primary six-lead panels.
            result = {
                **six_by_two,
                **(
                    {
                        "precordialLabelValidation": labeled_six_by_two.get(
                            "precordialLabelValidation"
                        ),
                        "limbLabelValidation": labeled_six_by_two.get(
                            "limbLabelValidation"
                        ),
                        "labelGeometryMethod": labeled_six_by_two.get("method"),
                    }
                    if labeled_six_by_two.get("layoutHint")
                    else {}
                ),
            }
        else:
            if calibration_six_by_two.get("layoutHint"):
                result = calibration_six_by_two
            else:
                six_by_two_plain = detect_six_by_two_geometry(image)
                if six_by_two_plain.get("layoutHint"):
                    result = six_by_two_plain
                else:
                    if labeled_six_by_two.get("layoutHint"):
                        result = labeled_six_by_two
                    else:
                        three_by_four_rhythm = (
                            detect_inset_three_by_four_rhythm_geometry(
                                image,
                                detect_three_by_four_rhythm_geometry(image),
                            )
                        )
                        if three_by_four_rhythm.get("layoutHint"):
                            result = three_by_four_rhythm
                        else:
                            three_by_four = detect_three_by_four_geometry(image)
                            if three_by_four.get("layoutHint"):
                                result = three_by_four
                            else:
                                result = (
                                    detect_calibration_anchored_three_by_four_geometry(
                                        image
                                    )
                                )
    result.update(detect_ecg_content_box(image))
    if result.get("layoutHint") == "standard_12x1":
        result["compoundPanels"] = None
    else:
        result.update(detect_compound_ecg_panels(image))
    result.update(detect_ecg_calibration(image))
    calibration = result.get("calibration") or {}
    row_grid_scales = horizontal_grid_scale_by_rows(
        image,
        [int(value) for value in result.get("rowCenters") or []],
        grid_scale_mm=(
            float(calibration["gridScaleMmX"])
            if calibration.get("gridScaleMmX") in {1.0, 5.0}
            else float(calibration["gridScaleMm"])
            if calibration.get("gridScaleMm") in {1.0, 5.0}
            else None
        ),
    )
    if row_grid_scales is not None:
        result["calibration"]["rowPixelsPerMmX"] = row_grid_scales
    if recognise_semantic_labels:
        result = attach_semantic_lead_identity(image, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read ECG image: {args.input}")
    print(
        json.dumps(
            detect_ecg_layout_geometry(
                image,
                recognise_semantic_labels=True,
            )
        )
    )


if __name__ == "__main__":
    main()
