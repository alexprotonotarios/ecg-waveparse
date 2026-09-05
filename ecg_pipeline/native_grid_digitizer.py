from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d
from scipy.signal import find_peaks

from scripts.detect_ecg_layout import detect_ecg_layout_geometry
from ecg_pipeline.domain import (
    LEAD_ORDER,
    PAGE_DURATION_SECONDS,
    SAMPLE_RATE_HZ,
    SIX_BY_TWO_PANEL_SAMPLES,
    THREE_BY_FOUR_PANEL_SAMPLES,
)
from ecg_pipeline.native_layout_strategies import (
    build_native_layout_strategy,
    row_bounds,
    sequential_row_bounds,
    six_row_bounds,
)


@dataclass(frozen=True)
class LeadFidelity:
    evidenceMedian: float
    evidenceP10: float
    maxNativeJumpPixels: float
    p95NativeJumpPixels: float
    coverage: float
    sourceStartPixel: int
    sourceEndPixel: int
    homeRowFraction: float = 1.0
    largeJumpCount: int = 0
    unsupportedLargeJumpCount: int = 0
    minimumLargeJumpSupport: float = 1.0
    qrsAnchorShiftPixels: int = 0
    qrsAnchorMatchCount: int = 0
    largeExcursionCount: int = 0
    recoveredExcursionCount: int = 0
    recoveredSampleCount: int = 0
    rejectedExcursionCount: int = 0
    rejectedSampleCount: int = 0
    disconnectedExcursionCount: int = 0
    verticalArtifactExcursionCount: int = 0
    unsafeExcursionCount: int = 0


def grid_residual_evidence(gray: np.ndarray) -> np.ndarray:
    """Fuse periodic-grid residuals with conservative dark-ink support."""
    if gray.ndim != 2:
        raise ValueError("Expected a grayscale image.")
    normalized = gray.astype(np.float32) / 255.0
    darkness = 1.0 - normalized
    row_grid = np.quantile(darkness, 0.5, axis=1)[:, None]
    column_grid = np.quantile(darkness, 0.5, axis=0)[None, :]
    residual = np.clip(
        (darkness - np.maximum(row_grid, column_grid) * 0.92) * 3.0,
        0.0,
        1.0,
    )
    # A trace drawn directly over a major horizontal grid line can disappear
    # from the profile residual even though the source pixel remains much
    # darker than the paper grid. Retain only the darkest tail as a second,
    # source-derived evidence channel. The high exponent keeps ordinary grid
    # strokes weak and does not create or interpolate waveform morphology.
    low = float(np.quantile(darkness, 0.50))
    high = float(np.quantile(darkness, 0.995))
    if high - low < 1e-6:
        dark_ink = np.zeros_like(darkness, dtype=np.float32)
    else:
        dark_ink = np.clip((darkness - low) / (high - low), 0.0, 1.0)
    evidence = np.maximum(residual, dark_ink**3.0 * 0.55)
    return cv2.GaussianBlur(evidence, (3, 3), 0.45)


def _centered_finite(values: np.ndarray) -> np.ndarray:
    result = values.astype(np.float64).copy()
    finite = np.isfinite(result)
    if np.any(finite):
        result[finite] -= float(np.median(result[finite]))
    return result


def validate_limb_lead_order(
    row_values: list[np.ndarray],
) -> dict[str, Any]:
    """Validate standard versus Cabrera limb order using ECG lead algebra."""

    if len(row_values) < 6:
        return {"passed": False, "reason": "fewer than six limb rows"}

    orderings: dict[str, dict[str, np.ndarray]] = {
        "standard": {
            "I": row_values[0],
            "II": row_values[1],
            "III": row_values[2],
            "aVR": row_values[3],
            "aVL": row_values[4],
            "aVF": row_values[5],
        },
        "cabrera": {
            "aVL": row_values[0],
            "I": row_values[1],
            "aVR": -row_values[2],
            "II": row_values[3],
            "aVF": row_values[4],
            "III": row_values[5],
        },
    }
    scored: list[dict[str, Any]] = []
    for name, values in orderings.items():
        centered = {lead: _centered_finite(value) for lead, value in values.items()}
        equations = (
            (centered["II"], centered["I"] + centered["III"]),
            (centered["aVR"], -(centered["I"] + centered["II"]) / 2.0),
            (centered["aVL"], centered["I"] - centered["II"] / 2.0),
            (centered["aVF"], centered["II"] - centered["I"] / 2.0),
        )
        normalized_errors: list[float] = []
        correlations: list[float] = []
        for observed, predicted in equations:
            finite = np.isfinite(observed) & np.isfinite(predicted)
            if np.count_nonzero(finite) < observed.size * 0.60:
                continue
            observed_finite = gaussian_filter1d(observed[finite], 1.0)
            predicted_finite = gaussian_filter1d(predicted[finite], 1.0)
            residual = observed_finite - predicted_finite
            scale = max(
                float(np.quantile(np.abs(observed_finite), 0.90)),
                float(np.quantile(np.abs(predicted_finite), 0.90)),
                25.0,
            )
            normalized_errors.append(
                float(np.median(np.abs(residual)) / scale)
            )
            if np.std(observed_finite) > 1e-6 and np.std(predicted_finite) > 1e-6:
                correlations.append(
                    float(np.corrcoef(observed_finite, predicted_finite)[0, 1])
                )
        median_error = float(np.median(normalized_errors)) if normalized_errors else 99.0
        median_correlation = float(np.median(correlations)) if correlations else -1.0
        score = max(0.0, median_correlation) * max(0.0, 1.0 - median_error)
        scored.append(
            {
                "order": name,
                "score": score,
                "medianNormalizedError": median_error,
                "medianCorrelation": median_correlation,
                "equationCount": len(normalized_errors),
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    best = scored[0]
    runner_up = scored[1]
    passed = bool(
        best["equationCount"] == 4
        and best["medianNormalizedError"] <= 0.62
        and best["medianCorrelation"] >= 0.38
        and best["score"] >= runner_up["score"] + 0.035
    )
    return {
        "passed": passed,
        "selectedOrder": best["order"] if passed else None,
        "best": best,
        "runnerUp": runner_up,
        "method": "simultaneous-limb-lead-algebra-v1",
    }


def trace_path(
    evidence: np.ndarray,
    *,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
    row_center: float,
    center_weight: float = 0.35,
    transition_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace one waveform with a smoothness-constrained native-pixel Viterbi path."""
    if y_end - y_start < 3 or x_end - x_start < 3:
        raise ValueError("The requested trace band is too small.")

    band = evidence[y_start:y_end, x_start:x_end]
    y_positions = np.arange(y_start, y_end, dtype=np.float32)
    half_height = max((y_end - y_start) / 2.0, 1.0)
    center_cost = center_weight * (
        np.abs(y_positions - float(row_center)) / half_height
    ) ** 4
    local_cost = 7.0 * (1.0 - band) + center_cost[:, None]

    displacement = np.abs(y_positions[:, None] - y_positions[None, :])
    transition_cost = transition_scale * (
        0.035 * displacement + 0.0018 * displacement**2
    )
    previous = local_cost[:, 0].astype(np.float64)
    backpointers = np.empty(
        (band.shape[1], band.shape[0]),
        dtype=np.int16,
    )
    backpointers[0] = -1

    for x_index in range(1, band.shape[1]):
        costs = previous[:, None] + transition_cost
        best_previous = np.argmin(costs, axis=0)
        previous = costs[best_previous, np.arange(band.shape[0])]
        previous += local_cost[:, x_index]
        backpointers[x_index] = best_previous.astype(np.int16)

    local_path = np.empty(band.shape[1], dtype=np.int32)
    local_path[-1] = int(np.argmin(previous))
    for x_index in range(band.shape[1] - 1, 0, -1):
        local_path[x_index - 1] = backpointers[x_index, local_path[x_index]]

    path = local_path + y_start
    columns = np.arange(x_start, x_end, dtype=np.int32)
    return columns, path


def _candidate_rows_for_column(
    column_evidence: np.ndarray,
    *,
    y_start: int,
    row_center: float,
    threshold: float = 0.28,
    maximum_candidates: int = 48,
) -> np.ndarray:
    active = np.flatnonzero(column_evidence >= threshold)
    candidates: set[int] = {
        int(np.argmax(column_evidence)),
        int(np.clip(round(row_center) - y_start, 0, column_evidence.size - 1)),
    }
    if active.size:
        runs = np.split(active, np.flatnonzero(np.diff(active) > 1) + 1)
        for run in runs:
            candidates.update((int(run[0]), int(run[-1])))
            candidates.add(int(run[np.argmax(column_evidence[run])]))
            if run.size >= 6:
                candidates.add(int(run[run.size // 3]))
                candidates.add(int(run[run.size * 2 // 3]))

    ordered = np.asarray(sorted(candidates), dtype=np.int32)
    if ordered.size <= maximum_candidates:
        return ordered

    strongest = ordered[
        np.argsort(column_evidence[ordered])[-(maximum_candidates - 4) :]
    ]
    anchors = np.asarray(
        [
            int(ordered[0]),
            int(ordered[-1]),
            int(np.argmin(np.abs(ordered + y_start - row_center))),
            int(np.argmax(column_evidence)),
        ],
        dtype=np.int32,
    )
    return np.unique(np.concatenate((strongest, anchors)))


def trace_crossing_path(
    evidence: np.ndarray,
    *,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
    row_center: float,
    row_spacing: float,
    transition_scale: float = 1.0,
    ink_connected_displacement_reward: float = 0.0,
    event_anchors: np.ndarray | None = None,
    event_radius_pixels: int = 0,
    event_excursion_reward: float = 0.0,
    event_reference_path: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace a waveform that may legitimately cross nominal ECG row bands."""
    if y_end - y_start < 3 or x_end - x_start < 3:
        raise ValueError("The requested trace band is too small.")
    if not np.isfinite(transition_scale) or transition_scale <= 0:
        raise ValueError("Crossing-path transition scale must be positive.")
    if (
        not np.isfinite(ink_connected_displacement_reward)
        or ink_connected_displacement_reward < 0
    ):
        raise ValueError(
            "Ink-connected displacement reward must be non-negative."
        )
    if event_radius_pixels < 0 or event_excursion_reward < 0:
        raise ValueError("Event-path recovery parameters must be non-negative.")

    columns = np.arange(x_start, x_end, dtype=np.int32)
    if event_reference_path is not None and event_reference_path.shape != columns.shape:
        raise ValueError("Event reference path must match crossing-path columns.")
    event_connected_support: np.ndarray | None = None
    if (
        event_reference_path is not None
        and event_anchors is not None
        and np.asarray(event_anchors).size
        and event_radius_pixels > 0
    ):
        event_connected_support = np.zeros(
            (y_end - y_start, columns.size), dtype=bool
        )
        for anchor in np.asarray(event_anchors, dtype=np.int32):
            left_index = max(0, int(anchor) - event_radius_pixels - x_start)
            right_index = min(
                columns.size,
                int(anchor) + event_radius_pixels + 1 - x_start,
            )
            if right_index - left_index < 3:
                continue
            region = (
                evidence[y_start:y_end, columns[left_index:right_index]]
                >= 0.24
            ).astype(np.uint8)
            component_count, labels = cv2.connectedComponents(
                region, connectivity=8
            )
            if component_count <= 1:
                continue
            allowed_labels: set[int] = set()
            boundary_indexes = list(
                range(left_index, min(right_index, left_index + 2))
            ) + list(range(max(left_index, right_index - 2), right_index))
            for index in boundary_indexes:
                local_y = int(event_reference_path[index]) - y_start
                local_x = index - left_index
                if 0 <= local_y < labels.shape[0]:
                    label = int(labels[local_y, local_x])
                    if label > 0:
                        allowed_labels.add(label)
            if not allowed_labels:
                continue
            local_support = np.isin(labels, list(allowed_labels))
            event_connected_support[:, left_index:right_index] |= local_support
    states = [
        _candidate_rows_for_column(
            evidence[y_start:y_end, column],
            y_start=y_start,
            row_center=row_center,
        )
        + y_start
        for column in columns
    ]
    anchors = (
        np.asarray(event_anchors, dtype=np.int32)
        if event_anchors is not None
        else np.asarray([], dtype=np.int32)
    )
    local_costs: list[np.ndarray] = []
    for state, column in zip(states, columns, strict=True):
        normalized_excursion = np.abs(
            (state.astype(np.float64) - row_center) / row_spacing
        )
        local = (
            5.0 * (1.0 - evidence[state, column])
            + 4.0 * normalized_excursion**2
        )
        if anchors.size and event_radius_pixels > 0 and event_excursion_reward > 0:
            distance = int(np.min(np.abs(anchors - int(column))))
            if distance <= event_radius_pixels:
                event_weight = 1.0 - distance / (event_radius_pixels + 1.0)
                connected = (
                    event_connected_support[
                        state - y_start,
                        int(column) - x_start,
                    ]
                    if event_connected_support is not None
                    else np.ones(state.size, dtype=bool)
                )
                local -= (
                    event_excursion_reward
                    * event_weight
                    * normalized_excursion
                    * connected
                )
        local_costs.append(local)
    previous = local_costs[0].astype(np.float64)
    backpointers: list[np.ndarray] = [
        np.full(states[0].size, -1, dtype=np.int16)
    ]
    for index in range(1, len(columns)):
        prior_states = states[index - 1].astype(np.float64)
        current_states = states[index].astype(np.float64)
        displacement = np.abs(
            prior_states[:, None] - current_states[None, :]
        )
        transition = transition_scale * (
            0.0025 * displacement + 0.000018 * displacement**2
        )
        if ink_connected_displacement_reward > 0:
            # Near-vertical QRS strokes can contain many source pixels in one
            # raster column. A one-state-per-column path otherwise stays near
            # the nominal baseline and skips the connected excursion. Reward
            # displacement only when the full transition is supported by an
            # almost continuous source-ink segment. Disconnected jumps retain
            # the stronger transition penalty above.
            active = evidence[y_start:y_end, columns[index]] >= 0.24
            inactive_prefix = np.concatenate(
                ([0], np.cumsum(~active, dtype=np.int32))
            )
            prior_indexes = states[index - 1] - y_start
            current_indexes = states[index] - y_start
            lower = np.minimum(
                prior_indexes[:, None], current_indexes[None, :]
            )
            upper = np.maximum(
                prior_indexes[:, None], current_indexes[None, :]
            )
            inactive_count = (
                inactive_prefix[upper + 1] - inactive_prefix[lower]
            )
            segment_length = upper - lower + 1
            connected = inactive_count <= np.maximum(
                2,
                np.floor(segment_length * 0.08),
            )
            transition -= (
                ink_connected_displacement_reward
                * displacement
                * connected
            )
        costs = previous[:, None] + transition
        best_previous = np.argmin(costs, axis=0)
        previous = (
            costs[best_previous, np.arange(current_states.size)]
            + local_costs[index]
        )
        backpointers.append(best_previous.astype(np.int16))

    state_indexes = np.empty(len(columns), dtype=np.int32)
    state_indexes[-1] = int(np.argmin(previous))
    for index in range(len(columns) - 1, 0, -1):
        state_indexes[index - 1] = backpointers[index][state_indexes[index]]
    path = np.asarray(
        [states[index][state_index] for index, state_index in enumerate(state_indexes)],
        dtype=np.int32,
    )
    return columns, path


def _true_runs(mask: np.ndarray) -> list[np.ndarray]:
    indexes = np.flatnonzero(mask)
    if indexes.size == 0:
        return []
    return [
        run
        for run in np.split(indexes, np.flatnonzero(np.diff(indexes) > 1) + 1)
        if run.size
    ]


def waveform_event_columns(
    columns: np.ndarray,
    path: np.ndarray,
    *,
    row_spacing: float,
) -> np.ndarray:
    """Return sharp waveform-event columns without assuming a fixed heart rate."""
    window = 6
    path_float = path.astype(np.float64)
    score = maximum_filter1d(path_float, window) - minimum_filter1d(
        path_float,
        window,
    )
    score = gaussian_filter1d(score, 0.8)
    peaks, _ = find_peaks(
        score,
        distance=10,
        height=max(12.0, row_spacing * 0.045),
        prominence=max(5.0, row_spacing * 0.018),
    )
    return columns[peaks]


def rhythm_qrs_anchors(
    columns: np.ndarray,
    path: np.ndarray,
    *,
    row_spacing: float,
    effective_sample_rate_hz: float,
) -> np.ndarray:
    """Detect page-time QRS anchors from the independent full-width rhythm row."""
    path_float = path.astype(np.float64)
    score = maximum_filter1d(path_float, 6) - minimum_filter1d(path_float, 6)
    score = gaussian_filter1d(score, 1.0)
    peaks, _ = find_peaks(
        score,
        distance=max(12, round(effective_sample_rate_hz * 0.32)),
        height=row_spacing * 0.07,
        prominence=row_spacing * 0.035,
    )
    return columns[peaks]


def cross_lead_event_anchors(
    paths: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    leads: tuple[str, ...] | list[str],
    *,
    panel_start: int,
    panel_end: int,
    row_spacing: float,
    tolerance_pixels: int,
    artifact_columns: np.ndarray | None = None,
    minimum_other_leads: int = 2,
) -> np.ndarray:
    """Return event times independently reproduced in other simultaneous leads.

    Only the event time is shared. Each lead keeps its independently traced
    amplitude and morphology. Page-spanning vertical artifact columns are
    explicitly excluded from this corroboration path.
    """

    observations: list[tuple[int, str]] = []
    for lead in leads:
        traced = paths.get(lead)
        if traced is None:
            continue
        columns, path, _ = traced
        events = waveform_event_columns(
            columns,
            path,
            row_spacing=row_spacing,
        )
        for event in events:
            column = int(event)
            if not panel_start <= column < panel_end:
                continue
            if (
                artifact_columns is not None
                and 0 <= column < artifact_columns.size
                and bool(artifact_columns[column])
            ):
                continue
            observations.append((column, lead))

    if not observations:
        return np.asarray([], dtype=np.int32)

    clusters: list[list[tuple[int, str]]] = []
    for observation in sorted(observations):
        if (
            clusters
            and observation[0]
            - int(np.median([item[0] for item in clusters[-1]]))
            <= tolerance_pixels
        ):
            clusters[-1].append(observation)
        else:
            clusters.append([observation])

    anchors = [
        round(float(np.median([column for column, _ in cluster])))
        for cluster in clusters
        if len({lead for _, lead in cluster}) >= minimum_other_leads
    ]
    return np.asarray(anchors, dtype=np.int32)


def normalized_cross_lead_event_times(
    paths: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    source_ranges: dict[str, tuple[int, int]],
    leads: tuple[str, ...] | list[str],
    *,
    row_spacing: float,
    minimum_leads: int = 4,
    tolerance_fraction: float = 0.006,
) -> np.ndarray:
    """Return simultaneous event times normalized across projective row spans."""

    observations: list[tuple[float, str]] = []
    for lead in leads:
        traced = paths.get(lead)
        source_range = source_ranges.get(lead)
        if traced is None or source_range is None:
            continue
        start, end = source_range
        if end <= start:
            continue
        columns, path, _ = traced
        for event in waveform_event_columns(
            columns,
            path,
            row_spacing=row_spacing,
        ):
            normalized = (float(event) - start) / (end - start)
            if 0.0 <= normalized <= 1.0:
                observations.append((normalized, lead))
    if not observations:
        return np.asarray([], dtype=np.float64)

    clusters: list[list[tuple[float, str]]] = []
    for observation in sorted(observations):
        if (
            clusters
            and observation[0]
            - float(np.median([item[0] for item in clusters[-1]]))
            <= tolerance_fraction
        ):
            clusters[-1].append(observation)
        else:
            clusters.append([observation])
    candidates = [
        (
            float(np.median([value for value, _ in cluster])),
            len({lead for _, lead in cluster}),
            len(cluster),
        )
        for cluster in clusters
        if len({lead for _, lead in cluster}) >= minimum_leads
    ]
    # QRS and T-wave slopes can both form cross-lead event clusters. Retain the
    # strongest simultaneous event within a 250 ms refractory window; this is
    # a timing-only ECG prior and does not impose a waveform shape or rate.
    selected: list[tuple[float, int, int]] = []
    for candidate in sorted(candidates, key=lambda item: (-item[1], -item[2])):
        if all(abs(candidate[0] - item[0]) >= 0.025 for item in selected):
            selected.append(candidate)
    return np.asarray(
        sorted(item[0] for item in selected),
        dtype=np.float64,
    )


def align_panel_qrs_anchors(
    event_columns: np.ndarray,
    page_anchors: np.ndarray,
    *,
    panel_start: int,
    panel_end: int,
    effective_sample_rate_hz: float,
) -> tuple[np.ndarray, int, int]:
    """Estimate a small scan/panel offset using the repeated QRS sequence."""
    expected = page_anchors[
        (page_anchors >= panel_start) & (page_anchors < panel_end)
    ]
    if expected.size == 0 or event_columns.size == 0:
        return expected, 0, 0

    search_radius = max(8, round(effective_sample_rate_hz * 0.15))
    match_tolerance = max(5, round(effective_sample_rate_hz * 0.065))
    best: tuple[tuple[int, int, int], int, int] | None = None
    for shift in range(-search_radius, search_radius + 1):
        distances = [
            int(np.min(np.abs(event_columns - (anchor + shift))))
            for anchor in expected
        ]
        matches = sum(distance <= match_tolerance for distance in distances)
        cost = sum(min(distance, match_tolerance * 2) for distance in distances)
        key = (-matches, cost, abs(shift))
        if best is None or key < best[0]:
            best = (key, shift, matches)

    if best is None:
        return expected, 0, 0
    return expected + best[1], best[1], best[2]


def vertical_artifact_columns(
    evidence: np.ndarray,
    *,
    primary_bottom: int,
    row_spacing: float,
) -> np.ndarray:
    """Flag long continuous vertical ink while retaining it for QRS adjudication."""
    ink = (evidence[:primary_bottom] >= 0.24).astype(np.uint8)
    ink = cv2.dilate(
        ink,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)),
    )
    longest_runs = np.zeros(ink.shape[1], dtype=np.int32)
    for column in range(ink.shape[1]):
        runs = _true_runs(ink[:, column].astype(bool))
        longest_runs[column] = max((run.size for run in runs), default=0)
    flagged = (longest_runs >= row_spacing * 0.45).astype(np.uint8)[None, :]
    expanded = cv2.dilate(
        flagged,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)),
    )
    return expanded.ravel().astype(bool)


def excursion_is_connected(
    evidence: np.ndarray,
    columns: np.ndarray,
    path: np.ndarray,
    run: np.ndarray,
    *,
    row_center: float,
    row_spacing: float,
) -> bool:
    """Require an out-of-band branch to connect back to its lead's home row."""
    padding_x = 4
    start = max(0, int(run[0]) - padding_x)
    end = min(columns.size, int(run[-1]) + padding_x + 1)
    x_start = int(columns[start])
    x_end = int(columns[end - 1]) + 1
    extreme_index = int(run[np.argmax(np.abs(path[run] - row_center))])
    extreme_y = int(path[extreme_index])
    vertical_margin = max(3, round(row_spacing * 0.10))
    y_start = max(0, round(min(extreme_y, row_center) - vertical_margin))
    y_end = min(
        evidence.shape[0],
        round(max(extreme_y, row_center) + vertical_margin + 1),
    )
    region = (evidence[y_start:y_end, x_start:x_end] >= 0.20).astype(np.uint8)
    region = cv2.dilate(
        region,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    component_count, labels = cv2.connectedComponents(region, connectivity=8)
    if component_count <= 1:
        return False

    extreme_local_y = int(np.clip(extreme_y - y_start, 0, labels.shape[0] - 1))
    extreme_local_x = int(
        np.clip(columns[extreme_index] - x_start, 0, labels.shape[1] - 1)
    )
    label_window = labels[
        max(0, extreme_local_y - 2) : extreme_local_y + 3,
        max(0, extreme_local_x - 2) : extreme_local_x + 3,
    ]
    candidate_labels = set(int(value) for value in np.unique(label_window) if value)
    if not candidate_labels:
        return False

    home_start = max(0, round(row_center - row_spacing * 0.12) - y_start)
    home_end = min(
        labels.shape[0],
        round(row_center + row_spacing * 0.12) - y_start + 1,
    )
    for label in candidate_labels:
        component = labels == label
        if np.any(component[home_start:home_end]):
            return True
    return False


def artifact_safe_validity(
    evidence: np.ndarray,
    columns: np.ndarray,
    path: np.ndarray,
    *,
    row_center: float,
    row_spacing: float,
    qrs_anchors: np.ndarray,
    qrs_anchor_shift: int,
    qrs_anchor_matches: int,
    artifact_columns: np.ndarray,
    effective_sample_rate_hz: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Mask uncorroborated excursions instead of publishing artificial morphology."""
    validity = np.ones(columns.size, dtype=bool)
    excursion_mask = (
        np.abs(path.astype(np.float64) - row_center) >= row_spacing * 0.30
    )
    excursion_runs = _true_runs(excursion_mask)
    anchor_tolerance = max(5, round(effective_sample_rate_hz * 0.075))
    rejected_regions: list[tuple[int, int]] = []
    disconnected = 0
    vertical_artifact_excursions = 0

    for run in excursion_runs:
        extreme_index = int(run[np.argmax(np.abs(path[run] - row_center))])
        event_column = int(columns[extreme_index])
        anchor_distance = (
            int(np.min(np.abs(qrs_anchors - event_column)))
            if qrs_anchors.size
            else np.iinfo(np.int32).max
        )
        connected = excursion_is_connected(
            evidence,
            columns,
            path,
            run,
            row_center=row_center,
            row_spacing=row_spacing,
        )
        x0 = max(0, int(run[0]) - 3)
        x1 = min(columns.size, int(run[-1]) + 4)
        overlaps_vertical_artifact = bool(
            np.any(artifact_columns[columns[x0:x1]])
        )
        if overlaps_vertical_artifact:
            vertical_artifact_excursions += 1
        if not connected:
            disconnected += 1
        if anchor_distance > anchor_tolerance or not connected:
            rejected_regions.append((x0, x1))

    jumps = np.abs(np.diff(path.astype(np.float64)))
    for index in np.flatnonzero(jumps > row_spacing * 0.25):
        event_column = int(columns[index])
        anchor_distance = (
            int(np.min(np.abs(qrs_anchors - event_column)))
            if qrs_anchors.size
            else np.iinfo(np.int32).max
        )
        if anchor_distance > anchor_tolerance:
            rejected_regions.append(
                (max(0, int(index) - 3), min(columns.size, int(index) + 5))
            )

    if rejected_regions:
        merged: list[tuple[int, int]] = []
        for start, end in sorted(rejected_regions):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        for start, end in merged:
            validity[start:end] = False
    else:
        merged = []

    return validity, {
        "qrsAnchorShiftPixels": int(qrs_anchor_shift),
        "qrsAnchorMatchCount": int(qrs_anchor_matches),
        "largeExcursionCount": len(excursion_runs),
        "recoveredExcursionCount": 0,
        "recoveredSampleCount": 0,
        "rejectedExcursionCount": len(merged),
        "rejectedSampleCount": int(np.count_nonzero(~validity)),
        "disconnectedExcursionCount": disconnected,
        "verticalArtifactExcursionCount": vertical_artifact_excursions,
        "unsafeExcursionCount": 0,
    }


def recover_rejected_intervals(
    evidence: np.ndarray,
    columns: np.ndarray,
    path: np.ndarray,
    validity: np.ndarray,
    strict_path: np.ndarray,
    safety_metrics: dict[str, int],
    *,
    row_center: float,
    row_spacing: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Recover rejected regions only from a conservative source-ink path.

    The strict path is independently traced inside the lead's home-row band.
    It may replace a rejected excursion only when it is ink-supported, remains
    close to the lead baseline, and joins the accepted path at both boundaries.
    """
    if not (
        columns.shape
        == path.shape
        == validity.shape
        == strict_path.shape
    ):
        raise ValueError("Recovery paths and validity must have matching shapes.")

    recovered_path = path.copy()
    recovered_validity = validity.copy()
    recovered_regions = 0
    recovered_samples = 0
    maximum_deviation = row_spacing * 0.25
    maximum_local_jump = row_spacing * 0.16
    maximum_boundary_jump = row_spacing * 0.12

    for run in _true_runs(~validity):
        if run[0] == 0 or run[-1] + 1 >= path.size:
            continue
        candidate = strict_path[run]
        support = evidence[candidate, columns[run]]
        local_jumps = np.abs(np.diff(candidate.astype(np.float64)))
        left_jump = abs(
            float(candidate[0]) - float(recovered_path[run[0] - 1])
        )
        right_jump = abs(
            float(candidate[-1]) - float(recovered_path[run[-1] + 1])
        )
        source_backed = (
            float(np.median(support)) >= 0.65
            and float(np.quantile(support, 0.10)) >= 0.30
        )
        geometry_safe = (
            float(np.max(np.abs(candidate.astype(np.float64) - row_center)))
            <= maximum_deviation
            and (
                not local_jumps.size
                or float(np.max(local_jumps)) <= maximum_local_jump
            )
            and left_jump <= maximum_boundary_jump
            and right_jump <= maximum_boundary_jump
        )
        if not (source_backed and geometry_safe):
            continue
        recovered_path[run] = candidate
        recovered_validity[run] = True
        recovered_regions += 1
        recovered_samples += int(run.size)

    remaining_runs = _true_runs(~recovered_validity)
    updated_metrics = {
        **safety_metrics,
        "recoveredExcursionCount": recovered_regions,
        "recoveredSampleCount": recovered_samples,
        "rejectedExcursionCount": len(remaining_runs),
        "rejectedSampleCount": int(np.count_nonzero(~recovered_validity)),
    }
    return recovered_path, recovered_validity, updated_metrics


def panel_margins(
    panel_width: int,
    column: int,
    *,
    column_count: int,
) -> tuple[int, int]:
    """Reserve label-heavy panel edges instead of inventing waveform samples."""
    if column_count == 2:
        start_fraction = 0.082 if column == 1 else 0.041
        start = max(3, round(panel_width * start_fraction))
        end = max(2, round(panel_width * 0.008))
    else:
        # Internal 3 x 4 transitions contain usable source ink almost to the
        # panel boundary. Keep the wider source-page margin only at the left
        # edge, and retain a small final margin for calibration geometry.
        start = max(3, round(panel_width * 0.045)) if column == 0 else 0
        end = (
            max(2, round(panel_width * 0.014))
            if column == column_count - 1
            else 0
        )
    return start, end


def labeled_six_by_two_time_ranges(
    evidence: np.ndarray,
    row_centers: list[int],
    *,
    pixels_per_mm: float,
    paper_speed_mm_per_second: float,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Locate the two five-second trace spans from cross-row source ink.

    A 6 x 2 page includes calibration/label margins and often a gutter, so
    splitting the raster into equal halves silently shifts and compresses
    time.  Repeated baseline ink across the six rows supplies a deterministic
    time origin, while the measured grid supplies the expected five-second
    width.  No waveform samples are created outside those source spans.
    """

    if evidence.ndim != 2 or len(row_centers) != 6:
        return None
    height, width = evidence.shape
    spacing = float(np.median(np.diff(row_centers)))
    radius = max(2, round(spacing * 0.02))
    row_support: list[np.ndarray] = []
    for center in row_centers:
        top = max(0, int(center) - radius)
        bottom = min(height, int(center) + radius + 1)
        if bottom <= top:
            return None
        row_support.append(np.max(evidence[top:bottom], axis=0))
    shared_support = np.median(np.stack(row_support), axis=0)
    active = (shared_support >= 0.24).astype(np.int32)
    duration_width = int(
        round(
            pixels_per_mm
            * paper_speed_mm_per_second
            * (SIX_BY_TWO_PANEL_SAMPLES / SAMPLE_RATE_HZ)
        )
    )
    nominal_panel_width = width / 2.0
    if duration_width < 40 or duration_width > width * 0.60:
        return None

    # A narrow baseline band is intentionally conservative, but it becomes
    # sparse when the six leads have different QRS/T amplitudes: their ink no
    # longer intersects the baselines in the same columns.  First look for
    # sustained source ink anywhere inside non-overlapping home-row bands.  A
    # calibration pulse is much shorter than one five-second panel, whereas a
    # genuine 6 x 2 trace appears either as two duration-sized runs separated
    # by a gutter or as one approximately two-duration run bridged by labels.
    row_band_radius = max(3, round(spacing * 0.42))
    row_band_support: list[np.ndarray] = []
    for center in row_centers:
        top = max(0, int(center) - row_band_radius)
        bottom = min(height, int(center) + row_band_radius + 1)
        if bottom <= top:
            return None
        row_band_support.append(np.max(evidence[top:bottom], axis=0))
    supported_row_count = np.sum(
        np.stack(row_band_support) >= 0.24,
        axis=0,
    )
    sustained_runs = [
        run
        for run in _true_runs(
            supported_row_count >= max(2, int(np.ceil(len(row_centers) / 2)))
        )
        if run.size >= duration_width * 0.75
    ]
    duration_matched_runs = [
        run
        for run in sustained_runs
        if duration_width * 0.85 <= run.size <= duration_width * 1.15
    ]
    if len(duration_matched_runs) >= 2:
        left_run = min(duration_matched_runs, key=lambda run: int(run[0]))
        right_run = max(duration_matched_runs, key=lambda run: int(run[-1]))
        left_range = (int(left_run[0]), int(left_run[0]) + duration_width)
        right_range = (int(right_run[-1]) + 1 - duration_width, int(right_run[-1]) + 1)
        if (
            0 <= left_range[0] < left_range[1] <= right_range[0]
            and right_range[1] <= width
        ):
            return left_range, right_range
    double_duration_runs = [
        run
        for run in sustained_runs
        if duration_width * 1.85 <= run.size <= duration_width * 2.15
    ]
    if double_duration_runs:
        run = min(
            double_duration_runs,
            key=lambda candidate: abs(candidate.size - duration_width * 2),
        )
        left_range = (int(run[0]), int(run[0]) + duration_width)
        right_range = (int(run[-1]) + 1 - duration_width, int(run[-1]) + 1)
        if left_range[1] <= right_range[0]:
            return left_range, right_range
        # A small grid-period bias can make two grid-derived panel widths
        # overlap even though the source ink supplies a well-bounded ten-
        # second span. In that case, split the observed span into two equal
        # source-time panels. The run has already passed the conservative
        # 1.85-2.15 duration check above, so this cannot turn arbitrary page
        # ink into a timing reference.
        source_start = int(run[0])
        source_end = int(run[-1]) + 1
        source_panel_width = int(round((source_end - source_start) / 2.0))
        if (
            source_panel_width >= 40
            and abs(source_panel_width - duration_width)
            <= duration_width * 0.08
        ):
            split = int(round((source_start + source_end) / 2.0))
            left_range = (source_start, split)
            right_range = (split, source_end)
            if left_range[0] < left_range[1] <= right_range[0] < right_range[1]:
                return left_range, right_range

    ranges: list[tuple[int, int]] = []
    for column in range(2):
        nominal_left = column * nominal_panel_width
        search_start = max(
            0,
            round(nominal_left + nominal_panel_width * 0.04),
        )
        search_end = min(
            width - duration_width,
            round(nominal_left + nominal_panel_width * 0.24),
        )
        if search_end < search_start:
            return None
        starts = np.arange(search_start, search_end + 1, dtype=np.int32)
        cumulative = np.concatenate(([0], np.cumsum(active, dtype=np.int64)))
        scores = cumulative[starts + duration_width] - cumulative[starts]
        best_index = int(np.argmax(scores))
        if int(scores[best_index]) < duration_width * 0.20:
            return None
        start = int(starts[best_index])
        ranges.append((start, start + duration_width))

    # On some rotated pages the left calibration pulse and lead labels form a
    # dense baseline-adjacent band, so the densest duration window starts near
    # the page edge even though the actual trace ends at the central gutter.
    # Use that gutter as the left panel's end only when all of the following
    # source checks agree: the original start is implausibly early, a sustained
    # inactive run exists immediately around the nominal split, the inferred
    # trace window stays separate from the right panel, and it retains nearly
    # the same source-ink support as the original window.
    if ranges[0][0] < width * 0.03:
        boundary_start = max(0, round(width * 0.485))
        boundary_end = min(width, round(width * 0.515))
        boundary_runs = [
            run + boundary_start
            for run in _true_runs(active[boundary_start:boundary_end] == 0)
            if run.size >= max(4, round(width * 0.004))
        ]
        if boundary_runs:
            gutter = max(boundary_runs, key=lambda run: int(run.size))
            inferred_end = int(gutter[0])
            inferred_start = inferred_end - duration_width
            original_score = int(
                cumulative[ranges[0][1]] - cumulative[ranges[0][0]]
            )
            inferred_score = (
                int(cumulative[inferred_end] - cumulative[inferred_start])
                if inferred_start >= 0
                else 0
            )
            if (
                inferred_start >= 0
                and inferred_start >= ranges[0][0] + width * 0.01
                and inferred_end <= ranges[1][0]
                and ranges[1][0] - inferred_end <= width * 0.08
                and inferred_score >= original_score * 0.85
            ):
                ranges[0] = (inferred_start, inferred_end)
    if ranges[0][1] > ranges[1][0]:
        return None
    return ranges[0], ranges[1]


def labeled_three_by_four_time_ranges(
    evidence: np.ndarray,
    row_centers: list[int] | tuple[int, ...],
    *,
    pixels_per_mm: float,
    paper_speed_mm_per_second: float,
    label_left_edges: list[float] | tuple[float, ...] | None = None,
) -> tuple[tuple[int, int], ...] | None:
    """Locate four 2.5-second panels without treating page margins as time.

    Label-validated 3 x 4 pages may place a left calibration/label margin before
    the first panel and may include a gap between later panels.  The measured
    grid fixes each panel's duration, while shared ink near all three row
    baselines selects the source start independently in each nominal quarter.
    """

    if evidence.ndim != 2 or len(row_centers) != 3:
        return None
    height, width = evidence.shape
    spacing = float(np.median(np.diff(row_centers)))
    radius = max(2, round(spacing * 0.055))
    row_support: list[np.ndarray] = []
    for center in row_centers:
        top = max(0, int(center) - radius)
        bottom = min(height, int(center) + radius + 1)
        if bottom <= top:
            return None
        row_support.append(np.max(evidence[top:bottom], axis=0))
    shared_support = np.median(np.stack(row_support), axis=0)
    active = (shared_support >= 0.24).astype(np.int32)
    duration_width = int(
        round(
            pixels_per_mm
            * paper_speed_mm_per_second
            * (THREE_BY_FOUR_PANEL_SAMPLES / SAMPLE_RATE_HZ)
        )
    )
    nominal_panel_width = width / 4.0
    if duration_width < 40 or duration_width > nominal_panel_width * 1.08:
        return None

    cumulative = np.concatenate(([0], np.cumsum(active, dtype=np.int64)))
    if label_left_edges is not None and len(label_left_edges) == 4:
        # Standard exporters place each printed lead name just inside the
        # panel origin.  Ink-onset timing is systematically late when the
        # label covers the baseline or the first samples contain a steep QRS:
        # rebasing that later onset to t=0 shifts every complex left.  The
        # repeated label grid supplies an independent origin one measured
        # minor square before the glyph edge.  Accept it only when all four
        # anchors reproduce the calibrated 2.5-second spacing and retain
        # sustained source ink across each proposed panel.
        label_starts = np.rint(
            np.asarray(label_left_edges, dtype=np.float64) - pixels_per_mm
        ).astype(np.int32)
        label_ranges = tuple(
            (int(start), int(start) + duration_width) for start in label_starts
        )
        spacing_tolerance = max(4, round(duration_width * 0.06))
        label_spacing_consistent = bool(
            np.all(np.isfinite(np.asarray(label_left_edges, dtype=np.float64)))
            and all(
                abs(int(label_starts[index + 1] - label_starts[index]) - duration_width)
                <= spacing_tolerance
                for index in range(3)
            )
        )
        label_ranges_in_bounds = bool(
            label_ranges[0][0] >= 0
            and label_ranges[-1][1] <= width
            and all(
                label_ranges[index][1] <= label_ranges[index + 1][0] + spacing_tolerance
                for index in range(3)
            )
        )
        label_source_supported = bool(
            all(
                int(cumulative[end] - cumulative[start])
                >= duration_width * 0.20
                for start, end in label_ranges
                if 0 <= start < end <= width
            )
            and all(0 <= start < end <= width for start, end in label_ranges)
        )
        if (
            label_spacing_consistent
            and label_ranges_in_bounds
            and label_source_supported
        ):
            return label_ranges

    ranges: list[tuple[int, int]] = []
    for column in range(4):
        nominal_left = column * nominal_panel_width
        search_start = (
            ranges[-1][1]
            if ranges
            else max(0, round(nominal_panel_width * 0.04))
        )
        search_end = min(
            width - duration_width,
            round(nominal_left + nominal_panel_width * 0.36),
        )
        if search_end < search_start:
            return None
        onset_window = max(12, round(duration_width * 0.04))
        sustained_onset: int | None = None
        for start in range(search_start, search_end + 1):
            window = active[start : start + onset_window]
            inactive_runs = _true_runs(window == 0)
            maximum_inactive_run = max(
                (int(run.size) for run in inactive_runs),
                default=0,
            )
            if (
                active[start]
                and float(np.mean(window)) >= 0.55
                and maximum_inactive_run <= max(4, onset_window // 3)
            ):
                sustained_onset = start
                break
        if sustained_onset is not None:
            ranges.append((sustained_onset, sustained_onset + duration_width))
            continue
        starts = np.arange(search_start, search_end + 1, dtype=np.int32)
        scores = cumulative[starts + duration_width] - cumulative[starts]
        best_score = int(np.max(scores))
        if best_score < duration_width * 0.20:
            return None
        # Prefer the earliest exact optimum so a label just before the
        # waveform cannot pull time zero into the first QRS complex.
        optimal = np.flatnonzero(scores == best_score)
        start = int(starts[int(optimal[0])])
        ranges.append((start, start + duration_width))

    return tuple(ranges)


def labeled_twelve_row_time_range(
    evidence: np.ndarray,
    row_centers: list[int] | tuple[int, ...],
    *,
    pixels_per_mm: float,
    paper_speed_mm_per_second: float,
) -> tuple[int, int] | None:
    """Locate a ten-second trace span on a label-anchored 12 x 1 page.

    Twelve-strip exports often include calibration and label space before the
    signal and a short page margin after it.  Mapping the full raster width to
    ten seconds shifts and compresses every lead.  Repeated source ink across
    the twelve validated rows establishes the time origin; the measured minor
    grid period establishes the expected duration.
    """

    if evidence.ndim != 2 or len(row_centers) != 12:
        return None
    height, width = evidence.shape
    spacing = float(np.median(np.diff(row_centers)))
    radius = max(2, round(spacing * 0.055))
    row_support: list[np.ndarray] = []
    for center in row_centers:
        top = max(0, int(center) - radius)
        bottom = min(height, int(center) + radius + 1)
        if bottom <= top:
            return None
        row_support.append(np.max(evidence[top:bottom], axis=0))
    shared_support = np.median(np.stack(row_support), axis=0)
    active = (shared_support >= 0.24).astype(np.int32)
    duration_width = int(
        round(
            pixels_per_mm
            * paper_speed_mm_per_second
            * PAGE_DURATION_SECONDS
        )
    )
    if duration_width < 80 or duration_width > width * 0.97:
        return None
    # Calibration pulses occupy the extreme left edge on each strip.  The
    # waveform itself begins near the validated label band, so exclude that
    # repeated non-signal geometry from the time-origin search.
    search_start = max(0, round(width * 0.045))
    search_end = min(width - duration_width, round(width * 0.16))
    if search_end < search_start:
        return None
    onset_window = max(8, round(duration_width * 0.012))
    sustained_onsets: list[int] = []
    for column in range(search_start, search_end + 1):
        window = active[column : column + onset_window]
        inactive_runs = _true_runs(window == 0)
        maximum_inactive_run = max(
            (int(run.size) for run in inactive_runs), default=0
        )
        if (
            active[column]
            and float(np.mean(window)) >= 0.55
            and maximum_inactive_run <= max(3, onset_window // 4)
        ):
            sustained_onsets.append(column)
    if sustained_onsets:
        start = int(sustained_onsets[0])
        return start, start + duration_width

    # Fall back to the densest duration window for faint or intermittently
    # interrupted traces where no sustained onset survives the evidence floor.
    starts = np.arange(search_start, search_end + 1, dtype=np.int32)
    cumulative = np.concatenate(([0], np.cumsum(active, dtype=np.int64)))
    scores = cumulative[starts + duration_width] - cumulative[starts]
    best_index = int(np.argmax(scores))
    if int(scores[best_index]) < duration_width * 0.20:
        return None
    start = int(starts[best_index])
    return start, start + duration_width


def labeled_twelve_source_consensus_time_range(
    evidence: np.ndarray,
    row_centers: list[int] | tuple[int, ...],
    *,
    raw_grid_period_x: float,
    paper_speed_mm_per_second: float,
) -> tuple[tuple[int, int], dict[str, Any]] | None:
    """Recover a 12 x 1 source span when calibrated grid scale is unavailable.

    The printed lead labels establish the layout, but JPEG degradation can
    erase the minor vertical grid and defeat calibration. A full-width trace
    remains simultaneously visible near all twelve row baselines. Its two
    sustained boundaries establish a candidate ten-second span; an observed
    horizontal grid period must independently agree with either a 1 mm minor
    or 5 mm major grid before the span is accepted.
    """

    if (
        evidence.ndim != 2
        or len(row_centers) != 12
        or not np.isfinite(raw_grid_period_x)
        or raw_grid_period_x <= 0
        or not np.isfinite(paper_speed_mm_per_second)
        or paper_speed_mm_per_second <= 0
    ):
        return None
    height, width = evidence.shape
    spacing = float(np.median(np.diff(row_centers)))
    radius = max(2, round(spacing * 0.055))
    row_support: list[np.ndarray] = []
    for center in row_centers:
        top = max(0, int(center) - radius)
        bottom = min(height, int(center) + radius + 1)
        if bottom <= top:
            return None
        row_support.append(np.max(evidence[top:bottom], axis=0))
    shared_support = np.median(np.stack(row_support), axis=0)
    active = shared_support >= 0.24
    edge_window = max(24, round(width * 0.024))
    minimum_edge_run = max(5, round(width * 0.004))
    maximum_inactive_run = max(4, edge_window // 4)

    start: int | None = None
    for column in range(round(width * 0.04), round(width * 0.20) + 1):
        window = active[column : column + edge_window]
        if window.size != edge_window or not window[0]:
            continue
        active_runs = _true_runs(window)
        inactive_runs = _true_runs(~window)
        if (
            active_runs
            and int(active_runs[0][0]) == 0
            and int(active_runs[0].size) >= minimum_edge_run
            and float(np.mean(window)) >= 0.72
            and max((int(run.size) for run in inactive_runs), default=0)
            <= maximum_inactive_run
        ):
            start = column
            break
    if start is None:
        return None

    end: int | None = None
    for column in range(
        round(width * 0.98),
        round(width * 0.80) - 1,
        -1,
    ):
        window_start = column - edge_window
        if window_start < 0:
            continue
        window = active[window_start:column]
        if window.size != edge_window or not window[-1]:
            continue
        active_runs = _true_runs(window)
        inactive_runs = _true_runs(~window)
        if (
            active_runs
            and int(active_runs[-1][-1]) == edge_window - 1
            and int(active_runs[-1].size) >= minimum_edge_run
            and float(np.mean(window)) >= 0.72
            and max((int(run.size) for run in inactive_runs), default=0)
            <= maximum_inactive_run
        ):
            end = column
            break
    if end is None or end <= start:
        return None

    span = end - start
    span_fraction = span / width
    if not 0.78 <= span_fraction <= 0.96:
        return None
    inferred_pixels_per_mm = span / (
        paper_speed_mm_per_second * PAGE_DURATION_SECONDS
    )
    if not 1.0 <= inferred_pixels_per_mm <= 40.0:
        return None
    raw_period_scale_ratio = raw_grid_period_x / inferred_pixels_per_mm
    inferred_grid_scale_mm = min(
        (1.0, 5.0),
        key=lambda scale: abs(raw_period_scale_ratio - scale),
    )
    scale_error_fraction = abs(
        raw_period_scale_ratio - inferred_grid_scale_mm
    ) / inferred_grid_scale_mm
    if scale_error_fraction > 0.12:
        return None

    return (start, end), {
        "method": "shared-row-span-plus-grid-ratio-v1",
        "sourceStartPixel": start,
        "sourceEndPixel": end,
        "sourceSpanFraction": span_fraction,
        "inferredPixelsPerMm": inferred_pixels_per_mm,
        "rawGridPeriodPixels": raw_grid_period_x,
        "inferredGridScaleMm": inferred_grid_scale_mm,
        "gridScaleErrorFraction": scale_error_fraction,
        "quantitativeCalibrationConfirmed": False,
    }


def _sustained_source_trace_span(active: np.ndarray) -> tuple[int, int] | None:
    if active.ndim != 1 or active.size < 100:
        return None
    width = int(active.size)
    edge_window = max(24, round(width * 0.024))
    minimum_edge_run = max(5, round(width * 0.004))
    maximum_inactive_run = max(4, edge_window // 4)

    start: int | None = None
    for column in range(round(width * 0.04), round(width * 0.20) + 1):
        window = active[column : column + edge_window]
        if window.size != edge_window or not window[0]:
            continue
        active_runs = _true_runs(window)
        inactive_runs = _true_runs(~window)
        if (
            active_runs
            and int(active_runs[0][0]) == 0
            and int(active_runs[0].size) >= minimum_edge_run
            and float(np.mean(window)) >= 0.72
            and max((int(run.size) for run in inactive_runs), default=0)
            <= maximum_inactive_run
        ):
            start = column
            break
    if start is None:
        return None

    end: int | None = None
    for column in range(
        round(width * 0.98),
        round(width * 0.80) - 1,
        -1,
    ):
        window_start = column - edge_window
        if window_start < 0:
            continue
        window = active[window_start:column]
        if window.size != edge_window or not window[-1]:
            continue
        active_runs = _true_runs(window)
        inactive_runs = _true_runs(~window)
        if (
            active_runs
            and int(active_runs[-1][-1]) == edge_window - 1
            and int(active_runs[-1].size) >= minimum_edge_run
            and float(np.mean(window)) >= 0.72
            and max((int(run.size) for run in inactive_runs), default=0)
            <= maximum_inactive_run
        ):
            end = column
            break
    return (start, end) if end is not None and end > start else None


def _source_trace_edge_span(active: np.ndarray) -> tuple[int, int] | None:
    """Find ECG trace edges without requiring a horizontal baseline onset."""

    if active.ndim != 1 or active.size < 100:
        return None
    width = int(active.size)
    left_start = round(width * 0.04)
    left_end = round(width * 0.20)
    right_start = round(width * 0.80)
    right_end = round(width * 0.98)
    continuity_window = max(32, round(width * 0.04))

    start_candidates = (
        np.flatnonzero(active[left_start : left_end + 1]) + left_start
    )
    start = next(
        (
            int(column)
            for column in start_candidates
            if float(
                np.mean(active[column : min(width, column + continuity_window)])
            )
            >= 0.45
        ),
        None,
    )
    if start is None:
        return None

    end_candidates = (
        np.flatnonzero(active[right_start : right_end + 1])
        + right_start
        + 1
    )
    end = next(
        (
            int(column)
            for column in end_candidates[::-1]
            if float(
                np.mean(active[max(0, column - continuity_window) : column])
            )
            >= 0.45
        ),
        None,
    )
    return (start, end) if end is not None and end > start else None


def labeled_twelve_source_consensus_time_ranges(
    evidence: np.ndarray,
    row_centers: list[int] | tuple[int, ...],
    *,
    raw_grid_period_x: float,
    paper_speed_mm_per_second: float,
    calibration_exclusion_end: int | None = None,
) -> tuple[tuple[tuple[int, int], ...], dict[str, Any]] | None:
    """Infer row-local source spans with grid-ratio and projective checks."""

    if (
        evidence.ndim != 2
        or len(row_centers) != 12
        or not np.isfinite(raw_grid_period_x)
        or raw_grid_period_x <= 0
        or not np.isfinite(paper_speed_mm_per_second)
        or paper_speed_mm_per_second <= 0
    ):
        return None
    height, width = evidence.shape
    spacing = float(np.median(np.diff(row_centers)))
    # The first source sample can be a nearly vertical QRS deflection, so a
    # narrow baseline band may not see sustained ink until tens of pixels
    # later. A wider non-overlapping home-row band captures that connected
    # onset while remaining far from the printed label row above it.
    radius = max(2, round(spacing * 0.20))
    ranges: list[tuple[int, int]] = []
    for center in row_centers:
        top = max(0, int(center) - radius)
        bottom = min(height, int(center) + radius + 1)
        if bottom <= top:
            return None
        support = np.max(evidence[top:bottom], axis=0)
        active = support >= 0.70
        if calibration_exclusion_end is not None:
            # A calibration tail can precede a long blank gap and still meet
            # the onset window's aggregate continuity fraction. Its detected
            # source extent is not waveform time, regardless of grid agreement.
            active[: max(0, int(calibration_exclusion_end) + 1)] = False
        source_span = _source_trace_edge_span(active)
        if source_span is None:
            return None
        ranges.append(source_span)

    starts = np.asarray([item[0] for item in ranges], dtype=np.float64)
    ends = np.asarray([item[1] for item in ranges], dtype=np.float64)
    lower_search_boundary = round(width * 0.04)
    upper_search_boundary = round(width * 0.98)
    if (
        int(np.count_nonzero(starts <= lower_search_boundary)) > 2
        or int(np.count_nonzero(ends >= upper_search_boundary)) > 2
    ):
        return None
    row_axis = np.asarray(row_centers, dtype=np.float64)

    def projective_predictions(values: np.ndarray) -> np.ndarray:
        slopes = np.asarray(
            [
                (values[right] - values[left])
                / (row_axis[right] - row_axis[left])
                for left in range(12)
                for right in range(left + 1, 12)
                if row_axis[right] != row_axis[left]
            ],
            dtype=np.float64,
        )
        if slopes.size == 0:
            return np.asarray([], dtype=np.int32)
        slope = float(np.median(slopes))
        intercept = float(np.median(values - slope * row_axis))
        return np.rint(intercept + slope * row_axis).astype(np.int32)

    predicted_starts = projective_predictions(starts)
    predicted_ends = projective_predictions(ends)
    if predicted_starts.size != 12 or predicted_ends.size != 12:
        return None
    initial_spans = ends - starts
    inferred_pixels_per_mm = float(np.median(initial_spans)) / (
        paper_speed_mm_per_second * PAGE_DURATION_SECONDS
    )
    if not 1.0 <= inferred_pixels_per_mm <= 40.0:
        return None
    boundary_tolerance = max(2, int(np.ceil(inferred_pixels_per_mm * 0.75)))
    start_outliers = np.abs(starts - predicted_starts) > boundary_tolerance
    end_outliers = np.abs(ends - predicted_ends) > boundary_tolerance
    corrected_rows = start_outliers | end_outliers
    correction_count = int(np.count_nonzero(corrected_rows))
    if correction_count > 3:
        return None
    # Time zero and ten seconds are straight page lines under a projective
    # transform. Once the independently detected boundaries have no more than
    # three gross outliers, project every row onto those robust lines instead
    # of preserving several pixels of row-local onset noise.
    corrected_starts = predicted_starts
    corrected_ends = predicted_ends
    corrected_spans = corrected_ends - corrected_starts
    if np.any(corrected_spans <= 0):
        return None
    span_fractions = corrected_spans / width
    if np.any(span_fractions < 0.78) or np.any(span_fractions > 0.96):
        return None
    inferred_pixels_per_mm = float(np.median(corrected_spans)) / (
        paper_speed_mm_per_second * PAGE_DURATION_SECONDS
    )
    raw_period_scale_ratio = raw_grid_period_x / inferred_pixels_per_mm
    inferred_grid_scale_mm = min(
        (1.0, 5.0),
        key=lambda scale: abs(raw_period_scale_ratio - scale),
    )
    scale_error_fraction = abs(
        raw_period_scale_ratio - inferred_grid_scale_mm
    ) / inferred_grid_scale_mm
    if scale_error_fraction > 0.12:
        return None

    corrected_ranges = tuple(
        (int(start), int(end))
        for start, end in zip(corrected_starts, corrected_ends, strict=True)
    )
    return corrected_ranges, {
        "method": "row-local-source-edge-plus-grid-ratio-v2",
        "correctedRowCount": correction_count,
        "projectedRowCount": int(
            np.count_nonzero(
                (corrected_starts != starts) | (corrected_ends != ends)
            )
        ),
        "maximumCorrectedRows": 3,
        "boundaryTolerancePixels": boundary_tolerance,
        "rawStarts": [int(value) for value in starts],
        "rawEnds": [int(value) for value in ends],
        "correctedStarts": [int(value) for value in corrected_starts],
        "correctedEnds": [int(value) for value in corrected_ends],
        "inferredPixelsPerMm": inferred_pixels_per_mm,
        "rawGridPeriodPixels": raw_grid_period_x,
        "inferredGridScaleMm": inferred_grid_scale_mm,
        "gridScaleErrorFraction": scale_error_fraction,
        "quantitativeCalibrationConfirmed": False,
    }


def calibration_confirms_labeled_twelve_source_timing(
    calibration: dict[str, Any],
    source_timing: dict[str, Any],
) -> bool:
    """Require pulse/grid calibration that agrees with the source span."""

    grid_scale = calibration.get("gridScaleMmX", calibration.get("gridScaleMm"))
    inferred_grid_scale = source_timing.get("inferredGridScaleMm")
    inferred_pixels_per_mm = float(
        source_timing.get("inferredPixelsPerMm", 0.0)
    )
    pixels_per_mm_x = float(calibration.get("pixelsPerMmX", 0.0))
    pixels_per_mm_y = float(calibration.get("pixelsPerMmY", 0.0))
    agreement_limit = 0.12
    return bool(
        calibration.get("detected")
        and float(calibration.get("confidence", 0.0)) >= 0.35
        and calibration.get("gridScaleDetected") is True
        and grid_scale in {1.0, 5.0}
        and inferred_grid_scale == grid_scale
        and float(source_timing.get("gridScaleErrorFraction", 1.0))
        <= agreement_limit
        and 1.0 <= pixels_per_mm_x <= 40.0
        and 1.0 <= pixels_per_mm_y <= 40.0
        and 1.0 <= inferred_pixels_per_mm <= 40.0
        and abs(inferred_pixels_per_mm - pixels_per_mm_x)
        / pixels_per_mm_x
        <= agreement_limit
        and abs(inferred_pixels_per_mm - pixels_per_mm_y)
        / pixels_per_mm_y
        <= agreement_limit
        and float(calibration.get("paperSpeedMmPerSecond", 0.0))
        in {25.0, 50.0}
        and float(calibration.get("gainMmPerMv", 0.0))
        in {5.0, 10.0, 20.0}
        and int(calibration.get("pulseEndX", 0))
        > int(calibration.get("pulseStartX", 0))
    )


def allow_labeled_twelve_global_time_fallback(
    calibrated_time_ranges: tuple[tuple[int, int], ...] | None,
    timing_diagnostics: dict[str, Any],
    *,
    grid_scale_detected: bool,
    raw_grid_period_scale_mm: float | None,
) -> bool:
    """Permit legacy global timing only when row consensus did not reject it."""

    return bool(
        calibrated_time_ranges is None
        and timing_diagnostics.get("accepted") is not False
        and (grid_scale_detected or raw_grid_period_scale_mm is not None)
    )


def labeled_twelve_source_timing_disagrees_with_global(
    source_time_ranges: tuple[tuple[int, int], ...],
    global_time_range: tuple[int, int],
    *,
    tolerance_pixels: int,
) -> bool:
    """Reject a global span contradicted by robust row-local source edges."""

    if len(source_time_ranges) != 12 or tolerance_pixels < 0:
        return True
    starts = np.asarray([item[0] for item in source_time_ranges], dtype=np.float64)
    ends = np.asarray([item[1] for item in source_time_ranges], dtype=np.float64)
    return bool(
        abs(float(np.median(starts)) - global_time_range[0])
        > tolerance_pixels
        or abs(float(np.median(ends)) - global_time_range[1])
        > tolerance_pixels
    )


def labeled_twelve_row_time_ranges(
    evidence: np.ndarray,
    row_centers: list[int] | tuple[int, ...],
    *,
    row_pixels_per_mm: list[float] | tuple[float, ...],
    paper_speed_mm_per_second: float,
    calibration_exclusion_end: int | None = None,
    timing_diagnostics: dict[str, Any] | None = None,
) -> tuple[tuple[int, int], ...] | None:
    """Locate perspective-aware ten-second spans for twelve labeled rows."""

    if (
        evidence.ndim != 2
        or len(row_centers) != 12
        or len(row_pixels_per_mm) != 12
    ):
        return None
    height, width = evidence.shape
    spacing = float(np.median(np.diff(row_centers)))
    radius = max(2, round(spacing * 0.055))
    ranges: list[tuple[int, int]] = []
    for center, pixels_per_mm in zip(
        row_centers, row_pixels_per_mm, strict=True
    ):
        if not np.isfinite(pixels_per_mm) or not 1.0 <= pixels_per_mm <= 40.0:
            return None
        duration_width = int(
            round(
                pixels_per_mm
                * paper_speed_mm_per_second
                * PAGE_DURATION_SECONDS
            )
        )
        if duration_width < 80 or duration_width > width * 0.97:
            return None
        top = max(0, int(center) - radius)
        bottom = min(height, int(center) + radius + 1)
        if bottom <= top:
            return None
        support = np.max(evidence[top:bottom], axis=0)
        active = (support >= 0.24).astype(np.int32)
        search_start = max(0, round(width * 0.012))
        if calibration_exclusion_end is not None:
            search_start = max(
                search_start,
                int(calibration_exclusion_end),
            )
        search_end = min(width - duration_width, round(width * 0.18))
        if search_end < search_start:
            return None
        # A standard calibration plateau is 5 mm wide (200 ms at 25 mm/s).
        # Require more than that much sustained trace support so a complete
        # rectangular pulse cannot become time zero. If the calibration pulse
        # was not localized globally, also step over compact leading source-ink
        # runs whose physical width is calibration-like.
        calibration_like_width = max(3, round(pixels_per_mm * 5.5))
        if calibration_exclusion_end is None:
            early_end = min(
                search_end + 1,
                max(search_start, round(width * 0.08)),
            )
            for run in _true_runs(active[search_start:early_end] > 0):
                run_start = int(run[0]) + search_start
                run_end = int(run[-1]) + search_start + 1
                if run_start > width * 0.045:
                    break
                if run.size <= calibration_like_width:
                    search_start = max(search_start, run_end)
        if search_end < search_start:
            return None
        onset_window = max(
            8,
            round(duration_width * 0.012),
            round(pixels_per_mm * 6.0),
        )
        start: int | None = None
        for column in range(search_start, search_end + 1):
            window = active[column : column + onset_window]
            inactive_runs = _true_runs(window == 0)
            maximum_inactive_run = max(
                (int(run.size) for run in inactive_runs), default=0
            )
            if (
                active[column]
                and float(np.mean(window)) >= 0.55
                and maximum_inactive_run <= max(3, onset_window // 4)
            ):
                start = column
                break
        if start is None:
            starts = np.arange(search_start, search_end + 1, dtype=np.int32)
            cumulative = np.concatenate(
                ([0], np.cumsum(active, dtype=np.int64))
            )
            scores = cumulative[starts + duration_width] - cumulative[starts]
            best_index = int(np.argmax(scores))
            if int(scores[best_index]) < duration_width * 0.20:
                return None
            start = int(starts[best_index])
        ranges.append((start, start + duration_width))

    starts = np.asarray([item[0] for item in ranges], dtype=np.float64)
    row_axis = np.asarray(row_centers, dtype=np.float64)
    pairwise_slopes = np.asarray(
        [
            (starts[right] - starts[left])
            / (row_axis[right] - row_axis[left])
            for left in range(12)
            for right in range(left + 1, 12)
            if row_axis[right] != row_axis[left]
        ],
        dtype=np.float64,
    )
    if pairwise_slopes.size == 0:
        return None
    robust_slope = float(np.median(pairwise_slopes))
    robust_intercept = float(np.median(starts - robust_slope * row_axis))
    predicted_starts = np.rint(
        robust_intercept + robust_slope * row_axis
    ).astype(np.int32)
    origin_tolerance = max(
        2,
        int(np.ceil(float(np.median(row_pixels_per_mm)) * 0.75)),
    )
    origin_outliers = np.abs(starts - predicted_starts) > origin_tolerance
    correction_count = int(np.count_nonzero(origin_outliers))
    corrected_starts = starts.astype(np.int32)
    consensus_accepted = correction_count <= 3
    if consensus_accepted:
        corrected_starts[origin_outliers] = predicted_starts[origin_outliers]
        ranges = [
            (int(start), int(start + (end - original_start)))
            for start, (original_start, end) in zip(
                corrected_starts,
                ranges,
                strict=True,
            )
        ]
    if timing_diagnostics is not None:
        timing_diagnostics.update(
            {
                "method": "robust-projective-row-origin-consensus-v1",
                "accepted": consensus_accepted,
                "applied": bool(consensus_accepted and correction_count),
                "correctedRowCount": correction_count if consensus_accepted else 0,
                "maximumCorrectedRows": 3,
                "originTolerancePixels": origin_tolerance,
                "maximumRawResidualPixels": int(
                    np.max(np.abs(starts - predicted_starts))
                ),
                "rawStarts": [int(value) for value in starts],
                "predictedStarts": [int(value) for value in predicted_starts],
                "correctedStarts": [
                    int(value) for value in corrected_starts
                ]
                if consensus_accepted
                else None,
            }
        )
    if not consensus_accepted:
        return None

    starts = np.asarray([item[0] for item in ranges], dtype=np.float64)
    ends = np.asarray([item[1] for item in ranges], dtype=np.float64)
    start_fit = np.polyval(np.polyfit(row_axis, starts, 1), row_axis)
    end_fit = np.polyval(np.polyfit(row_axis, ends, 1), row_axis)
    # A perspective warp moves both boundaries smoothly. Reject independent
    # artifact onsets rather than assigning twelve unrelated time origins.
    if (
        float(np.max(np.abs(starts - start_fit))) > width * 0.025
        or float(np.max(np.abs(ends - end_fit))) > width * 0.035
        or float(np.ptp(starts)) > width * 0.18
        or float(np.ptp(ends)) > width * 0.18
    ):
        return None
    return tuple(ranges)


def grid_period_to_pixels_per_mm(
    period_pixels: float,
    *,
    period_scale_mm: float | None,
) -> float | None:
    """Convert a detector period only when its physical grid scale is known.

    Pixel magnitude cannot distinguish a high-resolution 1 mm minor grid from
    a lower-resolution 5 mm major grid. The detector must therefore supply the
    independently established physical period instead of this layer guessing
    from a pixel threshold.
    """

    if (
        not np.isfinite(period_pixels)
        or period_pixels <= 0
        or period_scale_mm not in {1.0, 5.0}
    ):
        return None
    return period_pixels / float(period_scale_mm)


def timing_scale_pixels_per_mm(
    *,
    row_pixels_per_mm_x: list[float],
    calibrated_pixels_per_mm: float,
    raw_grid_period_x: float,
    raw_grid_period_scale_mm: float | None,
    prefer_raw_grid: bool,
) -> float:
    """Choose a timing scale without changing established 6 x 2 calibration."""

    if not prefer_raw_grid:
        return (
            min(row_pixels_per_mm_x)
            if len(row_pixels_per_mm_x) == 6
            else calibrated_pixels_per_mm
        )
    raw_grid_pixels_per_mm_x = grid_period_to_pixels_per_mm(
        raw_grid_period_x,
        period_scale_mm=raw_grid_period_scale_mm,
    )
    return (
        raw_grid_pixels_per_mm_x
        if raw_grid_pixels_per_mm_x is not None
        and 1.0 <= raw_grid_pixels_per_mm_x <= 40.0
        else calibrated_pixels_per_mm
    )


def contiguous_rhythm_time_range(
    source_time_ranges: tuple[tuple[int, int], ...] | None,
    *,
    page_width: int,
) -> tuple[int, int] | None:
    """Map a continuous rhythm strip without including segmented-panel gutters."""

    if not source_time_ranges or page_width <= 0:
        return None
    previous_end: int | None = None
    duration_width = 0
    for start, end in source_time_ranges:
        if (
            start < 0
            or end <= start
            or end > page_width
            or (previous_end is not None and start < previous_end)
        ):
            return None
        duration_width += end - start
        previous_end = end
    rhythm_start = source_time_ranges[0][0]
    rhythm_end = rhythm_start + duration_width
    if rhythm_end > page_width:
        return None
    return rhythm_start, rhythm_end


def native_path_to_canonical(
    columns: np.ndarray,
    path: np.ndarray,
    *,
    panel_start: int,
    panel_width: int,
    panel_samples: int,
    pixels_per_mm: float,
    gain_mm_per_mv: float = 10.0,
    validity: np.ndarray | None = None,
) -> np.ndarray:
    result = np.full(panel_samples, np.nan, dtype=np.float64)
    if validity is None:
        validity = np.ones(columns.size, dtype=bool)
    if validity.shape != columns.shape:
        raise ValueError("Native path validity must match its source columns.")
    if not np.any(validity):
        return result

    local_columns = columns.astype(np.float64) - float(panel_start)
    native_samples = local_columns / max(panel_width - 1, 1) * (panel_samples - 1)
    baseline = float(np.median(path[validity]))
    for run in _true_runs(validity):
        if run.size < 2:
            continue
        first_sample = max(0, int(np.ceil(native_samples[run[0]])))
        last_sample = min(
            panel_samples - 1,
            int(np.floor(native_samples[run[-1]])),
        )
        if last_sample < first_sample:
            continue
        sample_indexes = np.arange(first_sample, last_sample + 1)
        interpolated_path = np.interp(
            sample_indexes,
            native_samples[run],
            path[run],
        )
        result[sample_indexes] = (
            -(interpolated_path - baseline)
            * (1000.0 / gain_mm_per_mv)
            / pixels_per_mm
        )
    return result


def fidelity_for_path(
    evidence: np.ndarray,
    columns: np.ndarray,
    path: np.ndarray,
    *,
    panel_width: int,
    row_center: float,
    row_spacing: float,
    validity: np.ndarray | None = None,
    safety_metrics: dict[str, int] | None = None,
) -> LeadFidelity:
    if validity is None:
        validity = np.ones(columns.size, dtype=bool)
    if not np.any(validity):
        raise ValueError("A source-fidelity path must retain at least one sample.")
    safety_metrics = safety_metrics or {}
    support = evidence[path[validity], columns[validity]]
    all_jumps = np.abs(np.diff(path.astype(np.float64)))
    valid_pairs = validity[:-1] & validity[1:]
    jumps = all_jumps[valid_pairs]
    large_jump_indexes = np.flatnonzero(
        valid_pairs & (all_jumps > row_spacing * 0.25)
    )
    large_jump_support: list[float] = []
    for index in large_jump_indexes:
        lower = int(min(path[index], path[index + 1]))
        upper = int(max(path[index], path[index + 1])) + 1
        left = max(0, int(columns[index]) - 3)
        right = min(evidence.shape[1], int(columns[index + 1]) + 4)
        vertical_support = np.max(evidence[lower:upper, left:right], axis=1)
        large_jump_support.append(float(np.mean(vertical_support >= 0.24)))
    unsupported_large_jumps = sum(value < 0.35 for value in large_jump_support)
    return LeadFidelity(
        evidenceMedian=float(np.median(support)),
        evidenceP10=float(np.quantile(support, 0.10)),
        maxNativeJumpPixels=float(np.max(jumps)) if jumps.size else 0.0,
        p95NativeJumpPixels=float(np.quantile(jumps, 0.95))
        if jumps.size
        else 0.0,
        coverage=float(np.count_nonzero(validity) / max(panel_width, 1)),
        sourceStartPixel=int(columns[np.flatnonzero(validity)[0]]),
        sourceEndPixel=int(columns[np.flatnonzero(validity)[-1]] + 1),
        homeRowFraction=float(
            np.mean(
                np.abs(path[validity].astype(np.float64) - row_center)
                <= row_spacing * 0.42
            )
        ),
        largeJumpCount=len(large_jump_support),
        unsupportedLargeJumpCount=unsupported_large_jumps,
        minimumLargeJumpSupport=min(large_jump_support, default=1.0),
        qrsAnchorShiftPixels=safety_metrics.get("qrsAnchorShiftPixels", 0),
        qrsAnchorMatchCount=safety_metrics.get("qrsAnchorMatchCount", 0),
        largeExcursionCount=safety_metrics.get("largeExcursionCount", 0),
        recoveredExcursionCount=safety_metrics.get(
            "recoveredExcursionCount",
            0,
        ),
        recoveredSampleCount=safety_metrics.get("recoveredSampleCount", 0),
        rejectedExcursionCount=safety_metrics.get("rejectedExcursionCount", 0),
        rejectedSampleCount=safety_metrics.get("rejectedSampleCount", 0),
        disconnectedExcursionCount=safety_metrics.get(
            "disconnectedExcursionCount",
            0,
        ),
        verticalArtifactExcursionCount=safety_metrics.get(
            "verticalArtifactExcursionCount",
            0,
        ),
        unsafeExcursionCount=safety_metrics.get("unsafeExcursionCount", 0),
    )


def source_boundary_excursion_count(
    evidence: np.ndarray,
    *,
    x_start: int,
    x_end: int,
    boundary: str,
    band_pixels: int = 4,
) -> int:
    """Count source-ink runs clipped by the top or bottom raster boundary."""

    if evidence.ndim != 2 or boundary not in {"top", "bottom"}:
        raise ValueError("Boundary clipping requires 2D evidence and top/bottom.")
    left = max(0, int(x_start))
    right = min(evidence.shape[1], int(x_end))
    if right <= left:
        return 0
    band_size = max(1, min(int(band_pixels), evidence.shape[0]))
    band = evidence[:band_size] if boundary == "top" else evidence[-band_size:]
    active = np.max(band[:, left:right], axis=0) >= 0.70
    return sum(int(run.size) >= 2 for run in _true_runs(active))


def page_boundary_clipped_validity(
    evidence: np.ndarray,
    columns: np.ndarray,
    *,
    boundary: str,
    guard_columns: int = 2,
    band_pixels: int = 4,
) -> tuple[np.ndarray, dict[str, int]]:
    """Preserve a gap where the source page clips an ECG excursion.

    A path on the first or last raster row is a visible amplitude bound, not
    a measured ECG value.  Only trace bands that physically touch the source
    page are eligible; ordinary crossings of internal row bounds stay valid.
    """

    if evidence.ndim != 2 or boundary not in {"top", "bottom"}:
        raise ValueError("Boundary clipping requires 2D evidence and top/bottom.")
    validity = np.ones(columns.size, dtype=bool)
    band_size = max(1, min(int(band_pixels), evidence.shape[0]))
    band = evidence[:band_size] if boundary == "top" else evidence[-band_size:]
    clipped = np.max(band[:, columns], axis=0) >= 0.70
    runs = [run for run in _true_runs(clipped) if int(run.size) >= 2]
    for run in runs:
        left = max(0, int(run[0]) - guard_columns)
        right = min(columns.size, int(run[-1]) + guard_columns + 1)
        validity[left:right] = False
    return validity, {
        "rejectedExcursionCount": len(runs),
        "rejectedSampleCount": int(np.count_nonzero(~validity)),
        # Every detected page-edge run is removed from the quantitative trace;
        # none remains as an unsafe finite excursion.
        "unsafeExcursionCount": 0,
    }


def source_fidelity_report(
    lead_metrics: dict[str, LeadFidelity],
    *,
    pixels_per_mm: float,
    row_spacing: float,
    layout_confidence: float,
    method: str = "rhythm-anchored-connected-multievidence-native-path-v6",
    allow_ink_backed_crossings: bool = False,
    rhythm_anchor_count: int = 0,
    rhythm_trace_passed: bool = True,
    minimum_layout_confidence: float = 0.45,
    require_rhythm_anchors: bool = True,
    minimum_evidence_median: float = 0.70,
    minimum_evidence_p10: float = 0.45,
    maximum_crossing_p95_fraction: float = 0.45,
    maximum_crossing_jump_fraction: float = 1.15,
    unsupported_large_jump_fraction: float = 0.40,
    minimum_home_row_fraction: float = 0.55,
) -> dict[str, Any]:
    metrics = list(lead_metrics.values())
    evidence_median = float(np.median([item.evidenceMedian for item in metrics]))
    evidence_p10 = float(min(item.evidenceP10 for item in metrics))
    max_jump = float(max(item.maxNativeJumpPixels for item in metrics))
    p95_jump = float(max(item.p95NativeJumpPixels for item in metrics))
    minimum_coverage = float(min(item.coverage for item in metrics))
    observed_minimum_home_row_fraction = float(
        min(item.homeRowFraction for item in metrics)
    )
    large_jump_count = sum(item.largeJumpCount for item in metrics)
    unsupported_large_jump_count = sum(
        item.unsupportedLargeJumpCount for item in metrics
    )
    rejected_excursion_count = sum(item.rejectedExcursionCount for item in metrics)
    rejected_sample_count = sum(item.rejectedSampleCount for item in metrics)
    recovered_excursion_count = sum(
        item.recoveredExcursionCount for item in metrics
    )
    recovered_sample_count = sum(item.recoveredSampleCount for item in metrics)
    unsafe_excursion_count = sum(item.unsafeExcursionCount for item in metrics)
    maximum_unsupported_jumps = max(
        2,
        int(
            np.ceil(
                large_jump_count
                * (
                    unsupported_large_jump_fraction
                    if allow_ink_backed_crossings
                    else 0.12
                )
            )
        ),
    )
    minimum_required_coverage = 0.80 if allow_ink_backed_crossings else 0.85
    shared_checks_passed = (
        layout_confidence >= minimum_layout_confidence
        and evidence_median >= minimum_evidence_median
        and evidence_p10 >= minimum_evidence_p10
        and minimum_coverage >= minimum_required_coverage
    )
    if allow_ink_backed_crossings:
        jump_checks_passed = (
            p95_jump <= row_spacing * maximum_crossing_p95_fraction
            and max_jump <= row_spacing * maximum_crossing_jump_fraction
            and observed_minimum_home_row_fraction >= minimum_home_row_fraction
            and unsupported_large_jump_count <= maximum_unsupported_jumps
            and unsafe_excursion_count == 0
            and (not require_rhythm_anchors or rhythm_anchor_count >= 3)
            and rhythm_trace_passed
        )
    else:
        jump_checks_passed = (
            p95_jump <= row_spacing * 0.25
            and max_jump <= row_spacing * 0.50
        )
    passed = bool(shared_checks_passed and jump_checks_passed)
    return {
        "passed": passed,
        "method": method,
        "pixelsPerMm": pixels_per_mm,
        "layoutConfidence": layout_confidence,
        "minimumLayoutConfidence": minimum_layout_confidence,
        "evidenceMedian": evidence_median,
        "minimumEvidenceMedian": minimum_evidence_median,
        "evidenceP10": evidence_p10,
        "minimumEvidenceP10": minimum_evidence_p10,
        "maxNativeJumpPixels": max_jump,
        "p95NativeJumpPixels": p95_jump,
        "minimumCoverage": minimum_coverage,
        "minimumRequiredCoverage": minimum_required_coverage,
        "minimumHomeRowFraction": observed_minimum_home_row_fraction,
        "requiredMinimumHomeRowFraction": minimum_home_row_fraction,
        "largeJumpCount": large_jump_count,
        "unsupportedLargeJumpCount": unsupported_large_jump_count,
        "maximumUnsupportedLargeJumps": maximum_unsupported_jumps,
        "maximumCrossingP95Fraction": maximum_crossing_p95_fraction,
        "maximumCrossingJumpFraction": maximum_crossing_jump_fraction,
        "rhythmAnchorCount": rhythm_anchor_count,
        "rhythmTracePassed": rhythm_trace_passed,
        "recoveredExcursionCount": recovered_excursion_count,
        "recoveredSampleCount": recovered_sample_count,
        "rejectedExcursionCount": rejected_excursion_count,
        "rejectedSampleCount": rejected_sample_count,
        "unsafeExcursionCount": unsafe_excursion_count,
        "leadMetrics": {
            lead: asdict(metric) for lead, metric in lead_metrics.items()
        },
    }


def write_canonical_csv(
    output_path: Path,
    values: dict[str, np.ndarray],
    *,
    lead_panel_columns: dict[str, int],
    panel_samples: int,
) -> None:
    rows = np.full(
        (SAMPLE_RATE_HZ * PAGE_DURATION_SECONDS, len(LEAD_ORDER)),
        np.nan,
        dtype=np.float64,
    )
    for lead_index, lead in enumerate(LEAD_ORDER):
        if values[lead].size == rows.shape[0]:
            rows[:, lead_index] = values[lead]
            continue
        if values[lead].size != panel_samples:
            raise ValueError(
                f"{lead} has {values[lead].size} samples; expected "
                f"{panel_samples} panel samples or {rows.shape[0]} rhythm samples."
            )
        panel_offset = lead_panel_columns[lead] * panel_samples
        rows[
            panel_offset : panel_offset + panel_samples,
            lead_index,
        ] = values[lead]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(LEAD_ORDER)
        for row in rows:
            writer.writerow(
                "" if not np.isfinite(value) else f"{value:.8f}"
                for value in row
            )


def draw_overlay(
    source: np.ndarray,
    paths: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    lead_rows: dict[str, int],
) -> np.ndarray:
    overlay = source.copy()
    row_colors = (
        (40, 45, 225),
        (45, 170, 45),
        (225, 90, 35),
        (170, 65, 180),
        (30, 165, 210),
        (200, 150, 45),
    )
    for lead, (columns, path, validity) in paths.items():
        color = row_colors[lead_rows[lead] % len(row_colors)]
        for run in _true_runs(validity):
            if run.size < 2:
                continue
            points = np.column_stack((columns[run], path[run])).reshape(
                (-1, 1, 2)
            )
            cv2.polylines(
                overlay,
                [points],
                False,
                color,
                1,
                lineType=cv2.LINE_AA,
            )
    return overlay


def digitize_compound_six_row_panels(
    source_image: np.ndarray,
    evidence: np.ndarray,
    output_dir: Path,
    *,
    geometry: dict[str, Any],
    paper_speed_mm_per_second: float,
    gain_mm_per_mv: float,
    detected_pixels_per_mm: float | None,
) -> dict[str, Any]:
    """Trace paired limb/precordial six-row panels without global layout inference.

    The first panel is accepted only when its six simultaneous rows satisfy
    standard or Cabrera limb-lead algebra. The second panel is accepted only
    when six repeated compact labels localise V1-V6. This deliberately avoids
    assigning lead identities from row position alone.
    """

    panels = geometry.get("compoundPanels") or {}
    first_panel = panels.get("first") or {}
    second_panel = panels.get("second") or {}
    panel_specs = (first_panel, second_panel)
    if any(len(panel.get("rowCenters") or []) != 6 for panel in panel_specs):
        raise ValueError("Compound native extraction requires six rows per panel.")

    height, width = evidence.shape
    raw_values: list[list[np.ndarray]] = [[], []]
    raw_paths: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = [[], []]
    raw_metrics: list[list[LeadFidelity]] = [[], []]
    row_spacings: list[float] = []
    panel_widths: list[int] = []

    for panel_index, panel in enumerate(panel_specs):
        left = max(0, int(panel["left"]))
        top = max(0, int(panel["top"]))
        right = min(width, int(panel["right"]))
        bottom = min(height, int(panel["bottom"]))
        panel_width = right - left
        panel_height = bottom - top
        if panel_width < 40 or panel_height < 120:
            raise ValueError("Compound ECG panel is too small for native tracing.")
        centers = [int(value) for value in panel["rowCenters"]]
        if any(center < top or center >= bottom for center in centers):
            raise ValueError("Compound ECG row center lies outside its panel.")
        local_centers = [center - top for center in centers]
        local_bounds = sequential_row_bounds(local_centers, panel_height)
        spacing = float(np.median(np.diff(centers)))
        row_spacings.append(spacing)
        panel_widths.append(panel_width)
        label_localized = panel.get("rowMethod") == "compact-lead-label-sequence-v1"
        ordered_limb_labels = bool((panel.get("leadOrderHint") or {}).get("passed"))
        x_start = left + max(
            3,
            round(panel_width * (0.10 if label_localized or ordered_limb_labels else 0.045)),
        )
        x_end = right - max(2, round(panel_width * 0.018))
        for row_index, center in enumerate(centers):
            local_start, local_end = local_bounds[row_index]
            columns, path = trace_crossing_path(
                evidence,
                y_start=top + local_start,
                y_end=top + local_end,
                x_start=x_start,
                x_end=x_end,
                row_center=center,
                row_spacing=spacing,
            )
            validity = np.ones(columns.size, dtype=bool)
            pixels_per_mm = detected_pixels_per_mm or (
                panel_width
                / (paper_speed_mm_per_second * PAGE_DURATION_SECONDS)
            )
            raw_values[panel_index].append(
                native_path_to_canonical(
                    columns,
                    path,
                    panel_start=left,
                    panel_width=panel_width,
                    panel_samples=SIX_BY_TWO_PANEL_SAMPLES,
                    pixels_per_mm=pixels_per_mm,
                    gain_mm_per_mv=gain_mm_per_mv,
                    validity=validity,
                )
            )
            raw_paths[panel_index].append((columns, path, validity))
            raw_metrics[panel_index].append(
                fidelity_for_path(
                    evidence,
                    columns,
                    path,
                    panel_width=panel_width,
                    row_center=center,
                    row_spacing=spacing,
                    validity=validity,
                )
            )

    lead_order_validation = validate_limb_lead_order(raw_values[0])
    order_hint = first_panel.get("leadOrderHint") or {}
    scored_orders = {
        item.get("order"): item
        for item in (
            lead_order_validation.get("best") or {},
            lead_order_validation.get("runnerUp") or {},
        )
    }
    standard_score = scored_orders.get("standard") or {}
    cabrera_score = scored_orders.get("cabrera") or {}
    label_corroborated_standard = bool(
        order_hint.get("passed")
        and order_hint.get("order") == "standard"
        and float(order_hint.get("confidence", 0.0)) >= 0.08
        and standard_score.get("equationCount") == 4
        and float(standard_score.get("medianNormalizedError", 99.0)) <= 0.20
    )
    algebra_discriminative_standard = bool(
        standard_score.get("equationCount") == 4
        and float(standard_score.get("medianNormalizedError", 99.0)) <= 0.14
        and float(standard_score.get("medianNormalizedError", 99.0)) + 0.03
        <= float(cabrera_score.get("medianNormalizedError", 99.0))
    )
    if label_corroborated_standard or (
        not lead_order_validation.get("passed")
        and algebra_discriminative_standard
    ):
        lead_order_validation = {
            **lead_order_validation,
            "passed": True,
            "selectedOrder": "standard",
            "method": "limb-lead-algebra-plus-label-geometry-v1",
            "labelOrderHint": order_hint,
        }
    selected_order = lead_order_validation.get("selectedOrder")
    if selected_order == "cabrera":
        limb_row_for_lead = {
            "I": 1,
            "II": 3,
            "III": 5,
            "aVR": 2,
            "aVL": 0,
            "aVF": 4,
        }
    else:
        limb_row_for_lead = {
            "I": 0,
            "II": 1,
            "III": 2,
            "aVR": 3,
            "aVL": 4,
            "aVF": 5,
        }

    canonical_values: dict[str, np.ndarray] = {}
    paths: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    lead_metrics: dict[str, LeadFidelity] = {}
    lead_rows: dict[str, int] = {}
    for lead, row_index in limb_row_for_lead.items():
        values = raw_values[0][row_index]
        canonical_values[lead] = -values if selected_order == "cabrera" and lead == "aVR" else values
        paths[lead] = raw_paths[0][row_index]
        lead_metrics[lead] = raw_metrics[0][row_index]
        lead_rows[lead] = row_index
    for row_index, lead in enumerate(LEAD_ORDER[6:]):
        canonical_values[lead] = raw_values[1][row_index]
        paths[lead] = raw_paths[1][row_index]
        lead_metrics[lead] = raw_metrics[1][row_index]
        lead_rows[lead] = row_index

    row_spacing = float(np.median(row_spacings))
    pixels_per_mm = detected_pixels_per_mm or (
        float(np.median(panel_widths))
        / (paper_speed_mm_per_second * PAGE_DURATION_SECONDS)
    )
    confidence = float(panels.get("confidence", 0.0))
    fidelity = source_fidelity_report(
        lead_metrics,
        pixels_per_mm=pixels_per_mm,
        row_spacing=row_spacing,
        layout_confidence=confidence,
        method="row-local-compound-label-and-algebra-v1",
        allow_ink_backed_crossings=True,
        rhythm_anchor_count=0,
        rhythm_trace_passed=True,
        minimum_layout_confidence=0.08,
        require_rhythm_anchors=False,
        minimum_evidence_median=0.50,
        minimum_evidence_p10=0.0,
        maximum_crossing_p95_fraction=1.10,
        maximum_crossing_jump_fraction=1.55,
        unsupported_large_jump_fraction=0.50,
        minimum_home_row_fraction=0.05,
    )
    precordial_label_validation = {
        "passed": second_panel.get("rowMethod") == "compact-lead-label-sequence-v1",
        "method": second_panel.get("rowMethod"),
        "assignedLeads": list(LEAD_ORDER[6:]),
    }
    fidelity["leadOrderValidation"] = lead_order_validation
    fidelity["precordialLabelValidation"] = precordial_label_validation
    fidelity["passed"] = bool(
        fidelity["passed"]
        and lead_order_validation.get("passed")
        and precordial_label_validation["passed"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = output_dir / "native_grid_timeseries_canonical.csv"
    overlay_path = output_dir / "native_grid_overlay.png"
    metadata_path = output_dir / "digitization_metadata.csv"
    fidelity_path = output_dir / "source_fidelity.json"
    write_canonical_csv(
        canonical_path,
        canonical_values,
        lead_panel_columns={
            lead: 0 if lead in LEAD_ORDER[:6] else 1 for lead in LEAD_ORDER
        },
        panel_samples=SIX_BY_TWO_PANEL_SAMPLES,
    )
    if not cv2.imwrite(
        str(overlay_path),
        draw_overlay(source_image, paths, lead_rows=lead_rows),
    ):
        raise OSError(f"Could not write overlay: {overlay_path}")
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("file_path", "matching_cost", "is_flipped", "lead_layout"))
        writer.writerow(("native_grid", 1.0 - confidence, "False", "standard_6x2"))
    fidelity_path.write_text(f"{json.dumps(fidelity, indent=2)}\n", encoding="utf-8")

    return {
        "canonicalPath": str(canonical_path),
        "diagnosticPath": str(overlay_path),
        "metadataPath": str(metadata_path),
        "sourceFidelityPath": str(fidelity_path),
        "layout": "standard_6x2",
        "layoutCost": 1.0 - confidence,
        "effectiveSampleRateHz": (
            float(np.median(panel_widths)) / (PAGE_DURATION_SECONDS / 2.0)
        ),
        "sourceFidelity": fidelity,
    }


def digitize_native_grid(
    extraction_image: np.ndarray,
    source_image: np.ndarray,
    output_dir: Path,
    trace_evidence: np.ndarray | None = None,
    verified_rhythm_lead: str | None = None,
) -> dict[str, Any]:
    if extraction_image.shape[:2] != source_image.shape[:2]:
        raise ValueError("Extraction and source images must have identical dimensions.")
    geometry = detect_ecg_layout_geometry(extraction_image)
    if verified_rhythm_lead is not None:
        if verified_rhythm_lead != "II":
            raise ValueError("Only a source-verified lead II rhythm strip is supported.")
        geometry["verifiedRhythmLead"] = verified_rhythm_lead
    layout = geometry.get("layoutHint")
    compound_panels = geometry.get("compoundPanels") or {}
    compound_row_local = bool(
        (
            layout is None
            or compound_panels.get("method")
            == "separate-grid-panels-plus-label-geometry-v1"
        )
        and float(compound_panels.get("confidence", 0.0)) >= 0.08
        and all(
            len((compound_panels.get(name) or {}).get("rowCenters") or []) == 6
            for name in ("first", "second")
        )
    )
    if layout not in {
        "standard_6x2",
        "standard_6x2_with_r1_ignored",
        "standard_3x4",
        "standard_3x4_with_r1",
        "standard_12x1",
    } and not compound_row_local:
        raise ValueError(
            "Native grid extraction requires a detected 6 x 2, 3 x 4, or "
            "sequential/paired-panel 12-lead ECG."
        )

    gray = cv2.cvtColor(extraction_image, cv2.COLOR_BGR2GRAY)
    if trace_evidence is None:
        evidence = grid_residual_evidence(gray)
        evidence_method = "rhythm-anchored-connected-multievidence-native-path-v6"
    else:
        if trace_evidence.ndim != 2 or trace_evidence.shape != gray.shape:
            raise ValueError(
                "Prepared trace evidence must be grayscale and match the source."
            )
        prepared_evidence = trace_evidence.astype(np.float32) / 255.0
        raw_residual_evidence = grid_residual_evidence(gray)
        evidence = np.maximum(
            prepared_evidence,
            raw_residual_evidence * 0.75,
        )
        evidence_method = (
            "rhythm-anchored-connected-colour-plus-raw-grid-path-v2"
        )
    height, width = gray.shape
    calibration = geometry.get("calibration") or {}
    calibration_detected = bool(
        calibration.get("detected")
        and float(calibration.get("confidence", 0.0)) >= 0.35
    )
    paper_speed_mm_per_second = (
        float(calibration["paperSpeedMmPerSecond"])
        if calibration_detected
        else 25.0
    )
    gain_mm_per_mv = (
        float(calibration["gainMmPerMv"])
        if calibration_detected
        else 10.0
    )
    grid_pixels_x = float(calibration.get("pixelsPerMmX", 0.0) or 0.0)
    grid_pixels_y = float(calibration.get("pixelsPerMmY", 0.0) or 0.0)
    grid_scale_detected = bool(
        (
            (
                calibration.get("gridScaleDetected")
                and float(calibration.get("gridScaleConfidence", 0.0)) >= 0.10
            )
            or (
                calibration.get("detected")
                and float(calibration.get("confidence", 0.0)) >= 0.10
            )
        )
        and 1.0 <= grid_pixels_x <= 40.0
        and 1.0 <= grid_pixels_y <= 40.0
        and abs(grid_pixels_x - grid_pixels_y)
        <= max(grid_pixels_x, grid_pixels_y) * 0.15
    )
    detected_pixels_per_mm = grid_pixels_x if grid_scale_detected else None
    if compound_row_local:
        return digitize_compound_six_row_panels(
            source_image,
            evidence,
            output_dir,
            geometry=geometry,
            paper_speed_mm_per_second=paper_speed_mm_per_second,
            gain_mm_per_mv=gain_mm_per_mv,
            detected_pixels_per_mm=detected_pixels_per_mm,
        )
    effective_sample_rate_hz = (
        detected_pixels_per_mm * paper_speed_mm_per_second
        if detected_pixels_per_mm is not None
        else width / PAGE_DURATION_SECONDS
    )
    row_centers = [int(value) for value in geometry["rowCenters"]]
    row_spacing = float(geometry["medianRowSpacing"])
    strategy = build_native_layout_strategy(
        geometry,
        width=width,
        height=height,
        paper_speed_mm_per_second=paper_speed_mm_per_second,
        detected_pixels_per_mm=detected_pixels_per_mm,
    )
    row_leads = strategy.row_leads
    primary_centers = strategy.primary_centers
    primary_bounds = strategy.primary_bounds
    panel_edges = strategy.panel_edges
    panel_samples = strategy.panel_samples
    pixels_per_mm = strategy.pixels_per_mm
    diagnostic_amplitude_pixels_per_mm = pixels_per_mm
    primary_bottom = strategy.primary_bottom
    publish_rhythm_trace = strategy.publish_rhythm_trace
    allow_ink_backed_crossings = strategy.allow_ink_backed_crossings
    row_local_twelve = strategy.row_local_twelve
    label_anchored_twelve = strategy.label_anchored_twelve
    row_local_labeled_three = strategy.row_local_labeled_three
    row_local_labeled_six = strategy.row_local_labeled_six
    row_local_calibration_six = strategy.row_local_calibration_six
    row_local_geometry = strategy.row_local_geometry
    crossing_transition_scale = 2.0 if row_local_twelve else 1.0
    ink_connected_displacement_reward = (
        0.03 if row_local_twelve else 0.0
    )
    three_by_four_label_validated = bool(
        layout in {"standard_3x4", "standard_3x4_with_r1"}
        and (geometry.get("leadLabelValidation") or {}).get("passed")
        and (geometry.get("leadLabelValidation") or {}).get("order")
        == "standard"
    )
    rhythm_trace: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    rhythm_row_index = strategy.rhythm_row_index or 0
    rhythm_trace_passed = True
    page_qrs_anchors = np.asarray([], dtype=np.int32)
    artifact_column_mask = np.zeros(width, dtype=bool)
    if strategy.rhythm_row_index is not None:
        rhythm_start = max(
            primary_bottom,
            round(row_centers[rhythm_row_index] - row_spacing * 0.55),
        )
        rhythm_end = min(
            height,
            round(row_centers[rhythm_row_index] + row_spacing * 0.55),
        )
        rhythm_x_start = max(3, round(width * 0.012))
        rhythm_x_end = width - max(4, round(width * 0.035))
        rhythm_columns, rhythm_path = trace_path(
            evidence,
            y_start=rhythm_start,
            y_end=rhythm_end,
            x_start=rhythm_x_start,
            x_end=rhythm_x_end,
            row_center=row_centers[rhythm_row_index],
        )
        rhythm_validity = np.ones(rhythm_columns.size, dtype=bool)
        rhythm_trace = (rhythm_columns, rhythm_path, rhythm_validity)
        page_qrs_anchors = rhythm_qrs_anchors(
            rhythm_columns,
            rhythm_path,
            row_spacing=row_spacing,
            effective_sample_rate_hz=effective_sample_rate_hz,
        )
        artifact_column_mask = vertical_artifact_columns(
            evidence,
            primary_bottom=primary_bottom,
            row_spacing=row_spacing,
        )

    row_pixels_per_mm_x = [
        float(value)
        for value in calibration.get("rowPixelsPerMmX", [])
        if isinstance(value, (int, float)) and np.isfinite(value)
    ]
    raw_grid_period_x = float(
        calibration.get("gridSpacingXPixels", 0.0) or 0.0
    )
    raw_grid_period_scale_mm = (
        float(calibration["gridScaleMmX"])
        if calibration.get("gridScaleMmX") in {1.0, 5.0}
        else float(calibration["gridScaleMm"])
        if calibration.get("gridScaleMm") in {1.0, 5.0}
        else None
    )
    six_by_two_timing_pixels_per_mm = timing_scale_pixels_per_mm(
        row_pixels_per_mm_x=row_pixels_per_mm_x,
        calibrated_pixels_per_mm=pixels_per_mm,
        raw_grid_period_x=raw_grid_period_x,
        raw_grid_period_scale_mm=raw_grid_period_scale_mm,
        prefer_raw_grid=False,
    )
    twelve_row_timing_pixels_per_mm = timing_scale_pixels_per_mm(
        row_pixels_per_mm_x=row_pixels_per_mm_x,
        calibrated_pixels_per_mm=pixels_per_mm,
        raw_grid_period_x=raw_grid_period_x,
        raw_grid_period_scale_mm=raw_grid_period_scale_mm,
        prefer_raw_grid=True,
    )
    labeled_time_ranges = (
        labeled_six_by_two_time_ranges(
            evidence,
            row_centers[:6],
            pixels_per_mm=six_by_two_timing_pixels_per_mm,
            paper_speed_mm_per_second=paper_speed_mm_per_second,
        )
        # Source ink and the measured panel width establish time independently
        # of which layout recognizer supplied the row centers. Calibration and
        # label margins also exist on pages detected from repeated trace rows.
        if layout in {"standard_6x2", "standard_6x2_with_r1_ignored"}
        else None
    )
    labeled_three_by_four_ranges = (
        labeled_three_by_four_time_ranges(
            evidence,
            primary_centers,
            pixels_per_mm=twelve_row_timing_pixels_per_mm,
            paper_speed_mm_per_second=paper_speed_mm_per_second,
            label_left_edges=(
                geometry.get("leadLabelValidation", {}).get(
                    "columnLabelLeftEdgesX"
                )
            ),
        )
        if three_by_four_label_validated
        else None
    )
    labeled_twelve_timing_diagnostics: dict[str, Any] = {}
    calibrated_twelve_time_ranges = (
        labeled_twelve_row_time_ranges(
            evidence,
            primary_centers,
            row_pixels_per_mm=row_pixels_per_mm_x,
            paper_speed_mm_per_second=paper_speed_mm_per_second,
            calibration_exclusion_end=(
                int(calibration["pulseEndX"])
                if calibration_detected
                and isinstance(calibration.get("pulseEndX"), (int, float))
                else None
            ),
            timing_diagnostics=labeled_twelve_timing_diagnostics,
        )
        if label_anchored_twelve and len(row_pixels_per_mm_x) == 12
        else None
    )
    labeled_twelve_time_range = (
        labeled_twelve_row_time_range(
            evidence,
            primary_centers,
            pixels_per_mm=twelve_row_timing_pixels_per_mm,
            paper_speed_mm_per_second=paper_speed_mm_per_second,
        )
        if label_anchored_twelve
        and allow_labeled_twelve_global_time_fallback(
            calibrated_twelve_time_ranges,
            labeled_twelve_timing_diagnostics,
            grid_scale_detected=grid_scale_detected,
            raw_grid_period_scale_mm=raw_grid_period_scale_mm,
        )
        else None
    )
    labeled_twelve_source_consensus = (
        labeled_twelve_source_consensus_time_ranges(
            evidence,
            primary_centers,
            raw_grid_period_x=raw_grid_period_x,
            paper_speed_mm_per_second=paper_speed_mm_per_second,
            calibration_exclusion_end=(
                int(calibration["pulseEndX"])
                if calibration_detected
                and isinstance(calibration.get("pulseEndX"), (int, float))
                else None
            ),
        )
        # Timing and lead identity are independent claims.  A sequential page
        # can have weak/failed OCR while its twelve row-local trace edges,
        # measured grid period, and calibration pulse still provide a fully
        # source-backed ten-second mapping.  Requiring label OCR here caused
        # the calibration/label margin to be mapped as waveform time, shifting
        # every beat on otherwise legible 12 x 1 ECGs.
        if row_local_twelve
        and calibrated_twelve_time_ranges is None
        else None
    )
    labeled_twelve_source_global_disagreement = bool(
        labeled_twelve_source_consensus is not None
        and labeled_twelve_time_range is not None
        and labeled_twelve_source_timing_disagrees_with_global(
            labeled_twelve_source_consensus[0],
            labeled_twelve_time_range,
            tolerance_pixels=int(
                labeled_twelve_source_consensus[1][
                    "boundaryTolerancePixels"
                ]
            ),
        )
    )
    use_labeled_twelve_source_consensus = bool(
        labeled_twelve_source_consensus is not None
        and (
            labeled_twelve_time_range is None
            or labeled_twelve_source_global_disagreement
        )
    )
    labeled_twelve_source_time_ranges = (
        labeled_twelve_source_consensus[0]
        if use_labeled_twelve_source_consensus
        else None
    )
    source_timing_calibration_confirmed = bool(
        use_labeled_twelve_source_consensus
        and calibration_confirms_labeled_twelve_source_timing(
            calibration,
            labeled_twelve_source_consensus[1],
        )
    )
    labeled_twelve_source_timing_diagnostics = (
        {
            **labeled_twelve_source_consensus[1],
            "quantitativeCalibrationConfirmed": (
                source_timing_calibration_confirmed
            ),
            **(
                {
                    "quantitativeCalibrationMethod": (
                        "rectangular-pulse-plus-grid-and-source-span-v1"
                    ),
                    "calibrationConfidence": float(
                        calibration.get("confidence", 0.0)
                    ),
                    "calibrationPixelsPerMmX": grid_pixels_x,
                    "calibrationPixelsPerMmY": grid_pixels_y,
                    "calibrationAgreementLimitFraction": 0.12,
                }
                if source_timing_calibration_confirmed
                else {}
            ),
            **(
                {
                    "globalTimingRange": list(labeled_twelve_time_range),
                    "globalTimingConsistent": False,
                    "globalTimingTolerancePixels": int(
                        labeled_twelve_source_consensus[1][
                            "boundaryTolerancePixels"
                        ]
                    ),
                }
                if labeled_twelve_time_range is not None
                else {}
            ),
        }
        if use_labeled_twelve_source_consensus
        else None
    )
    labeled_twelve_time_ranges = (
        calibrated_twelve_time_ranges or labeled_twelve_source_time_ranges
    )
    source_time_ranges = labeled_time_ranges or labeled_three_by_four_ranges or (
        (labeled_twelve_time_range,)
        if labeled_twelve_time_range is not None
        else None
    )
    if labeled_twelve_source_timing_diagnostics is not None:
        if labeled_twelve_source_timing_diagnostics.get(
            "quantitativeCalibrationConfirmed"
        ):
            # The source edges establish the row-local ten-second mapping;
            # the independently detected pulse supplies vertical gain.
            diagnostic_amplitude_pixels_per_mm = grid_pixels_y
        else:
            # This scale makes the diagnostic waveform visually comparable
            # with the source. It is derived from an assumed ten-second page
            # span and must never replace quantitative calibration.
            diagnostic_amplitude_pixels_per_mm = float(
                labeled_twelve_source_timing_diagnostics[
                    "inferredPixelsPerMm"
                ]
            )
        effective_sample_rate_hz = float(
            np.median(
                [
                    end - start
                    for start, end in labeled_twelve_source_time_ranges
                ]
            )
            / PAGE_DURATION_SECONDS
        )
    rhythm_source_time_range = (
        contiguous_rhythm_time_range(source_time_ranges, page_width=width)
        if publish_rhythm_trace
        else None
    )
    panel_ranges = source_time_ranges or tuple(
        zip(panel_edges[:-1], panel_edges[1:], strict=True)
    )
    lead_panel_columns = {
        lead: column
        for leads in row_leads
        for column, lead in enumerate(leads)
    }
    lead_rows = {
        lead: row_index
        for row_index, leads in enumerate(row_leads)
        for lead in leads
    }

    canonical_values: dict[str, np.ndarray] = {}
    lead_metrics: dict[str, LeadFidelity] = {}
    paths: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    lead_source_ranges: dict[str, tuple[int, int]] = {}
    lead_order_validation: dict[str, Any] | None = None

    for row_index, leads in enumerate(row_leads):
        y_start, y_end = primary_bounds[row_index]
        for column, lead in enumerate(leads):
            if labeled_twelve_time_ranges is not None:
                panel_start, panel_end = labeled_twelve_time_ranges[row_index]
            else:
                panel_start, panel_end = panel_ranges[column]
            panel_width = panel_end - panel_start
            lead_source_ranges[lead] = (panel_start, panel_end)
            if (
                source_time_ranges is not None
                or labeled_twelve_time_ranges is not None
            ):
                # The source-derived range already begins at the signal origin
                # and excludes label/calibration margins.
                start_margin = 0
                end_margin = 0
            elif row_local_labeled_six:
                # Conservative fallback when a time origin cannot be found.
                start_margin = max(3, round(panel_width * 0.14))
                end_margin = max(2, round(panel_width * 0.012))
            elif label_anchored_twelve:
                # The labels are the trusted row anchors, not waveform ink.
                # Begin just beyond their validated left-margin band.
                start_margin = max(3, round(panel_width * 0.105))
                end_margin = max(2, round(panel_width * 0.014))
            else:
                start_margin, end_margin = panel_margins(
                    panel_width,
                    column,
                    column_count=len(panel_ranges),
                )
            x_start = panel_start + start_margin
            x_end = panel_end - end_margin
            weak_label_anchored_trace = bool(
                label_anchored_twelve
                and row_index >= 6
                and row_index < len(geometry.get("traceRowSupport") or [])
                and float(geometry["traceRowSupport"][row_index]) < 0.12
            )
            if weak_label_anchored_trace:
                columns, path = trace_path(
                    evidence,
                    y_start=y_start,
                    y_end=y_end,
                    x_start=x_start,
                    x_end=x_end,
                    row_center=primary_centers[row_index],
                    center_weight=3.0,
                    transition_scale=4.0,
                )
                validity = np.ones(columns.size, dtype=bool)
                safety_metrics = {}
            elif allow_ink_backed_crossings:
                columns, path = trace_crossing_path(
                    evidence,
                    y_start=y_start,
                    y_end=y_end,
                    x_start=x_start,
                    x_end=x_end,
                    row_center=primary_centers[row_index],
                    row_spacing=row_spacing,
                    transition_scale=crossing_transition_scale,
                    ink_connected_displacement_reward=(
                        ink_connected_displacement_reward
                    ),
                )
                if row_local_geometry:
                    validity = np.ones(columns.size, dtype=bool)
                    safety_metrics = {}
                    if row_local_labeled_three:
                        boundary_excursions = 0
                        if row_index == 0:
                            boundary_excursions += source_boundary_excursion_count(
                                evidence,
                                x_start=x_start,
                                x_end=x_end,
                                boundary="top",
                            )
                        if row_index == len(row_leads) - 1:
                            boundary_excursions += source_boundary_excursion_count(
                                evidence,
                                x_start=x_start,
                                x_end=x_end,
                                boundary="bottom",
                            )
                        safety_metrics["unsafeExcursionCount"] = (
                            boundary_excursions
                        )
                        if boundary_excursions > 0 and (
                            y_start <= 0 or y_end >= height
                        ):
                            boundary = "top" if y_start <= 0 else "bottom"
                            validity, clipped_metrics = page_boundary_clipped_validity(
                                evidence,
                                columns,
                                boundary=boundary,
                            )
                            safety_metrics.update(clipped_metrics)
                else:
                    event_columns = waveform_event_columns(
                        columns,
                        path,
                        row_spacing=row_spacing,
                    )
                    aligned_anchors, anchor_shift, anchor_matches = (
                        align_panel_qrs_anchors(
                            event_columns,
                            page_qrs_anchors,
                            panel_start=panel_start,
                            panel_end=panel_end,
                            effective_sample_rate_hz=effective_sample_rate_hz,
                        )
                    )
                    validity, safety_metrics = artifact_safe_validity(
                        evidence,
                        columns,
                        path,
                        row_center=primary_centers[row_index],
                        row_spacing=row_spacing,
                        qrs_anchors=aligned_anchors,
                        qrs_anchor_shift=anchor_shift,
                        qrs_anchor_matches=anchor_matches,
                        artifact_columns=artifact_column_mask,
                        effective_sample_rate_hz=effective_sample_rate_hz,
                    )
                    strict_y_start = max(
                        0,
                        round(primary_centers[row_index] - row_spacing * 0.38),
                    )
                    strict_y_end = min(
                        primary_bottom,
                        round(primary_centers[row_index] + row_spacing * 0.38),
                    )
                    strict_columns, strict_path = trace_path(
                        evidence,
                        y_start=strict_y_start,
                        y_end=strict_y_end,
                        x_start=x_start,
                        x_end=x_end,
                        row_center=primary_centers[row_index],
                    )
                    if not np.array_equal(strict_columns, columns):
                        raise ValueError(
                            f"{lead} conservative recovery columns do not match."
                        )
                    path, validity, safety_metrics = recover_rejected_intervals(
                        evidence,
                        columns,
                        path,
                        validity,
                        strict_path,
                        safety_metrics,
                        row_center=primary_centers[row_index],
                        row_spacing=row_spacing,
                    )
            else:
                columns, path = trace_path(
                    evidence,
                    y_start=y_start,
                    y_end=y_end,
                    x_start=x_start,
                    x_end=x_end,
                    row_center=primary_centers[row_index],
                )
                validity = np.ones(columns.size, dtype=bool)
                safety_metrics = {}
            paths[lead] = (columns, path, validity)
            canonical_values[lead] = native_path_to_canonical(
                columns,
                path,
                panel_start=panel_start,
                panel_width=panel_width,
                panel_samples=panel_samples,
                pixels_per_mm=diagnostic_amplitude_pixels_per_mm,
                gain_mm_per_mv=gain_mm_per_mv,
                validity=validity,
            )
            lead_metrics[lead] = fidelity_for_path(
                evidence,
                columns,
                path,
                panel_width=panel_width,
                row_center=primary_centers[row_index],
                row_spacing=row_spacing,
                validity=validity,
                safety_metrics=safety_metrics,
            )

    sequential_event_anchor_count = 0
    if row_local_twelve:
        normalized_event_times = normalized_cross_lead_event_times(
            paths,
            lead_source_ranges,
            list(LEAD_ORDER),
            row_spacing=row_spacing,
            minimum_leads=4,
        )
        sequential_event_anchor_count = int(normalized_event_times.size)
        if normalized_event_times.size:
            event_radius = max(3, round(effective_sample_rate_hz * 0.035))
            for row_index, leads in enumerate(row_leads):
                y_start, y_end = primary_bounds[row_index]
                for lead in leads:
                    panel_start, panel_end = lead_source_ranges[lead]
                    columns, previous_path, _previous_validity = paths[lead]
                    event_anchors = np.rint(
                        panel_start
                        + normalized_event_times * (panel_end - panel_start)
                    ).astype(np.int32)
                    recovered_columns, recovered_path = trace_crossing_path(
                        evidence,
                        y_start=y_start,
                        y_end=y_end,
                        x_start=int(columns[0]),
                        x_end=int(columns[-1]) + 1,
                        row_center=primary_centers[row_index],
                        row_spacing=row_spacing,
                        transition_scale=crossing_transition_scale,
                        ink_connected_displacement_reward=(
                            ink_connected_displacement_reward
                        ),
                        event_anchors=event_anchors,
                        event_radius_pixels=event_radius,
                        event_excursion_reward=6.0,
                        event_reference_path=previous_path,
                    )
                    if not np.array_equal(recovered_columns, columns):
                        raise ValueError(
                            f"{lead} event-recovery columns do not match."
                        )
                    validity = np.ones(columns.size, dtype=bool)
                    paths[lead] = (columns, recovered_path, validity)
                    panel_width = panel_end - panel_start
                    canonical_values[lead] = native_path_to_canonical(
                        columns,
                        recovered_path,
                        panel_start=panel_start,
                        panel_width=panel_width,
                        panel_samples=panel_samples,
                        pixels_per_mm=diagnostic_amplitude_pixels_per_mm,
                        gain_mm_per_mv=gain_mm_per_mv,
                        validity=validity,
                    )
                    lead_metrics[lead] = fidelity_for_path(
                        evidence,
                        columns,
                        recovered_path,
                        panel_width=panel_width,
                        row_center=primary_centers[row_index],
                        row_spacing=row_spacing,
                        validity=validity,
                        safety_metrics={
                            "qrsAnchorMatchCount": sequential_event_anchor_count,
                        },
                    )

    if allow_ink_backed_crossings and (
        not row_local_geometry or row_local_labeled_six
    ):
        # Re-adjudicate excursions with event times observed in at least two
        # *other* simultaneous leads. Rhythm-anchored layouts only revisit
        # previously rejected samples. Label-anchored 6 x 2 pages have no
        # independent rhythm row, so every path is checked here: this prevents
        # an isolated annotation stroke from being accepted merely because it
        # is connected to source ink while preserving cross-lead QRS events.
        for row_index, leads in enumerate(row_leads):
            y_start, y_end = primary_bounds[row_index]
            for column, lead in enumerate(leads):
                columns, path, previous_validity = paths[lead]
                if bool(np.all(previous_validity)) and not row_local_labeled_six:
                    continue
                panel_start, panel_end = panel_ranges[column]
                peer_leads = tuple(
                    row_leads[peer_row][column]
                    for peer_row in range(len(row_leads))
                    if peer_row != row_index
                )
                peer_anchors = cross_lead_event_anchors(
                    paths,
                    peer_leads,
                    panel_start=panel_start,
                    panel_end=panel_end,
                    row_spacing=row_spacing,
                    tolerance_pixels=max(
                        4,
                        round(effective_sample_rate_hz * 0.045),
                    ),
                    artifact_columns=artifact_column_mask,
                    minimum_other_leads=2,
                )
                event_columns = waveform_event_columns(
                    columns,
                    path,
                    row_spacing=row_spacing,
                )
                aligned_anchors, anchor_shift, anchor_matches = (
                    align_panel_qrs_anchors(
                        event_columns,
                        page_qrs_anchors,
                        panel_start=panel_start,
                        panel_end=panel_end,
                        effective_sample_rate_hz=effective_sample_rate_hz,
                    )
                )
                corroborated_anchors = np.unique(
                    np.concatenate((aligned_anchors, peer_anchors))
                ).astype(np.int32)
                validity, safety_metrics = artifact_safe_validity(
                    evidence,
                    columns,
                    path,
                    row_center=primary_centers[row_index],
                    row_spacing=row_spacing,
                    qrs_anchors=corroborated_anchors,
                    qrs_anchor_shift=anchor_shift,
                    qrs_anchor_matches=anchor_matches,
                    artifact_columns=artifact_column_mask,
                    effective_sample_rate_hz=effective_sample_rate_hz,
                )
                strict_y_start = max(
                    0,
                    round(primary_centers[row_index] - row_spacing * 0.38),
                )
                strict_y_end = min(
                    primary_bottom,
                    round(primary_centers[row_index] + row_spacing * 0.38),
                )
                strict_columns, strict_path = trace_path(
                    evidence,
                    y_start=strict_y_start,
                    y_end=strict_y_end,
                    x_start=int(columns[0]),
                    x_end=int(columns[-1]) + 1,
                    row_center=primary_centers[row_index],
                )
                if not np.array_equal(strict_columns, columns):
                    raise ValueError(
                        f"{lead} corroborated recovery columns do not match."
                    )
                path, validity, safety_metrics = recover_rejected_intervals(
                    evidence,
                    columns,
                    path,
                    validity,
                    strict_path,
                    safety_metrics,
                    row_center=primary_centers[row_index],
                    row_spacing=row_spacing,
                )
                panel_width = panel_end - panel_start
                paths[lead] = (columns, path, validity)
                canonical_values[lead] = native_path_to_canonical(
                    columns,
                    path,
                    panel_start=panel_start,
                    panel_width=panel_width,
                    panel_samples=panel_samples,
                    pixels_per_mm=diagnostic_amplitude_pixels_per_mm,
                    gain_mm_per_mv=gain_mm_per_mv,
                    validity=validity,
                )
                lead_metrics[lead] = fidelity_for_path(
                    evidence,
                    columns,
                    path,
                    panel_width=panel_width,
                    row_center=primary_centers[row_index],
                    row_spacing=row_spacing,
                    validity=validity,
                    safety_metrics=safety_metrics,
                )

    if row_local_twelve:
        raw_values = [canonical_values[lead] for lead in LEAD_ORDER]
        simultaneous_algebra_diagnostic = validate_limb_lead_order(raw_values)
        label_validation = geometry.get("leadLabelValidation") or {}
        if label_anchored_twelve:
            labels_passed = bool(
                label_validation.get("passed")
                and label_validation.get("order") == "standard"
            )
            lead_order_validation = {
                "passed": labels_passed,
                "selectedOrder": "standard" if labels_passed else None,
                "method": "sequential-lead-label-anchors-v2",
                "labelOrderHint": label_validation,
                # Sequential rows are recorded at different times. Limb-lead
                # equations are therefore diagnostic only and must never
                # approve, reject, or reorder this layout.
                "simultaneousAlgebraDiagnostic": (
                    simultaneous_algebra_diagnostic
                ),
            }
        else:
            lead_order_validation = {
                "passed": False,
                "selectedOrder": None,
                "method": "sequential-lead-order-unverified-v2",
                "reason": "Sequential 12x1 rows require explicit lead-label anchors.",
                "simultaneousAlgebraDiagnostic": (
                    simultaneous_algebra_diagnostic
                ),
            }
    elif row_local_labeled_six:
        lead_order_validation = validate_limb_lead_order(
            [canonical_values[lead] for lead in LEAD_ORDER[:6]]
        )
        limb_label_validation = geometry.get("limbLabelValidation") or {}
        algebra_standard = bool(
            lead_order_validation.get("passed")
            and lead_order_validation.get("selectedOrder") == "standard"
        )
        if not algebra_standard and limb_label_validation.get("passed"):
            scored_orders = {
                item.get("order"): item
                for item in (
                    lead_order_validation.get("best") or {},
                    lead_order_validation.get("runnerUp") or {},
                )
            }
            standard_score = scored_orders.get("standard") or {}
            if (
                standard_score.get("equationCount") == 4
                and float(standard_score.get("medianNormalizedError", 99.0))
                <= 0.20
            ):
                lead_order_validation = {
                    **lead_order_validation,
                    "passed": True,
                    "selectedOrder": "standard",
                    "method": "limb-lead-algebra-plus-label-geometry-v1",
                    "labelOrderHint": limb_label_validation,
                }
        if not lead_order_validation.get("passed"):
            scored_orders = {
                item.get("order"): item
                for item in (
                    lead_order_validation.get("best") or {},
                    lead_order_validation.get("runnerUp") or {},
                )
            }
            standard_score = scored_orders.get("standard") or {}
            cabrera_score = scored_orders.get("cabrera") or {}
            if (
                standard_score.get("equationCount") == 4
                and float(standard_score.get("medianNormalizedError", 99.0))
                <= 0.14
                and float(standard_score.get("medianNormalizedError", 99.0))
                + 0.03
                <= float(cabrera_score.get("medianNormalizedError", 99.0))
            ):
                lead_order_validation = {
                    **lead_order_validation,
                    "passed": True,
                    "selectedOrder": "standard",
                    "method": "discriminative-limb-lead-algebra-v1",
                }

    if rhythm_trace is not None:
        rhythm_columns, rhythm_path, rhythm_validity = rhythm_trace
        paths["rhythm II"] = rhythm_trace
        lead_rows["rhythm II"] = rhythm_row_index
        rhythm_metric = fidelity_for_path(
            evidence,
            rhythm_columns,
            rhythm_path,
            panel_width=width,
            row_center=row_centers[rhythm_row_index],
            row_spacing=row_spacing,
            validity=rhythm_validity,
            safety_metrics={
                "qrsAnchorMatchCount": int(page_qrs_anchors.size),
            },
        )
        rhythm_trace_passed = bool(
            rhythm_metric.coverage >= 0.90
            and rhythm_metric.evidenceP10 >= 0.45
            and rhythm_metric.homeRowFraction >= 0.95
            and rhythm_metric.p95NativeJumpPixels <= row_spacing * 0.20
            and rhythm_metric.maxNativeJumpPixels <= row_spacing * 0.35
        )
        if publish_rhythm_trace:
            rhythm_panel_start, rhythm_panel_end = (
                rhythm_source_time_range
                if rhythm_source_time_range is not None
                else (0, width)
            )
            canonical_values["II"] = native_path_to_canonical(
                rhythm_columns,
                rhythm_path,
                panel_start=rhythm_panel_start,
                panel_width=rhythm_panel_end - rhythm_panel_start,
                panel_samples=SAMPLE_RATE_HZ * PAGE_DURATION_SECONDS,
                pixels_per_mm=diagnostic_amplitude_pixels_per_mm,
                gain_mm_per_mv=gain_mm_per_mv,
                validity=rhythm_validity,
            )
            lead_metrics["II"] = rhythm_metric

    fidelity = source_fidelity_report(
        lead_metrics,
        pixels_per_mm=diagnostic_amplitude_pixels_per_mm,
        row_spacing=row_spacing,
        layout_confidence=float(geometry["confidence"]),
        method=strategy.fidelity.method or evidence_method,
        allow_ink_backed_crossings=allow_ink_backed_crossings,
        rhythm_anchor_count=int(page_qrs_anchors.size),
        rhythm_trace_passed=(
            True if row_local_geometry else rhythm_trace_passed
        ),
        minimum_layout_confidence=strategy.fidelity.minimum_layout_confidence,
        require_rhythm_anchors=strategy.fidelity.require_rhythm_anchors,
        minimum_evidence_median=strategy.fidelity.minimum_evidence_median,
        minimum_evidence_p10=strategy.fidelity.minimum_evidence_p10,
        maximum_crossing_p95_fraction=(
            strategy.fidelity.maximum_crossing_p95_fraction
        ),
        maximum_crossing_jump_fraction=(
            strategy.fidelity.maximum_crossing_jump_fraction
        ),
        unsupported_large_jump_fraction=(
            strategy.fidelity.unsupported_large_jump_fraction
        ),
    )
    fidelity["sourcePanelTimingDetected"] = bool(
        source_time_ranges is not None or labeled_twelve_time_ranges is not None
    )
    if row_local_twelve:
        fidelity["crossLeadEventAnchorCount"] = sequential_event_anchor_count
        fidelity["crossLeadEventRecovery"] = {
            "method": "normalized-simultaneous-event-excursion-v1",
            "minimumLeadCount": 4,
            "eventRadiusPixels": max(
                3, round(effective_sample_rate_hz * 0.035)
            ),
            "excursionReward": 6.0,
        }
    fidelity["rowLocalSourceTimingDetected"] = (
        labeled_twelve_time_ranges is not None
    )
    if labeled_twelve_timing_diagnostics:
        fidelity["rowTimeOriginConsensus"] = (
            labeled_twelve_timing_diagnostics
        )
    if labeled_twelve_source_timing_diagnostics is not None:
        fidelity["sourceTimingInference"] = (
            labeled_twelve_source_timing_diagnostics
        )
    fidelity["rhythmSourceTimingDetected"] = (
        rhythm_source_time_range is not None
    )
    if publish_rhythm_trace:
        fidelity["rhythmLeadValidation"] = {
            "passed": True,
            "lead": "II",
            "method": "upstream-source-pixel-rhythm-label-proof-v1",
            "semanticIdentityConfirmed": True,
        }
    if lead_order_validation is not None:
        fidelity["leadOrderValidation"] = lead_order_validation
        fidelity["passed"] = bool(
            fidelity["passed"] and lead_order_validation.get("passed")
        )
    if label_anchored_twelve:
        fidelity["inkConnectedTransitionRecovery"] = {
            "method": "near-continuous-vertical-source-ink-v1",
            "transitionScale": crossing_transition_scale,
            "displacementRewardPerPixel": (
                ink_connected_displacement_reward
            ),
            "maximumMissingInkFraction": 0.08,
            "maximumMissingInkPixelsFloor": 2,
        }
        fidelity["leadLabelValidation"] = geometry.get("leadLabelValidation")
        fidelity["passed"] = bool(
            fidelity["passed"]
            and (geometry.get("leadLabelValidation") or {}).get("passed")
            and lead_order_validation is not None
            and lead_order_validation.get("selectedOrder") == "standard"
        )
    if three_by_four_label_validated:
        fidelity["leadLabelValidation"] = geometry.get("leadLabelValidation")
        fidelity["passed"] = bool(
            fidelity["passed"]
            and (geometry.get("leadLabelValidation") or {}).get("passed")
            and (geometry.get("leadLabelValidation") or {}).get("order")
            == "standard"
        )
    if row_local_labeled_six:
        fidelity["precordialLabelValidation"] = geometry.get(
            "precordialLabelValidation"
        )
        fidelity["limbLabelValidation"] = geometry.get("limbLabelValidation")
        fidelity["passed"] = bool(
            fidelity["passed"]
            and (geometry.get("precordialLabelValidation") or {}).get("passed")
            and lead_order_validation is not None
            and lead_order_validation.get("selectedOrder") == "standard"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = output_dir / "native_grid_timeseries_canonical.csv"
    overlay_path = output_dir / "native_grid_overlay.png"
    metadata_path = output_dir / "digitization_metadata.csv"
    fidelity_path = output_dir / "source_fidelity.json"
    write_canonical_csv(
        canonical_path,
        canonical_values,
        lead_panel_columns=lead_panel_columns,
        panel_samples=panel_samples,
    )
    if not cv2.imwrite(
        str(overlay_path),
        draw_overlay(source_image, paths, lead_rows=lead_rows),
    ):
        raise OSError(f"Could not write overlay: {overlay_path}")
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("file_path", "matching_cost", "is_flipped", "lead_layout"))
        writer.writerow(
            (
                "native_grid",
                1.0 - float(geometry["confidence"]),
                "False",
                layout,
            )
        )
    fidelity_path.write_text(
        f"{json.dumps(fidelity, indent=2)}\n",
        encoding="utf-8",
    )

    return {
        "canonicalPath": str(canonical_path),
        "diagnosticPath": str(overlay_path),
        "metadataPath": str(metadata_path),
        "sourceFidelityPath": str(fidelity_path),
        "layout": layout,
        "layoutCost": 1.0 - float(geometry["confidence"]),
        "effectiveSampleRateHz": effective_sample_rate_hz,
        "sourceFidelity": fidelity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--upscale-max-dimension", type=int, default=0)
    parser.add_argument("--verified-rhythm-lead", choices=("II",))
    args = parser.parse_args()

    extraction_image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    source_image = (
        extraction_image
        if args.input.resolve() == args.source.resolve()
        else cv2.imread(str(args.source), cv2.IMREAD_COLOR)
    )
    trace_evidence = (
        cv2.imread(str(args.evidence), cv2.IMREAD_GRAYSCALE)
        if args.evidence
        else None
    )
    if extraction_image is None:
        raise SystemExit(f"Could not read extraction image: {args.input}")
    if source_image is None:
        raise SystemExit(f"Could not read source image: {args.source}")
    if args.evidence and trace_evidence is None:
        raise SystemExit(f"Could not read trace evidence: {args.evidence}")
    original_width = extraction_image.shape[1]
    original_max_dimension = max(extraction_image.shape[:2])
    if args.upscale_max_dimension > original_max_dimension:
        scale = args.upscale_max_dimension / original_max_dimension
        resized_shape = (
            max(1, round(extraction_image.shape[1] * scale)),
            max(1, round(extraction_image.shape[0] * scale)),
        )
        extraction_image = cv2.resize(
            extraction_image,
            resized_shape,
            interpolation=cv2.INTER_LANCZOS4,
        )
        source_image = cv2.resize(
            source_image,
            resized_shape,
            interpolation=cv2.INTER_LANCZOS4,
        )
        if trace_evidence is not None:
            trace_evidence = cv2.resize(
                trace_evidence,
                resized_shape,
                interpolation=cv2.INTER_LINEAR,
            )
    result = digitize_native_grid(
        extraction_image,
        source_image,
        args.output_dir,
        trace_evidence=trace_evidence,
        verified_rhythm_lead=args.verified_rhythm_lead,
    )
    if args.upscale_max_dimension > original_max_dimension:
        result["effectiveSampleRateHz"] = min(
            float(result["effectiveSampleRateHz"]),
            original_width / PAGE_DURATION_SECONDS,
        )
        result["sourceFidelity"]["deterministicUpscale"] = {
            "method": "lanczos-source-and-linear-evidence-v1",
            "originalMaxDimension": original_max_dimension,
            "targetMaxDimension": args.upscale_max_dimension,
            "effectiveSampleRateCappedToSource": True,
        }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
