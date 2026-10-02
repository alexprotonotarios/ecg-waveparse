"""Certify steep rhythm transitions from complete, unchanged source strokes."""
from __future__ import annotations

import hashlib
import math
from typing import Any

import cv2
import numpy as np


def _hash(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


def certify_source_transition(
    evidence: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    valid0: bool,
    valid1: bool,
    source_interval: tuple[int, int],
) -> dict[str, Any]:
    """Require one complete eight-connected component between endpoint regions.

    A certificate does not modify coordinates, recover a gap or establish lead
    identity. It only records whether a local source stroke supports this pair.
    """
    low, high = map(int, source_interval)
    height, width = evidence.shape
    eligible = bool(
        valid0 and valid1 and x1 == x0 + 1
        and 0 <= low <= x0 < x1 < high <= width
        and 0 <= min(y0, y1) <= max(y0, y1) < height
    )
    result: dict[str, Any] = {
        "eligible": eligible,
        "certified": False,
        "sourcePixels": [[int(x0), int(y0)], [int(x1), int(y1)]],
        "displacementPixels": abs(int(y1) - int(y0)),
        "sourceInterval": [low, high],
        "evidenceThreshold": 0.24,
        "horizontalMarginPixels": 3,
        "endpointRadiusPixels": 1,
        "pathOrValidityChanged": False,
    }
    if not eligible:
        return result
    left, right = max(low, x0 - 3), min(high, x1 + 4)
    top, bottom = min(y0, y1), max(y0, y1) + 1
    binary = (evidence[top:bottom, left:right] >= 0.24).astype(np.uint8)
    _, labels = cv2.connectedComponents(binary, connectivity=8)

    def endpoint(x: int, y: int) -> set[int]:
        values = labels[
            max(0, y - top - 1):min(bottom - top, y - top + 2),
            max(0, x - left - 1):min(right - left, x - left + 2),
        ]
        return set(int(value) for value in values.ravel()) - {0}

    shared = endpoint(x0, y0) & endpoint(x1, y1)
    complete = [
        value for value in sorted(shared)
        if np.all(np.any(labels == value, axis=1))
    ]
    result.update({
        "bounds": [left, top, right, bottom],
        "verticalSupportFraction": float(np.mean(np.any(binary, axis=1))),
        "sharedComponentCount": len(shared),
        "completeConnectedComponentCount": len(complete),
        "binarySha256": _hash(binary, "u1"),
        "certified": len(complete) == 1,
    })
    if len(complete) == 1:
        result["certifiedComponentSha256"] = _hash(labels == complete[0], "u1")
    return result


def source_rhythm_continuity(
    evidence: np.ndarray,
    columns: np.ndarray,
    path: np.ndarray,
    validity: np.ndarray,
    *,
    source_interval: tuple[int, int],
    row_spacing: float,
    decoded_raster_sha256: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Retain old QA and certify every measured pair exceeding its jump bound.

    Call after the same source-support filtering used by fidelity measurement.
    The original retained validity is bound separately from its recorded-domain
    intersection, matching source-grid conversion without recovering margins.
    """
    if (
        evidence.ndim != 2 or evidence.dtype != np.float32
        or not np.all(np.isfinite(evidence))
        or np.any(evidence < 0) or np.any(evidence > 1)
        or columns.ndim != 1 or columns.size < 1
        or path.shape != columns.shape or validity.shape != columns.shape
        or validity.dtype != np.bool_
        or not np.issubdtype(columns.dtype, np.integer)
        or not np.issubdtype(path.dtype, np.integer)
        or np.any(np.diff(columns) <= 0)
        or np.any(columns < 0) or np.any(columns >= evidence.shape[1])
        or np.any(path < 0) or np.any(path >= evidence.shape[0])
        or isinstance(row_spacing, (bool, np.bool_))
        or not isinstance(row_spacing, (float, int, np.floating, np.integer))
        or not math.isfinite(row_spacing) or row_spacing <= 0
        or not isinstance(source_interval, (tuple, list))
        or len(source_interval) != 2
        or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
               for v in source_interval)
        or not 0 <= source_interval[0] < source_interval[1] <= evidence.shape[1]
        or not isinstance(decoded_raster_sha256, str)
        or len(decoded_raster_sha256) != 64
        or any(c not in "0123456789abcdef" for c in decoded_raster_sha256)
    ):
        raise ValueError("Invalid source rhythm continuity evidence.")
    low, high = map(int, source_interval)
    recorded = validity & (columns >= low) & (columns < high)
    if not np.any(recorded):
        raise ValueError("No retained rhythm samples in the verified source interval.")
    measured_pairs = recorded[:-1] & recorded[1:]
    jumps = np.abs(np.diff(path.astype(np.float64)))
    measured = jumps[measured_pairs]
    raw_maximum = float(np.max(measured)) if measured.size else 0.0
    raw_p95 = float(np.quantile(measured, 0.95)) if measured.size else 0.0
    coverage = float(np.count_nonzero(recorded) / (high - low))
    support_p10 = float(np.quantile(evidence[path[recorded], columns[recorded]], 0.10))
    for key, expected in [
        ("maxNativeJumpPixels", raw_maximum),
        ("p95NativeJumpPixels", raw_p95),
        ("coverage", coverage),
        ("evidenceP10", support_p10),
    ]:
        if (isinstance(metrics.get(key), bool)
                or not isinstance(metrics.get(key), (int, float))
                or metrics[key] != expected):
            raise ValueError("Continuity evidence must use the measured source path.")
    home = metrics.get("homeRowFraction")
    if isinstance(home, bool) or not isinstance(home, (int, float)) or not 0 <= home <= 1:
        raise ValueError("Invalid rhythm home-row measurement.")
    bound = float(row_spacing * 0.35)
    indexes = np.flatnonzero(measured_pairs & (jumps > bound))
    transitions = [certify_source_transition(
        evidence, int(columns[i]), int(path[i]), int(columns[i + 1]), int(path[i + 1]),
        bool(recorded[i]), bool(recorded[i + 1]), (low, high),
    ) for i in indexes]
    original_passed = raw_maximum <= bound
    all_certified = bool(transitions) and all(t["certified"] for t in transitions)
    other_checks = bool(
        coverage >= 0.90 and support_p10 >= 0.45 and home >= 0.95
        and raw_p95 <= row_spacing * 0.20
    )
    maximum_passed = original_passed or all_certified
    return {
        "version": 1,
        "method": "complete-connected-source-rhythm-strokes-v1",
        "decodedRasterSha256": decoded_raster_sha256,
        "imageSize": [int(evidence.shape[1]), int(evidence.shape[0])],
        "sourceInterval": [low, high],
        "evidenceEncoding": "float32-le-row-major-v1",
        "evidenceSha256": _hash(evidence, "<f4"),
        "pathEncoding": "int32-le-y-by-source-column-v1",
        "columnsSha256": _hash(columns, "<i4"),
        "sourcePathSha256": _hash(path, "<i4"),
        "validitySha256": _hash(validity, "u1"),
        "recordedValiditySha256": _hash(recorded, "u1"),
        "retainedRecordedColumns": int(np.count_nonzero(recorded)),
        "measuredPairCount": int(np.count_nonzero(measured_pairs)),
        "rowSpacingPixels": float(row_spacing),
        "rawMaximumJumpPixels": raw_maximum,
        "originalMaximumAllowedJumpPixels": bound,
        "originalMaximumJumpPassed": bool(original_passed),
        "otherRhythmChecksPassed": other_checks,
        "legacyRhythmTracePassed": bool(other_checks and original_passed),
        "overLimitMeasuredPairCount": len(transitions),
        "overLimitTransitions": transitions,
        "allOverLimitMeasuredPairsCertified": all_certified,
        "rhythmTracePassed": bool(other_checks and maximum_passed),
        "sourcePathChanged": False,
        "sourceEvidenceChanged": False,
        "sourceTimingChanged": False,
        "validityChanged": False,
        "gapsRecovered": 0,
        "truthUsed": False,
    }
