"""Conservative timing proposals from source separators, grid and pulse edges.

This proposal is independent of waveform truth. A named grid only defines search windows: visible marks must supply
the boundaries, and independent physical calibration must corroborate spacing.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any

import numpy as np

METHOD = "source-separator-local-grid-timing-v2"


def _runs(active: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(active)
    return [part for part in np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1) if part.size]


def _fit_separator_centroids(observed: list[float]) -> tuple[list[float], float, float]:
    """Fit markers 1, 2, 3 without a platform-dependent linear algebra solver.

    Each observed centroid is the midpoint of integer source-pixel endpoints.
    With doubled centroids a,b,c, the exact fitted value at column j is
    (2*(4*a+b-2*c) + 3*j*(c-a))/12. Integer arithmetic preserves half-pixel
    ties before the existing round-to-even crop and publication operations.
    """
    if len(observed) != 3 or any(not np.isfinite(v) or v*2 != int(v*2) for v in observed):
        raise ValueError("separator centroids must be three finite pixel midpoints")
    a, b, c = (int(v*2) for v in observed)
    fitted = [(2*(4*a+b-2*c)+3*j*(c-a))/12 for j in range(5)]
    return fitted, (c-a)/4, abs(a-2*b+c)/6


def _propose_at_offset(image: np.ndarray, geometry: dict[str, Any], offset: int = 0) -> dict[str, Any]:
    from scripts.detect_ecg_layout import _grid_period

    rejected = {"version": 1, "method": METHOD, "state": "unresolved"}

    def reject(reason: str):
        return {**rejected, "failureReason": reason}

    grid = geometry.get("sourceLabelGrid") or {}
    if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
            or geometry.get("layoutHint") != "standard_3x4_with_r1"
            or grid.get("method") != "source-value-grid-below-trace-v1"
            or grid.get("imageSize") != [image.shape[1], image.shape[0]]
            or not (geometry.get("leadLabelValidation") or {}).get("semanticIdentityConfirmed")
            or not (geometry.get("rhythmLeadValidation") or {}).get("semanticIdentityConfirmed")):
        return reject("source-grid-identity-unconfirmed")
    from ecg_pipeline.source_label_geometry import valid_tiled_label_grid
    if not valid_tiled_label_grid(image, grid):
        return reject("source-tiled-label-evidence-invalid")
    calibration = geometry.get("calibration") or {}
    scale_mm = calibration.get("gridScaleMmX", calibration.get("gridScaleMm"))
    speed = calibration.get("paperSpeedMmPerSecond")
    confidence = calibration.get("confidence")
    physical = [speed, calibration.get("pixelsPerMmX"), calibration.get("pixelsPerMmY"),
                calibration.get("gainMmPerMv"), calibration.get("pulseEndX")]
    if (calibration.get("detected") is not True or calibration.get("gridScaleAmbiguous") is True
            or not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
            or not np.isfinite(confidence) or not .35 <= confidence <= 1
            or isinstance(scale_mm, bool) or scale_mm not in (1., 5.)
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) and v > 0 for v in physical)
            or (calibration.get("reconciliation") or {}).get("state") != "corroborated_inference"
            or (calibration.get("reconciliation") or {}).get("quantitativeBlocked") is not False):
        return reject("physical-calibration-unconfirmed")
    try:
        rows = np.asarray(geometry.get("rowCenters"), dtype=float)
        xfit = np.asarray(grid.get("xFit"), dtype=float)
    except (TypeError, ValueError):
        return reject("invalid-source-grid")
    font = grid.get("fontHeight")
    if (rows.shape != (4,) or xfit.shape != (3,) or not np.isfinite(rows).all()
            or not np.isfinite(xfit).all() or not np.all(np.diff(rows) > 0)
            or not isinstance(font, (int, float)) or isinstance(font, bool) or not np.isfinite(font) or font <= 0):
        return reject("invalid-source-grid")
    height, width = image.shape[:2]
    spacing = float(np.median(np.diff(rows)))
    mask = image.max(axis=2) < 160
    blue, green, red = np.moveaxis(image.astype(np.float32), 2, 0)
    grid_ink = np.clip(red-(green+blue)*.5, 0, 255)/255
    measured_grid = []
    for center in rows:
        radius = max(8, round(spacing*.07))
        top, bottom = max(0, round(center)-radius), min(height, round(center)+radius+1)
        period, confidence = _grid_period(grid_ink[top:bottom].mean(axis=0))
        if (period is None or not np.isfinite(period) or not np.isfinite(confidence)
                or confidence < .18 or not 1 <= period/scale_mm <= 40):
            return reject("row-grid-scale-unconfirmed")
        measured_grid.append({"rowCenter": float(center), "periodPixels": float(period),
                              "periodMm": scale_mm, "confidence": float(confidence)})
    scales = np.asarray([row["periodPixels"]/scale_mm for row in measured_grid])
    ppm = float(np.median(scales))
    if np.ptp(scales)/ppm > .02 or abs(ppm-calibration["pixelsPerMmX"])/ppm > .15:
        return reject("inconsistent-row-grid-scale")

    markers, boundaries = [], []
    for row, center in enumerate(rows[:3]):
        observed = []
        for column in (1, 2, 3):
            predicted = float(xfit@[1, row, column])
            left, right = max(0, round(predicted-font)), min(width, round(predicted+font)+1)
            top, bottom = max(0, round(center-font*.8)+offset), min(height, round(center+font*.55)+offset)
            if right-left < 5 or bottom-top < 5:
                return reject("separator-window-outside-image")
            profile = mask[top:bottom, left:right].mean(axis=0)
            candidates = [part for part in _runs(profile >= .8)
                          if max(3, round(font*.08)) <= part.size <= max(4, round(font*.30))]
            if len(candidates) != 1:
                return reject("source-separator-missing-or-ambiguous")
            part = candidates[0]
            x = float(left+(part[0]+part[-1])/2)
            observed.append(x)
            markers.append({"row": row, "column": column, "centerX": x,
                            "bounds": [int(left+part[0]), top, int(left+part[-1]+1), bottom],
                            "minimumBlackColumnSupport": float(np.min(profile[part]))})
        fitted, panel_width, residual = _fit_separator_centroids(observed)
        expected_width = scales[row]*speed*2.5
        if residual > max(1.0, ppm*.25) or abs(panel_width-expected_width)/expected_width > .03:
            return reject("separator-spacing-disagrees-with-grid")
        boundaries.append(fitted)
    boundaries = np.asarray(boundaries)
    if np.max(np.ptp(boundaries, axis=0)) > ppm:
        return reject("panel-boundaries-vary-by-row")
    common = np.median(boundaries, axis=0)
    if (common[0] < 0 or common[-1] > width
            or common[0] < calibration["pulseEndX"]-max(2, ppm*.25)
            or abs(common[0]-calibration["pulseEndX"]) > ppm):
        return reject("panel-origin-disagrees-with-pulse")

    # All four rows, including the separate rhythm, must retain the observed
    # descending edge near this origin. A blank margin cannot substantiate it.
    pulse_edges = []
    pulse_height = calibration["pixelsPerMmY"]*calibration["gainMmPerMv"]
    for row, center in enumerate(rows):
        left, right = max(0, round(common[0]-ppm)), min(width, round(common[0]+ppm)+1)
        top, bottom = max(0, round(center-pulse_height*.9)), min(height, round(center-pulse_height*.1))
        if bottom-top < 5:
            return reject("pulse-edge-window-outside-image")
        profile = mask[top:bottom, left:right].mean(axis=0)
        candidates = [part for part in _runs(profile >= .65) if 1 <= part.size <= max(3, round(ppm*.7))]
        if len(candidates) != 1:
            return reject("source-pulse-edge-missing-or-ambiguous")
        part = candidates[0]
        pulse_edges.append({"row": row, "centerX": float(left+(part[0]+part[-1])/2),
                            "bounds": [int(left+part[0]), top, int(left+part[-1]+1), bottom],
                            "minimumBlackColumnSupport": float(np.min(profile[part]))})
    # The final panel has no next separator. Measure its local paper grid
    # independently in all four rows instead of extrapolating equal pixel width
    # through scanner distortion. Calibration fixes physical units; no truth fit.
    nominal_end = float(common[-1])
    local_measurements = []
    left, right = round(common[3]), min(width, round(common[4]))
    for row, center in enumerate(rows):
        radius = max(8, round(spacing*.07))
        top, bottom = max(0, round(center)-radius), min(height, round(center)+radius+1)
        period, confidence = _grid_period(grid_ink[top:bottom, left:right].mean(axis=0))
        if (period is None or not np.isfinite(period) or not np.isfinite(confidence)
                or confidence < .18 or not 1 <= period/scale_mm <= 40):
            return reject("final-panel-grid-unconfirmed")
        local_measurements.append({"row": row, "periodPixels": float(period),
                                   "periodMm": scale_mm, "confidence": float(confidence)})
    local_scales = np.asarray([m["periodPixels"]/scale_mm for m in local_measurements])
    local_ppm = float(np.median(local_scales))
    final_width = local_ppm*speed*2.5
    nominal_width = float(common[4]-common[3])
    if (np.ptp(local_scales)/local_ppm > .02
            or abs(final_width-nominal_width)/nominal_width > .03):
        return reject("final-panel-grid-inconsistent")
    common[-1] = common[3]+final_width
    if common[-1] > width:
        return reject("physical-final-boundary-outside-image")
    edges = np.rint(common).astype(int)
    ranges = [[int(a), int(b)] for a, b in zip(edges[:-1], edges[1:])]
    ink_support = []
    for row, center in enumerate(rows):
        radius = max(3, round(spacing*.12))
        profile = mask[max(0, round(center)-radius):min(height, round(center)+radius+1)].any(axis=0)
        for column, (left, right) in enumerate(ranges):
            fraction = float(profile[left:right].mean())
            if fraction < .20:
                return reject("source-waveform-support-missing")
            ink_support.append({"row": row, "column": column, "fraction": fraction})
    return {"version": 2, "method": METHOD, "state": "source_supported",
            "separatorFitMethod": "integer-midpoint-three-marker-ols-v1",
            "imageSize": [width, height], "decodedRasterSha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "paperSpeedMmPerSecond": speed, "panelDurationSeconds": 2.5,
            "physicalCalibration": {key: calibration[key] for key in (
                "detected", "confidence", "pixelsPerMmX", "pixelsPerMmY",
                "paperSpeedMmPerSecond", "gainMmPerMv", "reconciliation")},
            "gridPixelsPerMmX": ppm, "rowGridMeasurements": measured_grid,
            "sourceSeparators": markers, "rowBoundariesX": boundaries.tolist(),
            "maximumRowBoundarySpreadPixels": float(np.max(np.ptp(boundaries, axis=0))),
            "sourcePulseEdges": pulse_edges, "calibrationPulseEndX": calibration["pulseEndX"],
            "panelRanges": ranges, "rhythmRange": [int(edges[0]), int(edges[-1])],
            "sourceInkSupport": ink_support, "truthUsed": False,
            "finalPanelLocalGrid": {"measurements": local_measurements, "pixelsPerMm": local_ppm,
                "nominalExtrapolatedEndX": nominal_end, "physicalEndX": float(common[-1]),
                "method": "four-row-local-grid-with-calibrated-speed-v1"}}


def _window_counts(image: np.ndarray, geometry: dict[str, Any], offset: int) -> list[int]:
    """Inspect every window so an earlier missing mark cannot hide ambiguity."""
    grid = geometry["sourceLabelGrid"]
    font, xfit = grid["fontHeight"], np.asarray(grid["xFit"])
    height, width = image.shape[:2]
    mask = image.max(axis=2) < 160
    counts = []
    for row, center in enumerate(geometry["rowCenters"][:3]):
        for column in (1, 2, 3):
            predicted = float(xfit @ [1, row, column])
            left = max(0, round(predicted-font))
            right = min(width, round(predicted+font)+1)
            top = max(0, round(center-font*.8)+offset)
            bottom = min(height, round(center+font*.55)+offset)
            if right-left < 5 or bottom-top < 5:
                counts.append(-1)
                continue
            profile = mask[top:bottom, left:right].mean(axis=0)
            counts.append(len([
                part for part in _runs(profile >= .8)
                if max(3, round(font*.08)) <= part.size <= max(4, round(font*.30))
            ]))
    return counts


def _window_consensus(attempts: list[dict[str, Any]], radius: int) -> dict[str, Any]:
    def reject(reason: str):
        return {"state": "unresolved", "failureReason": reason}

    if [a["offsetPixels"] for a in attempts] != list(range(-radius, radius+1)):
        return reject("incomplete-offset-enumeration")
    if any(len(a["windowCandidateCounts"]) != 9 or
           any(count < 0 or count > 1 for count in a["windowCandidateCounts"])
           for a in attempts):
        return reject("competing-or-invalid-source-windows")
    accepted = [a for a in attempts if a["timing"]["state"] == "source_supported"]
    offsets = [a["offsetPixels"] for a in accepted]
    if not any(right-left == 1 for left, right in zip(offsets, offsets[1:])):
        return reject("insufficient-adjacent-window-support")
    exact_fields = ["panelRanges", "rhythmRange", "sourcePulseEdges", "physicalCalibration",
                    "decodedRasterSha256", "rowGridMeasurements"]
    if any(any(a["timing"][key] != accepted[0]["timing"][key] for key in exact_fields)
           for a in accepted):
        return reject("source-window-published-timing-disagreement")
    representative = min(accepted, key=lambda a: (abs(a["offsetPixels"]), a["offsetPixels"]))
    markers = []
    for index in range(9):
        observed = [a["timing"]["sourceSeparators"][index] for a in accepted]
        centers = [mark["centerX"] for mark in observed]
        rectangles = [mark["bounds"] for mark in observed]
        if max(centers)-min(centers) > 1:
            return reject("source-center-uncertainty-exceeds-one-pixel")
        intersection = [max(box[0] for box in rectangles), min(box[2] for box in rectangles)]
        if intersection[1] <= intersection[0]:
            return reject("source-stroke-footprints-disjoint")
        union = [min(box[0] for box in rectangles), min(box[1] for box in rectangles),
                 max(box[2] for box in rectangles), max(box[3] for box in rectangles)]
        markers.append({
            "row": index//3, "column": index%3+1,
            "centerIntervalPixels": [min(centers), max(centers)],
            "horizontalIntersection": intersection, "unionBounds": union,
            "observations": [{"offsetPixels": a["offsetPixels"],
                              **a["timing"]["sourceSeparators"][index]} for a in accepted],
        })
    row_boundaries = np.stack([np.asarray(a["timing"]["rowBoundariesX"]) for a in accepted])
    intervals = np.stack([row_boundaries.min(axis=0), row_boundaries.max(axis=0)], axis=-1)
    receipt = {
        "version": 1, "method": "bounded-source-separator-window-consensus-v1",
        "radiusPixels": radius, "offsetsAttempted": list(range(-radius, radius+1)),
        "successfulOffsets": offsets, "selectedOffsetPixels": representative["offsetPixels"],
        "selectionRule": "minimum-absolute-offset-negative-first",
        "allPublishedTimingAndPulsesIdentical": True, "maximumCenterIntervalPixels": 1,
        "sourceSeparators": markers, "rowBoundaryIntervalsPixels": intervals.tolist(),
        "timingProposals": [{"offsetPixels": a["offsetPixels"], **{
            key: a["timing"][key] for key in
            ["panelRanges", "rhythmRange", "sourcePulseEdges", "rowBoundariesX"]
        }} for a in accepted],
        "attempts": [{"offsetPixels": a["offsetPixels"], "state": a["timing"]["state"],
                      "failureReason": a["timing"].get("failureReason"),
                      "windowCandidateCounts": a["windowCandidateCounts"]} for a in attempts],
        "truthUsed": False,
    }
    return {"state": "source_supported", "representative": copy.deepcopy(representative["timing"]),
            "consensus": receipt}


def _propose_source_window_timing(image: np.ndarray, geometry: dict[str, Any]) -> dict[str, Any]:
    """Preserve the original successful route; require consensus for a missing bar."""
    initial = _propose_at_offset(image, geometry)
    if initial["state"] == "source_supported" or initial.get("failureReason") != "source-separator-missing-or-ambiguous":
        return initial
    font = geometry["sourceLabelGrid"]["fontHeight"]
    radius = round(font * .25)
    # Source rasters are bounded before this route; also bound pathological metadata.
    if not 1 <= radius <= 64:
        return initial
    attempts = []
    for offset in range(-radius, radius + 1):
        timing = initial if offset == 0 else _propose_at_offset(image, geometry, offset)
        attempts.append({"offsetPixels": offset, "timing": timing,
                         "windowCandidateCounts": _window_counts(image, geometry, offset)})
    proposed = _window_consensus(attempts, radius)
    if proposed["state"] != "source_supported":
        return {**initial, "separatorWindowFallback": {"version": 1,
            "method": "bounded-source-separator-window-consensus-v1",
            "radiusPixels": radius, "failureReason": proposed["failureReason"],
            "attempts": [{"offsetPixels": a["offsetPixels"], "state": a["timing"]["state"],
                "failureReason": a["timing"].get("failureReason"),
                "windowCandidateCounts": a["windowCandidateCounts"]} for a in attempts]}}
    receipt = proposed["consensus"]
    receipt.update(fontHeightPixels=font, windowRadiusFraction=.25)
    return {**proposed["representative"], "version": 3,
            "method": "source-separator-window-consensus-timing-v3",
            "separatorWindowConsensus": receipt}


def propose_source_panel_timing(image: np.ndarray, geometry: dict[str, Any]) -> dict[str, Any]:
    """Keep complete old success exact; freshly measure an eligible joint fallback."""
    previous = _propose_source_window_timing(image, geometry)
    if (previous["state"] == "source_supported" or previous.get("failureReason") not in {
            "source-separator-missing-or-ambiguous", "panel-origin-disagrees-with-pulse"}):
        return previous
    from ecg_pipeline.joint_source_timing import propose_joint_source_timing
    proposed = propose_joint_source_timing(image, geometry)
    if proposed["state"] == "source_supported":
        return proposed
    return {**previous, "jointSourceFallback": {"version": 1,
            "method": proposed["method"], "failureReason": proposed["failureReason"]}}
