"""Static source-edge calibration proposals with explicit expanded periods."""
import itertools
from functools import partial
from typing import Any
import cv2
import numpy as np
from scripts.detect_ecg_layout import _as_gray, _chromatic_excess_planes
from . import _period as period

def _grid_period(profile):
    return period.grid_period(profile, 96)

def edge_evidence(neutral, q):
    top, bottom = (q['top'], q['bottom'])
    height, width = neutral.shape
    edges = []
    for x, y0, y1, line_width in q['orderedLines']:
        start = max(0, round(x - line_width / 2))
        end = min(width, start + line_width)
        bounds = [start, max(0, top), end, min(height, bottom)]
        left, upper, right, lower = bounds
        tile = neutral[upper:lower, left:right]
        rows = int(np.count_nonzero(np.any(tile, axis=1)))
        length = tile.shape[0]
        coverage = rows / length if length else 0.0
        edges.append({'bounds': bounds, 'supportedRows': rows, 'totalRows': length, 'coverage': coverage})
    bounds = [q['xStart'], max(0, top - 1), q['xEnd'], min(height, top + 2)]
    a, b, c, d = bounds
    tile = neutral[b:d, a:c]
    pixels = int(np.count_nonzero(tile))
    total = tile.size
    support = pixels / total if total else 0.0
    return {'verticalEdges': edges, 'topEdge': {'bounds': bounds, 'supportedPixels': pixels, 'totalPixels': total, 'support': support}, 'passed': bool(all((e['coverage'] >= 0.8 for e in edges)) and support >= 0.45)}

def _detect_pulse_proposals(image: np.ndarray, *, enforce_edges: bool):
    """Detect grid spacing and a rectangular 1 mV calibration pulse."""
    observations = []
    gray = _as_gray(image)
    height, width = gray.shape
    darkness = 1.0 - gray.astype(np.float32) / 255.0
    grayscale_periods = (*_grid_period(darkness.mean(axis=0)), *_grid_period(darkness.mean(axis=1)))
    period_candidates = [grayscale_periods]
    if image.ndim == 3 and image.shape[2] >= 3:
        for channel_excess in _chromatic_excess_planes(image):
            period_candidates.append((*_grid_period(channel_excess.mean(axis=0)), *_grid_period(channel_excess.mean(axis=1))))
    grid_candidates: list[dict[str, float]] = []
    for x_period, x_confidence, y_period, y_confidence in period_candidates:
        if x_period is None or y_period is None:
            continue
        disagreement = abs(x_period - y_period) / max(x_period, y_period, 1e-09)
        if disagreement > 0.15:
            continue
        grid_candidates.append({'xPeriod': float(x_period), 'yPeriod': float(y_period), 'xConfidence': float(x_confidence), 'yConfidence': float(y_confidence), 'confidence': float(min(x_confidence, y_confidence) * (1.0 - disagreement))})
    selected_grid = max(grid_candidates, key=lambda candidate: candidate['confidence']) if grid_candidates else None
    if selected_grid:
        x_period = selected_grid['xPeriod']
        y_period = selected_grid['yPeriod']
        x_confidence = selected_grid['xConfidence']
        y_confidence = selected_grid['yConfidence']
    else:
        x_period, x_confidence, y_period, y_confidence = grayscale_periods
    standalone_grid_scale_mm = 1.0 if selected_grid is not None and max(selected_grid['xPeriod'], selected_grid['yPeriod']) < 12.0 else None
    result: dict[str, Any] = {'calibration': {'method': 'grid-period-plus-rectangular-pulse-v1', 'detected': False, 'gridSpacingXPixels': x_period, 'gridSpacingYPixels': y_period, 'gridScaleDetected': standalone_grid_scale_mm is not None, 'gridScaleAmbiguous': bool(selected_grid is not None and standalone_grid_scale_mm is None), 'gridScaleConfidence': selected_grid['confidence'] if standalone_grid_scale_mm is not None and selected_grid else 0.0, **({'gridScaleMm': standalone_grid_scale_mm, 'gridScaleMmX': standalone_grid_scale_mm, 'gridScaleMmY': standalone_grid_scale_mm, 'pixelsPerMmX': selected_grid['xPeriod'] / standalone_grid_scale_mm, 'pixelsPerMmY': selected_grid['yPeriod'] / standalone_grid_scale_mm} if standalone_grid_scale_mm is not None and selected_grid else {}), 'confidence': 0.0}}
    if x_period is None or y_period is None:
        return (result, observations)
    otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, dark = cv2.threshold(gray, min(160.0, max(40.0, otsu_threshold * 0.75)), 255, cv2.THRESH_BINARY_INV)
    neutral = dark > 0
    if image.ndim == 3 and image.shape[2] >= 3:
        neutral &= np.max(image[:, :, :3], axis=2) - np.min(image[:, :, :3], axis=2) <= 28
    minimum_vertical = max(8, round(y_period * 1.4))
    vertical = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, minimum_vertical)))
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
            x0, x1 = (x1, x0)
            y0, y1, other_y0, other_y1 = (other_y0, other_y1, y0, y1)
            first_line_width, second_line_width = (second_line_width, first_line_width)
        pulse_width_pixels = x1 - x0
        pulse_height_pixels = min(y1, other_y1) - max(y0, other_y0) - max(first_line_width, second_line_width)
        if pulse_width_pixels <= 0 or pulse_height_pixels <= 0:
            continue
        top = round(max(y0, other_y0))
        x_start = max(0, round(x0))
        x_end = min(width, round(x1) + 1)
        if x_end - x_start < 3:
            continue
        horizontal_support = float(np.mean(dark[max(0, top - 1):min(height, top + 2), x_start:x_end] > 0))
        if horizontal_support < 0.45:
            continue
        grid_scales_mm_x = (1.0, 5.0) if x_period >= 12.0 else (1.0,)
        grid_scales_mm_y = (1.0, 5.0) if y_period >= 12.0 else (1.0,)
        for grid_scale_mm_x, grid_scale_mm_y in itertools.product(grid_scales_mm_x, grid_scales_mm_y):
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
            pulse_pixels_per_mm_x = pulse_width_pixels / (speed * 0.2)
            pulse_pixels_per_mm_y = pulse_height_pixels / gain
            refined_disagreement = abs(pulse_pixels_per_mm_x - pulse_pixels_per_mm_y) / max(pulse_pixels_per_mm_x, pulse_pixels_per_mm_y, 1e-09)
            if refined_disagreement <= 0.15:
                refined_pixels_per_mm_x = pulse_pixels_per_mm_x
                refined_pixels_per_mm_y = pulse_pixels_per_mm_y
            else:
                refined_pixels_per_mm_x = pixels_per_mm_x
                refined_pixels_per_mm_y = pixels_per_mm_y
            confidence = min(x_confidence, y_confidence) * horizontal_support * max(0.0, 1.0 - gain_error) * max(0.0, 1.0 - speed_error)
            candidate = {'method': 'grid-period-plus-rectangular-pulse-v1', 'detected': True, 'gridScaleDetected': True, 'gridScaleAmbiguous': False, 'gridScaleConfidence': float(confidence), **({'gridScaleMm': grid_scale_mm_x} if grid_scale_mm_x == grid_scale_mm_y else {}), 'gridScaleMmX': grid_scale_mm_x, 'gridScaleMmY': grid_scale_mm_y, 'gridSpacingXPixels': x_period, 'gridSpacingYPixels': y_period, 'pixelsPerMmX': float(refined_pixels_per_mm_x), 'pixelsPerMmY': float(refined_pixels_per_mm_y), 'paperSpeedMmPerSecond': speed, 'gainMmPerMv': gain, 'measuredGainMmPerMv': height_mm, 'pulseWidthMm': width_mm, 'pulseStartX': x_start, 'pulseEndX': x_end, 'confidence': float(confidence)}
            observations.append({'candidate': candidate.copy(), 'linePair': [list(first), list(second)], 'orderedLines': [[x0, y0, y1, first_line_width], [x1, other_y0, other_y1, second_line_width]], 'pulseHeightPixels': pulse_height_pixels, 'pulseWidthPixels': pulse_width_pixels, 'top': top, 'bottom': min(y1, other_y1), 'xStart': x_start, 'xEnd': x_end, 'horizontalSupport': horizontal_support, 'gainError': gain_error, 'speedError': speed_error, 'refinedDisagreement': refined_disagreement})
            observations[-1]['neutralEdgeEvidence'] = edge_evidence(neutral, observations[-1])
            observations[-1]['physicalConsistencyPassed'] = bool(grid_scale_mm_x == grid_scale_mm_y and refined_disagreement <= 0.15)
            observations[-1]['acceptedForSelection'] = bool(not enforce_edges or (observations[-1]['neutralEdgeEvidence']['passed'] and observations[-1]['physicalConsistencyPassed']))
            if not observations[-1]['acceptedForSelection']:
                continue
            if best is None or candidate['confidence'] > best['confidence']:
                best = candidate
    if best is not None:
        result['calibration'] = best
    return (result, observations)

def build(layout, arm):
    if arm not in {'baseline', 'edge-proof'}:
        raise ValueError('Unknown pulse proposal mode.')
    return partial(_detect_pulse_proposals, enforce_edges=arm == 'edge-proof')
