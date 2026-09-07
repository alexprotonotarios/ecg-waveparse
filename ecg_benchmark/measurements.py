"""Labelled interval/amplitude endpoints, separate from detector-on-truth error.

These engineering measurements are not a clinical ECG delineator. Search windows,
levels, polarity, reference values and source visibility are frozen annotations.
Neither windows nor thresholds are chosen from the reconstructed waveform.
"""
from __future__ import annotations

import numpy as np


def _finite(value) -> bool:
    return type(value) in (int, float) and bool(np.isfinite(value))


def validate_measurement(item: dict) -> None:
    if (not isinstance(item, dict) or not isinstance(item.get('id'), str) or not item['id']
            or item.get('kind') not in {'interval', 'amplitude'}
            or item.get('labelMethod') not in {'controlled', 'adjudicated'}
            or item.get('visibility') not in {'visible', 'ambiguous', 'unrecoverable'}
            or item.get('polarity') not in (-1, 1) or isinstance(item.get('polarity'), bool)
            or not _finite(item.get('referenceValue'))):
        raise ValueError('invalid_measurement_annotation')
    names = ['baselineRange', 'peakRange'] if item['kind'] == 'amplitude' else ['baselineRange', 'startRange', 'endRange']
    for name in names:
        window = item.get(name)
        if (not isinstance(window, dict) or not all(_finite(window.get(k)) for k in ('startSample', 'endSample'))
                or window['endSample'] <= window['startSample']):
            raise ValueError('invalid_measurement_window')
    if item['kind'] == 'interval' and (not _finite(item.get('levelUv')) or item['levelUv'] <= 0
                                     or item['startRange']['startSample'] > item['endRange']['startSample']
                                     or item['startRange']['endSample'] > item['endRange']['endSample']
                                     or item['referenceValue'] <= 0):
        raise ValueError('invalid_interval_annotation')


def _window(values: np.ndarray, bounds: dict) -> tuple[np.ndarray, int] | None:
    # Fractional endpoints can result from rate conversion. Never clip an
    # unavailable window or silently jump across a missing sample.
    first, last = int(np.ceil(bounds['startSample'])), int(np.ceil(bounds['endSample']))
    if first < 0 or last > values.size or first >= last:
        return None
    result = values[first:last]
    return (result, first) if np.isfinite(result).all() else None


def _crossing(values: np.ndarray, window: dict, baseline: float, item: dict, rising: bool) -> float | None:
    selected = _window(values, window)
    if selected is None or selected[0].size < 2:
        return None
    curve = item['polarity'] * (selected[0] - baseline) - item['levelUv']
    left, right = curve[:-1], curve[1:]
    transitions = (left < 0) & (right >= 0) if rising else (left > 0) & (right <= 0)
    indices = np.flatnonzero(transitions)
    if indices.size != 1:
        return None  # Multiple crossings are ambiguous, not a best-match search.
    i = int(indices[0])
    return float(selected[1] + i - left[i] / (right[i] - left[i]))


def detect_measurement(values: np.ndarray, item: dict, sample_rate_hz: float) -> float | None:
    baseline = _window(values, item['baselineRange'])
    if baseline is None:
        return None
    center = float(np.median(baseline[0]))
    if item['kind'] == 'amplitude':
        peak = _window(values, item['peakRange'])
        if peak is None:
            return None
        # Return signed amplitude; negative complexes remain negative.
        index = int(np.argmax(item['polarity'] * (peak[0] - center)))
        return float(peak[0][index] - center)
    start = _crossing(values, item['startRange'], center, item, True)
    end = _crossing(values, item['endRange'], center, item, False)
    return (end - start) * 1000 / sample_rate_hz if start is not None and end is not None and end > start else None


def score_measurements(truth: np.ndarray, candidate: np.ndarray, annotations: list[dict], *, sample_rate_hz: float) -> dict:
    if not _finite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError('invalid_measurement_sample_rate')
    rows, ids = [], set()
    for item in annotations:
        validate_measurement(item)
        if item['id'] in ids:
            raise ValueError('duplicate_measurement_identity')
        ids.add(item['id'])
        row = {k: item[k] for k in ('id', 'kind', 'labelMethod', 'visibility', 'referenceValue')}
        row['units'] = 'ms' if item['kind'] == 'interval' else 'uV'
        if item['visibility'] != 'visible':
            row.update(status='not_assessable_at_source_resolution', truthDetected=None, candidateDetected=None,
                       detectorOnTruthError=None, reconstructionIncrement=None, totalErrorAgainstLabel=None)
        else:
            reference = detect_measurement(truth, item, sample_rate_hz)
            result = detect_measurement(candidate, item, sample_rate_hz)
            row.update(status='completed' if reference is not None and result is not None else 'unavailable_or_ambiguous',
                       truthDetected=reference, candidateDetected=result,
                       detectorOnTruthError=reference-item['referenceValue'] if reference is not None else None,
                       reconstructionIncrement=result-reference if reference is not None and result is not None else None,
                       totalErrorAgainstLabel=result-item['referenceValue'] if result is not None else None)
        rows.append(row)
    return {'version': 1, 'method': 'frozen-window-signed-amplitude-and-level-duration-v1',
            'annotatedCount': len(rows), 'sourceVisibleCount': sum(r['visibility'] == 'visible' for r in rows),
            'completedCount': sum(r['status'] == 'completed' for r in rows), 'endpoints': rows,
            'clinicalDelineationValidated': False}


def source_feature_visibility(source: np.ndarray, counterfactual: np.ndarray, *, region: tuple[int, int, int, int],
                              horizontal_extent_pixels: float, vertical_extent_pixels: float) -> dict:
    """Conservative raster recoverability proxy using a controlled omitted feature.

    The counterfactual is generated only from known synthetic construction. It
    is never used to modify a patient image or supply reconstruction evidence.
    Two-pixel localisation support is an engineering label rule, not perception
    or clinical validation. A counterfactual difference alone is insufficient.
    """
    if source.shape != counterfactual.shape or source.dtype != np.uint8 or counterfactual.dtype != np.uint8 or source.ndim != 3:
        raise ValueError('invalid_visibility_rasters')
    x0, y0, x1, y1 = region
    if not (0 <= x0 < x1 <= source.shape[1] and 0 <= y0 < y1 <= source.shape[0]):
        raise ValueError('invalid_visibility_region')
    if not all(_finite(v) and v >= 0 for v in (horizontal_extent_pixels, vertical_extent_pixels)):
        raise ValueError('invalid_visibility_extent')
    changed = np.max(np.abs(source[y0:y1,x0:x1].astype(float)-counterfactual[y0:y1,x0:x1].astype(float)), axis=2) >= 32
    count = int(changed.sum())
    state = 'unrecoverable' if count == 0 else 'visible' if horizontal_extent_pixels >= 2 and vertical_extent_pixels >= 2 and count >= 4 else 'ambiguous'
    return {'version': 1, 'method': 'controlled-counterfactual-raster-contrast-v1', 'state': state,
            'changedPixels': count, 'contrastThreshold': 32, 'horizontalExtentPixels': horizontal_extent_pixels,
            'verticalExtentPixels': vertical_extent_pixels, 'minimumLocalisationExtentPixels': 2,
            'meaning': 'Conservative engineering localisation support; no human perceptual adjudication.'}
