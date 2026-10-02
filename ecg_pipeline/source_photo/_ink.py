"""Diagnostic pulse-referenced ink limit; never alter a source image."""
import hashlib
from ecg_pipeline.source_photo._masks import _maximum_channel
import cv2, numpy as np
from ecg_pipeline.native_grid_digitizer import trace_crossing_path, trace_path, refine_source_rhythm_path
from ecg_pipeline.source_waveform_support import source_supported_validity, connected_source_validity
shaarray = lambda a: hashlib.sha256(a.tobytes()).hexdigest()

def pulse_limit(image, bounds):
    l, t, r, b = map(int, bounds)
    maximum = _maximum_channel(image[t:b, l:r])
    core = maximum.min(axis=1).astype(float)
    median = float(np.median(core))
    mad = float(np.median(np.abs(core - median)))
    limit = min(159.0, median + 3.0 * 1.4826 * mad)
    return {'bounds': [l, t, r, b], 'coreMaximumRgbBySourceRow': core.astype(int).tolist(), 'median': median, 'medianAbsoluteDeviation': mad, 'robustUpperLimit': limit, 'method': 'median-plus-three-scaled-MAD-of-darkest-pulse-core-per-scanline', 'scalingFactor': 1.4826, 'maximumAllowedLimit': 159.0}

def prepare(image, neutral, excluded, pulse, limit):
    mask = (neutral.astype(bool) & (_maximum_channel(image) <= limit) & ~excluded.astype(bool)).astype(np.uint8)
    evidence = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0.45)
    evidence[pulse] = 0
    support = cv2.dilate(mask, np.ones((3, 3), np.uint8)).astype(bool)
    support[excluded.astype(bool)] = False
    return (mask, evidence, support)

def trace(evidence, support, publication, old, center, spacing, synthetic=False):
    lo, hi = old['range']
    top, bottom = old['yBounds']
    row = old['row']
    if row < 3 or synthetic:
        cols, path = trace_crossing_path(evidence, y_start=top, y_end=bottom, x_start=lo, x_end=hi, row_center=center, row_spacing=spacing)
    else:
        cols, path = trace_path(evidence, y_start=top, y_end=bottom, x_start=lo, x_end=hi, row_center=center)
    point = source_supported_validity(support, cols, path, None)
    connected, receipt = connected_source_validity(evidence, cols, path, point, source_interval=[lo, hi])
    old_valid = np.ones(len(cols), bool) if synthetic else np.asarray(old['valid'], bool)
    valid = connected & ~publication[path, cols] & old_valid
    refined, retained, refinement = refine_source_rhythm_path(evidence, support, cols, path, valid, source_interval=[lo, hi], y_start=top, y_end=bottom, row_center=center, row_spacing=spacing)
    retained &= ~publication[refined, cols] & old_valid
    assert not np.any(retained & ~old_valid)
    return {**{k: old[k] for k in ['lead', 'row', 'column', 'range', 'yBounds'] if k in old}, 'initialPathY': path.tolist(), 'initialValid': valid.astype(int).tolist(), 'initialConnectionEvidence': receipt, 'pathY': refined.tolist(), 'valid': retained.astype(int).tolist(), 'refinementReceipt': refinement, 'coverage': float(retained.mean()), 'retainedColumns': int(retained.sum()), 'newlyLostColumns': int(np.count_nonzero(old_valid & ~retained)), 'changedColumns': int(np.count_nonzero(refined != np.asarray(old['pathY']))), 'oldGapsRecovered': None if synthetic else 0, 'syntheticEvaluatedWithoutInheritedGaps': synthetic, 'pathSha256': shaarray(refined.astype('<i4')), 'validitySha256': shaarray(retained.astype('u1'))}
