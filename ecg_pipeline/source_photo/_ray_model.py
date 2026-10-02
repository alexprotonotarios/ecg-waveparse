"""Homogeneous source-ray model and diagnostic projection, without resampling."""
import numpy as np

def fit_model(tracks, image_size):
    width, height = image_size
    center = np.array([width / 2, height / 2])
    scale = float(max(image_size))
    rows = []
    for t in tracks:
        slope = t['dxPerDy']
        x = t['centerX'] - center[0]
        y = t['centerY'] - center[1]
        line = np.array([1.0, -slope, (-x + slope * y) / scale])
        line /= np.linalg.norm(line[:2])
        rows.append(line)
    assert len(rows) >= 3
    matrix = np.asarray(rows)
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    assert singular[1] > 1e-12 * singular[0], 'unidentifiable_ray_family'
    v = vt[-1]
    if v[1] < 0:
        v = -v
    physical = [scale * v[0] + center[0] * v[2], scale * v[1] + center[1] * v[2], v[2]]
    return {'normalizationCenter': center.tolist(), 'normalizationScale': scale, 'normalizedLines': matrix.tolist(), 'singularValues': singular.tolist(), 'normalizedVanishingPoint': v.tolist(), 'homogeneousVanishingPoint': physical, 'trackCount': len(tracks), 'fitMethod': 'unit-normalized_source_line_SVD_smallest_right_singular_vector', 'clinicalValidation': False}

def project_x(model, x, y, reference_y):
    vx, vy, vw = model['homogeneousVanishingPoint']
    den = vy - vw * y
    assert abs(den) > 1e-12, 'projection_pole'
    return float(x + (reference_y - y) * (vx - vw * x) / den)

def track_residual(model, track, reference_y):
    points = track['points']
    projected = [project_x(model, x, y, reference_y) for x, y in points]
    median = float(np.median(projected))
    raw = [q[0] for q in points]
    return {'trackId': track['trackId'], 'windowId': track['windowId'], 'referenceY': reference_y, 'projectedX': projected, 'referenceXMedian': median, 'maximumDeviationFromMedianPixels': max((abs(x - median) for x in projected)), 'projectedRangePixels': max(projected) - min(projected), 'originalXRangePixels': max(raw) - min(raw), 'withinOnePixelConstancy': max((abs(x - median) for x in projected)) <= 1.0}
