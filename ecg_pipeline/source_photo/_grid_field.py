"""Partial paper coordinates from directly observed grid-cell corners only."""
import numpy as np

def build_field(nodes, observations, major_indices, gap_passed, *, node_support):
    x = np.asarray(nodes, float)
    y = np.asarray(observations, float)
    indices = np.asarray(major_indices, int)
    gaps = np.asarray(gap_passed, bool)
    assert x.ndim == indices.ndim == gaps.ndim == 1
    assert y.shape == (len(indices), len(x)) and len(gaps) == len(indices) - 1
    assert np.all(np.diff(x) > 0) and np.all(np.diff(indices) > 0)
    assert np.isfinite(x).all()
    assert len(node_support) == len(x) and np.isfinite(node_support).all()
    assert np.all((np.asarray(node_support) >= 0) & (np.asarray(node_support) <= 1))
    if len(indices) < 8 or min(node_support) < 0.8:
        return {'state': 'unresolved', 'reason': 'Insufficient original seed/node support.', 'cells': [], 'nodeChecks': [], 'segmentChecks': [], 'candidateCells': []}
    for column in y.T:
        visible = column[np.isfinite(column)]
        if np.any(np.diff(visible) <= 0):
            return {'state': 'unresolved', 'reason': 'Observed grid rows cross.', 'cells': [], 'nodeChecks': [], 'segmentChecks': [], 'candidateCells': []}
    residuals = np.full_like(y, np.nan)
    node_checks = []
    for i in range(len(indices)):
        for j in range(1, len(x) - 1):
            if not np.isfinite(y[i, j - 1:j + 2]).all():
                continue
            if max(x[j] - x[j - 1], x[j + 1] - x[j]) > 64:
                continue
            fraction = (x[j] - x[j - 1]) / (x[j + 1] - x[j - 1])
            predicted = y[i, j - 1] * (1 - fraction) + y[i, j + 1] * fraction
            residual = abs(y[i, j] - predicted)
            residuals[i, j] = residual
            node_checks.append({'row': i, 'node': j, 'prediction': float(predicted), 'residualPixels': float(residual), 'passed': bool(residual <= 1.0)})
    segments = np.zeros((len(indices), len(x) - 1), bool)
    segment_checks = []
    for i in range(len(indices)):
        for j in range(len(x) - 1):
            if not np.isfinite(y[i, j:j + 2]).all():
                continue
            checks = residuals[i, j:j + 2]
            available = checks[np.isfinite(checks)]
            slope = abs((y[i, j + 1] - y[i, j]) / (x[j + 1] - x[j]))
            ok = bool(x[j + 1] - x[j] <= 64 and len(available) >= 1 and np.all(available <= 1.0) and (slope <= 0.25))
            segments[i, j] = ok
            segment_checks.append({'row': i, 'interval': j, 'slopeMagnitude': float(slope), 'omissionChecks': available.tolist(), 'passed': ok})
    cells = []
    candidates = []
    for j in range(len(x) - 1):
        common = np.flatnonzero(segments[:, j])
        for upper, lower in zip(common[:-1], common[1:]):
            steps = int(indices[lower] - indices[upper])
            heights = y[lower, j:j + 2] - y[upper, j:j + 2]
            gap_evidence = bool(gaps[upper:lower].all())
            monotone = bool(np.all(heights > 0))
            ppm = heights / (5 * steps)
            supported = bool(1 <= steps <= 4 and gap_evidence and monotone and np.all((ppm >= 2) & (ppm <= 40)))
            candidate = {'interval': j, 'upperRow': int(upper), 'lowerRow': int(lower), 'majorSteps': steps, 'seedGapsSupported': gap_evidence, 'positiveHeights': monotone, 'pixelsPerMmAtEdges': ppm.tolist(), 'passed': supported}
            candidates.append(candidate)
            if supported:
                cells.append({**candidate, 'x': x[j:j + 2].tolist(), 'upperY': y[upper, j:j + 2].tolist(), 'lowerY': y[lower, j:j + 2].tolist(), 'paperYmm': [float(5 * indices[upper]), float(5 * indices[lower])]})
    return {'state': 'partial' if cells else 'unresolved', 'reason': None if cells else 'No independently supported grid cells.', 'cells': cells, 'nodeChecks': node_checks, 'segmentChecks': segment_checks, 'candidateCells': candidates, 'maximumOmissionResidualPixels': 1.0, 'maximumSlopeMagnitude': 0.25, 'maximumNodeSeparationPixels': 64, 'maximumBracketMajorSteps': 4, 'unsupportedSeedGapIndices': np.flatnonzero(~gaps).tolist(), 'imputedGridCorners': 0, 'waveformOrTruthUsed': False, 'fullPageCoverageClaimed': False}

def forward(cell, source_x, source_y):
    x0, x1 = cell['x']
    if not x0 <= source_x <= x1:
        return None
    t = (source_x - x0) / (x1 - x0)
    top = cell['upperY'][0] * (1 - t) + cell['upperY'][1] * t
    bottom = cell['lowerY'][0] * (1 - t) + cell['lowerY'][1] * t
    if not top <= source_y <= bottom or not bottom > top:
        return None
    fraction = (source_y - top) / (bottom - top)
    return cell['paperYmm'][0] + fraction * (cell['paperYmm'][1] - cell['paperYmm'][0])

def inverse(cell, source_x, paper_y_mm):
    x0, x1 = cell['x']
    upper, lower = cell['paperYmm']
    if not x0 <= source_x <= x1 or not upper <= paper_y_mm <= lower:
        return None
    t = (source_x - x0) / (x1 - x0)
    top = cell['upperY'][0] * (1 - t) + cell['upperY'][1] * t
    bottom = cell['lowerY'][0] * (1 - t) + cell['lowerY'][1] * t
    return top + (paper_y_mm - upper) / (lower - upper) * (bottom - top)
