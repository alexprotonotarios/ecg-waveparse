"""Bounded source-window period predictions; no waveform or truth input."""
import numpy as np

def period_at(local_grids, source_x, source_y):
    rows = []
    x_edges = []
    for row in range(4):
        cells = sorted([q for q in local_grids if q['row'] == row], key=lambda q: q['column'])
        assert len(cells) == 4
        centers = [(q['bounds'][0] + q['bounds'][2]) / 2 for q in cells]
        ys = [(q['bounds'][1] + q['bounds'][3]) / 2 for q in cells]
        periods = [q['selectedPair']['yPeriod'] for q in cells]
        assert min((q['bounds'][0] for q in cells)) <= source_x < max((q['bounds'][2] for q in cells))
        rows.append((float(np.median(ys)), float(np.interp(source_x, centers, periods))))
        x_edges.append(bool(source_x < centers[0] or source_x > centers[-1]))
    rows.sort()
    top = min((q['bounds'][1] for q in local_grids))
    bottom = max((q['bounds'][3] for q in local_grids))
    assert top <= source_y < bottom
    value = float(np.interp(source_y, [r[0] for r in rows], [r[1] for r in rows]))
    if source_y < rows[0][0] or source_y > rows[-1][0]:
        a, b = rows[:2] if source_y < rows[0][0] else rows[-2:]
        value = float(a[1] + (b[1] - a[1]) * (source_y - a[0]) / (b[0] - a[0]))
    assert np.isfinite(value) and value > 0
    return (value, {'sourceX': float(source_x), 'sourceY': float(source_y), 'rowCenterAndInterpolatedPeriod': [[y, p] for y, p in rows], 'xEdgeContinuationByRow': x_edges, 'yEdgeContinuation': bool(source_y < rows[0][0] or source_y > rows[-1][0]), 'sourceWindowYDomain': [top, bottom]})

def calibrate(peaks, local_grids, source_x):
    peaks = np.asarray(peaks, int)
    top = min((q['bounds'][1] for q in local_grids))
    bottom = max((q['bounds'][3] for q in local_grids))
    kept = peaks[(peaks >= top) & (peaks < bottom)]
    checks = []
    for left, right in zip(kept[:-1], kept[1:]):
        predicted, receipt = period_at(local_grids, source_x, (left + right) / 2)
        gap = int(right - left)
        multiple = int(np.rint(gap / predicted))
        error = abs(gap - multiple * predicted)
        checks.append({'sourceRows': [int(left), int(right)], 'observedGapPixels': gap, 'predictedMajorPeriodPixels': predicted, 'roundedMajorMultiple': multiple, 'expectedGapPixels': multiple * predicted, 'absoluteErrorPixels': error, 'passed': bool(1 <= multiple <= 4 and error <= 2.0), 'prediction': receipt})
    fraction = float(np.mean([q['passed'] for q in checks])) if checks else 0.0
    accepted = bool(len(kept) >= 8 and fraction >= 0.9)
    return {'method': 'local-source-window-major-grid-calibration-edge-slope-v1', 'state': 'supported' if accepted else 'unresolved', 'sourceWindowYDomain': [top, bottom], 'sourceX': source_x, 'retainedSeeds': kept.tolist(), 'outsideDomainSeeds': peaks[(peaks < top) | (peaks >= bottom)].tolist(), 'gapChecks': checks, 'agreementFraction': fraction, 'minimumSeedCount': 8, 'minimumAgreementFraction': 0.9, 'absoluteAgreementBudgetPixels': 2.0, 'waveformOrTruthUsed': False, 'edgeContinuationLimitedToObservedWindowDomain': True}
