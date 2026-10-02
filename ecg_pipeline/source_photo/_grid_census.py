import math

def assess_pair(source_x, source_rows, major_steps, local_grids, period_fn):
    xlo = max((min((c['bounds'][0] for c in local_grids if c['row'] == row)) for row in range(4)))
    xhi = min((max((c['bounds'][2] for c in local_grids if c['row'] == row)) for row in range(4)))
    ylo = min((c['bounds'][1] for c in local_grids))
    yhi = max((c['bounds'][3] for c in local_grids))
    missing = any((v is None or not math.isfinite(v) for v in source_rows))
    result = {'sourceX': source_x, 'observedRows': source_rows, 'expectedMajorSteps': major_steps, 'modelXDomain': [xlo, xhi], 'modelYDomain': [ylo, yhi], 'xInsideModelDomain': xlo <= source_x < xhi, 'missingPairedObservation': missing}
    if missing:
        return result | {'status': 'missing-paired-observation'}
    y = sum(source_rows) / 2
    result['sourceMidY'] = y
    if not (xlo <= source_x < xhi and ylo <= y < yhi):
        return result | {'status': 'outside-model-domain'}
    predicted, receipt = period_fn(local_grids, source_x, y)
    gap = source_rows[1] - source_rows[0]
    rounded = round(gap / predicted)
    error = abs(gap - major_steps * predicted)
    return result | {'status': 'measured', 'gapPixels': gap, 'predictedMajorPeriodPixels': predicted, 'roundedMajorSteps': rounded, 'absoluteErrorPixels': error, 'passed': bool(1 <= major_steps <= 4 and rounded == major_steps and (error <= 2.0)), 'prediction': receipt}
