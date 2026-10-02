"""Bounded source-only joint geometry. Markers must be unique and pulse/grid corroborated."""
import hashlib
import itertools
import math

import cv2
import numpy as np
from scripts.detect_ecg_layout import _grid_period
from ecg_pipeline.source_panel_timing import _fit_separator_centroids
from ecg_pipeline.source_label_geometry import valid_tiled_label_grid
METHOD = 'joint-source-rectangle-pulse-geometry-proposal-v1'
POLICY = {
    'darkExclusive': 160,
    'minimumStrokeSupport': 0.8,
    'heightSpreadFraction': 0.05,
    'minimumHeightSpreadPixels': 2,
    'maximumSlope': 0.05,
    'maximumCandidatesPerWindow': 4,
    'maximumJointCombinations': 4096,
    'maximumRegionComponents': 64,
    'pulseWidthRelativeTolerance': 0.15,
    'pulseHeightRelativeTolerance': 0.15,
    'minimumPlateauSupport': 0.45,
}

def reject(reason, **evidence):
    return {'version': 1, 'method': METHOD, 'state': 'unresolved', 'failureReason': reason, 'nativeExtractionReady': False, 'truthUsed': False, **evidence}

def finite(v):
    return isinstance(v, (int, float)) and (not isinstance(v, bool)) and np.isfinite(v)

def admission(image, g):
    grid = g.get('sourceLabelGrid') or {}
    cal = g.get('calibration') or {}
    if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
            or g.get('layoutHint') != 'standard_3x4_with_r1'
            or grid.get('method') != 'source-value-grid-below-trace-v1'
            or grid.get('imageSize') != [image.shape[1], image.shape[0]]
            or (g.get('leadLabelValidation') or {}).get('semanticIdentityConfirmed') is not True
            or (g.get('rhythmLeadValidation') or {}).get('semanticIdentityConfirmed') is not True):
        return 'source-identity-unconfirmed'
    scale = cal.get('gridScaleMmX', cal.get('gridScaleMm'))
    physical = [cal.get(k) for k in ['pixelsPerMmX', 'pixelsPerMmY', 'paperSpeedMmPerSecond',
                                    'gainMmPerMv', 'pulseStartX', 'pulseEndX']]
    if (cal.get('detected') is not True or cal.get('gridScaleAmbiguous') is True
            or not finite(cal.get('confidence')) or not 0.35 <= cal['confidence'] <= 1
            or isinstance(scale, bool) or scale not in (1, 5)
            or not all(finite(v) and v > 0 for v in physical)
            or cal['pulseStartX'] >= cal['pulseEndX']
            or (cal.get('reconciliation') or {}).get('state') != 'corroborated_inference'
            or (cal.get('reconciliation') or {}).get('quantitativeBlocked') is not False):
        return 'physical-calibration-unconfirmed'
    rows = g.get('rowCenters')
    xf = grid.get('xFit')
    font = grid.get('fontHeight')
    if (not isinstance(rows, list) or len(rows) != 4 or not all(finite(v) for v in rows)
            or not all(a < b for a, b in zip(rows, rows[1:]))
            or not 0 <= rows[0] < rows[-1] < image.shape[0]
            or not isinstance(xf, list) or len(xf) != 3 or not all(finite(v) for v in xf)
            or not finite(font) or not 0 < font <= 256):
        return 'invalid-source-grid'
    return None

def runs(active):
    edges = np.flatnonzero(np.r_[False, active, False][1:] != np.r_[False, active, False][:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]

def region_components(image, bounds, minimum):
    l, t, r, b = bounds
    crop = image[t:b, l:r]
    dark = crop.max(axis=2) < POLICY['darkExclusive']
    kept = np.zeros(dark.shape, np.uint8)
    for x in range(dark.shape[1]):
        for a, z in runs(dark[:, x]):
            if z - a >= minimum:
                kept[a:z, x] = 255
    n, labels, stats, centres = cv2.connectedComponentsWithStatsWithAlgorithm(kept, 8, cv2.CV_32S, cv2.CCL_SAUF)
    result = []
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        box = [l + x, t + y, l + x + w, t + y + h]
        result.append({'id': i, 'bounds': box, 'width': w, 'height': h, 'pixels': area, 'midpoint': [(box[0] + box[2] - 1) / 2, (box[1] + box[3] - 1) / 2]})
    return result

def source_regions(image, g):
    h, w = image.shape[:2]
    grid = g['sourceLabelGrid']
    cal = g['calibration']
    ppm = cal['pixelsPerMmX']
    amp = cal['pixelsPerMmY'] * cal['gainMmPerMv']
    font = grid['fontHeight']
    regions = []
    for row, cy in enumerate(g['rowCenters']):
        box = [max(0, round(cal['pulseStartX'] - 2 * ppm)), max(0, round(cy - 1.5 * amp)), min(w, round(cal['pulseEndX'] + 2 * ppm) + 1), min(h, round(cy + 0.75 * amp) + 1)]
        regions.append({'kind': 'pulse', 'row': row, 'column': 0, 'bounds': box, 'minimumLength': max(3, round(amp * 0.45))})
    for row, cy in enumerate(g['rowCenters'][:3]):
        for col in (1, 2, 3):
            cx = float(grid['xFit'][0] + grid['xFit'][1] * row + grid['xFit'][2] * col)
            box = [max(0, round(cx - font)), max(0, round(cy - 2 * font)), min(w, round(cx + font) + 1), min(h, round(cy + 2 * font) + 1)]
            regions.append({'kind': 'separator', 'row': row, 'column': col, 'bounds': box, 'minimumLength': max(3, round(font * 0.8))})
    for r in regions:
        l, t, rr, b = r['bounds']
        if not (0 <= l < rr <= w and 0 <= t < b <= h):
            return None
        r['components'] = region_components(image, r['bounds'], r['minimumLength'])
    return regions

def pulse_pairs(image, region, ppm, cal):
    components = region['components']
    amp = cal['pixelsPerMmY'] * cal['gainMmPerMv']
    expected_width = ppm * cal['paperSpeedMmPerSecond'] * 0.2
    found = []
    for left, right in itertools.combinations(sorted(components, key=lambda c: c['midpoint'][0]), 2):
        if max(left['width'], right['width']) > max(5, round(ppm * 0.7)):
            continue
        a = left['bounds']
        b = right['bounds']
        width = right['midpoint'][0] - left['midpoint'][0]
        height = min(a[3], b[3]) - max(a[1], b[1]) - max(left['width'], right['width'])
        tol = max(3, 2 * max(left['width'], right['width']), 0.15 * height)
        if height <= 0 or abs(width - expected_width) / expected_width > POLICY['pulseWidthRelativeTolerance'] or abs(height - amp) / amp > POLICY['pulseHeightRelativeTolerance']:
            continue
        if abs(a[1] - b[1]) > tol or abs(a[3] - b[3]) > tol:
            continue
        top = max(a[1], b[1])
        lo = math.ceil(left['midpoint'][0])
        hi = math.floor(right['midpoint'][0]) + 1
        support = float((image[max(0, top - 1):top + 2, lo:hi].max(axis=2) < 160).mean())
        if support < POLICY['minimumPlateauSupport']:
            continue
        if any((left['midpoint'][0] < c['midpoint'][0] < right['midpoint'][0] and c['bounds'][1] <= top + tol and (c['bounds'][3] >= min(a[3], b[3]) - tol) for c in components if c not in (left, right))):
            continue
        found.append({'row': region['row'], 'rising': left, 'falling': right, 'endX': b[2], 'baselineY': min(a[3], b[3]) - 1, 'plateauSupport': support, 'widthPixels': width, 'heightPixels': height})
    return found

def separator_candidates(image, region, font):
    minimum = max(3, round(font * 0.08))
    maximum = max(4, round(font * 0.3))
    out = []
    for c in region['components']:
        if not minimum <= c['width'] <= maximum:
            continue
        l, t, r, b = c['bounds']
        profile = (image[t:b, l:r].max(axis=2) < 160).mean(axis=0)
        qualifying = [(a, z) for a, z in runs(profile >= POLICY['minimumStrokeSupport']) if minimum <= z - a <= maximum]
        if len(qualifying) != 1:
            continue
        a, z = qualifying[0]
        bounds = [l + a, t, l + z, b]
        center = (bounds[0] + bounds[2] - 1) / 2
        if abs(center - c['midpoint'][0]) > 1:
            continue
        out.append({'row': region['row'], 'column': region['column'], 'componentId': c['id'], 'observedBounds': c['bounds'], 'bounds': bounds, 'centerX': center, 'centerY': c['midpoint'][1], 'height': c['height'], 'minimumBlackColumnSupport': float(profile[a:z].min()), 'centerIntervalPixels': [min(center, c['midpoint'][0]), max(center, c['midpoint'][0])]})
    return out

def propose_after_identity(image, g):
    """Core geometry only. Corruption tests separately supply assumed label identity."""
    error = admission(image, g)
    if error:
        return reject(error)
    cal = g['calibration']
    scale = cal.get('gridScaleMmX', cal.get('gridScaleMm'))
    rows = g['rowCenters']
    font = g['sourceLabelGrid']['fontHeight']
    blue, green, red = np.moveaxis(image.astype(np.float32), 2, 0)
    grid_ink = np.clip(red - (green + blue) * 0.5, 0, 255) / 255
    grid = []
    radius = max(8, round(float(np.median(np.diff(rows))) * 0.07))
    for cy in rows:
        period, conf = _grid_period(grid_ink[max(0, round(cy) - radius):min(image.shape[0], round(cy) + radius + 1)].mean(axis=0))
        if period is None or not np.isfinite(period) or (not np.isfinite(conf)) or (conf < 0.18) or (not 1 <= period / scale <= 40):
            return reject('row-grid-scale-unconfirmed')
        grid.append({'rowCenter': cy, 'periodPixels': float(period), 'confidence': float(conf), 'periodMm': scale})
    scales = np.array([m['periodPixels'] / scale for m in grid])
    ppm = float(np.median(scales))
    tol = max(2, ppm * 0.25)
    if np.ptp(scales) / ppm > 0.02 or abs(ppm - cal['pixelsPerMmX']) / ppm > 0.15:
        return reject('inconsistent-row-grid-scale')
    regions = source_regions(image, g)
    if regions is None:
        return reject('source-region-outside-image')
    if any((len(r['components']) > POLICY['maximumRegionComponents'] for r in regions)):
        return reject('source-region-component-limit', regions=regions)
    pulse_choices = [pulse_pairs(image, r, float(scales[i]), cal) for i, r in enumerate(regions[:4])]
    evidence = {'regions': regions, 'rowGridMeasurements': grid, 'pulseChoices': pulse_choices}
    if any((len(options) != 1 for options in pulse_choices)):
        return reject('pulse-pair-missing-or-ambiguous', **evidence)
    pulses = [a[0] for a in pulse_choices]
    if min((abs(p['endX'] - cal['pulseEndX']) for p in pulses)) > tol:
        return reject('global-pulse-inconsistent-with-observed-rows', **evidence)
    candidates = [separator_candidates(image, r, font) for r in regions[4:]]
    return resolve_observed_candidates(image, g, evidence, candidates)


def resolve_observed_candidates(image, g, observations, candidates):
    """Apply the original bounded joint gates to already observed source candidates.

    This is an internal arithmetic step, not source admission. Its callers must
    reconstruct the pixel observations before using a result for native timing.
    """
    cal = g['calibration']
    rows = g['rowCenters']
    grid = observations['rowGridMeasurements']
    scales = np.array([m['periodPixels'] / m['periodMm'] for m in grid])
    ppm = float(np.median(scales))
    tol = max(2, ppm * 0.25)
    pulses = [options[0] for options in observations['pulseChoices']]
    evidence = {key: observations[key] for key in
                ('regions', 'rowGridMeasurements', 'pulseChoices')}
    evidence['separatorChoices'] = candidates
    if any((not options or len(options) > POLICY['maximumCandidatesPerWindow'] for options in candidates)):
        return reject('separator-components-missing-or-excessive', **evidence)
    count = math.prod(map(len, candidates))
    evidence['candidateCombinations'] = count
    if count > POLICY['maximumJointCombinations']:
        return reject('bounded-search-limit', **evidence)
    row_choices = []
    for row in range(3):
        options = []
        for triple in itertools.product(*candidates[row * 3:row * 3 + 3]):
            heights = [m['height'] for m in triple]
            height_tol = max(2, round(float(np.median(heights)) * 0.05))
            if max(heights) - min(heights) > height_tol:
                continue
            xs = [m['centerX'] for m in triple]
            ys = np.array([m['centerY'] for m in triple])
            bounds, width, residual = _fit_separator_centroids(xs)
            if residual > max(1, ppm * 0.25) or abs(width - scales[row] * cal['paperSpeedMmPerSecond'] * 2.5) / (scales[row] * cal['paperSpeedMmPerSecond'] * 2.5) > 0.03:
                continue
            if abs(bounds[0] - pulses[row]['endX']) > tol:
                continue
            slope = float((ys[2] - ys[0]) / (xs[2] - xs[0]))
            intercept = float(ys.mean() - slope * np.mean(xs))
            vertical_residual = float(np.max(abs(ys - (intercept + slope * np.asarray(xs)))))
            if abs(slope) > 0.05 or vertical_residual > max(1, ppm * 0.25):
                continue
            if abs(intercept + slope * pulses[row]['falling']['midpoint'][0] - pulses[row]['baselineY']) > cal['pixelsPerMmY']:
                continue
            options.append({'separators': list(triple), 'rowBoundaries': bounds, 'width': width, 'horizontalResidual': residual, 'slope': slope, 'intercept': intercept, 'verticalResidual': vertical_residual})
        row_choices.append(options)
    evidence['rowProposalCounts'] = list(map(len, row_choices))
    solutions = []
    for group in itertools.product(*row_choices):
        heights = [m['height'] for row in group for m in row['separators']]
        if max(heights) - min(heights) > max(2, round(float(np.median(heights)) * 0.05)):
            continue
        boundaries = np.array([r['rowBoundaries'] for r in group])
        if np.max(np.ptp(boundaries, axis=0)) > ppm:
            continue
        if np.ptp([r['slope'] for r in group]) * float(np.median([r['width'] for r in group])) > 2:
            continue
        cy = np.asarray(rows[:3], float)
        origins = boundaries[:, 0]
        sx = float((origins[-1] - origins[0]) / (cy[-1] - cy[0]))
        ix = float(origins.mean() - sx * cy.mean())
        rhythm_origin = ix + sx * rows[3]
        if float(np.max(abs(origins - (ix + sx * cy)))) > max(1, ppm * 0.25) or abs(rhythm_origin - pulses[3]['endX']) > tol:
            continue
        solutions.append({'rows': list(group), 'rhythmOriginPrediction': rhythm_origin, 'rowOriginShear': sx, 'originShearResidual': float(np.max(abs(origins - (ix + sx * cy))))})
    evidence['solutionCount'] = len(solutions)
    if len(solutions) != 1:
        return reject('joint-geometry-missing-or-ambiguous', solutions=solutions, **evidence)
    return {
        'version': 1, 'method': METHOD, 'state': 'joint_geometry_supported',
        'nativeExtractionReady': False, 'truthUsed': False,
        'decodedRasterSha256': hashlib.sha256(image.tobytes()).hexdigest(),
        'imageSize': [image.shape[1], image.shape[0]],
        'selected': solutions[0], **evidence,
    }

def propose_joint_geometry(image, geometry):
    """Fresh source observations after complete semantic/calibration admission."""
    if not isinstance(geometry, dict):
        return reject('source-identity-unconfirmed')
    try:
        error = admission(image, geometry)
        if error:
            return reject(error)
        if not valid_tiled_label_grid(image, geometry['sourceLabelGrid']):
            return reject('source-tiled-label-evidence-invalid')
        return propose_after_identity(image, geometry)
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return reject('invalid-joint-source-geometry')
