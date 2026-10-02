"""Diagnostic physical timing from local paper grids and observed source marks."""
import copy, hashlib
import numpy as np
import cv2
import math

def runs(profile, threshold):
    idx = np.flatnonzero(profile >= threshold)
    return [a for a in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if a.size]

def propose(image, case):
    original = case['existingTiming']
    cal = case['calibration']
    evidence = {'version': 1, 'method': 'diagnostic-row-panel-local-grid-timing-v1', 'state': 'unresolved', 'productionFormatSupported': False, 'failureReasons': []}

    def finish(passed):
        evidence['state'] = 'diagnostic_source_supported' if passed else 'unresolved'
        return {'branch': 'diagnostic-local-timing' if passed else 'fallback-unresolved', 'candidatePassed': passed, 'candidate': evidence, 'finalResult': evidence if passed else copy.deepcopy(original), 'originalTimingExact': not passed, 'productionTimingPromoted': False}
    if original.get('state') == 'source_supported':
        return {'branch': 'existing-source-timing-retained', 'candidatePassed': False, 'finalResult': copy.deepcopy(original), 'originalTimingExact': True, 'productionTimingPromoted': False}
    if not (cal.get('detected') is True and 0.35 <= cal.get('confidence', 0) <= 1 and ((cal.get('reconciliation') or {}).get('state') == 'corroborated_inference') and ((cal.get('reconciliation') or {}).get('quantitativeBlocked') is False) and (cal.get('gridScaleMmX') == cal.get('gridScaleMmY')) and (cal.get('gridScaleMmX') in [1.0, 5.0]) and (case['diagnosticContext'].get('sourceIdentityVerifiedForDiagnostic') is True)):
        evidence['failureReasons'].append('physical-calibration-or-source-context-unconfirmed')
        return finish(False)
    scale = cal['gridScaleMmX']
    speed = cal['paperSpeedMmPerSecond']
    gain = cal['gainMmPerMv']
    height, width = image.shape[:2]
    context = case['diagnosticContext']
    attempts = case['separatorAttempts']
    if any((n < 0 or n > 1 for a in attempts for n in a['candidateCounts'])):
        evidence['failureReasons'].append('competing-source-separator-window')
        return finish(False)
    complete = [a for a in attempts if all((n == 1 for n in a['candidateCounts']))]
    offsets = [a['offsetPixels'] for a in complete]
    if not any((b - a == 1 for a, b in zip(offsets, offsets[1:]))):
        evidence['failureReasons'].append('insufficient-adjacent-complete-offsets')
        return finish(False)
    markers = []
    for i in range(9):
        found = [next((q for q in a['windows'][i]['supportRuns'] if q['widthAdmissible'])) for a in complete]
        centers = [q['centerX'] for q in found]
        boxes = [q['bounds'] for q in found]
        interval = [min(centers), max(centers)]
        intersection = [max((b[0] for b in boxes)), min((b[2] for b in boxes))]
        if interval[1] - interval[0] > 1 or intersection[1] <= intersection[0]:
            evidence['failureReasons'].append('separator-location-uncertain')
        markers.append({'row': i // 3, 'column': i % 3 + 1, 'centerX': float(np.median(centers)), 'centerIntervalPixels': interval, 'horizontalIntersection': intersection, 'unionBounds': [min((b[0] for b in boxes)), min((b[1] for b in boxes)), max((b[2] for b in boxes)), max((b[3] for b in boxes))]})
    evidence.update(completeOffsets=offsets, sourceSeparators=markers, localGrids=case['localGrids'])
    if evidence['failureReasons']:
        return finish(False)
    local = case['localGrids']
    assert len(local) == 16
    if any((q['selectedPair'] is None or q['selectedPair']['confidence'] < 0.18 for q in local)):
        evidence['failureReasons'].append('local-grid-unconfirmed')
        return finish(False)
    xppm = np.asarray([q['selectedPair']['xPeriod'] / scale for q in local]).reshape(4, 4)
    yppm = np.asarray([q['selectedPair']['yPeriod'] / scale for q in local]).reshape(4, 4)
    if not (np.isfinite(xppm).all() and np.isfinite(yppm).all() and (np.min(xppm) >= 1) and (np.max(xppm) <= 40) and (np.min(yppm) >= 1) and (np.max(yppm) <= 40)):
        evidence['failureReasons'].append('invalid-local-physical-scale')
        return finish(False)
    widths = xppm * speed * 2.5
    observed = np.asarray([q['centerX'] for q in markers]).reshape(3, 3)
    initial = []
    gap_checks = []
    for row in range(3):
        initial.append(float(observed[row, 0] - widths[row, 0]))
        for col in [1, 2]:
            measured = float(observed[row, col] - observed[row, col - 1])
            expected = float(widths[row, col])
            err = abs(measured - expected) / expected
            gap_checks.append({'row': row, 'column': col, 'observedPixels': measured, 'expectedPixels': expected, 'relativeDifference': err, 'passed': err <= 0.03})
    initial.append(float(context['firstColumnRows'][3]['labelX']))
    evidence.update(expectedPanelWidthsPixels=widths.tolist(), interiorPanelGridChecks=gap_checks, pulseOriginProposals=initial)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[0]
    cutoff = min(160., max(40., otsu * .75))
    mask = (gray <= math.floor(cutoff)) & (np.ptp(image.astype(np.int16), axis=2) <= 28)
    pulse_edges = []
    for row, origin in enumerate(initial):
        center = context['firstColumnRows'][row]['waveformCenterY']
        ppm = float(xppm[row, 0])
        pulse_height = float(yppm[row, 0] * gain)
        left, right = (max(0, round(origin - ppm)), min(width, round(origin + ppm) + 1))
        top, bottom = (max(0, round(center - pulse_height * 0.9)), min(height, round(center - pulse_height * 0.1)))
        profile = mask[top:bottom, left:right].mean(axis=0)
        parts = runs(profile, 0.65)
        limit = max(3, round(ppm * 0.7))
        found = []
        for part in parts:
            found.append({'centerX': float(left + (part[0] + part[-1]) / 2), 'bounds': [int(left + part[0]), top, int(left + part[-1] + 1), bottom], 'widthPixels': int(part.size), 'minimumBlackColumnSupport': float(np.min(profile[part])), 'widthAdmissible': bool(1 <= part.size <= limit)})
        valid = [q for q in found if q['widthAdmissible']]
        pulse_edges.append({'row': row, 'proposalX': origin, 'localPixelsPerMmX': ppm, 'pulseHeightPixels': pulse_height, 'windowBounds': [left, top, right, bottom], 'profileSha256': hashlib.sha256(profile.tobytes()).hexdigest(), 'widthLimit': limit, 'supportRuns': found, 'candidateCount': len(valid), 'selected': valid[0] if len(valid) == 1 else None})
    evidence['sourcePulseEdges'] = pulse_edges
    if not all((q['passed'] for q in gap_checks)):
        evidence['failureReasons'].append('interior-panel-grid-disagreement')
    if any((q['candidateCount'] != 1 for q in pulse_edges)):
        evidence['failureReasons'].append('source-pulse-edge-missing-or-ambiguous')
        return finish(False)
    origins = [float(q['selected']['bounds'][2]) for q in pulse_edges]
    first_checks = []
    boundaries = []
    for row in range(3):
        first_width = float(observed[row, 0] - origins[row])
        expected = float(widths[row, 0])
        difference = abs(first_width - expected) / expected
        origin_difference = abs(origins[row] - initial[row])
        first_checks.append({'row': row, 'observedFirstPanelPixels': first_width, 'expectedFirstPanelPixels': expected, 'relativeDifference': difference, 'originDifferencePixels': origin_difference, 'originTolerancePixels': float(xppm[row, 0]), 'passed': difference <= 0.03 and origin_difference <= xppm[row, 0]})
        boundaries.append([origins[row], *observed[row].tolist(), float(observed[row, 2] + widths[row, 3])])
    rhythm = [origins[3]]
    for panel_width in widths[3]:
        rhythm.append(float(rhythm[-1] + panel_width))
    boundaries.append(rhythm)
    start_fit = np.polyfit([0, 1, 2], origins[:3], 1)
    end_fit = np.polyfit([0, 1, 2], [q[-1] for q in boundaries[:3]], 1)
    expected_start = float(np.polyval(start_fit, 3))
    expected_end = float(np.polyval(end_fit, 3))
    rhythm_check = {'predictedOriginFromPrimaryRows': expected_start, 'observedOrigin': origins[3], 'originDifferencePixels': abs(origins[3] - expected_start), 'originTolerancePixels': float(xppm[3, 0]), 'predictedEndFromPrimaryRows': expected_end, 'localGridEnd': rhythm[-1], 'endDifferencePixels': abs(rhythm[-1] - expected_end), 'endTolerancePixels': float(xppm[3, 3]), 'passed': abs(origins[3] - expected_start) <= xppm[3, 0] and abs(rhythm[-1] - expected_end) <= xppm[3, 3]}
    reference_row = case['pulseCalibrationSourceRow']
    reference = {'row': reference_row, 'calibrationPulseEndX': cal['pulseEndX'], 'observedPulseEndX': origins[reference_row], 'differencePixels': abs(origins[reference_row] - cal['pulseEndX']), 'tolerancePixels': float(xppm[reference_row, 0])}
    reference['passed'] = reference['differencePixels'] <= reference['tolerancePixels']
    evidence.update(firstPanelGridChecks=first_checks, rowBoundariesX=boundaries, rhythmPrimaryRowTrendCheck=rhythm_check, calibrationPulseAgreement=reference, originRule='right-exclusive edge of observed descending source pulse stroke', finalPanelRule='observed third separator plus calibrated local grid width; no terminal separator', rhythmRule='observed rhythm pulse plus four calibrated local grid spans; divisions are inferred, not observed separators')
    if not all((q['passed'] for q in first_checks)):
        evidence['failureReasons'].append('first-panel-grid-or-pulse-origin-disagreement')
    if not rhythm_check['passed']:
        evidence['failureReasons'].append('rhythm-origin-or-end-disagrees-with-primary-trend')
    if not reference['passed']:
        evidence['failureReasons'].append('pulse-origin-disagrees-with-verified-calibration')
    if any((q[0] < 0 or q[-1] > width or any((b <= a for a, b in zip(q[:-1], q[1:]))) for q in boundaries)):
        evidence['failureReasons'].append('invalid-source-panel-ranges')
    if evidence['failureReasons']:
        return finish(False)
    rounded = np.rint(boundaries).astype(int)
    evidence.update(rowPanelRanges=[[[int(a), int(b)] for a, b in zip(row[:-1], row[1:])] for row in rounded], rhythmRange=[int(rounded[3, 0]), int(rounded[3, -1])], imageSize=[width, height], decodedRasterSha256=hashlib.sha256(image.tobytes()).hexdigest(), paperSpeedMmPerSecond=speed, gainMmPerMv=gain, panelDurationSeconds=2.5, gridScaleMm=scale, truthUsed=False)
    return finish(True)
