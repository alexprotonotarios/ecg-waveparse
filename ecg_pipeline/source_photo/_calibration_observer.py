"""Fresh label -> physical calibration -> source-local timing, image-only.

Source file hash is provenance metadata only. No file paths, case ids, cached
measurements or outcome-based switches are accepted by this observer.
"""
import copy, hashlib
import numpy as np
from . import _pulse_local_grid, _neutral_timing_replay

def pack(v):
    if isinstance(v, np.ndarray):
        return {'shape': list(v.shape), 'dtype': str(v.dtype), 'rasterSha256': hashlib.sha256(v.tobytes()).hexdigest()}
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {str(k): pack(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [pack(x) for x in v]
    return v

def local_grids(image, geometry, layout, period):
    grid = geometry['sourceLabelGrid']
    rows = geometry['rowCenters']
    w, h = grid['imageSize']
    spacing = grid['rowSpacing']
    fit = np.asarray(grid['xFit'])
    records = []
    gray = 1.0 - layout._as_gray(image).astype(np.float32) / 255.0
    planes = dict(zip(['gray', 'red-excess', 'green-excess', 'blue-excess'], [gray, *layout._chromatic_excess_planes(image)]))
    for row, center in enumerate(rows):
        sr = min(row, 2)
        columns = fit[2] + (sr * fit[3] if len(fit) == 4 else 0)
        for col in range(4):
            basis = [1, sr, col, sr * col] if len(fit) == 4 else [1, sr, col]
            start = float(np.asarray(basis) @ fit)
            bounds = [max(0, round(start + 0.08 * columns)), max(0, round(center - 0.75 * spacing)), min(w, round(start + 0.92 * columns)), min(h, round(center + 0.65 * spacing))]
            left, top, right, bottom = bounds
            assert right - left >= 80 and bottom - top >= 80
            candidates = []
            profiles = []
            for name, plane in planes.items():
                tile = plane[top:bottom, left:right]
                xp = tile.mean(axis=0)
                yp = tile.mean(axis=1)
                x, xc = period.grid_period(xp, 96)
                y, yc = period.grid_period(yp, 96)
                profiles.append({'plane': name, 'xProfileSha256': hashlib.sha256(xp.tobytes()).hexdigest(), 'yProfileSha256': hashlib.sha256(yp.tobytes()).hexdigest(), 'xPeriod': x, 'yPeriod': y, 'xConfidence': xc, 'yConfidence': yc})
                if x is None or y is None:
                    continue
                difference = abs(x - y) / max(x, y, 1e-09)
                if difference <= 0.15:
                    candidates.append({'plane': name, 'xPeriod': x, 'yPeriod': y, 'relativeAxisDifference': difference, 'confidence': min(xc, yc) * (1 - difference)})
            records.append({'row': row, 'column': col, 'bounds': bounds, 'selectedPair': max(candidates, key=lambda q: q['confidence']) if candidates else None, 'profiles': profiles})
    return records

def source_context(labels, image):
    baseline = labels['baseline']
    if baseline.get('layoutHint'):
        assert baseline['leadLabelValidation']['semanticIdentityConfirmed'] and baseline['rhythmLeadValidation']['semanticIdentityConfirmed']
        geometry = baseline
        kind = 'fresh-maintained-source-identity'
    elif labels['candidatePassed']:
        candidate = labels['candidate']
        assert candidate['candidatePassed'] and all((q['passed'] for q in candidate['finalSourceBindings']))
        geometry = candidate['diagnosticGeometry']
        kind = 'fresh-diagnostic-source-proof-not-production'
    else:
        return (None, None)
    grid = geometry['sourceLabelGrid']
    fit = grid['xFit']
    rows = geometry['rowCenters']
    context = {'sourceIdentityVerifiedForDiagnostic': True, 'evidenceKind': kind, 'fontHeight': grid['fontHeight'], 'rowSpacing': grid['rowSpacing'], 'columnSpacing': grid['columnSpacing'], 'firstColumnRows': [{'row': i, 'lead': ['I', 'II', 'III', 'II'][i], 'labelX': fit[0] + i * fit[1], 'waveformCenterY': center} for i, center in enumerate(rows)], 'sourceContextProvenance': {'method': 'fresh-image-only-label-observer', 'sourceRasterSha256': hashlib.sha256(image.tobytes()).hexdigest()}}
    return (geometry, context)

def observe(image, geometry, identity, crops, constraints, composite, label_observer, edge_calibration, pulse_fallback, period, windows, local_timing, layout, printed_calibration, maintained_timing, source_sha):
    labels = label_observer.observe(image, geometry, identity, crops, constraints, composite)
    g, context = source_context(labels, image)
    result = {'labels': labels, 'branch': 'no-source-label-context', 'candidatePassed': False, 'sourceContext': context, 'diagnosticOnly': True, 'productionChange': False}
    if g is None:
        return result
    raw = layout.detect_ecg_calibration(image)
    edge_result, proposals = edge_calibration.build(layout, 'edge-proof')(image)
    replay = _pulse_local_grid.replay(raw, proposals, context, image)
    selected = replay['candidate']
    if any(q.get('changed') for q in replay['refinements']) or replay['gridFallbacks']:
        result['pulseCalibrationRefinement'] = {
            'bottomCandidate': replay['bottomCandidate'],
            'bottomRefinements': replay['refinements'],
            'gridFallbacks': replay['gridFallbacks'],
            'originalChoice': replay['original'],
        }
    printed = printed_calibration.read_printed_settings(image)
    printed['sourceSha256'] = source_sha
    reconciled = printed_calibration.reconcile_printed_settings(selected['finalResult']['calibration'], printed)
    timed_geometry = {**copy.deepcopy(g), 'calibration': reconciled}
    original = maintained_timing.propose_source_panel_timing(image, timed_geometry)
    result.update(rawCalibration=raw, edgeCalibrationResult=edge_result, pulseProposals=proposals, pulseSelection=selected, printedSettings=printed, reconciledCalibration=reconciled, existingTiming=original)
    attempts = []
    grids = []
    source_row = None
    physical = bool(reconciled.get('detected') and (reconciled.get('reconciliation') or {}).get('state') == 'corroborated_inference' and ((reconciled.get('reconciliation') or {}).get('quantitativeBlocked') is False))
    if original.get('state') != 'source_supported' and physical:
        radius = round(g['sourceLabelGrid']['fontHeight'] * 0.25)
        for offset in range(-radius, radius + 1):
            observed = windows.windows(image, g, offset)
            counts = [q['candidateCount'] for q in observed]
            attempts.append({'offsetPixels': offset, 'windows': observed, 'candidateCounts': counts, 'allNineUnique': all((n == 1 for n in counts)), 'originalCountsExact': None if len(g['sourceLabelGrid']['xFit']) == 4 else counts == maintained_timing._window_counts(image, g, offset)})
        grids = local_grids(image, g, layout, period)
        if selected['newCalibration']:
            source_row = selected['candidateAssessments'][selected['selectedProposalIndex']]['sourceContext']['matchedRows'][0]
    case = {'existingTiming': original, 'calibration': reconciled, 'diagnosticContext': context, 'separatorAttempts': attempts, 'localGrids': [{k: q[k] for k in ['row', 'column', 'bounds', 'selectedPair']} for q in grids], 'pulseCalibrationSourceRow': source_row}
    if physical and original.get('state') != 'source_supported' and (source_row is None):
        final = {'branch': 'fallback-unresolved', 'candidatePassed': False, 'finalResult': copy.deepcopy(original), 'originalTimingExact': True, 'productionTimingPromoted': False, 'failureReason': 'calibration-source-row-unconfirmed'}
    else:
        final = local_timing.propose(image, case)
    result.update(branch=final['branch'], candidatePassed=final['candidatePassed'], timing=final, separatorAttempts=attempts, localGridObservations=grids, pulseCalibrationSourceRow=source_row)
    if not final['candidatePassed']:
        neutral = _neutral_timing_replay.replay(image, result)
        if neutral['newTimingAdmission']:
            final = neutral['finalTiming']
            result.update(branch=final['branch'], candidatePassed=final['candidatePassed'], timing=final, neutralTimingFallback=neutral)
    return result
