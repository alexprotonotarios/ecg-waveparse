"""Ambiguity gaps with observed absolute returns and partially observed medians."""
import numpy as np

def median_bounds(values, valid):
    values = np.asarray(values, float)
    valid = np.asarray(valid, bool)
    n = len(values)
    known = int(valid.sum())
    if not n:
        return {'sampleCount': 0, 'knownCount': 0, 'missingCount': 0, 'lower': None, 'upper': None, 'finite': False}
    lower = float(np.median(np.where(valid, values, -np.inf)))
    upper = float(np.median(np.where(valid, values, np.inf)))
    return {'sampleCount': n, 'knownCount': known, 'missingCount': n - known, 'lower': lower if np.isfinite(lower) else None, 'upper': upper if np.isfinite(upper) else None, 'finite': bool(np.isfinite(lower) and np.isfinite(upper))}

def classify(y, times, valid, event, pulse_height_pixels, peer_context):
    y = np.asarray(y, float)
    times = np.asarray(times, float)
    valid = np.asarray(valid, bool)
    i = int(np.argmin(abs(times - event['extremeTimeSeconds'])))
    distance = times - event['timeSeconds']
    lf = np.flatnonzero((distance <= -0.08) & (distance >= -0.12))
    rf = np.flatnonzero((distance >= 0.08) & (distance <= 0.12))
    lb = median_bounds(y[lf], valid[lf])
    rb = median_bounds(y[rf], valid[rf])
    bounded = bool(lb['finite'] and rb['finite'] and (lb['knownCount'] >= 2) and (rb['knownCount'] >= 2))
    interval = [min(lb['lower'], rb['lower']), max(lb['upper'], rb['upper'])] if bounded else None
    amplitude = max(interval[0] - y[i], y[i] - interval[1], 0.0) if bounded else None
    band = max(2.0, 0.1 * amplitude) if bounded else None
    candidates = np.flatnonzero(abs(times - times[i]) <= 0.04)
    near = np.zeros(len(y), bool)
    if bounded:
        near = valid & (np.maximum(abs(y - interval[0]), abs(y - interval[1])) <= band)
    left = candidates[(candidates < i) & near[candidates]]
    right = candidates[(candidates > i) & near[candidates]]
    li = int(left[-1]) if len(left) else None
    ri = int(right[0]) if len(right) else None
    returned = li is not None and ri is not None
    width = float(times[ri] - times[li]) if returned else None
    core = np.flatnonzero(abs(distance) <= 0.04)
    paired = bool(any((q['withinExisting75msTolerance'] and q['centerValid'] and q['extremeValid'] for q in event['nearestOtherLeads'])))
    checks = {'flankMedianBoundsFiniteWithTwoObservedSamples': bounded, 'allPossibleFlankMediansAgree': bool(bounded and interval[1] - interval[0] <= band), 'minimumAmplitudeAtLeastThirtyPercentCalibrationPulse': bool(bounded and amplitude >= 0.3 * pulse_height_pixels), 'actualValidBaselineReturnBothSidesWithin40ms': returned, 'returnToReturnDurationAtMost40ms': bool(width is not None and width <= 0.04), 'entireReturnIntervalSourceValid': bool(returned and np.all(valid[li:ri + 1])), 'coreFullySourceValid': bool(len(core) and np.all(valid[core])), 'threePeersHaveObservedLocalContext': bool(len(peer_context) == 3 and all((q['columnCount'] >= 2 and q['validFraction'] >= 0.9 for q in peer_context))), 'noValidPeerEventWithin75ms': not paired}
    reject = all(checks.values())
    return {'extremeIndex': i, 'leftFlankIndices': lf.tolist(), 'rightFlankIndices': rf.tolist(), 'leftMedianBounds': lb, 'rightMedianBounds': rb, 'possibleBaselineInterval': interval, 'minimumAmplitudePixels': float(amplitude) if amplitude is not None else None, 'pulseHeightPixels': float(pulse_height_pixels), 'absoluteReturnBandPixels': band, 'leftReturnIndex': li, 'rightReturnIndex': ri, 'returnToReturnDurationSeconds': width, 'peerContext': peer_context, 'checks': checks, 'rejectAsUncorroboratedNarrowExcursion': reject, 'gapIndexRange': [li, ri + 1] if reject else None}

def apply(paths, events, ranges, spacing, local_grids, grid_mm, gain, event_module, synthetic=False):
    coordinates = [event_module.coordinates(p, ranges, synthetic) for p in paths]
    out = []
    for n, (path, record, (cols, times, spans)) in enumerate(zip(paths, events, coordinates, strict=True)):
        old = np.asarray(path['valid'], bool)
        new = old.copy()
        details = []
        if path['row'] < 3 or synthetic:
            for event in record['events']:
                peers = []
                for k, other in enumerate(paths):
                    if k == n or (not synthetic and (not (other['row'] != path['row'] and (other['row'] == 3 or other['column'] == path['column'])))):
                        continue
                    ot = coordinates[k][1]
                    v = np.asarray(other['valid'], bool)
                    ix = np.flatnonzero(abs(ot - event['timeSeconds']) <= 0.04)
                    peers.append({'lead': other.get('lead', 'synthetic_row_' + str(other['row'])), 'columnCount': len(ix), 'validCount': int(v[ix].sum()), 'validFraction': float(v[ix].mean()) if len(ix) else 0.0})
                ppm = 10.0 if synthetic else local_grids[4 * path['row'] + path['column']]['selectedPair']['yPeriod'] / grid_mm
                result = classify(path['pathY'], times, old, event, ppm * gain, peers)
                details.append({'eventSourceX': event['sourceX'], 'eventTimeSeconds': event['timeSeconds'], **result})
                if result['gapIndexRange'] is not None:
                    l, r = result['gapIndexRange']
                    new[l:r] = False
        assert not np.any(new & ~old)
        out.append({'lead': path.get('lead', 'synthetic_row_' + str(path['row'])), 'row': path['row'], 'column': path.get('column'), 'range': path['range'], 'pathY': path['pathY'], 'oldValid': path['valid'], 'valid': new.astype(int).tolist(), 'newlyRejectedColumns': int(np.count_nonzero(old & ~new)), 'coverage': float(new.mean()), 'decisions': details, 'oldGapsRecovered': 0})
    return out
