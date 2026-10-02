"""Observe source-path events and peer timing; never edit source paths or validity."""
import hashlib
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter1d
from ecg_pipeline.native_grid_digitizer import waveform_event_columns, rhythm_qrs_anchors

def coordinates(path, ranges, synthetic=False):
    lo, hi = path['range']
    cols = np.arange(lo, hi, dtype=np.int32)
    if synthetic:
        return (cols, (cols - 270) / 250.0, [{'sourceRange': [270, 1990], 'startSeconds': 0.0, 'endSeconds': 6.88}])
    if path['row'] < 3:
        k = path['column']
        spans = [{'sourceRange': [lo, hi], 'startSeconds': 2.5 * k, 'endSeconds': 2.5 * (k + 1)}]
    else:
        spans = [{'sourceRange': r, 'startSeconds': k * 2.5, 'endSeconds': (k + 1) * 2.5} for k, r in enumerate(ranges[3])]
    times = np.full(len(cols), np.nan)
    for q in spans:
        l, r = q['sourceRange']
        inside = (cols >= l) & (cols < r)
        times[inside] = q['startSeconds'] + (cols[inside] - l) / (r - l) * (q['endSeconds'] - q['startSeconds'])
    assert np.isfinite(times).all() and np.all(np.diff(times) > 0)
    return (cols, times, spans)

def observe_path(path, ranges, spacing, synthetic=False):
    cols, times, spans = coordinates(path, ranges, synthetic)
    ys = np.asarray(path['pathY'], np.int32)
    valid = np.asarray(path['valid'], bool)
    sharp = waveform_event_columns(cols, ys, row_spacing=spacing)
    is_rhythm = not synthetic and path['row'] == 3
    used = rhythm_qrs_anchors(cols, ys, row_spacing=spacing, effective_sample_rate_hz=len(cols) / 10.0) if is_rhythm else sharp
    rawscore = maximum_filter1d(ys.astype(float), 6) - minimum_filter1d(ys.astype(float), 6)
    score = gaussian_filter1d(rawscore, 1.0 if is_rhythm else 0.8)
    entries = []
    for x in used:
        i = int(x - cols[0])
        core = np.flatnonzero(np.abs(times - times[i]) <= 0.04)
        wide = np.flatnonzero(np.abs(times - times[i]) <= 0.12)
        flank = wide[np.abs(times[wide] - times[i]) >= 0.08]
        baseline = float(np.median(ys[flank])) if len(flank) else float(np.median(ys[wide]))
        extreme = int(core[np.argmax(np.abs(ys[core] - baseline))])
        entries.append({'sourceX': int(x), 'sourceY': int(ys[i]), 'timeSeconds': float(times[i]), 'eventScorePixels': float(score[i]), 'centerValid': bool(valid[i]), 'coreSourceRange': [int(cols[core[0]]), int(cols[core[-1]]) + 1], 'coreValidCount': int(valid[core].sum()), 'coreColumnCount': len(core), 'localBaselinePixels': baseline, 'extremeSourceX': int(cols[extreme]), 'extremeSourceY': int(ys[extreme]), 'extremeValid': bool(valid[extreme]), 'extremeTimeSeconds': float(times[extreme]), 'excursionFromLocalBaselinePixels': float(ys[extreme] - baseline), 'currentRowSpacingExcursionGatePixels': spacing * 0.3, 'currentRowSpacingJumpGatePixels': spacing * 0.25, 'maximumCoreJumpPixels': float(np.max(np.abs(np.diff(ys[core].astype(float))))) if len(core) > 1 else 0.0})
    return {'lead': path.get('lead', 'synthetic_row_' + str(path['row'])), 'row': path['row'], 'column': path.get('column'), 'sourcePathSha256': hashlib.sha256(ys.astype('<i4').tobytes()).hexdigest(), 'validitySha256': hashlib.sha256(valid.astype('u1').tobytes()).hexdigest(), 'spans': spans, 'nativeSharpEventColumns': sharp.tolist(), 'usedEventColumns': used.tolist(), 'eventDetector': 'existing-rhythm-qrs-anchors' if is_rhythm else 'existing-waveform-event-columns', 'events': entries}

def peer_census(records, synthetic=False):
    for r in records:
        if not synthetic and r['row'] == 3:
            continue
        peers = [q for q in records if q['row'] != r['row'] and (synthetic or q['row'] == 3 or q['column'] == r['column'])]
        for e in r['events']:
            observations = []
            for q in peers:
                choices = q['events']
                nearest = min(choices, key=lambda z: abs(z['timeSeconds'] - e['timeSeconds'])) if choices else None
                distance = abs(nearest['timeSeconds'] - e['timeSeconds']) if nearest else None
                observations.append({'lead': q['lead'], 'sourceX': nearest['sourceX'] if nearest else None, 'timeSeconds': nearest['timeSeconds'] if nearest else None, 'absoluteDistanceSeconds': distance, 'centerValid': nearest['centerValid'] if nearest else None, 'extremeValid': nearest['extremeValid'] if nearest else None, 'withinExisting75msTolerance': bool(distance is not None and distance <= 0.075)})
            e['nearestOtherLeads'] = observations
            e['otherLeadsWithin75ms'] = sum((o['withinExisting75msTolerance'] for o in observations))
            e['otherValidCentersWithin75ms'] = sum((o['withinExisting75msTolerance'] and bool(o['centerValid']) for o in observations))
    return records
