"""Fresh source paths and qualitative flags after fresh label/calibration timing."""
import copy, hashlib
import numpy as np

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

def trace_fresh(image, context, initial_trace, refinement, ink, masks, events, excursions, review_policy):
    labels = context['labels']
    g = labels['candidate']['diagnosticGeometry']
    timing = context['timing']['candidate']
    case = {'caseId': 'image-only-diagnostic', 'kind': 'diagnostic-source', 'geometry': g, 'timing': timing, 'sourceBindings': labels['candidate']['finalSourceBindings'], 'localGrids': [{k: q[k] for k in ['row', 'column', 'bounds', 'selectedPair']} for q in context['localGridObservations']], 'gridScaleMm': context['reconciledCalibration']['gridScaleMmY']}
    before = ink.shaarray(image)
    initial = initial_trace.run_case(image, case, masks)
    prepared, excluded, publication, receipt = initial_trace.prepare_masks(image, case, masks)
    rows = g['rowCenters']
    spacing = float(np.median(np.diff(rows)))
    refined = {}
    for arm, (evidence, support, clean) in prepared.items():
        assert ink.shaarray(clean) == initial['arms'][arm]['maskSha256'] and ink.shaarray(evidence) == initial['arms'][arm]['evidenceSha256'] and (ink.shaarray(support) == initial['arms'][arm]['supportSha256'])
        refined[arm] = [refinement.refine(evidence, support, publication, q, row_center=rows[q['row']], row_spacing=spacing) for q in initial['arms'][arm]['paths']]
    neutral = masks.masks(image, g['sourceLabelGrid']['fontHeight'])['neutral-local-contrast']
    pulse = np.zeros(excluded.shape, bool)
    for box in receipt['boxes']:
        if box['kind'] == 'diagnostic-observed-pulse-columns':
            l, t, r, b = box['bounds']
            pulse[t:b, l:r] = True
    pulse_models = []
    row_masks = []
    pulse_paths = []
    for row in range(4):
        edge = timing['sourcePulseEdges'][row]
        model = ink.pulse_limit(image, (edge.get('selected') or edge)['bounds'])
        model['row'] = row
        pulse_models.append(model)
        mask, evidence, support = ink.prepare(image, neutral, excluded, pulse, model['robustUpperLimit'])
        row_masks.append({'row': row, 'maskSha256': ink.shaarray(mask), 'evidenceSha256': ink.shaarray(evidence), 'supportSha256': ink.shaarray(support), 'cleanPixelCount': int(mask.sum())})
        for path in refined['neutral-local-contrast']:
            if path['row'] == row:
                pulse_paths.append(ink.trace(evidence, support, publication, path, rows[row], spacing))
    paths = pulse_paths
    ranges = timing['rowPanelRanges']
    observed_events = events.peer_census([events.observe_path(q, ranges, spacing) for q in paths])
    historical = excursions.apply(paths, observed_events, ranges, spacing, case['localGrids'], case['gridScaleMm'], context['reconciledCalibration']['gainMmPerMv'], events)
    retained = [review_policy.retain_with_review_flags(q) for q in historical]
    for old, new in zip(paths, retained, strict=True):
        assert old['pathY'] == new['pathY'] and old['valid'] == new['valid']
    assert ink.shaarray(image) == before
    return {'initial': initial, 'refined': refined, 'pulseModels': pulse_models, 'rowMasks': row_masks, 'pulsePaths': pulse_paths, 'events': observed_events, 'historicalExcursionDecisions': historical, 'retainedPaths': retained, 'sourceRasterUnchanged': True, 'calibratedSignalExported': False, 'clinicalMeaning': 'unclassified', 'quantitativeUncertainty': 'unavailable', 'sourceGapPolicy': 'all prior connected/source/publication gaps retained; morphology-only exclusions become review flags'}

def observe(image, geometry, identity, crops, constraints, composite, label_observer, edge_calibration, pulse_fallback, period, windows, local_timing, calibration_observer, initial_trace, refinement, ink, masks, events, excursions, review_policy, layout, printed_calibration, maintained_timing, source_sha):
    context = calibration_observer.observe(image, geometry, identity, crops, constraints, composite, label_observer, edge_calibration, pulse_fallback, period, windows, local_timing, layout, printed_calibration, maintained_timing, source_sha)
    result = {'calibration': context, 'branch': context['branch'], 'candidatePassed': False, 'diagnosticOnly': True, 'productionChange': False}
    if context['candidatePassed']:
        result.update(branch='fresh-diagnostic-source-paths', candidatePassed=True, sourceTrace=trace_fresh(image, context, initial_trace, refinement, ink, masks, events, excursions, review_policy))
    return result
