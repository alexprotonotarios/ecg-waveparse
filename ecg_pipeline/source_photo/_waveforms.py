"""Fresh diagnostic ray geometry, source tracing and calibrated waveform arrays."""
import hashlib
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

def clean(v):
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, np.generic):
        return clean(v.item())
    if isinstance(v, float) and (not np.isfinite(v)):
        return None
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [clean(x) for x in v]
    return v

def observe_rays(image, cal, vertical_ridges, ray_model):
    geometry = cal['labels']['candidate']['diagnosticGeometry']
    rows = geometry['rowCenters']
    windows = []
    for q in sorted(cal['localGridObservations'], key=lambda v: (v['row'], v['column'])):
        x0, y0, x1, y1 = q['bounds']
        out = vertical_ridges.observe(image[y0:y1, x0:x1], q['selectedPair']['xPeriod'])
        windows.append({'row': q['row'], 'column': q['column'], 'sourceBounds': q['bounds'], 'rowBaselineSourceY': rows[q['row']], **out})
    tracks = []
    duplicates = []
    unfitted = []
    envelopes = []
    for window in windows:
        wid = f"{window['row']}:{window['column']}"
        seen = {}
        x0, y0, _, _ = window['sourceBounds']
        allpoints = []
        for i, ridge in enumerate(window['ridges']):
            tid = f'{wid}:{i}'
            key = tuple(((z['nodeY'], z['measuredX']) for z in ridge['profiles']))
            if ridge['fit'] is None:
                unfitted.append({'trackId': tid, 'measuredNodes': ridge['measuredNodeCount']})
                continue
            if key in seen:
                duplicates.append({'trackId': tid, 'sameAs': seen[key]})
                continue
            seen[key] = tid
            f = ridge['fit']
            points = [[x0 + z['measuredX'], y0 + z['nodeY']] for z in ridge['profiles'] if z['measuredX'] is not None]
            allpoints.extend(points)
            tracks.append({'trackId': tid, 'windowId': wid, 'row': window['row'], 'column': window['column'], 'centerX': x0 + f['centerX'], 'centerY': y0 + f['centerY'], 'dxPerDy': f['dxPerDy'], 'points': points, 'referenceY': window['rowBaselineSourceY']})
        if not allpoints:
            return {'state': 'unresolved', 'reason': 'source-window-without-fitted-rays', 'windows': windows, 'tracks': tracks, 'duplicates': duplicates, 'unfittedTracks': unfitted}
        envelopes.append({'windowId': wid, 'bounds': [min((x for x, y in allpoints)), min((y for x, y in allpoints)), max((x for x, y in allpoints)), max((y for x, y in allpoints))], 'meaning': 'Observed-node bounding envelope only, not a validated interpolation domain.'})
    result = {'state': 'unresolved', 'windows': windows, 'tracks': tracks, 'duplicates': duplicates, 'unfittedTracks': unfitted, 'observedNodeEnvelopes': envelopes}
    if len(tracks) < 3:
        return result | {'reason': 'fewer-than-three-source-ray-tracks'}
    size = [image.shape[1], image.shape[0]]
    try:
        model = ray_model.fit_model(tracks, size)
        direct = [ray_model.track_residual(model, t, t['referenceY']) for t in tracks]
        held = []
        for envelope in envelopes:
            wid = envelope['windowId']
            train = [t for t in tracks if t['windowId'] != wid]
            test = [t for t in tracks if t['windowId'] == wid]
            local = ray_model.fit_model(train, size)
            held.append({'windowId': wid, 'trainingTrackIds': [t['trackId'] for t in train], 'model': local, 'residuals': [ray_model.track_residual(local, t, t['referenceY']) for t in test]})
        vx, vy, vw = model['homogeneousVanishingPoint']
        jacobians = [(vy - vw * y0) / (vy - vw * y) for y0 in rows for y in [0, size[1] - 1]]
        if min(jacobians) <= 0:
            return result | {'reason': 'nonpositive-source-ray-jacobian'}
    except (AssertionError, ValueError, ZeroDivisionError, np.linalg.LinAlgError) as e:
        return result | {'reason': 'unresolved-source-ray-model', 'detail': str(e)}
    return result | {'state': 'diagnostic-approximation', 'model': model, 'directResiduals': direct, 'heldOutWindows': held, 'fullImageHorizontalJacobianCornerValues': jacobians, 'clinicalValidation': False}

def trace_rays(image, context, rays, initial_trace, ink, masks, ray_sampling, source_sha):
    cal = context['sourcePaths']['calibration']
    trace = context['sourcePaths']['sourceTrace']
    geometry = cal['labels']['candidate']['diagnosticGeometry']
    rows = geometry['rowCenters']
    spacing = float(np.median(np.diff(rows)))
    case = {'caseId': 'image-only-diagnostic', 'kind': 'diagnostic-source', 'geometry': geometry, 'timing': cal['timing']['candidate'], 'sourceBindings': cal['labels']['candidate']['finalSourceBindings']}
    excluded, publication, pulse, exclusion_receipt = initial_trace.diagnostic_exclusions(image, case)
    neutral = masks.masks(image, geometry['sourceLabelGrid']['fontHeight'])['neutral-local-contrast']
    arms = {'guarded-identity': [], 'guarded-rays': []}
    row_masks = []
    for row in range(4):
        model = trace['pulseModels'][row]
        mask, ev, support = ink.prepare(image, neutral, excluded, pulse, model['robustUpperLimit'])
        receipt = {'row': row, 'maskSha256': ink.shaarray(mask), 'evidenceSha256': ink.shaarray(ev), 'supportSha256': ink.shaarray(support), 'cleanPixelCount': int(mask.sum())}
        assert receipt == trace['rowMasks'][row]
        row_masks.append(receipt)
        for previous in trace['refined']['neutral-local-contrast']:
            if previous['row'] != row:
                continue
            retained = next((q for q in trace['retainedPaths'] if q['lead'] == previous['lead']))
            spec = {**previous, 'pathY': retained['pathY'], 'valid': [int(x) for x in retained['valid']]}
            flags = retained['reviewFlags']
            review_ranges = [q['sourceXRangeHalfOpen'] for q in flags]
            for arm, geometry_model in [('guarded-identity', {'homogeneousVanishingPoint': [0.0, 1.0, 0.0]}), ('guarded-rays', rays['model'])]:
                E, S, P, footprint, receipt = ray_sampling.planes(ev, support, publication, spec, geometry_model, rows[row], review_ranges)
                traced = ink.trace(E, S, P, spec, rows[row], spacing)
                q = ray_sampling.retain_path(traced, footprint, spec)
                q.update(ray_sampling.chord_transport(q, spec, geometry_model, rows[row], review_ranges))
                q.update(referenceY=rows[row], samplingReceipt=receipt, sourceReviewFlags=flags, sourceImageSha256=source_sha, clinicalMeaning='unclassified', quantitativeUncertainty='unavailable', geometryModelIsApproximate=arm == 'guarded-rays', completePhysicalExtractionAccepted=False)
                arms[arm].append(q)
    return {'arms': arms, 'rowMasks': row_masks, 'sourceGapTransportApplied': True, 'oldSamplesSorted': False}

def convert_paths(context, rays, source, converter, coordinate_mesh, coordinate_map, curve_domain):
    grid = context['grid']
    cal = context['sourcePaths']['calibration']
    cells = grid['field']['cells']
    mesh = grid['mesh']
    extension = grid['extension']
    offset = grid['sourceInterval'][0]
    base = coordinate_mesh.mapper_class(converter.CellMap, mesh)
    factory = coordinate_map.mapper_class(base, mesh, extension, coordinate_mesh.triangle_value)
    domain = curve_domain.CurveDomain(cells, mesh, extension, offset)
    ranges = cal['timing']['candidate']['rowPanelRanges']
    rows = cal['labels']['candidate']['diagnosticGeometry']['rowCenters']
    gain = cal['reconciledCalibration']['gainMmPerMv']
    out = {}
    for arm, paths in source['arms'].items():
        model = {'homogeneousVanishingPoint': [0.0, 1.0, 0.0]} if arm == 'guarded-identity' else rays['model']
        converted = []
        reviews = []
        for path in paths:
            wrapped = curve_domain.path_mapper_factory(factory, domain, path, model, rows[path['row']])
            q = clean(converter.convert(path, cells, offset, ranges, gain=gain, mapper_factory=wrapped))
            q.update(originalSourceX=path['sourceX'], referenceY=rows[path['row']], timeCoordinate='ray-reference x at frozen source row baseline', sourceChordGeometry='inverse quadratic camera ray between consecutive reference-x/integer-source-y samples', sourceTransportBrokenConnections=[z['sourceIndices'] for z in q['brokenConnections'] if not path['sourceGapConnectionValid'][z['sourceIndices'][0]]])
            flags = [False] * len(q['canonicalMv'])
            times = np.asarray(q['sourceTimesSeconds'])
            for segment in q['segments']:
                lo, hi = segment['sourceIndices']
                for j in segment['targetIndices']:
                    t = j / q['canonicalSampleRate']
                    i = max(lo, min(hi - 2, int(np.searchsorted(times[lo:hi], t, side='right')) + lo - 1))
                    assert q['connectionValid'][i]
                    flags[j] = bool(path['reviewRequired'][i] or path['reviewRequired'][i + 1] or path['sourceChordReviewRequired'][i])
            converted.append(q)
            reviews.append({'lead': path['lead'], 'canonicalReviewRequired': flags, 'sourceReviewFlags': path['sourceReviewFlags'], 'clinicalMeaning': 'unclassified', 'quantitativeUncertainty': 'unavailable', 'unflaggedDoesNotMeanSafe': True})
        out[arm] = {'convertedPaths': converted, 'canonicalReview': reviews}
    return {'arms': out, 'sampleRateHz': 500, 'gainMmPerMv': gain, 'physicalExtractionAccepted': False, 'clinicalMeaning': 'unclassified'}

def observe(image, geometry, identity, grid_observer, vertical_ridges, ray_model, ray_sampling, converter, coordinate_mesh, coordinate_map, curve_domain, **parts):
    before = hashlib.sha256(image.tobytes()).hexdigest()
    context = grid_observer.observe(image, geometry, identity, **parts)
    result = {'sourceGrid': context, 'branch': context['branch'], 'candidatePassed': False, 'diagnosticOnly': True, 'productionChange': False}
    if context['candidatePassed']:
        rays = observe_rays(image, context['sourcePaths']['calibration'], vertical_ridges, ray_model)
        result.update(branch='fresh-ray-model-unresolved', rays=rays)
        if rays['state'] == 'diagnostic-approximation':
            source = trace_rays(image, context, rays, parts['initial_trace'], parts['ink'], parts['masks'], ray_sampling, parts['source_sha'])
            converted = convert_paths(context, rays, source, converter, coordinate_mesh, coordinate_map, curve_domain)
            result.update(branch='fresh-diagnostic-ray-waveforms', candidatePassed=True, raySourcePaths=source, waveforms=converted)
    assert hashlib.sha256(image.tobytes()).hexdigest() == before
    result['sourceRasterUnchanged'] = True
    return clean(result)
