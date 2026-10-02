"""Fresh diagnostic grid domains from the image and freshly retained source paths."""
import hashlib
import cv2
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

def grid_fresh(image, context, grid_baseline, grid_ridges, grid_calibration, grid_components, grid_crossings, grid_census, grid_consensus, grid_field, grid_mesh, grid_holes, masks, ink):
    cal = context['calibration']
    trace = context['sourceTrace']
    timing = cal['timing']['candidate']
    geometry = cal['labels']['candidate']['diagnosticGeometry']
    local = [{k: q[k] for k in ['row', 'column', 'bounds', 'selectedPair']} for q in cal['localGridObservations']]
    lo, hi = timing['rhythmRange']
    crop = image[:, lo:hi]
    before = ink.shaarray(image)
    first = next((q for q in local if q['row'] == 3 and q['column'] == 0))
    ppm = first['selectedPair']['yPeriod'] / cal['reconciledCalibration']['gridScaleMmY']
    limit = max((q['robustUpperLimit'] for q in trace['pulseModels']))
    channels = crop if crop.dtype == np.uint8 else crop.astype(np.int16)
    maximum, minimum = masks._channel_extrema(channels)
    colour = (maximum.astype(np.int16) - minimum.astype(np.int16) >= 18).astype(np.uint8)
    neutral = masks.masks(crop, geometry['sourceLabelGrid']['fontHeight'])['neutral-local-contrast'].astype(bool)
    old_maximum = maximum if crop.dtype == np.uint8 else masks._maximum_channel(crop)
    old_dark = (neutral & (old_maximum <= limit)).astype(np.float32)
    dark, components = grid_components.anchored_dark_components(old_dark, colour, neutral)
    source_x = lo + (hi - lo - 1) // 2
    callback = lambda peaks: grid_calibration.calibrate(peaks, local, source_x)
    result = {'state': 'unresolved', 'sourceInterval': [lo, hi], 'sourceImageArraySha256': ink.shaarray(crop), 'pixelsPerMm': ppm, 'occlusionPulseLimit': limit, 'colourOverrideSha256': ink.shaarray(colour), 'occlusionOverrideSha256': ink.shaarray(old_dark), 'darkMaskSha256': ink.shaarray(dark), 'componentReceipt': components, 'sourcePathsIncludeReviewFlaggedSamples': True, 'caseSpecificReviewControlsUsedDuringInference': False, 'calibratedSignalExported': False}
    for name, module, mask in [('baseline', grid_baseline, old_dark), ('componentLimited', grid_ridges, dark)]:
        cv2.setRNGSeed(0)
        try:
            evidence, arrays = module.observe_grid(crop, ppm, colour_mask=colour, dark_mask=mask, seed_calibration=callback)
            result[name] = {'refusedEarly': False, 'evidence': evidence, 'arrays': arrays}
        except module.ObservationError as error:
            result[name] = {'refusedEarly': True, 'reason': str(error), 'observations': error.observations}
            return clean(result)
    old = result['baseline']['arrays']['observedRowsAll']
    raw = result['componentLimited']['arrays']
    observed = raw['observedRowsAll'].copy()
    assert np.array_equal(observed[np.isfinite(old)], old[np.isfinite(old)])
    checks = []
    eligible = ~(np.isfinite(raw['occludedColumnFraction']) & (raw['occludedColumnFraction'] > 0.1))
    for receipt in raw['ridgeReceipts']:
        row, node = (receipt['row'], receipt['node'])
        if np.isfinite(old[row, node]) or not np.isfinite(observed[row, node]):
            continue
        check = grid_crossings.crossings(receipt, trace['retainedPaths'], lo, raw['xNodes'], result['componentLimited']['evidence']['coarseSlope'])
        checks.append(check)
        if check['crossesKnownSourcePath']:
            observed[row, node] = np.nan
            eligible[row, node] = False
    support = np.isfinite(observed).sum(0) / np.maximum(1, eligible.sum(0))
    seed = result['componentLimited']['evidence']['localSeedCalibration']
    indices = np.r_[0, np.cumsum([q['roundedMajorMultiple'] for q in seed['gapChecks']])]
    spatial = []
    for i in range(len(indices) - 1):
        for j, node in enumerate(raw['xNodes']):
            q = grid_census.assess_pair(lo + node, clean(observed[i:i + 2, j]), int(indices[i + 1] - indices[i]), local, grid_calibration.period_at)
            q.update(gapIndex=i, nodeIndex=j, originalSeedIds=seed['retainedSeeds'][i:i + 2], originalIntegerGapPassed=seed['gapChecks'][i]['passed'])
            spatial.append(q)
    distributed = [grid_consensus.consensus([q for q in spatial if q['gapIndex'] == i]) for i in range(len(indices) - 1)]
    gaps = [q['eligible'] for q in distributed]
    field = grid_field.build_field(raw['xNodes'], observed, indices, gaps, node_support=support)
    result.update(guardedObservedRows=observed, crossingChecks=checks, nodeSupport=support, majorIndices=indices, spatialChecks=spatial, distributedCalibration=distributed, field=field, oldObservationsExact=int(np.isfinite(old).sum()), rawNewRidges=len(checks), vetoedNewRidges=sum((q['crossesKnownSourcePath'] for q in checks)), retainedNewRidges=int((np.isfinite(observed) & ~np.isfinite(old)).sum()))
    if field['cells']:
        mesh = grid_mesh.build_mesh(field['cells'])
        extension = grid_holes.extend_holes(mesh, indices, gaps)
        result.update(state='partial', mesh=mesh, extension=extension)
    assert ink.shaarray(image) == before
    result['sourceRasterUnchanged'] = True
    return clean(result)

def observe(image, geometry, identity, path_observer, grid_baseline, grid_ridges, grid_calibration, grid_components, grid_crossings, grid_census, grid_consensus, grid_field, grid_mesh, grid_holes, **parts):
    context = path_observer.observe(image, geometry, identity, **parts)
    result = {'sourcePaths': context, 'branch': context['branch'], 'candidatePassed': False, 'diagnosticOnly': True, 'productionChange': False}
    if context['candidatePassed']:
        grid = grid_fresh(image, context, grid_baseline, grid_ridges, grid_calibration, grid_components, grid_crossings, grid_census, grid_consensus, grid_field, grid_mesh, grid_holes, parts['masks'], parts['ink'])
        if not grid.get('field', {}).get('cells') and 'guardedObservedRows' in grid:
            from . import _supported_grid_fallback
            grid = _supported_grid_fallback.propose(image, context, grid)
        result.update(branch='fresh-diagnostic-source-grid', candidatePassed=grid['state'] == 'partial', grid=grid)
    return result
