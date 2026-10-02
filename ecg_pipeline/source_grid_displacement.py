"""Paper-grid displacement measured from original colours and visible ridges."""
from __future__ import annotations

import hashlib
from typing import Any
import numpy as np
import cv2
from scipy.signal import find_peaks


def _refine_seed_ridges(profile: np.ndarray, peaks: np.ndarray, period: float):
    """Observe bounded half-height ridge centroids; do not infer missing centers."""
    centers, supports = ([], [])
    radius = max(3, int(np.floor(0.4 * period)))
    for peak in peaks:
        lo, hi = (max(0, int(peak) - radius), min(len(profile), int(peak) + radius + 1))
        local = profile[lo:hi]
        at = int(peak) - lo
        if len(local) < 3 or local[at] < 0.55 or local.max() - local.min() < 0.25:
            raise ValueError('Fractional seed ridge has insufficient support.')
        cutoff = float(local.min() + 0.5 * (local.max() - local.min()))
        a, b = (at, at + 1)
        while a > 0 and local[a - 1] >= cutoff:
            a -= 1
        while b < len(local) and local[b] >= cutoff:
            b += 1
        if a == 0 or b == len(local) or b - a > 0.4 * period:
            raise ValueError('Fractional seed ridge is clipped or merged.')
        other = local.copy()
        other[a:b] = local.min()
        if other.max() >= local[at] - 0.05:
            raise ValueError('Fractional seed ridge is ambiguous.')
        weights = np.maximum(0, local[a:b] - local.min())
        center = float(np.average(np.arange(lo + a, lo + b), weights=weights))
        centers.append(center)
        supports.append([lo + a, lo + b])
    centers = np.asarray(centers)
    if len(centers) < 8 or np.any(np.diff(centers) <= 0):
        raise ValueError('Fractional seed ordering is unsupported.')
    return (centers, np.asarray(supports, dtype=np.int32))


def _chromatic_centroid(values, bounds, window):
    a, b = map(int, bounds)
    lo, hi = map(int, window)
    if not 0 <= lo < a < b < hi <= len(values):
        raise ValueError('Chromatic position requires interior binary support.')
    local = values[lo:hi]
    if not np.isfinite(local).all() or np.any(local < 0):
        raise ValueError('Invalid original chromatic profile.')
    weights = np.maximum(0, values[a:b] - local.min())
    if weights.sum() <= 0:
        raise ValueError('No chromatic weight within binary support.')
    center = float(np.average(np.arange(a, b), weights=weights))
    if not a <= center <= b - 1:
        raise ValueError('Source ridge observations do not match the original binary support.')
    return center


def _observe_chromatic_ridges(image, pixels_per_mm, prior_evidence, prior):
    """Remeasure every existing valid observation, without deleting or adding one."""
    h, w = image.shape[:2]
    channels = image.astype(np.int16)
    binary = ((channels.max(2) - channels.min(2) >= 18) & (channels.max(2) >= 150)).astype(np.float32)
    chroma = (channels.max(2) - channels.min(2)).astype(np.float32) * binary
    dark = (channels.max(2) < 150).astype(np.float32)
    slope = prior_evidence['coarseSlope']
    period = 5 * float(pixels_per_mm)
    mid = (w - 1) // 2
    nodes = prior['xNodes']
    peaks = prior['seedRows']
    old = prior['observedRowsAll']
    observed = np.full(old.shape, np.nan)
    supports = np.full(old.shape + (2,), -1, dtype=np.int32)
    windows = np.full(old.shape + (2,), -1, dtype=np.int32)
    profiles = []
    for j, node in enumerate(nodes):
        xs = np.arange(max(0, int(node) - 32), min(w, int(node) + 33), dtype=np.float32)
        yy = np.arange(h, dtype=np.float32)[:, None] + slope * (xs - int(node))[None, :]
        xx = np.broadcast_to(xs, yy.shape).copy()

        def remap(field, interpolation):
            return cv2.remap(field, xx, yy, interpolation, borderMode=cv2.BORDER_CONSTANT)
        values = remap(binary, cv2.INTER_LINEAR).mean(1)
        intensities = remap(chroma, cv2.INTER_LINEAR).mean(1)
        dark_strip = remap(dark, cv2.INTER_NEAREST)
        profiles.append(intensities)
        for i in np.flatnonzero(np.isfinite(old[:, j])):
            center = int(round(peaks[i] + slope * (node - mid)))
            radius = max(3, int(np.floor(0.4 * period)))
            lo, hi = (max(0, center - radius), min(h, center + radius + 1))
            local = values[lo:hi]
            at = int(np.argmax(local))
            a, b = (at, at + 1)
            cutoff = local.min() + 0.5 * (local.max() - local.min())
            while a > 0 and local[a - 1] >= cutoff:
                a -= 1
            while b < len(local) and local[b] >= cutoff:
                b += 1
            if not (0 < a < b < len(local) and local.max() >= 0.55 and (local.max() - local.min() >= 0.25)):
                raise ValueError('Source ridge observations do not match the original binary support.')
            old_center = np.average(np.arange(lo + a, lo + b), weights=np.maximum(0, local[a:b] - local.min()))
            if not old_center == old[i, j]:
                raise ValueError('Source ridge observations do not match the original binary support.')
            fraction = float(np.any(dark_strip[lo + a:lo + b] > 0.5, axis=0).mean())
            if not (fraction == prior['occludedColumnFraction'][i, j] and fraction <= 0.1):
                raise ValueError('Source ridge observations do not match the original binary support.')
            other = local.copy()
            other[a:b] = local.min()
            if b - a > 0.4 * period or other.max() >= local[at] - 0.05:
                raise ValueError('Chromatic observation has merged or ambiguous binary support.')
            bounds = (lo + a, lo + b)
            observed[i, j] = _chromatic_centroid(intensities, bounds, (lo, hi))
            supports[i, j] = bounds
            windows[i, j] = (lo, hi)
    if not np.array_equal(np.isfinite(observed), np.isfinite(old)):
        raise ValueError('Source ridge observations do not match the original binary support.')
    complete = np.isfinite(observed).all(1)
    reason = None
    if prior_evidence['minimumNodeSupportFraction'] < 0.8:
        reason = 'Less than 80 percent row support at a node.'
    elif complete.sum() < 8:
        reason = 'Fewer than eight complete grid rows.'
    elif np.ptp(observed[complete, 0]) < 0.75 * np.ptp(peaks):
        reason = 'Complete rows cover less than 75 percent of the seeded vertical span.'
    else:
        rows = observed[complete]
        delta = rows - rows[:, 0, None]
        displacement = np.median(delta, axis=0)
        residual = float(np.max(abs(delta - displacement)))
        if residual > 1:
            reason = 'Rows disagree with shared displacement by more than one pixel.'
        elif np.min(np.diff(rows, axis=0)) <= 0:
            reason = 'Grid rows cross.'
        elif np.max(abs(np.diff(displacement) / np.diff(nodes))) > 0.25:
            reason = 'Local displacement slope exceeds existing budget.'
    evidence = {k: v for k, v in prior_evidence.items() if k not in ('maxRowResidualPixels', 'endpointDisplacementPixels', 'referenceBoundsY')}
    evidence.update(state='unresolved' if reason else 'supported', reason=reason, method='unoccluded-binary-support-chromatic-grid-ridges-v1', positionMethod='positive-source-channel-range-within-unchanged-half-height-support-v1', binarySupportUnchanged=True, priorMethod=prior_evidence['method'], priorState=prior_evidence['state'], priorReason=prior_evidence['reason'])
    result = {k: v.copy() for k, v in prior.items() if k not in ('observedRows', 'displacement', 'referenceRows')}
    result.update(observedRowsAll=observed, binaryObservedRowsAll=old.copy(), ridgeSupportBounds=supports, ridgeWindowBounds=windows, chromaticProfiles=np.asarray(profiles))
    if not reason:
        evidence.update(maxRowResidualPixels=residual, endpointDisplacementPixels=float(displacement[-1]), referenceBoundsY=[float(rows[0, 0]), float(rows[-1, 0])])
        result.update(observedRows=rows, displacement=displacement, referenceRows=rows[:, 0])
    return (evidence, result)


def _detect_grid_displacement_binary(image: np.ndarray, pixels_per_mm: float, *, node_spacing: int = 64, refine_seeds: bool = False) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if type(node_spacing) is not int or node_spacing not in (32, 64):
        raise ValueError('Unsupported source observation node spacing.')
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected original BGR uint8 raster.')
    if not np.isfinite(pixels_per_mm) or not 2 <= pixels_per_mm <= 40:
        raise ValueError('Unsupported independently measured grid calibration.')
    height, width = image.shape[:2]
    if min(height, width) < 200 or image.size > 36_000_000:
        raise ValueError('Unsupported image bounds.')
    channels = image.astype(np.int16)
    mask = ((channels.max(axis=2)-channels.min(axis=2) >= 18) &
            (channels.max(axis=2) >= 150)).astype(np.uint8)
    # Closing is only a coarse proposal. Ridge observations use untouched mask.
    proposal = cv2.morphologyEx(mask*255, cv2.MORPH_CLOSE,
                              np.ones((1, 9), np.uint8))
    lines = cv2.HoughLinesP(proposal, 1, np.pi/1800,
                           threshold=max(40, width//5),
                           minLineLength=max(80, width//4), maxLineGap=12)
    accepted = []
    if lines is not None:
        for x0, y0, x1, y1 in lines[:, 0]:
            if x0 > x1:
                x0, y0, x1, y1 = x1, y1, x0, y0
            if x1-x0 < width/4:
                continue
            slope = (int(y1)-int(y0))/(int(x1)-int(x0))
            if abs(slope) <= np.tan(np.deg2rad(2)):
                accepted.append((x0, y0, x1, y1))
    if len(accepted) < 8:
        raise ValueError('Fewer than eight long coloured lines.')
    accepted = np.asarray(accepted, np.int32)
    slopes = (accepted[:,3]-accepted[:,1])/(accepted[:,2]-accepted[:,0])
    slope = float(np.median(slopes))
    coarse_spread = float(np.median(np.abs(slopes-slope))*(width-1))
    period = 5*float(pixels_per_mm)
    field = mask.astype(np.float32)
    dark = (channels.max(axis=2) < 150).astype(np.float32)

    def profile(node, *, return_dark=False):
        xs = np.arange(max(0, node-32), min(width, node+33), dtype=np.float32)
        yy = np.arange(height, dtype=np.float32)[:,None]+slope*(xs-node)[None,:]
        xx = np.broadcast_to(xs, yy.shape).copy()
        values = cv2.remap(field, xx, yy, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT).mean(axis=1)
        if return_dark:
            return values, cv2.remap(dark, xx, yy, cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT)
        return values

    mid = (width-1)//2
    seed = profile(mid)
    peaks, _ = find_peaks(seed, height=.55, prominence=.25,
                         distance=max(3, int(.6*period)))
    margin = abs(slope)*(width-1)/2 + period
    peaks = peaks[(peaks >= margin) & (peaks < height-margin)]
    if len(peaks) < 8:
        raise ValueError('Fewer than eight major-grid seeds.')
    periods = np.diff(peaks)
    multiples = np.rint(periods/period)
    seed_refinement = None
    if np.mean((multiples >= 1) & (multiples <= 4) &
               (np.abs(periods-multiples*period) <= 2)) < .9:
        if not refine_seeds:
            raise ValueError('Major-grid seed spacing is inconsistent with calibration.')
        original_peaks = peaks.copy()
        peaks, seed_supports = _refine_seed_ridges(seed, peaks, period)
        periods = np.diff(peaks)
        multiples = np.rint(periods/period)
        if np.mean((multiples >= 1) & (multiples <= 4) &
                   (np.abs(periods-multiples*period) <= 2)) < .9:
            raise ValueError('Fractional major-grid seed spacing is inconsistent with calibration.')
        seed_refinement = {
            'original': original_peaks, 'refined': peaks.copy(),
            'supports': seed_supports, 'profile': seed.copy(),
        }
    nodes = np.unique(np.append(np.arange(0,width,node_spacing),width-1)).astype(float)
    observed = np.full((len(peaks),len(nodes)),np.nan)
    strengths = np.zeros(observed.shape)
    occluded = np.full(observed.shape,np.nan)
    support = []
    all_seed_support = []
    for j, node in enumerate(nodes):
        values, dark_strip = profile(int(node),return_dark=True)
        expected = peaks+slope*(node-mid)
        for i, location in enumerate(expected):
            center = int(round(location))
            radius = max(3, int(np.floor(.4*period)))
            lo, hi = max(0,center-radius), min(height,center+radius+1)
            local = values[lo:hi]
            if len(local) < 3 or local.max() < .55 or local.max()-local.min() < .25:
                continue
            at = int(np.argmax(local))
            if at == 0 or at == len(local)-1:
                continue
            a, b = at, at+1
            cutoff = local.min()+.5*(local.max()-local.min())
            while a > 0 and local[a-1] >= cutoff:
                a -= 1
            while b < len(local) and local[b] >= cutoff:
                b += 1
            if a == 0 or b == len(local):
                continue
            occluded[i,j] = float(np.any(dark_strip[lo+a:lo+b] > .5,axis=0).mean())
            if occluded[i,j] > .1:
                continue
            weights = np.maximum(0,local[a:b]-local.min())
            observed[i,j] = np.average(np.arange(lo+a,lo+b),weights=weights)
            strengths[i,j] = local.max()
        valid = np.isfinite(observed[:,j])
        eligible = ~(np.isfinite(occluded[:,j]) & (occluded[:,j] > .1))
        all_seed_support.append(float(valid.mean()))
        support.append(float(valid.sum()/eligible.sum()) if eligible.any() else 0.0)
    result = {'coarseLines':accepted,'xNodes':nodes,'seedRows':peaks,
              'observedRowsAll':observed,'strengths':strengths,'occludedColumnFraction':occluded}
    # Return observations even when the geometry is refused.
    reason = None
    complete = np.isfinite(observed).all(axis=1)
    if min(support) < .8:
        reason = 'Less than 80 percent row support at a node.'
    elif complete.sum() < 8:
        reason = 'Fewer than eight complete grid rows.'
    elif np.ptp(observed[complete,0]) < .75*np.ptp(peaks):
        reason = 'Complete rows cover less than 75 percent of the seeded vertical span.'
    else:
        rows = observed[complete]
        row_delta = rows-rows[:,0,None]
        displacement = np.median(row_delta,axis=0)
        residual = float(np.max(np.abs(row_delta-displacement)))
        if residual > 1:
            reason = 'Rows disagree with shared displacement by more than one pixel.'
        elif np.min(np.diff(rows,axis=0)) <= 0:
            reason = 'Grid rows cross.'
        elif np.max(np.abs(np.diff(displacement)/np.diff(nodes))) > .25:
            reason = 'Local displacement slope exceeds existing budget.'
        else:
            result.update(observedRows=rows,displacement=displacement,
                          referenceRows=rows[:,0])
    evidence = {'state':'unresolved' if reason else 'supported','reason':reason,
                'method':'unoccluded-major-grid-ridges-v1',
                'coarseSlope':slope,'coarseEndpointMadPixels':coarse_spread,'coarseLineCount':len(accepted),
                'majorPeriodPixels':period,'seedRowCount':len(peaks),
                'completeRowCount':int(complete.sum()),'nodeCount':len(nodes),
                'minimumNodeSupportFraction':min(support),'minimumAllSeedSupportFraction':min(all_seed_support),'supportDenominator':'Seeds not directly observed to be dark-occluded; failed or unavailable ridge observations remain in denominator','boundsX':[0,width-1],
                'candidateSignalOrTruthUsed':False,'maximumAllowedOccludedColumnFraction':.1}
    if not reason:
        evidence.update(maxRowResidualPixels=residual,
                        endpointDisplacementPixels=float(displacement[-1]),
                        referenceBoundsY=[float(rows[0,0]),float(rows[-1,0])])
    if seed_refinement is not None:
        evidence.update(
            method='unoccluded-major-grid-fractional-seed-ridges-v2',
            seedObservationMethod='bounded-half-height-weighted-source-profile-v1',
            integerSpacingRefused=True,
        )
        result.update(
            seedOriginalRows=seed_refinement['original'],
            seedRefinedRows=seed_refinement['refined'],
            seedSupportBounds=seed_refinement['supports'],
            seedProfile=seed_refinement['profile'],
        )
    return evidence,result


def detect_grid_displacement(
    image: np.ndarray, pixels_per_mm: float, *, node_spacing: int = 64,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Retain original successes; refine positions only after explicit refusals."""
    try:
        evidence, observations = _detect_grid_displacement_binary(
            image, pixels_per_mm, node_spacing=node_spacing,
        )
    except ValueError as error:
        if str(error) != 'Major-grid seed spacing is inconsistent with calibration.':
            raise
        evidence, observations = _detect_grid_displacement_binary(
            image, pixels_per_mm, node_spacing=node_spacing, refine_seeds=True,
        )
    if evidence['reason'] != 'Rows disagree with shared displacement by more than one pixel.':
        return evidence, observations
    return _observe_chromatic_ridges(image, pixels_per_mm, evidence, observations)


def source_grid_rhythm_coordinates(
    image: np.ndarray,
    columns: np.ndarray,
    path: np.ndarray,
    validity: np.ndarray,
    *,
    source_interval: list[int] | tuple[int, int],
    pixels_per_mm: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map supported rhythm points to paper coordinates, retaining source paths.

    Unsupported geometry returns the original path with an explicit reason.
    Neither source gaps nor off-canvas contributors to the legacy median origin
    are mapped or extrapolated. Source QA and overlays must use the input path.
    """
    if (
        image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
        or columns.ndim != 1 or path.shape != columns.shape
        or validity.shape != columns.shape or validity.dtype != np.bool_
        or not np.issubdtype(columns.dtype, np.integer)
        or not np.issubdtype(path.dtype, np.integer)
        or columns.size == 0 or np.any(np.diff(columns) != 1)
        or np.any(columns < 0) or np.any(columns >= image.shape[1])
        or np.any(path < 0) or np.any(path >= image.shape[0])
        or len(source_interval) != 2
        or any(type(value) is not int for value in source_interval)
        or not 0 <= source_interval[0] < source_interval[1] <= image.shape[1]
    ):
        raise ValueError("Grid conversion requires valid original source coordinates.")
    lo, hi = source_interval
    receipt = {
        "version": 1, "method": "source-grid-rhythm-coordinate-conversion-v1",
        "decodedRasterSha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "imageSize": [image.shape[1], image.shape[0]],
        "sourceInterval": list(source_interval),
        "sourcePathChanged": False, "sourceTimingChanged": False,
        "validityChanged": False, "sourceImageChanged": False,
        "extrapolationUsed": False, "truthUsed": False,
    }

    def unavailable(reason: str) -> tuple[np.ndarray, dict[str, Any]]:
        return path, {**receipt, "state": "unavailable", "reason": reason}

    try:
        geometry, observations = detect_grid_displacement(image[:, lo:hi], pixels_per_mm)
    except ValueError as error:
        return unavailable(str(error))
    if geometry["state"] != "supported":
        return unavailable(geometry["reason"])
    mapped = validity & (columns >= lo) & (columns < hi)
    if not np.any(mapped):
        return unavailable("No valid recorded rhythm coordinates.")
    nodes = observations["xNodes"]
    if nodes[0] != 0 or nodes[-1] != hi-lo-1:
        return unavailable("Grid nodes do not cover the verified recording interval.")
    shift = np.zeros(columns.size)
    shift[mapped] = np.interp(columns[mapped]-lo, nodes, observations["displacement"])
    corrected = path.astype(float)-shift
    lower, upper = geometry["referenceBoundsY"]
    if np.any((corrected[mapped] < lower) | (corrected[mapped] > upper)):
        return unavailable("Rhythm coordinates lie outside independently observed grid rows.")
    inverse_error = float(np.max(np.abs(corrected[mapped]+shift[mapped]-path[mapped])))
    if inverse_error > 1e-12:
        return unavailable("Grid conversion does not reconstruct original source coordinates.")
    visible = ~(np.isfinite(observations["occludedColumnFraction"])
                & (observations["occludedColumnFraction"] > .1))
    observed = np.isfinite(observations["observedRowsAll"])
    array_hash = lambda values, dtype: hashlib.sha256(
        np.asarray(values, dtype=dtype).tobytes()
    ).hexdigest()
    receipt.update({
        "state": "applied", "grid": {
            **geometry,
            "xNodes": (nodes+lo).tolist(),
            "observedGridY": observations["observedRows"].tolist(),
            "referenceGridY": observations["referenceRows"].tolist(),
            "displacementAtNodes": observations["displacement"].tolist(),
            "seedReferenceBoundsY": [
                float(observations["seedRows"][0]) if "seedRefinedRows" in observations
                else int(observations["seedRows"][0]),
                float(observations["seedRows"][-1]) if "seedRefinedRows" in observations
                else int(observations["seedRows"][-1]),
            ],
            "eligibleRowsPerNode": visible.sum(axis=0).tolist(),
            "observedRowsPerNode": observed.sum(axis=0).tolist(),
        },
        "mapping": "paper_y = source_y - displacement(source_x)",
        "interpolation": "piecewise-linear-no-extrapolation",
        "columnsSha256": array_hash(columns, "<i4"),
        "sourcePathSha256": array_hash(path, "<i4"),
        "paperPathSha256": array_hash(corrected, "<f8"),
        "validitySha256": array_hash(validity, "u1"),
        "mappedSourceColumns": int(np.count_nonzero(mapped)),
        "unmappedValidOffCanvasColumns": int(np.count_nonzero(validity & ~mapped)),
        "originalMedianPixels": float(np.median(path[validity])),
        "paperMedianPixels": float(np.median(corrected[validity])),
        "sourceInverseMaximumErrorPixels": inverse_error,
        "pathEncoding": "float64-le-paper-y-by-source-column-v1",
    })
    if "seedRefinedRows" in observations:
        receipt["grid"]["seedObservation"] = {
            "version": 1,
            "method": "bounded-half-height-weighted-source-profile-v1",
            "integerSpacingRefused": True,
            "originalRows": observations["seedOriginalRows"].tolist(),
            "refinedRows": observations["seedRefinedRows"].tolist(),
            "supportBounds": observations["seedSupportBounds"].tolist(),
            "sourceProfileSha256": array_hash(observations["seedProfile"], "<f4"),
            "sourceProfileLength": image.shape[0],
        }
    if "binaryObservedRowsAll" in observations:
        receipt["grid"]["ridgePositionEvidence"] = {
            "version": 1,
            "method": "positive-source-channel-range-within-unchanged-half-height-support-v1",
            "binarySupportUnchanged": True,
            "sourceWindowWidth": 65,
            "channelRangeMinimum": 18,
            "brightnessMinimum": 150,
            "weightNormalization": "subtract-local-profile-minimum",
            "observedPositionCount": int(observed.sum()),
            "supportShape": [geometry["seedRowCount"], geometry["nodeCount"], 2],
            "supportBoundsSha256": array_hash(observations["ridgeSupportBounds"], "<i4"),
            "windowBoundsSha256": array_hash(observations["ridgeWindowBounds"], "<i4"),
            "sourceProfilesSha256": array_hash(observations["chromaticProfiles"], "<f4"),
            "binaryPositionsSha256": array_hash(observations["binaryObservedRowsAll"], "<f8"),
            "positionsSha256": array_hash(observations["observedRowsAll"], "<f8"),
            "finiteMaskSha256": array_hash(observed, "u1"),
        }
    return corrected, receipt
