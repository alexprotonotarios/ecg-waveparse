"""Generated from the frozen observer; only two mask overrides and error receipts."""
from typing import Any
import numpy as np
import cv2
from scipy.signal import find_peaks

class ObservationError(ValueError):

    def __init__(self, reason, state):
        super().__init__(reason)
        self.observations = {key: state[key] for key in ['accepted', 'slope', 'coarse_spread', 'period', 'peaks', 'periods', 'multiples', 'nodes', 'observed', 'strengths', 'occluded', 'seed_calibration_receipt'] if key in state}

def observe_grid(image: np.ndarray, pixels_per_mm: float, *, colour_mask=None, dark_mask=None, seed_calibration=None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ObservationError('Expected original BGR uint8 raster.', locals())
    if not np.isfinite(pixels_per_mm) or not 2 <= pixels_per_mm <= 40:
        raise ObservationError('Unsupported independently measured grid calibration.', locals())
    height, width = image.shape[:2]
    if min(height, width) < 200 or image.size > 36000000:
        raise ObservationError('Unsupported image bounds.', locals())
    channels = None
    if colour_mask is None:
        channels = image.astype(np.int16)
        mask = ((channels.max(axis=2) - channels.min(axis=2) >= 18) & (channels.max(axis=2) >= 150)).astype(np.uint8)
    else:
        assert colour_mask.dtype == np.uint8 and colour_mask.shape == (height, width)
        mask = colour_mask.copy()
    proposal = cv2.morphologyEx(mask * 255, cv2.MORPH_CLOSE, np.ones((1, 9), np.uint8))
    lines = cv2.HoughLinesP(proposal, 1, np.pi / 1800, threshold=max(40, width // 5), minLineLength=max(80, width // 4), maxLineGap=12)
    accepted = []
    if lines is not None:
        for x0, y0, x1, y1 in lines[:, 0]:
            if x0 > x1:
                x0, y0, x1, y1 = (x1, y1, x0, y0)
            if x1 - x0 < width / 4:
                continue
            slope = (int(y1) - int(y0)) / (int(x1) - int(x0))
            if abs(slope) <= np.tan(np.deg2rad(2)):
                accepted.append((x0, y0, x1, y1))
    if len(accepted) < 8:
        raise ObservationError('Fewer than eight long coloured lines.', locals())
    accepted = np.asarray(accepted, np.int32)
    slopes = (accepted[:, 3] - accepted[:, 1]) / (accepted[:, 2] - accepted[:, 0])
    slope = float(np.median(slopes))
    coarse_spread = float(np.median(np.abs(slopes - slope)) * (width - 1))
    period = 5 * float(pixels_per_mm)
    field = mask.astype(np.float32)
    if dark_mask is None:
        if channels is None:
            channels = image.astype(np.int16)
        dark = (channels.max(axis=2) < 150).astype(np.float32)
    else:
        assert dark_mask.dtype == np.float32 and dark_mask.shape == (height, width)
        dark = dark_mask.copy()

    def profile(node, *, return_dark=False):
        xs = np.arange(max(0, node - 32), min(width, node + 33), dtype=np.float32)
        yy = np.arange(height, dtype=np.float32)[:, None] + slope * (xs - node)[None, :]
        xx = np.broadcast_to(xs, yy.shape).copy()
        values = cv2.remap(field, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).mean(axis=1)
        if return_dark:
            return (values, cv2.remap(dark, xx, yy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT))
        return values
    mid = (width - 1) // 2
    seed = profile(mid)
    peaks, _ = find_peaks(seed, height=0.55, prominence=0.25, distance=max(3, int(0.6 * period)))
    margin = abs(slope) * (width - 1) / 2 + period
    peaks = peaks[(peaks >= margin) & (peaks < height - margin)]
    if len(peaks) < 8:
        raise ObservationError('Fewer than eight major-grid seeds.', locals())
    seed_calibration_receipt = None
    if seed_calibration is not None:
        seed_calibration_receipt = seed_calibration(peaks)
        peaks = np.asarray(seed_calibration_receipt['retainedSeeds'], np.int32)
        if seed_calibration_receipt['state'] != 'supported':
            raise ObservationError('Local source-window grid calibration is unresolved.', locals())
    periods = np.diff(peaks)
    multiples = np.rint(periods / period)
    if seed_calibration is None and np.mean((multiples >= 1) & (multiples <= 4) & (np.abs(periods - multiples * period) <= 2)) < 0.9:
        raise ObservationError('Major-grid seed spacing is inconsistent with calibration.', locals())
    nodes = np.unique(np.append(np.arange(0, width, 64), width - 1)).astype(float)
    observed = np.full((len(peaks), len(nodes)), np.nan)
    strengths = np.zeros(observed.shape)
    occluded = np.full(observed.shape, np.nan)
    support = []
    all_seed_support = []
    for j, node in enumerate(nodes):
        values, dark_strip = profile(int(node), return_dark=True)
        expected = peaks + slope * (node - mid)
        for i, location in enumerate(expected):
            center = int(round(location))
            radius = max(3, int(np.floor(0.4 * period)))
            lo, hi = (max(0, center - radius), min(height, center + radius + 1))
            local = values[lo:hi]
            if len(local) < 3 or local.max() < 0.55 or local.max() - local.min() < 0.25:
                continue
            at = int(np.argmax(local))
            if at == 0 or at == len(local) - 1:
                continue
            a, b = (at, at + 1)
            cutoff = local.min() + 0.5 * (local.max() - local.min())
            while a > 0 and local[a - 1] >= cutoff:
                a -= 1
            while b < len(local) and local[b] >= cutoff:
                b += 1
            if a == 0 or b == len(local):
                continue
            occluded[i, j] = float(np.any(dark_strip[lo + a:lo + b] > 0.5, axis=0).mean())
            if occluded[i, j] > 0.1:
                continue
            weights = np.maximum(0, local[a:b] - local.min())
            observed[i, j] = np.average(np.arange(lo + a, lo + b), weights=weights)
            strengths[i, j] = local.max()
        valid = np.isfinite(observed[:, j])
        eligible = ~(np.isfinite(occluded[:, j]) & (occluded[:, j] > 0.1))
        all_seed_support.append(float(valid.mean()))
        support.append(float(valid.sum() / eligible.sum()) if eligible.any() else 0.0)
    result = {'coarseLines': accepted, 'xNodes': nodes, 'seedRows': peaks, 'observedRowsAll': observed, 'strengths': strengths, 'occludedColumnFraction': occluded}
    reason = None
    complete = np.isfinite(observed).all(axis=1)
    if min(support) < 0.8:
        reason = 'Less than 80 percent row support at a node.'
    elif complete.sum() < 8:
        reason = 'Fewer than eight complete grid rows.'
    elif np.ptp(observed[complete, 0]) < 0.75 * np.ptp(peaks):
        reason = 'Complete rows cover less than 75 percent of the seeded vertical span.'
    else:
        rows = observed[complete]
        row_delta = rows - rows[:, 0, None]
        displacement = np.median(row_delta, axis=0)
        residual = float(np.max(np.abs(row_delta - displacement)))
        if residual > 1:
            reason = 'Rows disagree with shared displacement by more than one pixel.'
        elif np.min(np.diff(rows, axis=0)) <= 0:
            reason = 'Grid rows cross.'
        elif np.max(np.abs(np.diff(displacement) / np.diff(nodes))) > 0.25:
            reason = 'Local displacement slope exceeds existing budget.'
        else:
            result.update(observedRows=rows, displacement=displacement, referenceRows=rows[:, 0])
    evidence = {'state': 'unresolved' if reason else 'supported', 'reason': reason, 'method': 'unoccluded-major-grid-ridges-v1', 'coarseSlope': slope, 'coarseEndpointMadPixels': coarse_spread, 'coarseLineCount': len(accepted), 'majorPeriodPixels': period, 'seedRowCount': len(peaks), 'completeRowCount': int(complete.sum()), 'nodeCount': len(nodes), 'minimumNodeSupportFraction': min(support), 'minimumAllSeedSupportFraction': min(all_seed_support), 'supportDenominator': 'Seeds not directly observed to be dark-occluded; failed or unavailable ridge observations remain in denominator', 'boundsX': [0, width - 1], 'candidateSignalOrTruthUsed': False, 'maximumAllowedOccludedColumnFraction': 0.1}
    if not reason:
        evidence.update(maxRowResidualPixels=residual, endpointDisplacementPixels=float(displacement[-1]), referenceBoundsY=[float(rows[0, 0]), float(rows[-1, 0])])
    if seed_calibration_receipt is not None:
        evidence['localSeedCalibration'] = seed_calibration_receipt
    return (evidence, result)
