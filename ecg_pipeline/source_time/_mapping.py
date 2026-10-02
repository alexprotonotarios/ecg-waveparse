"""Map captured native contributors through observed source-grid cells."""
import numpy as np
from . import _field


def clipped_interval(cell, start, end, crop):
    u0, v0 = start[1] - crop[1], start[0] - crop[0]
    du, dv = end[1] - start[1], end[0] - start[0]
    x0, x1 = cell['x']
    upper_slope = (cell['upperY'][1] - cell['upperY'][0]) / (x1 - x0)
    lower_slope = (cell['lowerY'][1] - cell['lowerY'][0]) / (x1 - x0)
    inequalities = [(u0 - x0, du), (x1 - u0, -du),
                    (v0 - cell['upperY'][0] - upper_slope * (u0 - x0), dv - upper_slope * du),
                    (cell['lowerY'][0] + lower_slope * (u0 - x0) - v0, lower_slope * du - dv)]
    lo, hi = 0., 1.
    for offset, slope in inequalities:
        if slope == 0:
            if offset < 0: return None
        elif slope > 0:
            lo = max(lo, -offset / slope)
        else:
            hi = min(hi, -offset / slope)
        if hi < lo: return None
    return lo, hi


def segment_supported(cells, start, end, crop, component):
    intervals = []
    for cell in cells:
        if cell['supportedComponent'] != component:
            continue
        if max(start[1], end[1]) - crop[1] < cell['x'][0] or min(start[1], end[1]) - crop[1] > cell['x'][1]:
            continue
        interval = clipped_interval(cell, start, end, crop)
        if interval is not None: intervals.append(interval)
    covered = 0.
    for lo, hi in sorted(intervals):
        if lo > covered + 1e-12: return False
        covered = max(covered, hi)
    return bool(covered >= 1. - 1e-12)


def map_publication(grid, arrays, published):
    """A publication can remove raw samples; it cannot silently move/change them."""
    raw = arrays['canonicalUv']; native = arrays['nativeSourceXY']
    if (raw.shape != (12, 5000) or published.shape != raw.shape
            or native.ndim != 3 or native.shape[0] != 4 or native.shape[2] != 2
            or not np.array_equal(np.isfinite(raw), arrays['canonicalFinite'])):
        raise ValueError('invalid_source_time_capture_shape')
    mask = np.isfinite(published)
    if not np.array_equal(raw[mask], published[mask]):
        raise ValueError('publication_values_do_not_match_captured_decoder')
    cells, crop = grid['preparedCells'], grid['cropBox']
    mm = np.full(native.shape[:2], np.nan)
    components = np.full(mm.shape, -1, np.int32)
    for row, index in zip(*np.where(np.isfinite(native).all(2))):
        x, y = map(float, native[row, index])
        hits = [(_field.forward(cell, y - crop[1], x - crop[0]), cell['supportedComponent']) for cell in cells]
        hits = [(value, component) for value, component in hits if value is not None]
        if hits:
            if len({c for _, c in hits}) != 1 or max(v for v, _ in hits) - min(v for v, _ in hits) >= 1e-10:
                components[row, index] = -2
                continue
            mm[row, index], components[row, index] = hits[0]
    edges = np.zeros((native.shape[0], native.shape[1] - 1), bool)
    for row, index in zip(*np.where(np.isfinite(mm[:, :-1]) & np.isfinite(mm[:, 1:]) & (components[:, :-1] == components[:, 1:]))):
        edges[row, index] = segment_supported(cells, native[row, index], native[row, index + 1], crop, int(components[row, index]))
    if np.any(edges & (np.diff(mm, axis=1) <= 0)):
        raise ValueError('non_increasing_native_source_time')
    paper = np.full(raw.shape, np.nan); canonical_components = np.full(raw.shape, -1, np.int32)
    for key in ['rowIndices', 'nativeLeftIndices', 'nativeRightIndices', 'rightWeights']:
        if arrays[key].shape != raw.shape: raise ValueError('invalid_source_time_contributor_shape')
        if key != 'rightWeights' and arrays[key].dtype.kind not in 'iu':
            raise ValueError('source_time_contributor_indices_must_be_integers')
    for lead, sample in zip(*np.where(mask)):
        row, left, right = [int(arrays[key][lead, sample]) for key in ['rowIndices', 'nativeLeftIndices', 'nativeRightIndices']]
        weight = float(arrays['rightWeights'][lead, sample])
        if (not 0 <= row < 4 or not 0 <= left < native.shape[1] or not 0 <= right < native.shape[1]
                or not np.isfinite(weight) or not 0 <= weight <= 1 or (weight > 0 and right != left + 1)):
            raise ValueError('invalid_source_time_native_contributors')
        if not np.isfinite(mm[row, left]) or (weight > 0 and (not np.isfinite(mm[row, right]) or components[row, left] != components[row, right] or not edges[row, left])):
            continue
        paper[lead, sample] = mm[row, left] if weight == 0 else mm[row, left] * (1 - weight) + mm[row, right] * weight
        canonical_components[lead, sample] = components[row, left]
    connected = (np.isfinite(paper[:, :-1]) & np.isfinite(paper[:, 1:])
                 & (canonical_components[:, :-1] == canonical_components[:, 1:])
                 & (arrays['rowIndices'][:, :-1] == arrays['rowIndices'][:, 1:]))
    # Canonical downsampling must not jump over an unsupported native edge even
    # when both returned endpoint coordinates lie in the same field component.
    for lead, sample in zip(*np.where(connected)):
        row = int(arrays['rowIndices'][lead, sample])
        first = int(arrays['nativeLeftIndices'][lead, sample])
        last = int(arrays['nativeRightIndices'][lead, sample + 1] if arrays['rightWeights'][lead, sample + 1] > 0
                   else arrays['nativeLeftIndices'][lead, sample + 1])
        connected[lead, sample] = last >= first and bool(np.all(edges[row, first:last]))
    return {'nativePaperXmm': mm, 'nativeComponent': components, 'nativeEdgeSupported': edges,
            'publishedPaperXmm': paper, 'publishedComponent': canonical_components, 'connectedAdjacent': connected}
