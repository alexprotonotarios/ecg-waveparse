"""Diagnostic vertical calibration and gap-preserving sampling of fixed source paths."""
import bisect
import numpy as np

class CellMap:

    def __init__(self, cells, source_offset):
        self.cells = cells
        self.offset = source_offset
        self.edges = sorted(set((v for c in cells for v in c['x'])))
        self.by_left = {}
        for i, c in enumerate(cells):
            self.by_left.setdefault(c['x'][0], []).append(i)

    def candidates(self, source_x):
        x = source_x - self.offset
        at = bisect.bisect_right(self.edges, x) - 1
        result = []
        for j in [at - 1, at]:
            if 0 <= j < len(self.edges):
                result.extend((i for i in self.by_left.get(self.edges[j], []) if self.cells[i]['x'][0] <= x <= self.cells[i]['x'][1]))
        return sorted(set(result))

    def point(self, source_x, y):
        x = source_x - self.offset
        hits = []
        for i in self.candidates(source_x):
            c = self.cells[i]
            left, right = c['x']
            t = (x - left) / (right - left)
            upper = c['upperY'][0] * (1 - t) + c['upperY'][1] * t
            lower = c['lowerY'][0] * (1 - t) + c['lowerY'][1] * t
            if upper <= y <= lower:
                mm = c['paperYmm'][0] + (y - upper) / (lower - upper) * (c['paperYmm'][1] - c['paperYmm'][0])
                hits.append((i, float(mm)))
        if not hits:
            return (None, [], 'no-supported-cell')
        if max((q[1] for q in hits)) - min((q[1] for q in hits)) > 1e-12:
            return (None, [q[0] for q in hits], 'inconsistent-cell-boundary')
        return (hits[0][1], [q[0] for q in hits], None)

    def connection(self, x0, y0, x1, y1):
        """Require the entire source chord to lie in the union of valid cells."""
        local0, local1 = (x0 - self.offset, x1 - self.offset)
        intervals = []
        for i in sorted(set(self.candidates(x0) + self.candidates(x1))):
            c = self.cells[i]
            left, right = c['x']
            width = right - left

            def margins(x, y):
                t = (x - left) / width
                upper = c['upperY'][0] * (1 - t) + c['upperY'][1] * t
                lower = c['lowerY'][0] * (1 - t) + c['lowerY'][1] * t
                return [x - left, right - x, y - upper, lower - y]
            a = margins(local0, y0)
            b = margins(local1, y1)
            lo, hi = (0.0, 1.0)
            for start, end in zip(a, b):
                delta = end - start
                if delta == 0:
                    if start < 0:
                        hi = -1.0
                        break
                elif delta > 0:
                    lo = max(lo, -start / delta)
                else:
                    hi = min(hi, -start / delta)
            if lo <= hi:
                intervals.append([float(lo), float(hi), i])
        extent = 0.0
        for lo, hi, _ in sorted(intervals):
            if lo > extent + 1e-12:
                return (False, intervals)
            extent = max(extent, hi)
        return (bool(extent >= 1.0 - 1e-12), intervals)

def source_times(path, row_ranges, duration):
    x = np.arange(*path['range'], dtype=float)
    t = np.full(len(x), np.nan)
    columns = [path['column']] if path['column'] is not None else range(4)
    for column in columns:
        left, right = row_ranges[path['row']][column]
        use = (x >= left) & (x < right)
        t[use] = column * duration + (x[use] - left) / (right - left) * duration
    assert np.isfinite(t).all() and np.all(np.diff(t) > 0)
    return t

def sample_contiguous(times, values, valid, connections, target):
    """Linear resampling within connected observed spans, never across their gaps."""
    result = np.full(len(target), np.nan)
    segments = []
    start = None
    for i in range(len(valid) + 1):
        active = i < len(valid) and valid[i]
        join = active and start is not None and connections[i - 1]
        if start is not None and (not join):
            stop = i
            if stop - start >= 2:
                use = (target >= times[start]) & (target <= times[stop - 1])
                result[use] = np.interp(target[use], times[start:stop], values[start:stop])
                segments.append({'sourceIndices': [start, stop], 'timeBounds': [float(times[start]), float(times[stop - 1])], 'targetIndices': np.flatnonzero(use).tolist()})
            start = None
        if active and start is None:
            start = i
    return (result, segments)

def convert(path, cells, offset, row_ranges, *, duration=2.5, gain=10.0, sample_rate=500, mapper_factory=CellMap):
    assert duration > 0 and gain > 0 and (sample_rate > 0)
    mapper = mapper_factory(cells, offset)
    x = np.arange(*path['range'])
    y = np.asarray(path['pathY'], float)
    old_valid = np.asarray(path['valid'], bool)
    times = source_times(path, row_ranges, duration)
    assert len(x) == len(y) == len(old_valid)
    mm = np.full(len(x), np.nan)
    cell_ids = [[] for _ in x]
    reasons = []
    for i in range(len(x)):
        if not old_valid[i]:
            reasons.append('prior-source-gap')
            continue
        value, ids, reason = mapper.point(float(x[i]), float(y[i]))
        cell_ids[i] = ids
        reasons.append(reason)
        if value is not None:
            mm[i] = value
    valid = old_valid & np.isfinite(mm)
    connections = np.zeros(max(0, len(x) - 1), bool)
    broken = []
    for i in range(len(connections)):
        if not (valid[i] and valid[i + 1]):
            continue
        ok, intervals = mapper.connection(float(x[i]), float(y[i]), float(x[i + 1]), float(y[i + 1]))
        connections[i] = ok
        if not ok:
            broken.append({'sourceIndices': [i, i + 1], 'cellIntervals': intervals})
    baseline = float(np.median(mm[valid])) if valid.any() else None
    mv = (baseline - mm) / gain if baseline is not None else np.full(len(x), np.nan)
    target = np.arange(round(4 * duration * sample_rate)) / sample_rate
    canonical, segments = sample_contiguous(times, mv, valid, connections, target)
    expected = (target >= path['column'] * duration) & (target < (path['column'] + 1) * duration) if path['column'] is not None else np.ones(len(target), bool)
    return {'lead': path['lead'], 'row': path['row'], 'column': path['column'], 'range': path['range'], 'sourceTimesSeconds': times.tolist(), 'sourcePathY': path['pathY'], 'priorValid': path['valid'], 'mappedValid': valid.tolist(), 'paperYmm': mm.tolist(), 'centeredMv': mv.tolist(), 'cellIds': cell_ids, 'gapReasons': reasons, 'connectionValid': connections.tolist(), 'brokenConnections': broken, 'baselineMedianPaperYmm': baseline, 'gainMmPerMv': gain, 'canonicalSampleRate': sample_rate, 'canonicalMv': canonical.tolist(), 'segments': segments, 'sourceColumns': len(x), 'priorValidColumns': int(old_valid.sum()), 'mappedColumns': int(valid.sum()), 'newCoordinateGaps': int((old_valid & ~valid).sum()), 'oldGapsRecovered': int((valid & ~old_valid).sum()), 'canonicalExpectedSamples': int(expected.sum()), 'canonicalFiniteSamples': int(np.isfinite(canonical).sum()), 'canonicalCoverage': float(np.isfinite(canonical[expected]).mean()), 'fullPhysicalExtractionAccepted': False, 'baselineConvention': 'Median of mapped valid source columns; original paper coordinates retained.'}
