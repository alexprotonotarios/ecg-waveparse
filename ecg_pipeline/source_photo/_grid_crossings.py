"""Conservative source-path crossings of a deskewed candidate ridge band."""
import math

def crossings(receipt, paths, crop_offset, x_nodes, slope, radius=1.0):
    xlo, xhi = receipt['profileXRange']
    xmin = crop_offset + xlo - 0.5 - radius
    xmax = crop_offset + xhi - 0.5 + radius
    a, b = receipt['bandInNodeCoordinates']
    dy = radius * (1 + abs(slope))
    ymin, ymax = (a - 0.5 - dy, b - 0.5 + dy)
    node_x = crop_offset + x_nodes[receipt['node']]
    hits = []
    for path in paths:
        left, right = path['range']
        ys, valid = (path['pathY'], path['valid'])
        assert right - left == len(ys) == len(valid)
        # Samples and adjacent chords outside this profile's x interval cannot
        # intersect it. Keep a two-column margin and the original scalar tests
        # so endpoint rounding, hit ordering and all source gaps remain exact.
        if (isinstance(left, int) and abs(left) + len(ys) < 2**40
                and all(math.isfinite(v) and abs(v) < 2**40 for v in (xmin, xmax))):
            start = max(0, math.floor(xmin - left) - 2)
            stop = min(len(ys), math.ceil(xmax - left) + 3)
        else:
            start, stop = 0, len(ys)
        for i in range(start, stop):
            y, ok = ys[i], valid[i]
            if not ok or y is None or (not math.isfinite(y)):
                continue
            x = left + i
            z = y - slope * (x - node_x)
            if xmin <= x <= xmax and ymin <= z <= ymax:
                hits.append({'lead': path['lead'], 'kind': 'sample', 'indices': [i], 'sourceX': [x], 'sourceY': [y]})
            if i + 1 >= len(ys) or not valid[i + 1]:
                continue
            ynext = ys[i + 1]
            if ynext is None or not math.isfinite(ynext):
                continue
            t0, t1 = (max(0.0, xmin - x), min(1.0, xmax - x))
            if t0 > t1:
                continue
            znext = ynext - slope * (x + 1 - node_x)
            za, zb = (z + t0 * (znext - z), z + t1 * (znext - z))
            if min(za, zb) <= ymax and max(za, zb) >= ymin:
                hits.append({'lead': path['lead'], 'kind': 'adjacent-valid-chord', 'indices': [i, i + 1], 'sourceX': [x, x + 1], 'sourceY': [y, ynext], 'xClip': [t0, t1]})
    return {'key': [receipt['row'], receipt['node']], 'profileSourceBounds': [xmin, xmax], 'deskewedBandBounds': [ymin, ymax], 'supportRadiusPixels': radius, 'crossesKnownSourcePath': bool(hits), 'hits': hits}
