"""Conforming vertical paper coordinates from existing supported cell corners."""
import math

def build_mesh(cells):
    registry = {}
    for ci, cell in enumerate(cells):
        assert cell['x'][0] < cell['x'][1]
        assert cell['paperYmm'][0] < cell['paperYmm'][1]
        for side, x in enumerate(cell['x']):
            for end, ys in enumerate([cell['upperY'], cell['lowerY']]):
                y, mm = (ys[side], cell['paperYmm'][end])
                assert all((math.isfinite(v) for v in [x, y, mm]))
                key = (x, mm)
                if key in registry:
                    assert registry[key]['source'][1] == y, 'Conflicting observed corner.'
                else:
                    registry[key] = {'source': [x, y], 'paperYmm': mm, 'sourceCorners': []}
                registry[key]['sourceCorners'].append([ci, side, end])
    vertices = [registry[k] for k in sorted(registry)]
    by_x = {}
    for i, vertex in enumerate(vertices):
        by_x.setdefault(vertex['source'][0], []).append(i)
    for ids in by_x.values():
        assert all((vertices[b]['source'][1] > vertices[a]['source'][1] for a, b in zip(ids[:-1], ids[1:]))), 'Non-monotone observed edge.'
    triangles, partitions = ([], [])
    for ci, cell in enumerate(cells):
        lo, hi = cell['paperYmm']
        chains = [[i for i in by_x[x] if lo <= vertices[i]['paperYmm'] <= hi] for x in cell['x']]
        assert all((len(c) >= 2 for c in chains))
        for side, ids in enumerate(chains):
            assert vertices[ids[0]]['source'][1] == cell['upperY'][side]
            assert vertices[ids[-1]]['source'][1] == cell['lowerY'][side]
        left, right = chains
        il = ir = 0
        tids = []
        area = 0.0
        while il + 1 < len(left) or ir + 1 < len(right):
            advance_left = ir + 1 == len(right) or (il + 1 < len(left) and vertices[left[il + 1]]['paperYmm'] <= vertices[right[ir + 1]]['paperYmm'])
            ids = [left[il], right[ir], left[il + 1] if advance_left else right[ir + 1]]
            if advance_left:
                il += 1
            else:
                ir += 1
            pts = [vertices[i]['source'] for i in ids]
            mm = [vertices[i]['paperYmm'] for i in ids]
            dx1, dy1 = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
            dx2, dy2 = (pts[2][0] - pts[0][0], pts[2][1] - pts[0][1])
            det = dx1 * dy2 - dx2 * dy1
            assert det > 0
            gx = ((mm[1] - mm[0]) * dy2 - (mm[2] - mm[0]) * dy1) / det
            gy = (dx1 * (mm[2] - mm[0]) - dx2 * (mm[1] - mm[0])) / det
            assert gy > 0 and 2 <= 1 / gy <= 40, 'Unsupported triangle vertical scale.'
            tids.append(len(triangles))
            area += det / 2
            triangles.append({'cellId': ci, 'vertices': ids, 'gradientMmPerPixel': [gx, gy], 'areaPixelsSquared': det / 2})
        expected = (cell['x'][1] - cell['x'][0]) * sum((b - a for a, b in zip(cell['upperY'], cell['lowerY']))) / 2
        assert math.isclose(area, expected, rel_tol=1e-12, abs_tol=1e-08)
        partitions.append({'cellId': ci, 'leftVertices': left, 'rightVertices': right, 'triangleIds': tids, 'areaPixelsSquared': area})
    return {'vertices': vertices, 'triangles': triangles, 'cellPartitions': partitions, 'inventedSourceVertices': 0, 'originalCellCount': len(cells), 'originalCellAreaPixelsSquared': sum((q['areaPixelsSquared'] for q in partitions))}

def triangle_value(mesh, triangle, x, y):
    a, b, c = [mesh['vertices'][i]['source'] for i in triangle['vertices']]
    dx, dy = (x - a[0], y - a[1])
    ux, uy = (b[0] - a[0], b[1] - a[1])
    vx, vy = (c[0] - a[0], c[1] - a[1])
    det = ux * vy - vx * uy
    u = (dx * vy - vx * dy) / det
    v = (ux * dy - dx * uy) / det
    if u < -1e-12 or v < -1e-12 or u + v > 1 + 1e-12:
        return None
    origin = mesh['vertices'][triangle['vertices'][0]]['paperYmm']
    gx, gy = triangle['gradientMmPerPixel']
    return origin + gx * dx + gy * dy

def mapper_class(base_class, mesh):

    class ConformingMap(base_class):

        def point(self, source_x, y):
            x = source_x - self.offset
            hits = []
            for ci in self.candidates(source_x):
                cell = self.cells[ci]
                left, right = cell['x']
                t = (x - left) / (right - left)
                upper = cell['upperY'][0] * (1 - t) + cell['upperY'][1] * t
                lower = cell['lowerY'][0] * (1 - t) + cell['lowerY'][1] * t
                if not upper <= y <= lower:
                    continue
                values = []
                for ti in mesh['cellPartitions'][ci]['triangleIds']:
                    value = triangle_value(mesh, mesh['triangles'][ti], x, y)
                    if value is not None:
                        values.append(value)
                if not values:
                    return (None, [ci], 'unresolved-mesh-interior')
                if max(values) - min(values) > 1e-12:
                    return (None, [ci], 'inconsistent-triangle-boundary')
                hits.append((ci, float(values[0])))
            if not hits:
                return (None, [], 'no-supported-cell')
            if max((q[1] for q in hits)) - min((q[1] for q in hits)) > 1e-12:
                return (None, [q[0] for q in hits], 'inconsistent-mesh-boundary')
            return (hits[0][1], [q[0] for q in hits], None)
    return ConformingMap
