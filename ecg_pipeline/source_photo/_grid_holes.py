"""Bounded enclosed coordinate holes; every vertex is an existing observation."""
import collections, math
import shapely
from shapely.geometry import Polygon

def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

def extend_holes(mesh, major_indices, gap_eligible):
    vertices = mesh['vertices']
    known = {tuple(v['source']): i for i, v in enumerate(vertices)}
    assert len(known) == len(vertices) and len(gap_eligible) == len(major_indices) - 1
    shapes = [Polygon([vertices[i]['source'] for i in t['vertices']]) for t in mesh['triangles']]
    union = shapely.union_all(shapes)
    assert union.is_valid
    components = [union] if union.geom_type == 'Polygon' else list(union.geoms)
    assert all((poly.geom_type == 'Polygon' for poly in components))
    edge_counts = collections.Counter((tuple(sorted((a, b))) for t in mesh['triangles'] for a, b in zip(t['vertices'], t['vertices'][1:] + t['vertices'][:1])))
    boundary_edges = [e for e, n in edge_counts.items() if n == 1]
    boundary_vertices = sorted(set((i for e in boundary_edges for i in e)))
    out = []
    for component_id, component in enumerate(components):
        for ring in component.interiors:
            raw = list(ring.coords)[:-1]
            assert all((tuple(q) in known for q in raw))
            ids = []
            for a, b in zip(raw, raw[1:] + raw[:1]):
                dx, dy = (b[0] - a[0], b[1] - a[1])
                den = dx * dx + dy * dy
                assert den > 0
                along = []
                for i in boundary_vertices:
                    q = vertices[i]['source']
                    t = ((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / den
                    if cross(a, b, q) == 0 and 0 <= t < 1:
                        along.append((t, i))
                assert along and min(along)[1] == known[tuple(a)]
                ids.extend((i for _, i in sorted(along)))
            assert len(set(ids)) == len(ids)
            coordinates = [vertices[i]['source'] for i in ids]
            hole = Polygon(coordinates)
            assert hole.is_valid and (not hole.interiors)
            assert hole.equals(Polygon(raw))
            xs = [q[0] for q in coordinates]
            physical = [vertices[i]['paperYmm'] for i in ids]
            lo, hi = (min(physical), max(physical))
            relevant = [i for i, (a, b) in enumerate(zip(major_indices, major_indices[1:])) if 5 * a < hi and 5 * b > lo]
            reasons = []
            if max(xs) - min(xs) > 128:
                reasons.append('enclosed-hole-width-exceeds128pixels')
            if not relevant or not all((gap_eligible[i] for i in relevant)):
                reasons.append('unsupported-physical-grid-identity')
            result = {'componentId': component_id, 'boundaryVertexIds': ids, 'sourceBounds': list(hole.bounds), 'sourceAreaPixelsSquared': hole.area, 'physicalYRangeMm': [lo, hi], 'interveningGapIndices': relevant, 'rejectionReasons': reasons, 'triangles': [], 'accepted': False}
            if not reasons:
                triangles = []
                for shape in shapely.constrained_delaunay_triangles(hole).geoms:
                    points = list(shape.exterior.coords)[:-1]
                    assert len(points) == 3 and all((tuple(q) in known for q in points))
                    ti = [known[tuple(q)] for q in points]
                    if cross(*points) < 0:
                        ti[1], ti[2] = (ti[2], ti[1])
                        points[1], points[2] = (points[2], points[1])
                    det = cross(*points)
                    assert det > 0
                    values = [vertices[i]['paperYmm'] for i in ti]
                    dx1, dy1 = (points[1][0] - points[0][0], points[1][1] - points[0][1])
                    dx2, dy2 = (points[2][0] - points[0][0], points[2][1] - points[0][1])
                    gx = ((values[1] - values[0]) * dy2 - (values[2] - values[0]) * dy1) / det
                    gy = (dx1 * (values[2] - values[0]) - dx2 * (values[1] - values[0])) / det
                    tr = {'vertices': ti, 'areaPixelsSquared': det / 2, 'gradientMmPerPixel': [gx, gy], 'physicalYSpanMm': max(values) - min(values), 'sourceXSpanPixels': max((q[0] for q in points)) - min((q[0] for q in points)), 'positiveSupportedScale': bool(gy > 0 and 2 <= 1 / gy <= 40)}
                    tr['passed'] = bool(tr['positiveSupportedScale'] and 0 < tr['physicalYSpanMm'] <= 20 and (tr['sourceXSpanPixels'] <= 128))
                    triangles.append(tr)
                tris = shapely.union_all([Polygon([vertices[i]['source'] for i in t['vertices']]) for t in triangles])
                used = {i for t in triangles for i in t['vertices']}
                assert hole.symmetric_difference(tris).area <= 1e-07 and abs(sum((t['areaPixelsSquared'] for t in triangles)) - hole.area) <= 1e-07
                assert tris.intersection(union).area <= 1e-07
                if not set(ids) <= used:
                    reasons.append('observed-boundary-knot-omitted')
                if not all((t['passed'] for t in triangles)):
                    reasons.append('unsupported-triangle-scale-or-span')
                tedges = collections.Counter((tuple(sorted((a, b))) for t in triangles for a, b in zip(t['vertices'], t['vertices'][1:] + t['vertices'][:1])))
                wanted = {tuple(sorted(e)) for e in zip(ids, ids[1:] + ids[:1])}
                if {e for e, n in tedges.items() if n == 1} != wanted:
                    reasons.append('boundary-edge-mismatch')
                result['triangles'] = triangles
                result['accepted'] = not reasons
            out.append(result)
    return {'method': 'enclosed-observed-boundary-triangulation-v1', 'sourceComponentCount': len(components), 'sourceAreaPixelsSquared': union.area, 'enclosedHoleCount': len(out), 'holes': out, 'acceptedHoleCount': sum((q['accepted'] for q in out)), 'maximumHoleWidthPixels': 128, 'maximumTrianglePhysicalYSpanMm': 20, 'minimumPixelsPerMm': 2, 'maximumPixelsPerMm': 40, 'inventedSourceVertices': 0, 'oldMeshUnchanged': True, 'openBoundaryExtrapolation': False, 'newWaveformSamples': 0, 'shapelyVersion': shapely.__version__, 'geosVersion': shapely.geos_version_string}
