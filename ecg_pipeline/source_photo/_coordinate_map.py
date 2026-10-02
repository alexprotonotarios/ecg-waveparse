"""Preserve original points; add only verified enclosed domains and full-chord support."""
from shapely.geometry import Polygon, Point

def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

def mapper_class(original_class, mesh, extension, triangle_value):
    vertices = mesh['vertices']
    accepted = [(i, h, Polygon([vertices[j]['source'] for j in h['boundaryVertexIds']])) for i, h in enumerate(extension['holes']) if h['accepted']]

    class ExtendedMap(original_class):

        def point(self, source_x, y):
            old = super().point(source_x, y)
            if old[0] is not None or old[2] != 'no-supported-cell':
                return old
            x = source_x - self.offset
            hits = []
            for i, h, polygon in accepted:
                if not polygon.covers(Point(x, y)):
                    continue
                values = [value for t in h['triangles'] if (value := triangle_value(mesh, t, x, y)) is not None]
                domain_id = len(self.cells) + i
                if not values:
                    return (None, [domain_id], 'unresolved-hole-interior')
                if max(values) - min(values) > 1e-12:
                    return (None, [domain_id], 'inconsistent-hole-triangle-boundary')
                hits.append((domain_id, values[0]))
            if not hits:
                return old
            if max((q[1] for q in hits)) - min((q[1] for q in hits)) > 1e-12:
                return (None, [q[0] for q in hits], 'inconsistent-hole-domain-boundary')
            return (hits[0][1], [q[0] for q in hits], None)

        def connection(self, x0, y0, x1, y1):
            old_ok, old_intervals = super().connection(x0, y0, x1, y1)
            if old_ok:
                return (old_ok, old_intervals)
            a = [x0 - self.offset, y0]
            b = [x1 - self.offset, y1]
            intervals = list(old_intervals)
            for i, h, polygon in accepted:
                for triangle in h['triangles']:
                    pts = [vertices[j]['source'] for j in triangle['vertices']]
                    lo, hi = (0.0, 1.0)
                    for u, v in zip(pts, pts[1:] + pts[:1]):
                        start, end = (cross(u, v, a), cross(u, v, b))
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
                        intervals.append([float(lo), float(hi), len(self.cells) + i])
            extent = 0.0
            for lo, hi, _ in sorted(intervals):
                if lo > extent + 1e-12:
                    return (False, intervals)
                extent = max(extent, hi)
            return (bool(extent >= 1.0 - 1e-12), intervals)
    return ExtendedMap
