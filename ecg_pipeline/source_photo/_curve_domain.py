"""Exact-domain interval checks for inverse quadratic source chords."""
import math

def roots(a, b, c):
    if a == 0:
        return [] if b == 0 else [-c / b]
    d = b * b - 4 * a * c
    if d < 0:
        return []
    if d == 0:
        return [-b / (2 * a)]
    q = -0.5 * (b + math.copysign(math.sqrt(d), b))
    return [q / a, c / q]

def positive_intervals(margins):
    cuts = {0.0, 1.0}
    for a, b, c in margins:
        cuts.update((t for t in roots(a, b, c) if 0 < t < 1))
    cuts = sorted(cuts)
    result = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        t = (lo + hi) / 2
        if all(((a * t + b) * t + c >= 0 for a, b, c in margins)):
            result.append([lo, hi])
    return result

def bounds(a, b, c):
    values = [c, a + b + c]
    if a != 0:
        t = -b / (2 * a)
        if 0 < t < 1:
            values.append((a * t + b) * t + c)
    return (min(values), max(values))

def covered(intervals):
    extent = 0.0
    for lo, hi, *_ in sorted(intervals):
        if lo > extent + 1e-12:
            return False
        extent = max(extent, hi)
    return bool(extent >= 1.0 - 1e-12)

class CurveDomain:

    def __init__(self, cells, mesh, extension, offset):
        self.offset = offset
        self.domains = []
        for i, c in enumerate(cells):
            l, r = c['x']
            u0, u1 = c['upperY']
            v0, v1 = c['lowerY']
            us = (u1 - u0) / (r - l)
            vs = (v1 - v0) / (r - l)
            self.domains.append({'id': i, 'bounds': [l, min(u0, u1), r, max(v0, v1)], 'margins': [(1.0, 0.0, -l), (-1.0, 0.0, r), (-us, 1.0, us * l - u0), (vs, -1.0, v0 - vs * l)]})
        for i, h in enumerate(extension['holes']):
            if not h['accepted']:
                continue
            for tri in h['triangles']:
                pts = [mesh['vertices'][j]['source'] for j in tri['vertices']]
                margins = []
                for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
                    dx = x1 - x0
                    dy = y1 - y0
                    margins.append((-dy, dx, dy * x0 - dx * y0))
                self.domains.append({'id': len(cells) + i, 'bounds': [min((x for x, y in pts)), min((y for x, y in pts)), max((x for x, y in pts)), max((y for x, y in pts))], 'margins': margins})

    def connection(self, xcoef, ycoef):
        a, b, c = xcoef
        c -= self.offset
        dy, y0 = ycoef
        xlo, xhi = bounds(a, b, c)
        ylo, yhi = sorted([y0, y0 + dy])
        intervals = []
        for domain in self.domains:
            l, t, r, bot = domain['bounds']
            if xhi < l or xlo > r or yhi < t or (ylo > bot):
                continue
            polys = [(ax * a, ax * b + ay * dy, ax * c + ay * y0 + k) for ax, ay, k in domain['margins']]
            intervals.extend(([lo, hi, domain['id']] for lo, hi in positive_intervals(polys)))
        return (covered(intervals), intervals)

def path_mapper_factory(base_factory, domain, path, model, reference):

    class PathMap:

        def __init__(self, cells, offset):
            self.original = base_factory(cells, offset)

        def point(self, u, y):
            i = int(u) - path['range'][0]
            assert u == int(u) and y == path['pathY'][i]
            return self.original.point(float(path['sourceX'][i]), y)

        def connection(self, u0, y0, u1, y1):
            i = int(u0) - path['range'][0]
            assert u1 == u0 + 1
            if not path['sourceGapConnectionValid'][i]:
                return (False, [])
            vx, vy, vw = model['homogeneousVanishingPoint']
            den = vy - vw * reference
            dy = y1 - y0
            a = -vw * dy / den
            b = (vy - vw * y0 + dy * (vx - vw * u0)) / den
            c = path['sourceX'][i]
            if a == 0:
                return self.original.connection(float(path['sourceX'][i]), y0, float(path['sourceX'][i + 1]), y1)
            return domain.connection((a, b, c), (dy, y0))
    return PathMap
