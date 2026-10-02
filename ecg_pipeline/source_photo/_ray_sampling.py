"""Inverse camera-ray sampling; each result retains its original-pixel footprint."""
import hashlib
import numpy as np
shaarray = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def inverse_x(model, u, y, reference):
    vx, vy, vw = model['homogeneousVanishingPoint']
    den = vy - vw * reference
    if abs(den) <= 1e-12 * max(1.0, abs(vy), abs(vw * reference)):
        raise ValueError('reference row intersects projective pole')
    out = u + (y - reference) * (vx - vw * u) / den
    if np.any((vy - vw * y) / den <= 0):
        raise ValueError('nonpositive inverse horizontal Jacobian')
    return out

def footprint(source_x, width):
    floor = np.floor(source_x).astype(np.int64)
    ceil = np.ceil(source_x).astype(np.int64)
    inside = (floor >= 0) & (ceil < width)
    return (np.clip(floor, 0, width - 1), np.clip(ceil, 0, width - 1), source_x - floor, inside)

def planes(evidence, support, publication, old, model, reference, review_ranges):
    h, w = evidence.shape
    lo, hi = old['range']
    top, bottom = old['yBounds']
    yy = np.arange(top, bottom)[:, None]
    uu = np.arange(lo, hi)[None, :]
    sx = inverse_x(model, uu, yy, reference)
    left, right, weight, inside = footprint(sx, w)
    prior = np.zeros(w, bool)
    prior[lo:hi] = np.asarray(old['valid'], bool)
    flag = np.zeros(w, bool)
    for a, b in review_ranges:
        flag[a:b] = True
    guard = inside & (left >= lo) & (right < hi) & prior[left] & prior[right]
    mixed = ((1 - weight) * evidence[yy, left] + weight * evidence[yy, right]).astype(np.float32)
    mixed[~guard] = 0
    ss = support[yy, left] & support[yy, right] & guard
    pp = publication[yy, left] | publication[yy, right] | ~guard
    ff = (flag[left] | flag[right]) & inside
    E = np.zeros_like(evidence)
    S = np.zeros_like(support, bool)
    P = np.ones_like(publication, bool)
    E[top:bottom, lo:hi] = mixed
    S[top:bottom, lo:hi] = ss
    P[top:bottom, lo:hi] = pp
    return (E, S, P, {'sourceX': sx, 'sourceLeft': left, 'sourceRight': right, 'rightWeight': weight, 'sourceFootprintAllowed': guard, 'reviewFootprint': ff}, {'evidenceSha256': shaarray(E), 'supportSha256': shaarray(S), 'publicationSha256': shaarray(P), 'sampledPixelQueries': int(sx.size), 'sourceFootprintsAllowed': int(guard.sum()), 'sourceFootprintsRefused': int((~guard).sum()), 'allEarlierSourceGapsMasked': True, 'sampling': 'linear_horizontal_float32_evidence; all_contributing_source_support; any_contributing_publication_or_review_flag', 'sourceImageResampled': False})

def retain_path(path, footprints, old):
    lo, hi = old['range']
    top, _ = old['yBounds']
    ys = np.asarray(path['pathY'])
    index = (ys - top, np.arange(hi - lo))
    allowed = footprints['sourceFootprintAllowed'][index]
    initial = np.asarray(path['valid'], bool)
    valid = initial & allowed
    result = {**path, 'preTransportValid': path['valid'], 'valid': valid.astype(int).tolist(), 'retainedColumns': int(valid.sum()), 'coverage': float(valid.mean()), 'sourceX': footprints['sourceX'][index].tolist(), 'sourceLeft': footprints['sourceLeft'][index].tolist(), 'sourceRight': footprints['sourceRight'][index].tolist(), 'sourceRightWeight': footprints['rightWeight'][index].tolist(), 'sourceFootprintAllowed': allowed.tolist(), 'reviewRequired': (footprints['reviewFootprint'][index] & valid).tolist(), 'newTransportRefusals': int((initial & ~valid).sum()), 'validitySha256': shaarray(valid.astype('u1')), 'oldGapsRecovered': 0, 'sourcePositionMeaning': 'original floating x and integer y; x is not a source column index', 'coordinateSystem': 'camera_ray_reference_x_original_source_y', 'newlyLostColumnsMeaning': 'compared to the same baseline-x prior mask; not a loss count at identical physical coordinates'}
    return result

def chord_transport(path, old, model, reference, review_ranges):
    lo, hi = old['range']
    valid = np.asarray(path['valid'], bool)
    prior = np.asarray(old['valid'], bool)
    vx, vy, vw = model['homogeneousVanishingPoint']
    den = vy - vw * reference
    connections = []
    review = []
    bounds = []
    for i in range(hi - lo - 1):
        u = lo + i
        y = path['pathY'][i]
        dy = path['pathY'][i + 1] - y
        a = path['sourceX'][i]
        b = (vy - vw * y + dy * (vx - vw * u)) / den
        c = -vw * dy / den
        values = [a, path['sourceX'][i + 1]]
        if c != 0:
            t = -b / (2 * c)
            if 0 < t < 1:
                values.append(a + b * t + c * t * t)
        low, high = (min(values), max(values))
        first, last = (int(np.floor(low)), int(np.ceil(high)))
        ok = bool(valid[i] and valid[i + 1] and (first >= lo) and (last < hi) and prior[first - lo:last - lo + 1].all())
        connections.append(ok)
        bounds.append([low, high])
        review.append(ok and any((first < b and last >= a for a, b in review_ranges)))
    return {'sourceGapConnectionValid': connections, 'sourceChordXBounds': bounds, 'sourceChordReviewRequired': review, 'sourceGapConnectionMeaning': 'Both sampled endpoints valid and every source column touched by the inverse quadratic chord had prior source validity. Other coordinate-field support remains unevaluated.'}
