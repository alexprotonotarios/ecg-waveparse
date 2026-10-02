"""Bounded local label coordinates with row-by-column interaction, not image warping."""
import numpy as np
ROWS = [['I', 'aVR', 'V1', 'V4'], ['II', 'aVL', 'V2', 'V5'], ['III', 'aVF', 'V3', 'V6']]
POSITIONS = {n: (r, c) for r, row in enumerate(ROWS) for c, n in enumerate(row)}

def vector(row, column):
    return np.asarray([1, row, column, row * column], float)

def coordinate(model, row, column):
    return [float(vector(row, column) @ np.asarray(model[k])) for k in ['xFit', 'yFit']]

def local_tolerances(model, row, column):
    return (0.04 * (model['xFit'][2] + row * model['xFit'][3]), max(0.6 * model['fontHeight'], 0.04 * (model['yFit'][1] + column * model['yFit'][3])))

def fit(points, image_size, font):
    w, h = image_size
    design = np.asarray([vector(*POSITIONS[p['lead']]) for p in points])
    target = np.asarray([[p['x'], p['y']] for p in points], float)
    beta, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank != 4:
        return {'passed': False, 'reason': 'rank-deficient', 'rank': int(rank)}
    x, y = (beta[:, 0], beta[:, 1])
    residual = np.abs(design @ beta - target)
    corners = []
    for row in [0, 3]:
        for col in [0, 3]:
            columns = float(x[2] + row * x[3])
            spacing = float(y[1] + col * y[3])
            dx = float(x[1] + col * x[3])
            dy = float(y[2] + row * y[3])
            checks = {'columnSpacing': 0.12 * w < columns < 0.3 * w, 'rowSpacing': 0.09 * h < spacing < 0.3 * h, 'horizontalSkew': abs(dx) < columns * 0.06, 'verticalSkew': abs(dy) < spacing * 0.06, 'fontHeight': 0.02 * spacing < font < 0.18 * spacing, 'positiveJacobian': columns * spacing - dx * dy > 0}
            corners.append({'row': row, 'column': col, 'columnSpacing': columns, 'rowSpacing': spacing, 'rowXDerivative': dx, 'columnYDerivative': dy, 'checks': checks})
    model = {'xFit': x.tolist(), 'yFit': y.tolist(), 'fontHeight': font, 'imageSize': image_size}
    residuals = []
    for p, error in zip(points, residual, strict=True):
        r, c = POSITIONS[p['lead']]
        xt, yt = local_tolerances(model, r, c)
        residuals.append({'lead': p['lead'], 'x': float(error[0]), 'y': float(error[1]), 'xTolerance': xt, 'yTolerance': yt, 'passed': bool(error[0] < xt and error[1] < yt)})
    model.update(rank=int(rank), corners=corners, residuals=residuals, points=points, passed=all((all(c['checks'].values()) for c in corners)) and all((r['passed'] for r in residuals)), meanRowSpacing=float(y[1] + 1.5 * y[3]), meanColumnSpacing=float(x[2] + 1.5 * x[3]))
    return model

def rhythm_check(model, rhythm):
    if model.get('rank') != 4:
        return {'passed': False, 'reason': 'rank-deficient'}
    x, y = coordinate(model, 3, 0)
    dx = abs(rhythm['x'] - x)
    dy = abs(rhythm['y'] - y)
    xt = 0.12 * (model['xFit'][2] + 3 * model['xFit'][3])
    yt = 0.25 * model['yFit'][1]
    return {'passed': dx < xt and dy < yt, 'predictedX': x, 'predictedY': y, 'xDistance': dx, 'yDistance': dy, 'xTolerance': xt, 'yTolerance': yt}

def proposal(model, rhythm):
    w, h = model['imageSize']
    font = model['fontHeight']
    slots = []
    for r, row in enumerate(ROWS):
        for c, n in enumerate(row):
            x, y = coordinate(model, r, c)
            b = [round(x - 0.65 * font), round(y - 0.6 * font), round(x + 3.4 * font), round(y + 0.6 * font)]
            if not 0 <= b[0] < b[2] <= w or not 0 <= b[1] < b[3] <= h:
                return None
            slots.append({'lead': n, 'row': r, 'column': c, 'box': b})
    x, y = (rhythm['x'], rhythm['y'])
    b = [round(x - 0.65 * font), round(y - 0.6 * font), round(x + 3.4 * font), round(y + 0.6 * font)]
    if not 0 <= b[0] < b[2] <= w or not 0 <= b[1] < b[3] <= h:
        return None
    return {'version': 2, 'method': 'diagnostic-bilinear-label-grid-v1', 'imageSize': [w, h], 'xFit': model['xFit'], 'yFit': model['yFit'], 'fontHeight': font, 'rowSpacing': model['meanRowSpacing'], 'columnSpacing': model['meanColumnSpacing'], 'slots': slots, 'rhythmSlot': {'lead': 'II', 'row': 3, 'column': 0, 'box': b}, 'labelRowCenters': [coordinate(model, r, 1.5)[1] for r in range(3)] + [rhythm['y']], 'productionFormatSupported': False}

def bindings(grid, points, rhythm):
    observations = {p['lead']: p for p in points}
    w, h = grid['imageSize']
    font = grid['fontHeight']
    rows = []
    for s in grid['slots'] + [grid['rhythmSlot']]:
        p = rhythm if s['row'] == 3 else observations.get(s['lead'])
        if p is None:
            continue
        b = s['box']
        expanded = [max(0, round(b[0] - font)), max(0, round(b[1] - 0.5 * font)), min(w, round(b[2] + 0.5 * font)), min(h, round(b[3] + 0.5 * font))]
        observed = p['sourceBox']
        passed = expanded[0] <= observed[0] < observed[2] <= expanded[2] and expanded[1] <= observed[1] < observed[3] <= expanded[3] and (b[1] <= p['y'] < b[3])
        rows.append({'lead': s['lead'], 'row': s['row'], 'passed': passed, 'sourceBox': observed, 'strictBox': b, 'expandedBox': expanded})
    return rows

def supported_rows(image, grid):
    """Keep original ink/support rules, evaluating x ranges with row-dependent spacing."""
    h, w = image.shape[:2]
    mask = (image[..., :3].max(axis=2) < 160).astype(float) if image.ndim == 3 else (image < 160).astype(float)
    spacing, font = (grid['rowSpacing'], grid['fontHeight'])
    centers = []
    details = []
    for row, label_y in enumerate(grid['labelRowCenters']):
        top = max(0, round(label_y - spacing * 0.35))
        bottom = min(h, round(label_y - max(font * 0.85, spacing * 0.04)))
        if bottom - top < 4:
            return {'centers': None, 'reason': 'short-row-band', 'rows': details}
        source_row = min(row, 2)
        columns = grid['xFit'][2] + source_row * grid['xFit'][3]
        ranges = []
        for col in range(4):
            start = coordinate(grid, source_row, col)[0]
            left, right = (max(0, round(start + max(font * 3.5, columns * 0.12))), min(w, round(start + columns * 0.88)))
            if right - left < 20:
                return {'centers': None, 'reason': 'short-column-range', 'rows': details}
            ranges.append((left, right))
        profiles = [mask[top:bottom, left:right].mean(axis=1) for left, right in ranges]
        profile = np.mean(profiles, axis=0)
        smooth = np.convolve(profile, np.ones(3) / 3, mode='same')
        local = int(np.argmax(smooth))
        center = top + local
        radius = max(2, round(spacing * 0.025))
        support = [float(np.max(p[max(0, local - radius):min(len(p), local + radius + 1)])) for p in profiles]
        details.append({'row': row, 'band': [top, bottom], 'ranges': [list(r) for r in ranges], 'support': support, 'smoothedPeak': float(smooth[local]), 'center': center})
        if min(support) < 0.025 or float(smooth[local]) < 0.04:
            return {'centers': None, 'reason': 'insufficient-source-row-ink', 'rows': details}
        centers.append(center)
    gaps = np.diff(centers)
    if not np.all((gaps > spacing * 0.7) & (gaps < spacing * 1.3)):
        return {'centers': None, 'reason': 'inconsistent-row-gaps', 'rows': details}
    return {'centers': centers, 'reason': None, 'rows': details}
