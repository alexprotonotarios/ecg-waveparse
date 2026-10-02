"""Source-only vertical colour-ridge census; not a coordinate correction."""
import math
import cv2
import numpy as np

def fit_points(points):
    y = np.array([p[0] for p in points], float)
    x = np.array([p[1] for p in points], float)
    yc = float(y.mean())
    xc = float(x.mean())
    slope = float(np.dot(y - yc, x - xc) / np.dot(y - yc, y - yc))
    residual = x - (xc + slope * (y - yc))
    return {'centerY': yc, 'centerX': xc, 'dxPerDy': slope, 'maximumResidualPixels': float(np.max(np.abs(residual)))}

def observe(image, x_period):
    assert image.dtype == np.uint8 and image.ndim == 3 and (image.shape[2] == 3)
    height, width = image.shape[:2]
    assert min(height, width) >= 300 and 20 <= x_period <= 100
    channels = image.astype(np.int16)
    mask = (channels.max(2) - channels.min(2) >= 18).astype(np.uint8)
    proposal = cv2.morphologyEx(mask * 255, cv2.MORPH_CLOSE, np.ones((9, 1), np.uint8))
    raw = cv2.HoughLinesP(proposal, 1, np.pi / 1800, threshold=max(40, height // 5), minLineLength=max(80, height // 4), maxLineGap=12)
    center_y = (height - 1) // 2
    lines = []
    for line in [] if raw is None else raw[:, 0].tolist():
        x0, y0, x1, y1 = line
        if y0 > y1:
            x0, y0, x1, y1 = (x1, y1, x0, y0)
        if y1 - y0 < height / 4:
            continue
        slope = (x1 - x0) / (y1 - y0)
        if abs(slope) > math.tan(math.radians(10)):
            continue
        lines.append({'endpoints': [x0, y0, x1, y1], 'dxPerDy': slope, 'crossesCenter': y0 <= center_y <= y1, 'centerX': x0 + slope * (center_y - y0), 'outsideHistoricalTwoDegreeBound': abs(slope) > math.tan(math.radians(2))})
    eligible = sorted([q for q in lines if q['crossesCenter']], key=lambda q: (q['centerX'], q['endpoints']))
    groups = []
    for line in eligible:
        if not groups or line['centerX'] - groups[-1][0]['centerX'] > 3:
            groups.append([])
        groups[-1].append(line)
    nodes = [center_y + i for i in [-128, -64, 0, 64, 128]]
    ridges = []
    for group in groups:
        representative = sorted(group, key=lambda q: (-(q['endpoints'][3] - q['endpoints'][1]), q['endpoints']))[0]
        slope = representative['dxPerDy']
        profiles = []
        for node in nodes:
            expected = representative['centerX'] + slope * (node - center_y)
            middle = round(expected)
            radius = max(3, math.floor(0.25 * x_period))
            xs = list(range(max(0, middle - radius), min(width, middle + radius + 1)))
            ys = list(range(node - 8, node + 9))
            counts = []
            in_bounds = True
            for x in xs:
                pixels = [(round(x + slope * (y - node)), y) for y in ys]
                if any((not (0 <= xx < width and 0 <= yy < height) for xx, yy in pixels)):
                    in_bounds = False
                counts.append(sum((int(mask[yy, xx]) for xx, yy in pixels if 0 <= xx < width and 0 <= yy < height)))
            receipt = {'nodeY': node, 'expectedX': expected, 'sampleXs': xs, 'sampleYs': ys, 'shearDxPerDy': slope, 'colourCounts': counts, 'denominator': len(ys), 'allSamplesInsideWindow': in_bounds, 'measuredX': None, 'reason': None}
            if not in_bounds or len(xs) < 3:
                receipt['reason'] = 'profile_boundary'
            elif max(counts) / len(ys) < 0.55 or (max(counts) - min(counts)) / len(ys) < 0.25:
                receipt['reason'] = 'insufficient_original_colour_support'
            else:
                at = int(np.argmax(counts))
                a, b = (at, at + 1)
                cut = min(counts) + 0.5 * (max(counts) - min(counts))
                while a > 0 and counts[a - 1] >= cut:
                    a -= 1
                while b < len(counts) and counts[b] >= cut:
                    b += 1
                receipt['halfHeightIndexRange'] = [a, b]
                if a == 0 or b == len(counts):
                    receipt['reason'] = 'ridge_touches_search_edge'
                else:
                    weights = [v - min(counts) for v in counts[a:b]]
                    measured = sum((x * w for x, w in zip(xs[a:b], weights))) / sum(weights)
                    receipt['candidateX'] = measured
                    if abs(measured - expected) > 0.1 * x_period:
                        receipt['reason'] = 'ambiguous_proposal_identity'
                    else:
                        receipt['measuredX'] = measured
            profiles.append(receipt)
        points = [(z['nodeY'], z['measuredX']) for z in profiles if z['measuredX'] is not None]
        fit = fit_points(points) if len(points) >= 3 and points[-1][0] - points[0][0] >= 128 else None
        loo = []
        if fit:
            for i, (y, x) in enumerate(points):
                f = fit_points(points[:i] + points[i + 1:])
                loo.append(abs(x - (f['centerX'] + f['dxPerDy'] * (y - f['centerY']))))
            fit['maximumOmittedNodeResidualPixels'] = max(loo)
        ridges.append({'proposalCount': len(group), 'proposal': representative, 'profiles': profiles, 'measuredNodeCount': len(points), 'fit': fit})
    slopes = [r['fit']['dxPerDy'] for r in ridges if r['fit']]
    summary = {'proposalLines': len(lines), 'centerCrossingProposals': len(eligible), 'ridgeGroups': len(ridges), 'fittedRidges': len(slopes), 'attemptedNodes': 5 * len(ridges), 'measuredNodes': sum((r['measuredNodeCount'] for r in ridges)), 'medianDxPerDy': float(np.median(slopes)) if slopes else None, 'minimumDxPerDy': min(slopes) if slopes else None, 'maximumDxPerDy': max(slopes) if slopes else None, 'candidateCoordinateCorrectionAccepted': False}
    if slopes:
        summary['medianHorizontalDriftPer100VerticalPixels'] = 100 * summary['medianDxPerDy']
        summary['equivalentMillisecondsPer100VerticalPixels'] = 100 * summary['medianDxPerDy'] / (5 * x_period) * 1000
    return {'imageSize': [width, height], 'colourPixelCount': int(mask.sum()), 'xPeriodPixels': x_period, 'sourceNodeYs': nodes, 'coarseLines': lines, 'ridges': ridges, 'summary': summary, 'limitations': ['Colour ridge candidates are not independent observations; Hough fragments can duplicate a physical line.', 'Nearest-source-pixel profiles preserve actual mask samples; no missing grid node is filled.', 'Fits summarize observed drift only; omitted-node residuals and identity refusals remain.', 'No coordinate field, trace reordering, resampling, correction or waveform timing fit.']}
