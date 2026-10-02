"""Additional directly measured evidence for original candidate grid cells."""
import copy
from ecg_pipeline.source_photo import _grid_field as original

def evidence(brackets, upper, lower, x, observed, indices, gaps, support, offset):
    if lower <= upper + 1 or not 1 <= indices[lower] - indices[upper] <= 4:
        return None
    candidates = [b for b in brackets if (b['upperRow'], b['lowerRow']) == (upper, lower)]
    if len(candidates) != 1:
        return None
    b = candidates[0]
    steps = int(indices[lower] - indices[upper])
    if b['majorSteps'] != steps:
        return None
    failed = [i for i in range(upper, lower) if not gaps[i]]
    if not failed:
        return None
    for i in failed:
        adjacent = [v for v in brackets if (v['upperRow'], v['lowerRow']) == (i, i + 1)]
        if len(adjacent) != 1:
            return None
        measured = [q for q in adjacent[0]['checks'] if q['status'] == 'measured']
        if len(measured) >= 8 or any(not q.get('passed', False) for q in measured):
            return None
    allowed = set(b['supportedNodeIds'])
    measured = [q for q in b['checks'] if q['nodeIndex'] in allowed and q['status'] == 'measured']
    passing = [q for q in measured if q.get('passed', False)]
    if len(measured) < 8 or len(passing) / len(measured) < .9:
        return None
    endpoints = []
    for k in range(2):
        found = [q for q in measured if q['sourceX'] == float(x[k] + offset)]
        if len(found) != 1:
            return None
        q = found[0]
        if support[k] < .8 or not q.get('passed', False) or q['expectedMajorSteps'] != steps or q['roundedMajorSteps'] != steps or q['absoluteErrorPixels'] > 2:
            return None
        if q['observedRows'] != [float(observed[0][k]), float(observed[1][k])]:
            return None
        endpoints.append({'nodeIndex': q['nodeIndex'], 'sourceX': q['sourceX'], 'observedRows': q['observedRows'], 'absoluteErrorPixels': q['absoluteErrorPixels']})
    return {'upperRow': upper, 'lowerRow': lower, 'majorSteps': steps, 'measuredCount': len(measured), 'passingCount': len(passing), 'agreementFraction': len(passing) / len(measured), 'sparseUncontradictedInnerGaps': failed, 'endpoints': endpoints}

def build_field(nodes, observations, major_indices, gap_passed, *, node_support, brackets=(), offset=0):
    result = original.build_field(nodes, observations, major_indices, gap_passed, node_support=node_support)
    if not result['candidateCells']:
        return result
    out = copy.deepcopy(result)
    old_cells = {(q['interval'], q['upperRow'], q['lowerRow']): q for q in result['cells']}
    cells = []
    for q in out['candidateCells']:
        j, u, l = q['interval'], q['upperRow'], q['lowerRow']
        key = (j, u, l)
        if key in old_cells:
            cells.append(copy.deepcopy(old_cells[key]))
            continue
        if q['seedGapsSupported'] or not q['positiveHeights'] or not 1 <= q['majorSteps'] <= 4 or not all(2 <= v <= 40 for v in q['pixelsPerMmAtEdges']):
            continue
        xy = [[observations[u][j], observations[u][j + 1]], [observations[l][j], observations[l][j + 1]]]
        receipt = evidence(brackets, u, l, nodes[j:j + 2], xy, major_indices, gap_passed, node_support[j:j + 2], offset)
        if receipt is None:
            continue
        q.update(passed=True, directBracketEvidence=receipt)
        cells.append({**q, 'x': [float(v) for v in nodes[j:j + 2]], 'upperY': [float(v) for v in xy[0]], 'lowerY': [float(v) for v in xy[1]], 'paperYmm': [float(5 * major_indices[u]), float(5 * major_indices[l])]})
    if len(cells) == len(result['cells']):
        return result
    out.update(cells=cells, state='partial', reason=None)
    return out
