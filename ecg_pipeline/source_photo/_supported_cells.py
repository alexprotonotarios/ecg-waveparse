"""Add uniquely supported outer cells without replacing accepted geometry."""
import copy

import numpy as np

from . import _supported_direct as direct


def overlaps(a, b):
    return a['interval'] == b['interval'] and max(a['upperRow'], b['upperRow']) < min(a['lowerRow'], b['lowerRow'])


def build_field(nodes, observations, major_indices, gap_passed, *, node_support, brackets=(), offset=0):
    original = direct.build_field(nodes, observations, major_indices, gap_passed,
                                  node_support=node_support, brackets=brackets, offset=offset)
    if not original['segmentChecks']:
        return original
    x = np.asarray(nodes, float)
    y = np.asarray(observations, float)
    indices = np.asarray(major_indices, int)
    known = {(c['interval'], c['upperRow'], c['lowerRow']) for c in original['candidateCells']}
    proposed = []
    for j in range(len(x) - 1):
        common = sorted(q['row'] for q in original['segmentChecks'] if q['interval'] == j and q['passed'])
        for pos, upper in enumerate(common):
            for lower in common[pos + 2:]:
                if (j, upper, lower) in known:
                    continue
                steps = int(indices[lower] - indices[upper])
                if not 1 <= steps <= 4:
                    continue
                heights = y[lower, j:j + 2] - y[upper, j:j + 2]
                ppm = heights / (5 * steps)
                if not np.all(heights > 0) or not np.all((ppm >= 2) & (ppm <= 40)):
                    continue
                xy = y[[upper, lower], j:j + 2].tolist()
                receipt = direct.evidence(brackets, upper, lower, x[j:j + 2], xy,
                                          indices, gap_passed, node_support[j:j + 2], offset)
                if receipt is None:
                    continue
                cell = {'interval': j, 'upperRow': upper, 'lowerRow': lower, 'majorSteps': steps,
                        'seedGapsSupported': bool(np.asarray(gap_passed)[upper:lower].all()),
                        'positiveHeights': True, 'pixelsPerMmAtEdges': ppm.tolist(), 'passed': True,
                        'directBracketEvidence': receipt, 'x': x[j:j + 2].tolist(),
                        'upperY': xy[0], 'lowerY': xy[1],
                        'paperYmm': [float(5 * indices[upper]), float(5 * indices[lower])]}
                if not any(overlaps(cell, old) for old in original['cells']):
                    proposed.append(cell)
    accepted = [c for c in proposed if not any(c is not other and overlaps(c, other) for other in proposed)]
    if not accepted:
        return original
    out = copy.deepcopy(original)
    out['candidateCells'].extend(copy.deepcopy(accepted))
    out['cells'].extend(copy.deepcopy(accepted))
    out.update(state='partial', reason=None,
               outerCellConstruction={'proposed': len(proposed), 'accepted': len(accepted),
                                      'overlapRefused': len(proposed) - len(accepted),
                                      'existingCellsUnchanged': True, 'newCorners': 0})
    return out
