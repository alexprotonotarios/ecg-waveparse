"""Transport observed pulse/separator extents without fitting a time origin."""
from . import _field, _ranges


def observe(grid, timing):
    cells = grid['preparedCells']; crop = grid['cropBox']
    pieces, domain = _ranges.build(cells)
    marks = []
    for edge in timing['sourcePulseEdges']:
        left, top, right, bottom = edge['selected']['bounds']
        marks.append({'id': f"row{edge['row']}:pulse", 'row': edge['row'], 'column': 0,
                      'locusBounds': [right, top, right, bottom], 'inkBounds': [left, top, right, bottom]})
    for q in timing['sourceSeparators']:
        left, top, right, bottom = q['unionBounds']; lo, hi = q['centerIntervalPixels']
        marks.append({'id': f"row{q['row']}:separator{q['column']}", 'row': q['row'], 'column': q['column'],
                      'locusBounds': [lo, top, hi, bottom], 'inkBounds': [left, top, right, bottom]})
    if len(marks) != 13 or len({m['id'] for m in marks}) != 13:
        raise ValueError('incomplete_source_time_marks')
    for mark in marks:
        mark['locus'] = _ranges.transport(pieces, domain, mark['locusBounds'], crop)
        mark['inkFootprint'] = _ranges.transport(pieces, domain, mark['inkBounds'], crop)
        left, top, right, bottom = mark['locusBounds']
        xy = [(left + right) / 2, (top + bottom) / 2]
        hits = [(i, _field.forward(c, xy[1] - crop[1], xy[0] - crop[0]), c['supportedComponent']) for i, c in enumerate(cells)]
        hits = [q for q in hits if q[1] is not None]
        consistent = bool(hits) and len({q[2] for q in hits}) == 1 and max(q[1] for q in hits) - min(q[1] for q in hits) < 1e-10
        mark.update(sourceMidpointXY=xy, originPaperXmm=hits[0][1] if consistent and mark['locus']['admitted'] else None)
    by_id = {q['id']: q for q in marks}; spans = []
    for row in range(3):
        names = [f'row{row}:pulse', *[f'row{row}:separator{i}' for i in [1, 2, 3]]]
        for column in range(3):
            a, b = [by_id[names[i]]['locus'] for i in [column, column + 1]]
            ready = a['admitted'] and b['admitted'] and a['components'] == b['components']
            span = [b['paperXmmRange'][0] - a['paperXmmRange'][1], b['paperXmmRange'][1] - a['paperXmmRange'][0]] if ready else None
            spans.append({'row': row, 'column': column, 'observedSpanMmRange': span,
                          'passed': bool(ready and 62.5 * .97 <= span[0] <= span[1] <= 62.5 * 1.03)})
    return {'marks': marks, 'physicalSpanChecks': spans,
            'admitted': all(m['originPaperXmm'] is not None and m['inkFootprint']['admitted'] for m in marks) and all(s['passed'] for s in spans),
            'originRule': 'midpoint_of_observed_source_locus', 'originRangesAreGeometricNotStatistical': True}
