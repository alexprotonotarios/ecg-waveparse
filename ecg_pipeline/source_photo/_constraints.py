"""Bounded group selection only; never alter glyph pixels or value thresholds."""

def signature(group):
    return frozenset(((c['index'], c['x'], c['y'], c['width'], c['height']) for c in group['components']))

def select(groups, *, mode, source_top, original_top, original_bottom):
    assert mode in ['baseline', 'text-band', 'subset', 'combined']
    selected = {n: list(items) for n, items in groups.items()}
    if mode in ['text-band', 'combined']:
        selected = {n: [g for g in items if all((original_top <= c['centerY'] + source_top < original_bottom for c in g['components']))] for n, items in selected.items()}
    if mode in ['subset', 'combined']:
        signatures = {n: [signature(g) for g in items] for n, items in selected.items()}
        selected = {n: [g for g in items if not any((signature(g) < other for larger, sets in signatures.items() if int(larger) > int(n) for other in sets))] for n, items in selected.items()}
    return selected
