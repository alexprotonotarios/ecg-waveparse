from . import _label_model as LOCAL, _row_model as ROWS
'Bind observed label values to source positions; no truth or file names enter.'
import copy
import math
import numpy as np
NAMES = ['I', 'aVR', 'V1', 'V4', 'II', 'aVL', 'V2', 'V5', 'III', 'aVF', 'V3', 'V6', 'II']
NON_ROMAN = {'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'}

def observed_coverage(proposal, whole_page, local, *, preferred_coverage=None):
    """Retain all tokens; bind exact non-Roman values by observed source center."""
    assert len(proposal['crops']) == len(local) == 13
    rows = []
    selected = []
    unbound = []
    for token in whole_page:
        if token['lead'] not in NON_ROMAN:
            continue
        locations = []
        for i, crop in enumerate(proposal['crops']):
            b = crop['bounds']
            band = crop['originalBounds']
            x = token['x'] + token['width'] / 2
            if b[0] <= x < b[2] and band[1] <= token['y'] < band[3]:
                locations.append(i)
        if len(locations) != 1:
            unbound.append(copy.deepcopy(token))
    for i, (crop, observations) in enumerate(zip(proposal['crops'], local, strict=True)):
        b = crop['bounds']
        band = crop['originalBounds']
        tokens = []
        for origin, items in [('whole-page', whole_page), ('local', observations)]:
            for original in items:
                t = copy.deepcopy(original)
                if origin == 'local':
                    t['x'] += b[0]
                    t['y'] += b[1]
                finite = all((isinstance(t.get(k), (int, float)) and (not isinstance(t[k], bool)) and math.isfinite(t[k]) for k in ['x', 'y', 'width', 'height', 'confidence']))
                valid = finite and t['width'] > 0 and (t['height'] > 0) and (0.65 <= t['confidence'] <= 1)
                bound = bool(valid and b[0] <= t['x'] + t['width'] / 2 < b[2] and (band[1] <= t['y'] < band[3]))
                if origin == 'local' or bound:
                    tokens.append({'token': t, 'origin': origin, 'bound': bound, 'originalToken': original})
        matches = [t for t in tokens if t['bound'] and t['token']['lead'] == NAMES[i] and (NAMES[i] in NON_ROMAN)]
        conflicts = [t for t in tokens if t['bound'] and t['token']['lead'] in NON_ROMAN and (t['token']['lead'] != NAMES[i])]
        best = max(matches, key=lambda t: t['token']['confidence']) if matches else None
        if best and (not conflicts):
            selected.append(best['token'])
        rows.append({'sourceCropIndex': i, 'expectedPositionValue': NAMES[i], 'tokens': tokens, 'matches': matches, 'conflicts': conflicts, 'selected': best})
    preferences = []
    if preferred_coverage is not None:
        selected = []
        for index, (current, previous) in enumerate(zip(rows, preferred_coverage['rows'], strict=True)):
            preferred = previous['selected']
            if preferred is not None:
                assert preferred in current['matches']
                current['selected'] = copy.deepcopy(preferred)
            winner = current['selected']
            if winner is not None and not current['conflicts']:
                selected.append(winner['token'])
            preferences.append({'sourceCropIndex': index,
                                'basis': 'existing-observed-selection' if preferred is not None else 'missing-value-recovery',
                                'previous': previous['selected'], 'selected': current['selected']})
    passed = len(selected) == 9 and (not unbound) and (not any((r['conflicts'] for r in rows)))
    result = {'passed': passed, 'nineExactBoundValues': len(selected), 'selectedTokens': selected, 'allBoundNonRomanTokens': [t['token'] for r in rows for t in r['matches']], 'rows': rows, 'unboundWholePageNonRomanTokens': unbound, 'missing': [NAMES[i] for i in range(12) if NAMES[i] in NON_ROMAN and (not rows[i]['matches'])], 'usedRomanOcrValues': False}

    if preferred_coverage is not None:
        result.update(selectionPreference=preferences, selectionRule='existing-source-observation-before-tight-crop-recovery-v1')
    return result

def group_box(group, source_box):
    components = group['components']
    x0 = min((c['x'] for c in components)) + source_box[0]
    y0 = min((c['y'] for c in components)) + source_box[1]
    x1 = max((c['x'] + c['width'] for c in components)) + source_box[0]
    y1 = max((c['y'] + c['height'] for c in components)) + source_box[1]
    return [x0, y0, x1, y1]

def candidate(image, proposal, whole_page, local, glyph_observation, geometry_module, *, preferred_coverage=None):
    """Return a diagnostic proof only; the maintained primary verifier is not bypassed in production."""
    coverage = observed_coverage(proposal, whole_page, local, preferred_coverage=preferred_coverage)
    result = {'candidatePassed': False, 'coverage': coverage, 'failureReasons': [], 'productionAdmissionPerformed': False, 'fullMaintainedIdentityHelperPassed': False}
    if not coverage['passed']:
        result['failureReasons'].append('nine-exact-source-bound-values-required')
        return result
    roman = glyph_observation['primary']['romanValidation']
    rhythm = glyph_observation['rhythm']
    if not roman['passed']:
        result['failureReasons'].append('roman-sequence-unconfirmed')
        return result
    if rhythm['sourceGlyphSimilarity'] < 0.72 or rhythm['sourceGlyphAmbiguityMargin'] < 0.04:
        result['failureReasons'].append('rhythm-source-glyph-match-unconfirmed')
        return result
    glyph_boxes = {}
    for crop_index, n in [(0, 1), (4, 2), (8, 3), (12, 2)]:
        bounds = proposal['crops'][crop_index]['bounds']
        key = ','.join(map(str, bounds))
        groups = glyph_observation['groups'][key]['selected'][str(n)]
        if len(groups) != 1:
            result['failureReasons'].append('non-unique-selected-roman-source-group')
            return result
        glyph_boxes[str(crop_index)] = group_box(groups[0], bounds)
    rhythm_box = glyph_boxes['12']
    token = {'lead': 'II', 'confidence': rhythm['sourceGlyphSimilarity'], 'x': rhythm_box[0], 'y': (rhythm_box[1] + rhythm_box[3]) / 2, 'width': rhythm_box[2] - rhythm_box[0], 'height': rhythm_box[3] - rhythm_box[1], 'evidenceKind': 'source-derived-repeated-roman-glyphs', 'ocrObserved': False}
    points = [{'lead': t['lead'], 'x': t['x'], 'y': t['y'], 'sourceBox': [t['x'], t['y'] - t['height'] / 2, t['x'] + t['width'], t['y'] + t['height'] / 2], 'evidenceKind': 'exact-non-Roman-source-OCR', 'confidence': t['confidence']} for t in coverage['selectedTokens']]
    for index, lead in [('0', 'I'), ('4', 'II'), ('8', 'III')]:
        b = glyph_boxes[index]
        points.append({'lead': lead, 'x': b[0], 'y': (b[1] + b[3]) / 2, 'sourceBox': b, 'evidenceKind': 'verified-Roman-source-group', 'sourceCropIndex': int(index)})
    for t in coverage['allBoundNonRomanTokens']:
        selected = next((p for p in points if p['lead'] == t['lead']))
        if abs(t['x'] - selected['x']) > 0.025 * image.shape[1] or abs(t['y'] - selected['y']) > 0.025 * image.shape[0]:
            result['failureReasons'].append('duplicate-source-label-locations')
            return result
    font = float(np.median([t['height'] for t in coverage['selectedTokens']]))
    model = LOCAL.fit(points, [image.shape[1], image.shape[0]], font)
    rhythm_point = {'lead': 'II', 'x': token['x'], 'y': token['y'], 'sourceBox': rhythm_box, 'evidenceKind': 'verified-Roman-source-group', 'sourceCropIndex': 12}
    rhythm_geometry = LOCAL.rhythm_check(model, rhythm_point)
    result.update(localLabelModel=model, rhythmGeometry=rhythm_geometry, rhythmPointExcludedFromFit=rhythm_point)
    if model['passed'] and rhythm_geometry['passed']:
        grid = LOCAL.proposal(model, rhythm_point)
        strict = {'layoutHint': 'standard_3x4_with_r1', 'method': 'diagnostic-bilinear-label-grid-v1', 'sourceLabelGrid': grid, 'productionFormatSupported': False} if grid else {'layoutHint': None, 'failureReason': 'source-label-crop-outside-image'}
    else:
        strict = {'layoutHint': None, 'failureReason': 'local-label-geometry-unconfirmed'}
    result.update(romanValidation=roman, rhythmSourceSimilarity=rhythm['sourceGlyphSimilarity'], rhythmSourceMargin=rhythm['sourceGlyphAmbiguityMargin'], romanSourceBoxes=glyph_boxes, rhythmSourceToken=token, strictProposal=strict)
    if not strict.get('layoutHint'):
        result['failureReasons'].append(strict['failureReason'])
        return result
    grid = strict['sourceLabelGrid']
    font = grid['fontHeight']
    checks = []
    selected = {t['lead']: t for t in coverage['selectedTokens']}
    for i, slot in enumerate(grid['slots'] + [grid['rhythmSlot']]):
        b = slot['box']
        expanded = [max(0, round(b[0] - font)), max(0, round(b[1] - 0.5 * font)), min(image.shape[1], round(b[2] + 0.5 * font)), min(image.shape[0], round(b[3] + 0.5 * font))]
        if str(i) in glyph_boxes:
            measured = glyph_boxes[str(i)]
            center = [(measured[0] + measured[2]) / 2, (measured[1] + measured[3]) / 2]
        else:
            t = selected[slot['lead']]
            measured = [t['x'], t['y'] - t['height'] / 2, t['x'] + t['width'], t['y'] + t['height'] / 2]
            center = [t['x'] + t['width'] / 2, t['y']]
        passed = expanded[0] <= measured[0] < measured[2] <= expanded[2] and expanded[1] <= measured[1] < measured[3] <= expanded[3] and (b[1] <= center[1] < b[3])
        checks.append({'sourceCropIndex': i, 'lead': slot['lead'], 'strictBox': b, 'expandedBox': expanded, 'measuredSourceBox': measured, 'center': center, 'passed': passed})
    result['finalSourceBindings'] = checks
    if not all((c['passed'] for c in checks)):
        result['failureReasons'].append('observed-label-outside-final-source-context')
        return result
    masks, _ = ROWS.masks(image, grid)
    row_observation = ROWS.rows(masks['horizontal-only'], grid, prominence=True)
    centers = row_observation['centers']
    result['sourceRowObservation'] = row_observation
    result['waveformRowCenters'] = centers
    if centers is None:
        result['failureReasons'].append('source-waveform-rows-unconfirmed')
        return result
    result.update(candidatePassed=True, primaryIdentityEvidenceKind='nine-exact-source-values-plus-roman-source-sequence', rhythmIdentityEvidenceKind='unique-primary-ii-and-repeated-source-glyph-match', diagnosticGeometry={**strict, 'rowCenters': centers})
    return result
