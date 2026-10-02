"""Use only fresh, source-bound OCR already observed within this invocation."""
import copy
import hashlib
import math

def _recover_missing_source_values(image, proposal, whole, local, glyph, original, anchor_calls, geometry, composite):
    """Query only missing image-derived source values; the complete diagnostic verifier stays unchanged."""
    result = {'attempted': False, 'candidatePassed': False,
              'productionAdmissionPerformed': False, 'savedObservationsUsed': False,
              'fullTightPrefixAvailable': False, 'expectedNamesProvidedToRecognizer': False}
    before = hashlib.sha256(image.tobytes()).hexdigest()
    if len(anchor_calls) != 1 or len(proposal['crops']) != 13 or len(local) != 13:
        return {**result, 'reason': 'no-complete-fresh-expanded-source-observations'}
    current_whole = anchor_calls[0]
    if current_whole['raster']['rasterSha256'] != before or current_whole['tokens'] != whole:
        return {**result, 'reason': 'fresh-whole-source-correspondence-failed'}
    coverage = original['coverage']
    if (len(coverage['rows']) != 13 or coverage.get('unboundWholePageNonRomanTokens')
            or any(row['conflicts'] for row in coverage['rows'])):
        return {**result, 'reason': 'conflicting-source-values-cannot-be-recovered'}
    missing = set(coverage['missing'])
    indices = []
    for i, row in enumerate(coverage['rows']):
        if row['sourceCropIndex'] != i or row['expectedPositionValue'] != composite.NAMES[i]:
            return {**result, 'reason': 'source-position-correspondence-failed'}
        if row['expectedPositionValue'] in missing and not row['matches']:
            indices.append(i)
    if not indices or {composite.NAMES[i] for i in indices} != missing or not missing <= composite.NON_ROMAN:
        return {**result, 'reason': 'no-eligible-missing-source-values'}
    union = copy.deepcopy(local)
    bindings = []
    for i in indices:
        expanded = proposal['crops'][i]
        box = expanded['originalBounds']
        l,t,r,b = box
        if not (0 <= l < r <= image.shape[1] and 0 <= t < b <= image.shape[0]):
            return {**result, 'reason': 'tight-source-crop-outside-image', 'sourceCropIndex': i}
        raster = image[t:b,l:r].copy()
        identity = {'shape': list(raster.shape), 'dtype': str(raster.dtype),
                    'rasterSha256': hashlib.sha256(raster.tobytes()).hexdigest()}
        tokens = geometry.read_label_anchors(raster)
        if hashlib.sha256(raster.tobytes()).hexdigest() != identity['rasterSha256']:
            raise RuntimeError('Fresh tight recognizer changed source pixels.')
        try:
            translated = geometry._translate_local_label_observations(tokens, box)
        except ValueError:
            return {**result, 'reason': 'fresh-tight-token-outside-source', 'sourceCropIndex': i}
        transferred = [{**q, 'x': q['x']-expanded['bounds'][0],
                        'y': q['y']-expanded['bounds'][1]} for q in translated]
        width, height = expanded['bounds'][2]-expanded['bounds'][0], expanded['bounds'][3]-expanded['bounds'][1]
        if not all(geometry._valid_local_label_token(q,width,height) for q in transferred):
            return {**result, 'reason': 'transferred-token-outside-expanded-source', 'sourceCropIndex': i}
        union[i].extend(transferred)
        bindings.append({'index':i,'originalBounds':box,'expandedBounds':expanded['bounds'],
                         'raster':identity,'tokens':copy.deepcopy(tokens),
                         'sourceTokens':translated,'transferredTokens':transferred})
    candidate = composite.candidate(image,proposal,whole,union,glyph,geometry,
                                    preferred_coverage=coverage)
    assert hashlib.sha256(image.tobytes()).hexdigest() == before
    return {**result,'attempted':True,'candidatePassed':candidate['candidatePassed'],
            'queriedTightCropIndices':indices,'unqueriedTightCropIndices':[i for i in range(13) if i not in indices],
            'unqueriedTightObservationsCreated':False,'tightCropBindings':bindings,
            'unionLocalObservations':union,'candidate':candidate,
            'selectionRule':'existing-source-values-before-fresh-missing-tight-recovery-v1'}

def recover(image, proposal, whole, local, glyph, original, anchor_calls, geometry, composite):
    result = {'attempted': False, 'candidatePassed': False,
              'productionAdmissionPerformed': False, 'savedObservationsUsed': False}
    if original['candidatePassed'] or not original['coverage']['missing']:
        return {**result, 'reason': 'no-missing-rejected-label-values'}
    if len(anchor_calls) == 1:
        rows = original['coverage'].get('rows')
        required = {'sourceCropIndex', 'expectedPositionValue', 'matches', 'conflicts'}
        if not isinstance(rows, list) or any(
            not isinstance(row, dict) or not required <= row.keys()
            or not isinstance(row['matches'], list) or not isinstance(row['conflicts'], list)
            for row in rows
        ):
            return {**result, 'reason': 'no-complete-fresh-tight-crop-observations'}
        return _recover_missing_source_values(image, proposal, whole, local, glyph, original, anchor_calls, geometry, composite)
    if len(anchor_calls) < 14 or len(proposal['crops']) != 13 or len(local) != 13:
        return {**result, 'reason': 'no-complete-fresh-tight-crop-observations'}
    # The detector may append tile observations after its whole page and thirteen
    # tight crops. Consume only that complete, ordered, source-verified prefix.
    current_whole = anchor_calls[0]
    before = hashlib.sha256(image.tobytes()).hexdigest()
    if current_whole['raster']['rasterSha256'] != before or current_whole['tokens'] != whole:
        return {**result, 'reason': 'fresh-whole-source-correspondence-failed'}
    union = copy.deepcopy(local)
    bindings = []
    for i, (expanded, saved) in enumerate(zip(proposal['crops'], anchor_calls[1:14], strict=True)):
        box = expanded['originalBounds']
        l,t,r,b = box
        crop = image[t:b,l:r]
        identity = {'shape': list(crop.shape), 'dtype': str(crop.dtype),
                    'rasterSha256': hashlib.sha256(crop.tobytes()).hexdigest()}
        if identity != saved['raster']:
            return {**result, 'reason': 'fresh-tight-source-correspondence-failed', 'sourceCropIndex': i}
        try:
            translated = geometry._translate_local_label_observations(saved['tokens'], box)
        except ValueError:
            return {**result, 'reason': 'fresh-tight-token-outside-source', 'sourceCropIndex': i}
        transferred = [{**q, 'x': q['x']-expanded['bounds'][0],
                        'y': q['y']-expanded['bounds'][1]} for q in translated]
        width, height = expanded['bounds'][2]-expanded['bounds'][0], expanded['bounds'][3]-expanded['bounds'][1]
        if not all(geometry._valid_local_label_token(q,width,height) for q in transferred):
            return {**result, 'reason': 'transferred-token-outside-expanded-source', 'sourceCropIndex': i}
        union[i].extend(transferred)
        bindings.append({'index':i,'originalBounds':box,'expandedBounds':expanded['bounds'],
                         'raster':saved['raster'],'tokens':saved['tokens'],
                         'sourceTokens':translated,'transferredTokens':transferred})
    candidate = composite.candidate(image,proposal,whole,union,glyph,geometry,
                                    preferred_coverage=original['coverage'])
    assert hashlib.sha256(image.tobytes()).hexdigest() == before
    return {**result,'attempted':True,'candidatePassed':candidate['candidatePassed'],
            'tightCropBindings':bindings,'unionLocalObservations':union,'candidate':candidate,
            'selectionRule':'existing-source-observation-before-tight-crop-recovery-v1'}
