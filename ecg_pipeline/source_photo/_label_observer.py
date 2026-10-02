"""Image-only diagnostic label/row composition; no file or case inputs.

Fresh whole-page admission runs first; local/tiled recovery follows diagnostic rejection.
Diagnostic fallback depends only on
its fresh whole-page anchors, fresh local OCR and actual glyph/row source pixels.
Nothing in this module reads saved observations or sets production identity flags.
"""
import copy
from . import _tight_label_recovery
import hashlib
import cv2
import numpy as np

def pack(v):
    if isinstance(v, np.ndarray):
        return {'shape': list(v.shape), 'dtype': str(v.dtype), 'rasterSha256': hashlib.sha256(v.tobytes()).hexdigest()}
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {str(k): pack(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [pack(x) for x in v]
    return v

def tentative_geometry(proposal, image):
    fit = proposal['fit']
    names = ['I', 'aVR', 'V1', 'V4', 'II', 'aVL', 'V2', 'V5', 'III', 'aVF', 'V3', 'V6', 'II']
    slots = [{'lead': names[i], 'row': c['row'], 'column': c['column'], 'box': c['bounds']} for i, c in enumerate(proposal['crops'])]
    y = fit['yFit']
    centers = [y[0] + r * y[1] + 1.5 * y[2] for r in range(3)] + [y[0] + 3 * y[1]]
    grid = {'version': 1, 'method': 'source-value-grid-below-trace-v1', 'imageSize': [image.shape[1], image.shape[0]], 'maskMethod': 'maximum-channel-below-160-v1', **fit, 'slots': slots[:12], 'rhythmSlot': slots[12], 'labelRowCenters': centers, 'anchors': proposal['originalAnchors']}
    return {'layoutHint': 'standard_3x4_with_r1', 'method': 'diagnostic-tentative-crop-container-v1', 'rowCenters': [round(v - 0.16 * fit['rowSpacing']) for v in centers], 'sourceLabelGrid': grid, 'geometryConfirmed': False}

def glyph_observation(image, proposal, identity, constraints):
    geometry = tentative_geometry(proposal, image)
    bands = {tuple(c['bounds']): c['originalBounds'] for c in proposal['crops']}
    original_slot = identity._default_source_grid_slot
    original_groups = identity._default_roman_groups
    groups_record = {}
    slots_record = {}
    cache = {}

    def source_slot(raster, grid, item):
        slot = original_slot(raster, grid, item)
        if slot is None:
            return None
        box = tuple(item['box'])
        assert box in bands
        x0, y0, x1, y1 = box
        raw = raster[y0:y1, x0:x1]
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5 * grid['fontHeight'], sigmaY=1.5 * grid['fontHeight'])
        threshold = (background - gray >= 12).astype(np.uint8)
        side = max(1, round(0.08 * grid['fontHeight']))
        slot.mask = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, np.ones((side, side), np.uint8))
        key = ','.join(map(str, box))
        slots_record[key] = {'box': list(box), 'band': bands[box], 'spacing': slot.spacing, 'fontHeight': grid['fontHeight'], 'mask': pack(slot.mask)}
        return slot

    def groups(slot, n):
        key = ','.join(map(str, slot.source_box))
        digest = pack(slot.mask)['rasterSha256']
        ck = (key, digest)
        if ck not in cache:
            raw = {str(k): original_groups(slot, k) for k in [1, 2, 3]}
            band = bands[tuple(slot.source_box)]
            selected = constraints.select(raw, mode='combined', source_top=slot.source_box[1], original_top=band[1], original_bottom=band[3])
            cache[ck] = selected
            groups_record[key] = {'maskSha256': digest, 'raw': pack(raw), 'selected': pack(selected)}
        return cache[ck][str(n)]
    with identity.source_glyph_observers(source_slot, groups):
        slots = {s.expected: s for s in identity._layout_slots(image, geometry)}
        roman = identity._verify_roman_sequence(slots)
        rhythm = identity.recognise_rhythm_lead_identity(image, geometry, ocr_runner=lambda sheet: {'engine': 'D116-rhythm-source-glyph-only', 'observations': []}, primary_identity_confirmed=False)
        return {'primary': {'romanValidation': roman}, 'rhythm': rhythm, 'groups': groups_record, 'slots': slots_record, 'primaryFullOcrHelperUsed': False, 'productionIdentityConfirmed': False}

def observe(image, geometry, identity, crops, constraints, composite):
    before = pack(image)
    anchor_calls = []
    original_read = geometry.read_label_anchors

    def anchor_observed(raster, tokens):
        anchor_calls.append({'raster': pack(raster), 'tokens': copy.deepcopy(tokens)})
    with geometry.source_anchor_observer(anchor_observed):
        baseline = geometry.detect_source_label_geometry(image, recover_tiles=False, recover_local=False)
    result = {'baseline': baseline, 'baselineAnchorCalls': anchor_calls, 'fallbackAttempted': False, 'candidatePassed': False, 'finalResult': baseline, 'diagnosticOnly': True}
    if baseline.get('layoutHint'):
        result['branch'] = 'fresh-baseline-admission'
    elif not anchor_calls:
        result['branch'] = 'no-fresh-whole-page-observation'
    else:
        whole = anchor_calls[0]['tokens']
        assert anchor_calls[0]['raster'] == before
        proposal = crops.expand(geometry.propose_local_label_crops(image, whole), image.shape[1], image.shape[0])
        result['proposal'] = proposal
        if proposal.get('state') != 'crop_proposal':
            result['branch'] = 'no-eligible-fresh-source-proposal'
        else:
            local = []
            records = []
            for crop in proposal['crops']:
                l, t, r, b = crop['bounds']
                raster = image[t:b, l:r].copy()
                tokens = original_read(raster)
                local.append(tokens)
                records.append({'bounds': crop['bounds'], 'raster': pack(raster), 'tokens': tokens})
            glyph = glyph_observation(image, proposal, identity, constraints)
            candidate = composite.candidate(image, proposal, whole, local, glyph, geometry)
            if not candidate['candidatePassed'] and candidate['coverage']['missing']:
                recovery = _tight_label_recovery.recover(image, proposal, whole, local, glyph, candidate, anchor_calls, geometry, composite)
                result['tightCropRecovery'] = recovery
                if recovery['candidatePassed']:
                    result['originalRejectedCandidate'] = copy.deepcopy(candidate)
                    candidate = recovery['candidate']
            result.update(branch='fresh-diagnostic-candidate' if candidate['candidatePassed'] else 'fresh-fallback-rejected', fallbackAttempted=True, localObservations=records, glyphObservation=glyph, candidate=candidate, candidatePassed=candidate['candidatePassed'])
    if (not baseline.get('layoutHint') and not result['candidatePassed'] and anchor_calls
            and baseline.get('failureReason') != 'source-label-recognizer-unavailable-or-invalid'):
        # Reuse only this invocation's original source tokens; all maintained
        # local/tiled identity and waveform-row gates run unchanged after rejection.
        with geometry.source_anchor_observer(anchor_observed):
            recovered = geometry._complete_deferred_source_label_geometry(image, anchor_calls[0]['tokens'], baseline)
        result['deferredSourceRecovery'] = {'attempted': True, 'result': recovered}
        if recovered.get('layoutHint'):
            result.update(baselineBeforeSourceRecovery=baseline, baseline=recovered,
                          finalResult=recovered, branch='fresh-baseline-admission')
    assert pack(image) == before
    result.update(inputRaster=before, inputRasterUnchanged=True)
    return result
