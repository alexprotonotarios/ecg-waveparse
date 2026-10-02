"""Diagnostic source paths only; no production identity or calibrated CSV export."""
import hashlib
import cv2, numpy as np
from ecg_pipeline.source_waveform_support import build_source_ink, source_supported_validity, connected_source_validity
from ecg_pipeline.source_separator_publication import separator_publication_mask, apply_separator_publication_validity
from ecg_pipeline.native_grid_digitizer import trace_crossing_path, trace_path
from ecg_pipeline.source_grid_displacement import detect_grid_displacement
shaarray = lambda a: hashlib.sha256(a.tobytes()).hexdigest()
LEADS = [['I', 'aVR', 'V1', 'V4'], ['II', 'aVL', 'V2', 'V5'], ['III', 'aVF', 'V3', 'V6']]

def diagnostic_exclusions(image, case):
    h, w = image.shape[:2]
    g = case['geometry']['sourceLabelGrid']
    rows = case['geometry']['rowCenters']
    timing = case['timing']
    excluded = np.zeros((h, w), np.uint8)
    publication = np.zeros((h, w), bool)
    pulse = np.zeros((h, w), bool)
    boxes = []
    mid = [0, *np.ceil((np.asarray(rows[:-1]) + np.asarray(rows[1:])) / 2).astype(int).tolist(), h]
    for slot, binding in zip([*g['slots'], g['rhythmSlot']], case['sourceBindings'], strict=True):
        assert slot['lead'] == binding['lead'] and binding['passed']
        l, t, r, b = slot['box']
        x0, y0, x1, y1 = binding['measuredSourceBox']
        bounds = [max(l, int(np.floor(x0)) - 2), max(t, int(np.floor(y0)) - 2), min(r, int(np.ceil(x1)) + 2), min(b, int(np.ceil(y1)) + 2)]
        assert bounds[0] < bounds[2] and bounds[1] < bounds[3]
        boxes.append({'kind': 'diagnostic-source-bound-label', 'lead': slot['lead'], 'bounds': bounds, 'binding': binding})
    for mark in timing['sourceSeparators']:
        l, t, r, b = mark['unionBounds']
        row = mark['row']
        boxes.append({'kind': 'diagnostic-observed-separator-union', 'row': row, 'bounds': [max(0, l - 1), t, min(w, r + 1), b]})
        publication[mid[row]:mid[row + 1], max(0, l - 1):min(w, r + 1)] = True
    for edge in timing['sourcePulseEdges']:
        l, t, r, b = edge['selected']['bounds']
        row = edge['row']
        bounds = [max(0, l - 1), mid[row], min(w, r + 1), mid[row + 1]]
        boxes.append({'kind': 'diagnostic-observed-pulse-columns', 'row': row, 'bounds': bounds})
        pulse[bounds[1]:bounds[3], bounds[0]:bounds[2]] = True
    for box in boxes:
        l, t, r, b = box['bounds']
        excluded[t:b, l:r] = 1
    return (excluded, publication, pulse, {'method': 'diagnostic-source-bound-exclusions-v1', 'boxes': boxes, 'productionIdentityAsserted': False})

def prepare_masks(image, case, mask_module):
    if case['kind'] == 'control':
        original_evidence, original_support, excluded, receipt = build_source_ink(image, case['geometry'], case['timing'])
        publication, pub_receipt = separator_publication_mask(image, case['geometry'], case['timing'], original_support)
        receipt = {'sourceInk': receipt, 'publication': pub_receipt}
        pulse = np.zeros(excluded.shape, bool)
        for b in receipt['sourceInk']['exclusionBoxes']:
            if b['kind'] == 'observed-calibration-edge-columns':
                l, t, r, bottom = b['bounds']
                pulse[t:bottom, l:r] = True
    else:
        excluded, publication, pulse, receipt = diagnostic_exclusions(image, case)
    masks = mask_module.masks(image, case['geometry']['sourceLabelGrid']['fontHeight'])
    out = {}
    for arm, mask in masks.items():
        clean = mask * (1 - excluded)
        evidence = cv2.GaussianBlur(clean.astype(np.float32), (3, 3), 0.45)
        evidence[pulse] = 0
        support = cv2.dilate(clean, np.ones((3, 3), np.uint8)).astype(bool)
        support[excluded.astype(bool)] = False
        if case['kind'] == 'control' and arm == 'baseline':
            assert np.array_equal(evidence, original_evidence) and np.array_equal(support, original_support)
        out[arm] = (evidence, support, clean)
    return (out, excluded, publication, receipt)

def run_case(image, case, mask_module):
    before = shaarray(image)
    masks, excluded, publication, receipt = prepare_masks(image, case, mask_module)
    rows = case['geometry']['rowCenters']
    spacing = float(np.median(np.diff(rows)))
    h, w = image.shape[:2]
    bottom = min(h, round((rows[2] + rows[3]) / 2))
    ranges = case['timing'].get('rowPanelRanges') or [case['timing']['panelRanges']] * 4
    specs = [{'lead': lead, 'row': row, 'column': col, 'range': ranges[row][col]} for row in range(3) for col, lead in enumerate(LEADS[row])]
    specs.append({'lead': 'rhythm_II', 'row': 3, 'column': None, 'range': case['timing']['rhythmRange']})
    result = {'caseId': case['caseId'], 'kind': case['kind'], 'imageArraySha256': before, 'exclusions': receipt, 'excludedMaskSha256': shaarray(excluded), 'publicationMaskSha256': shaarray(publication), 'arms': {}}
    for arm, (evidence, support, clean) in masks.items():
        records = []
        for s in specs:
            row = s['row']
            lo, hi = s['range']
            if row < 3:
                top = max(0, round(rows[row] - spacing * 2))
                end = min(bottom, round(rows[row] + spacing * 1.5))
                columns, path = trace_crossing_path(evidence, y_start=top, y_end=end, x_start=lo, x_end=hi, row_center=rows[row], row_spacing=spacing)
            else:
                top = max(0, round(rows[row] - spacing * 0.55))
                end = min(h, round(rows[row] + spacing * 0.55))
                columns, path = trace_path(evidence, y_start=top, y_end=end, x_start=lo, x_end=hi, row_center=rows[row])
            point = source_supported_validity(support, columns, path, None)
            connected, connected_receipt = connected_source_validity(evidence, columns, path, point, source_interval=[lo, hi])
            valid = connected & ~publication[path, columns]
            records.append({**s, 'yBounds': [top, end], 'pathY': path.tolist(), 'pointSupported': point.astype(int).tolist(), 'connectedSupported': connected.astype(int).tolist(), 'valid': valid.astype(int).tolist(), 'pathSha256': shaarray(path.astype('<i4')), 'validitySha256': shaarray(valid.astype('u1')), 'sourceColumnCount': len(columns), 'retainedColumns': int(valid.sum()), 'coverage': float(valid.mean()), 'connectionEvidence': connected_receipt})
        result['arms'][arm] = {'maskSha256': shaarray(clean), 'evidenceSha256': shaarray(evidence), 'supportSha256': shaarray(support), 'cleanPixelCount': int(clean.sum()), 'paths': records}
    baseline = result['arms']['baseline']['paths']
    candidate = result['arms']['neutral-local-contrast']['paths']
    result['armDifferences'] = [{'lead': a['lead'], 'changedPathColumns': sum((x != y for x, y in zip(a['pathY'], b['pathY']))), 'changedValidityColumns': sum((x != y for x, y in zip(a['valid'], b['valid']))), 'coverageDifference': b['coverage'] - a['coverage']} for a, b in zip(baseline, candidate)]
    ppm = case['localGrids'][12]['selectedPair']['yPeriod'] / case['gridScaleMm']
    lo, hi = case['timing']['rhythmRange']
    try:
        ge, observations = detect_grid_displacement(image[:, lo:hi], ppm)
        result['existingPaperGridConversionPrerequisite'] = {'pixelsPerMm': ppm, 'evidence': ge, 'arrayHashes': {k: shaarray(v) for k, v in observations.items()}}
    except ValueError as e:
        result['existingPaperGridConversionPrerequisite'] = {'pixelsPerMm': ppm, 'state': 'unavailable', 'reason': str(e)}
    assert shaarray(image) == before
    result.update(sourceImageUnchanged=True, calibratedSignalExported=False, productionAdmissionPerformed=False)
    return result
