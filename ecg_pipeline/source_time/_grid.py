"""Observe a source-time grid from source label geometry and physical calibration.

No coordinates are filled when a ridge is hidden. Disconnected identities remain
separate; shared boundaries use only accepted source vertices.
"""
from __future__ import annotations

import math
import numpy as np
import cv2

from ecg_pipeline.source_grid_displacement import _detect_grid_displacement_binary
from ecg_pipeline.source_photo import _grid_field
from . import _field, _identity, _leading


def source_crop(image, geometry, calibration):
    grid = geometry['sourceLabelGrid']
    width, height = grid['imageSize']
    centers = np.asarray(geometry['rowCenters'], float)
    fit = np.asarray(grid['xFit'], float)
    period = calibration['gridSpacingXPixels']
    spacing = geometry['medianRowSpacing']
    columns = grid['columnSpacing']
    if (list(image.shape[:2]) != [height, width] or centers.shape != (4,)
            or fit.shape != (3,) or not np.isfinite(centers).all() or not np.isfinite(fit).all()
            or not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in [period, spacing, columns])
            or calibration.get('gridScaleMmX') != 5 or calibration.get('paperSpeedMmPerSecond') != 25):
        raise ValueError('unsupported_source_time_grid_geometry')
    origins = [fit[0] + fit[1] * i for i in range(4)]
    crop = [max(0, math.floor(min(origins) - 2 * period)),
            max(0, math.floor(min(centers) - .6 * spacing)),
            min(width, math.ceil(max(origins) + 4 * columns + 2 * period)),
            min(height, math.ceil(max(centers) + .45 * spacing))]
    if min(crop[2] - crop[0], crop[3] - crop[1]) < 200:
        raise ValueError('unsupported_source_time_crop')
    return crop, period / 5


def build_observed_field(evidence, arrays):
    nodes = np.asarray(arrays['xNodes'], float)
    observed = np.asarray(arrays['observedRowsAll'], float)
    seeds = np.asarray(arrays['seedRows'], float)
    occluded = np.asarray(arrays['occludedColumnFraction'], float)
    period = evidence['majorPeriodPixels']
    widths = np.diff(seeds); steps = np.rint(widths / period).astype(int)
    original = (steps >= 1) & (steps <= 4) & (np.abs(widths - steps * period) <= 2)
    indices = np.r_[0, np.cumsum(steps)]
    eligible = ~(np.isfinite(occluded) & (occluded > .1))
    support = np.isfinite(observed).sum(0) / np.maximum(1, eligible.sum(0))
    initial = _grid_field.build_field(nodes, observed, indices, original, node_support=support)
    segments = {(q['row'], q['interval']): q['passed'] for q in initial['segmentChecks']}
    identities = []; admitted = []
    for gap, (width, step, passed) in enumerate(zip(widths, steps, original)):
        checks = []
        for left, right in zip(observed[gap], observed[gap + 1]):
            width_at_node = float(right) - float(left) if np.isfinite([left, right]).all() else None
            count = round(width_at_node / period) if width_at_node is not None else None
            residual = abs(width_at_node - count * period) if count is not None else None
            checks.append({'width': width_at_node, 'step': count,
                           'passed': bool(width_at_node is not None and width_at_node > 0 and 1 <= count <= 4 and residual <= 2)})
        anchors = [{'edge': j, 'step': checks[j]['step']} for j in range(len(nodes) - 1)
                   if segments.get((gap, j), False) and segments.get((gap + 1, j), False)
                   and checks[j]['passed'] and checks[j + 1]['passed'] and checks[j]['step'] == checks[j + 1]['step']]
        original_check = {'gap': gap, 'pixels': float(width), 'roundedMajorSteps': int(step),
                          'residualPixels': float(abs(width - step * period)), 'passed': bool(passed)}
        proof = _identity.identity_candidate({'originalSeedGapCheck': original_check, 'anchors': anchors, 'nodeChecks': checks}, nodes, period)
        admitted.append(bool(passed or proof['candidate']))
        identities.append({'originalSeedGapCheck': original_check, 'candidateFallback': proof, 'candidateGapPassed': admitted[-1]})
    components = np.r_[0, np.cumsum(~np.asarray(admitted, bool))]
    field = _grid_field.build_field(nodes, observed, indices, admitted, node_support=support)
    for cell in field['cells']:
        cell['supportedComponent'] = int(components[cell['upperRow']])
    return {'field': field, 'indices': indices, 'admittedGaps': admitted,
            'identityChecks': identities, 'nodeSupport': support}


def observe(image, geometry, calibration):
    crop, ppm = source_crop(image, geometry, calibration)
    left, top, right, bottom = crop
    transposed = image[top:bottom, left:right].transpose(1, 0, 2).copy()
    cv2.setRNGSeed(0)
    # This consumer admits its own partial field from the original binary ridges.
    # Native extraction's chromatic fallback uses a different position contract
    # and must not replace these observations before that independent admission.
    evidence, arrays = _detect_grid_displacement_binary(transposed, ppm, node_spacing=32)
    built = build_observed_field(evidence, arrays)
    old_cells = built['field']['cells']
    leading = _leading.observe(image[top:bottom, :right].transpose(1, 0, 2).copy(),
                               {'evidence': evidence, 'arrays': arrays}, left)
    proof = (_leading.identity(leading['observations'], np.asarray(arrays['observedRowsAll'][0], float) + left,
                               arrays['xNodes'], evidence['majorPeriodPixels'])
             if leading['seedAdmitted'] else {'accepted': False, 'reason': leading['reason']})
    field = built['field']
    if proof['accepted']:
        observations = leading['observations']
        row = np.array([q['coordinate'] for q in observations], float) - left
        values = np.vstack([row, np.asarray(arrays['observedRowsAll'], float)])
        occluded = np.vstack([np.array([q['occludedFraction'] for q in observations], float), np.asarray(arrays['occludedColumnFraction'], float)])
        eligible = ~(np.isfinite(occluded) & (occluded > .1))
        support = np.isfinite(values).sum(0) / np.maximum(1, eligible.sum(0))
        gaps = [True, *built['admittedGaps']]
        field = _grid_field.build_field(arrays['xNodes'], values, [-1, *built['indices']], gaps, node_support=support)
        components = np.r_[0, np.cumsum(~np.asarray(gaps, bool))]
        for cell in field['cells']:
            cell['supportedComponent'] = int(components[cell['upperRow']])
            cell['upperRow'] -= 1; cell['lowerRow'] -= 1
        for key in ['nodeChecks', 'segmentChecks']:
            for check in field[key]: check['row'] -= 1
        for cell in field['candidateCells']:
            cell['upperRow'] -= 1; cell['lowerRow'] -= 1
        if [c for c in field['cells'] if c['upperRow'] >= 0] != old_cells:
            raise ValueError('source_time_margin_changes_existing_cells')
    prepared = _field.prepare(field['cells'])
    if [c for c in prepared if c['upperRow'] >= 0] != _field.prepare(old_cells):
        raise ValueError('source_time_margin_changes_existing_profiles')
    return {'cropBox': crop, 'pixelsPerMmX': ppm, 'evidence': evidence, 'arrays': arrays,
            'identityChecks': built['identityChecks'], 'leadingObservation': leading,
            'leadingIdentity': proof, 'field': field, 'preparedCells': prepared}
