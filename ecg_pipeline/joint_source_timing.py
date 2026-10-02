"""Row-specific source timing with independently corroborated local grid and pulse origins."""
import copy
import hashlib
import json
import math
import struct
import numpy as np
from scripts.detect_ecg_layout import _grid_period
from ecg_pipeline.joint_source_geometry import propose_joint_geometry
METHOD = 'source-joint-corresponding-grid-row-timing-v4'
PROOF_ENCODING = 'typed-json-finite-f64be-v1'

class TimingRefusal(ValueError):
    pass

def require(ok, reason):
    if not ok:
        raise TimingRefusal(reason)

def digest(image):
    return hashlib.sha256(image.tobytes()).hexdigest()

def stable(value):
    """Hash JSON values independently of Python/JavaScript number formatting.

    Numbers use finite IEEE-754 doubles, with signed zero normalized. Tags keep
    numbers distinct from strings and booleans; UTF-8 key order is explicit.
    """
    def tagged(v):
        if v is None:
            return ['null']
        if isinstance(v, bool):
            return ['bool', v]
        if isinstance(v, (int, float)):
            require(math.isfinite(v) and (not float(v).is_integer() or abs(v) <= 2**53 - 1),
                    'invalid-proof-number')
            return ['number', struct.pack('>d', 0.0 if v == 0 else float(v)).hex()]
        if isinstance(v, str):
            return ['string', v]
        if isinstance(v, list):
            return ['array', [tagged(x) for x in v]]
        if isinstance(v, dict) and all(isinstance(k, str) for k in v):
            return ['object', [[k, tagged(v[k])] for k in sorted(v, key=lambda k: k.encode('utf-8'))]]
        raise TimingRefusal('invalid-proof-value')

    encoded = json.dumps(tagged(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256((PROOF_ENCODING + '\0' + encoded).encode('utf-8')).hexdigest()

def measure_grid(image, row, column, bounds, scale):
    l, t, r, b = bounds
    require(0 <= l < r <= image.shape[1] and 0 <= t < b <= image.shape[0], 'grid-window-outside-image')
    channels = image[t:b, l:r].astype(np.float32)
    blue, green, red = np.moveaxis(channels, 2, 0)
    profile = (np.clip(red - (green + blue) * 0.5, 0, 255) / 255).mean(axis=0)
    period, confidence = _grid_period(profile)
    require(period is not None and np.isfinite(period) and np.isfinite(confidence) and (confidence >= 0.18) and (1 <= period / scale <= 40), 'local-grid-unconfirmed')
    return {'row': row, 'column': column, 'bounds': bounds, 'periodPixels': float(period), 'periodMm': scale, 'pixelsPerMm': float(period / scale), 'confidence': float(confidence), 'profileSha256': digest(profile)}

def physical_ranges(edges, width):
    require(all((np.isfinite(v) for v in edges)) and 0 <= edges[0] < edges[-1] <= width, 'physical-boundary-outside-image')
    rounded = [round(float(v)) for v in edges]
    require(all((a < b for a, b in zip(rounded, rounded[1:]))), 'non-increasing-panel-bounds')
    return [[a, b] for a, b in zip(rounded, rounded[1:])]


def _observed_pulse_edges(image, geometry, joint):
    """Probe the core, but retain every column of the observed falling stroke."""
    rows = np.asarray(geometry['rowCenters'])
    row_bounds = [0, *np.ceil((rows[:-1] + rows[1:]) / 2).astype(int).tolist(), image.shape[0]]
    pulses = []
    for row, choices in enumerate(joint['pulseChoices']):
        require(len(choices) == 1, 'pulse-solution-not-unique')
        bounds = choices[0]['falling']['bounds']
        left, top, right, bottom = bounds
        require(0 <= left < right <= image.shape[1] and row_bounds[row] <= top < bottom <= row_bounds[row + 1], 'pulse-footprint-invalid')
        margin = round((bottom - top) * .2)
        probe_top, probe_bottom = top + margin, bottom - margin
        profile = (image[probe_top:probe_bottom, left:right].max(axis=2) < 160).mean(axis=0)
        active = np.r_[False, profile >= .65, False]
        edges = np.flatnonzero(active[1:] != active[:-1])
        parts = [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)
                 if 1 <= b - a <= max(3, round(geometry['calibration']['pixelsPerMmX'] * .7))]
        require(len(parts) == 1, 'pulse-core-missing-or-ambiguous')
        a, b = parts[0]
        core = [left + a, probe_top, left + b, probe_bottom]
        pulses.append({'row': row, 'bounds': core, 'observedBounds': bounds.copy(),
                       'centerX': (core[0] + core[2] - 1) / 2,
                       'minimumBlackColumnSupport': float(profile[a:b].min()), 'sourceEndX': right})
    return pulses

def _timing_fields(image, g, joint):
    require(joint.get('state') == 'joint_geometry_supported' and joint.get('solutionCount') == 1 and (joint.get('decodedRasterSha256') == digest(image)), 'joint-source-binding-invalid')
    require(joint.get('imageSize') == [image.shape[1], image.shape[0]] and g['sourceLabelGrid']['imageSize'] == joint['imageSize'], 'source-frame-mismatch')
    cal = g['calibration']
    speed = cal['paperSpeedMmPerSecond']
    scale = cal.get('gridScaleMmX', cal.get('gridScaleMm'))
    rows = g['rowCenters']
    h, w = image.shape[:2]
    require(cal['detected'] is True and cal['confidence'] >= 0.35 and (cal['reconciliation']['state'] == 'corroborated_inference') and (cal['reconciliation']['quantitativeBlocked'] is False), 'calibration-unconfirmed')
    radius = max(8, round(float(np.median(np.diff(rows))) * 0.07))
    local = []
    row_ranges = []
    row_edges = []
    for row, line in enumerate(joint['selected']['rows']):
        edges = line['rowBoundaries'].copy()
        nominal_width = edges[4] - edges[3]
        left, right = (round(edges[3]), round(edges[4]))
        top = max(0, round(rows[row]) - radius)
        bottom = min(h, round(rows[row]) + radius + 1)
        m = measure_grid(image, row, 3, [left, top, right, bottom], scale)
        physical_width = m['pixelsPerMm'] * speed * 2.5
        require(abs(physical_width - nominal_width) / nominal_width <= 0.03, 'final-panel-spacing-conflict')
        m.update(nominalEndX=edges[4], physicalEndX=edges[3] + physical_width)
        edges[-1] = m['physicalEndX']
        row_ranges.append(physical_ranges(edges, w))
        row_edges.append(edges)
        local.append(m)
    pulse_end = joint['pulseChoices'][3][0]['endX']
    nominal = joint['rowGridMeasurements'][3]['periodPixels'] / scale * speed * 2.5
    rhythm_edges = [float(pulse_end)]
    rhythm_measures = []
    for column in range(4):
        left, right = (round(pulse_end + column * nominal), round(pulse_end + (column + 1) * nominal))
        top = max(0, round(rows[3]) - radius)
        bottom = min(h, round(rows[3]) + radius + 1)
        m = measure_grid(image, 3, column, [left, top, right, bottom], scale)
        physical_width = m['pixelsPerMm'] * speed * 2.5
        require(abs(physical_width - nominal) / nominal <= 0.03, 'rhythm-quarter-spacing-conflict')
        rhythm_edges.append(rhythm_edges[-1] + physical_width)
        m.update(physicalStartX=rhythm_edges[-2], physicalEndX=rhythm_edges[-1])
        rhythm_measures.append(m)
    final_scales = [m['pixelsPerMm'] for m in [*local, rhythm_measures[-1]]]
    require(np.ptp(final_scales) / np.median(final_scales) <= 0.02, 'final-panel-row-grid-conflict')
    primary_measures = []
    for row, line in enumerate(joint['selected']['rows']):
        edges = line['rowBoundaries']
        quarter = []
        for column in range(3):
            left, right = (round(edges[column]), round(edges[column + 1]))
            top = max(0, round(rows[row]) - radius)
            bottom = min(h, round(rows[row]) + radius + 1)
            m = measure_grid(image, row, column, [left, top, right, bottom], scale)
            nominal_width = edges[column + 1] - edges[column]
            physical_width = m['pixelsPerMm'] * speed * 2.5
            require(abs(physical_width - nominal_width) / nominal_width <= 0.03, 'primary-panel-spacing-conflict')
            quarter.append(m)
        primary_measures.append([*quarter, local[row]])
    column_consistency = []
    for column in range(4):
        scales = [m[column]['pixelsPerMm'] for m in primary_measures] + [rhythm_measures[column]['pixelsPerMm']]
        spread = float(np.ptp(scales) / np.median(scales))
        require(spread <= 0.02, 'corresponding-column-row-grid-conflict')
        column_consistency.append({'column': column, 'rowPixelsPerMm': scales, 'relativeSpread': spread, 'maximumRelativeSpread': 0.02})
    rhythm_ranges = physical_ranges(rhythm_edges, w)
    pulses = _observed_pulse_edges(image, g, joint)
    report = {
        'version': 4,
        'method': METHOD,
        'state': 'source_supported',
        'nativeExtractionReady': True,
        'truthUsed': False,
        'imageSize': [w, h],
        'decodedRasterSha256': digest(image),
        'geometryProofSha256': stable(joint),
        'geometryProofEncoding': PROOF_ENCODING,
        'physicalCalibration': copy.deepcopy(cal),
        'paperSpeedMmPerSecond': speed,
        'panelDurationSeconds': 2.5,
        'rounding': 'nearest-integer-ties-to-even-v1',
        'coordinateSpace': 'original-source',
        'rowPhysicalBoundariesX': row_edges,
        'rowPanelRanges': row_ranges,
        'rowRecordedRanges': [[r[0][0], r[-1][1]] for r in row_ranges],
        'rhythmPhysicalBoundariesX': rhythm_edges,
        'rhythmPanelRanges': rhythm_ranges,
        'rhythmRange': [rhythm_ranges[0][0], rhythm_ranges[-1][1]],
        'rhythmOrigin': 'observed-rhythm-pulse-falling-stroke-right-edge-v1',
        'rhythmMapping': 'four-local-grid-quarters-shared-baseline-v2',
        'primaryQuarterMeasurements': primary_measures,
        'correspondingColumnConsistency': column_consistency,
        'gridConsistencyMethod': 'four-row-corresponding-quarter-grid-v1',
        'finalPanelMeasurements': local,
        'rhythmQuarterMeasurements': rhythm_measures,
        'sourcePulseEdges': pulses,
        'jointGeometry': copy.deepcopy(joint),
        'sourceGridContext': {'rowCenters': copy.deepcopy(rows),
                              'fontHeight': g['sourceLabelGrid']['fontHeight'],
                              'xFit': copy.deepcopy(g['sourceLabelGrid']['xFit'])},
        'rowGridMeasurements': copy.deepcopy(joint['rowGridMeasurements']),
        'gridPixelsPerMmX': float(np.median([m['periodPixels'] / m['periodMm'] for m in joint['rowGridMeasurements']])),
        'sourceSeparators': [copy.deepcopy(mark) for line in joint['selected']['rows'] for mark in line['separators']],
    }
    return report


FAMILY_METHOD = 'source-endpoint-family-corresponding-grid-row-timing-v5'


def _family_timing_fields(image, geometry, family):
    """Keep only physical fields shared exactly by every endpoint combination."""
    require(family.get('state') == 'joint_geometry_supported'
            and family.get('geometryKind') == 'endpoint_family', 'no-source-family')
    common = None
    digests = []
    for variant in family['variantGeometries']:
        measured = _timing_fields(image, geometry, variant)
        fields = {key: value for key, value in measured.items() if key not in (
            'jointGeometry', 'geometryProofSha256', 'geometryProofEncoding',
            'sourceSeparators', 'version', 'method', 'nativeExtractionReady')}
        if common is None:
            common = fields
        require(fields == common, 'endpoint-variant-physical-timing-disagreement')
        digests.append(stable(measured))
    require(bool(digests), 'missing-endpoint-variants')
    return {**common, 'version': 5, 'method': FAMILY_METHOD,
            'nativeExtractionReady': True, 'jointGeometry': copy.deepcopy(family),
            'geometryProofSha256': stable(family), 'geometryProofEncoding': PROOF_ENCODING,
            'sourceSeparatorFamilies': copy.deepcopy(family['selectedFamilies']),
            'allVariantPhysicalTimingExact': True, 'variantCount': len(digests),
            'variantTimingDigests': digests}


def validate_joint_timing_fields(image, geometry, timing):
    """Reconstruct the source-derived fields before masks consume their bounds."""
    if timing.get('version') == 5:
        from ecg_pipeline.source_separator_families import propose
        joint = propose(image, geometry)
        expected = _family_timing_fields(image, geometry, joint)
        require('sourceSeparators' not in timing, 'family-has-representative-separator')
    else:
        joint = propose_joint_geometry(image, geometry)
        expected = _timing_fields(image, geometry, joint) if joint.get('state') == 'joint_geometry_supported' else {}
    require(joint.get('state') == 'joint_geometry_supported', 'joint-geometry-no-longer-supported')
    if timing.get('version') == 4 and 'geometryProofEncoding' not in timing:
        # Retained pre-encoding v4 reports keep their original representation
        # contract. New reports always declare the portable encoding above.
        del expected['geometryProofEncoding']
        expected['geometryProofSha256'] = hashlib.sha256(json.dumps(
            joint, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(all(timing.get(key) == value for key, value in expected.items()),
            'joint-timing-does-not-match-source')
    require('panelRanges' not in timing and 'separatorWindowConsensus' not in timing,
            'mixed-source-timing-contracts')


def source_panel_ranges(timing, row):
    """Use the same explicit source row at every extraction/export boundary."""
    require(type(row) is int and 0 <= row < 4, 'invalid-source-row')
    if timing.get('version') in (4, 5):
        ranges = timing['rhythmPanelRanges'] if row == 3 else timing['rowPanelRanges'][row]
    else:
        ranges = timing['panelRanges']
    require(isinstance(ranges, list) and len(ranges) == 4, 'invalid-source-panel-ranges')
    width = timing['imageSize'][0]
    for index, interval in enumerate(ranges):
        require(isinstance(interval, list) and len(interval) == 2
                and all(type(value) is int for value in interval)
                and 0 <= interval[0] < interval[1] <= width
                and (index == 0 or interval[0] == ranges[index - 1][1]),
                'invalid-source-panel-ranges')
    return tuple(tuple(interval) for interval in ranges)


def source_recorded_interval(timing, row):
    # Old transition proofs deliberately retain their original common domain.
    if timing.get('version') not in (4, 5):
        return tuple(timing['rhythmRange'])
    ranges = source_panel_ranges(timing, row)
    return ranges[0][0], ranges[-1][1]


def map_source_events(events, source_interval, destination_interval):
    """Transfer observed event times, without sharing any amplitude or path ink."""
    start, end = source_interval
    target, target_end = destination_interval
    require(end - start >= 2 and target_end - target >= 2, 'invalid-event-timing-domain')
    events = np.asarray(events)
    selected = events[(events >= start) & (events < end)]
    mapped = target + (selected.astype(float) - start) * (target_end - target - 1) / (end - start - 1)
    return np.unique(np.rint(mapped).astype(np.int32))


def source_ink_support(geometry, timing, support):
    rows = geometry['rowCenters']
    radius = max(3, round(float(np.median(np.diff(rows))) * .12))
    observations = []
    for row in range(4):
        top = max(0, round(rows[row]) - radius)
        bottom = min(support.shape[0], round(rows[row]) + radius + 1)
        profile = support[top:bottom].any(axis=0)
        for column, (left, right) in enumerate(source_panel_ranges(timing, row)):
            fraction = float(profile[left:right].mean())
            require(fraction >= .2, 'recorded-panel-waveform-support-missing')
            observations.append({'row': row, 'column': column, 'fraction': fraction,
                                 'bounds': [left, top, right, bottom]})
    return observations


def propose_joint_source_timing(image, geometry):
    """Fresh bounded fallback; old successful timing is handled by the caller."""
    from ecg_pipeline.source_waveform_support import build_source_ink

    from ecg_pipeline.source_separator_families import propose
    joint = propose(image, geometry)
    if joint.get('state') != 'joint_geometry_supported':
        return {'version': 4, 'method': METHOD, 'state': 'unresolved',
                'failureReason': joint['failureReason']}
    try:
        timing = (_family_timing_fields(image, geometry, joint)
                  if joint.get('geometryKind') == 'endpoint_family'
                  else _timing_fields(image, geometry, joint))
        _, support, _, _ = build_source_ink(image, geometry, timing)
        timing['sourceInkSupport'] = source_ink_support(geometry, timing, support)
        return timing
    except (ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
        return {'version': 4, 'method': METHOD, 'state': 'unresolved',
                'failureReason': str(error)}
