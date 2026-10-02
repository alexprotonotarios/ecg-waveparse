"""Source-bound time conversion with preserved uncertainty and native lineage."""
from __future__ import annotations
import copy
import math
import numpy as np
from . import _mapping
from ._conversion import convert, runs

LEADS = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
ROWS = (('I', 'aVR', 'V1', 'V4'), ('II', 'aVL', 'V2', 'V5'), ('III', 'aVF', 'V3', 'V6'))
METHOD = 'source_grid_time_publication_v1'


class ConversionRefused(ValueError):
    """Insufficient source evidence; preserve the ordinary publication."""


def number(value):
    return float(value) if value is not None and value != '' else float('nan')


def uncertainty_index(rows, canonical, segments):
    result = {}; expected = set()
    for li, lead in enumerate(LEADS):
        segment = next(s for s in segments if s['lead'] == lead)
        start, count = segment['canonicalStartSample'], segment['sampleCount']
        expected.update((lead, j) for j in range(start, start + count))
    for row in rows:
        lead, sample = row['lead'], int(row['canonical_sample'])
        key = (lead, sample)
        if key in result or key not in expected: raise ValueError('invalid_publication_uncertainty_namespace')
        li = LEADS.index(lead); value = number(row['value_uv'])
        segment = next(s for s in segments if s['lead'] == lead)
        local = sample - segment['canonicalStartSample']
        if (int(row['lead_sample']) != local or number(row['t_s_at_500hz']) != local / 500
                or not (value == canonical[li, sample] or (math.isnan(value) and math.isnan(canonical[li, sample])))):
            raise ValueError('uncertainty_does_not_match_published_values')
        annotation = str(row['annotation_overlap'])
        if annotation not in ('0', '1'): raise ValueError('invalid_annotation_uncertainty_flag')
        if math.isfinite(value) and (annotation != '0' or row['status'] not in ('observed', 'uncertain_candidate_disagreement', 'uncertainty_unavailable')):
            raise ConversionRefused('published_value_has_unresolved_annotation_or_uncertainty_status')
        count = number(row['candidate_count']); spread = number(row['candidate_spread_uv'])
        if (math.isfinite(count) and (count < 0 or count != int(count))) or (math.isfinite(spread) and spread < 0):
            raise ValueError('invalid_candidate_uncertainty')
        result[key] = row
    if set(result) != expected: raise ValueError('publication_uncertainty_is_incomplete')
    return result


def native_contributors(arrays, lead_index, left, right, left_weight, right_weight):
    native = {}
    for sample, weight in [(left, left_weight), (right, right_weight)]:
        if weight == 0: continue
        row = int(arrays['rowIndices'][lead_index, sample])
        lo = int(arrays['nativeLeftIndices'][lead_index, sample]); hi = int(arrays['nativeRightIndices'][lead_index, sample])
        w = float(arrays['rightWeights'][lead_index, sample])
        for index, inner in [(lo, 1 - w), (hi, w)]:
            if inner == 0: continue
            native[(row, index)] = native.get((row, index), 0.) + weight * inner
    if not 1 <= len(native) <= 4 or abs(sum(native.values()) - 1) >= 1e-12:
        raise ValueError('invalid_converted_native_weights')
    for (row, sample), weight in native.items():
        if not math.isfinite(weight) or weight <= 0 or not np.isfinite(arrays['nativeSourceXY'][row, sample]).all():
            raise ValueError('invalid_converted_native_coordinate')
    return native


def convert_publication(observation, arrays, metadata, canonical, uncertainty_rows, segment_map):
    """Inputs are ordinary published values, with exclusions already applied."""
    source = observation.get('sourceSha256')
    if source != metadata.get('sourceSha256') or source != segment_map.get('sourceSha256'):
        raise ValueError('source_time_input_identity_mismatch')
    if observation.get('state') != 'source_time_supported':
        raise ConversionRefused(observation.get('state', 'source_time_unresolved'))
    if (observation.get('method') != 'observed_source_grid_time_v1' or observation.get('version') != 1
            or observation.get('origins', {}).get('admitted') is not True
            or metadata.get('coordinateScope') != 'original_source_raster'
            or metadata.get('coordinateConvention') != 'pixel_centers_zero_based'
            or metadata.get('preparationCorrespondenceVerified') is not True
            or metadata.get('waveformChanged') is not False
            or metadata.get('metadata', {}).get('match', {}).get('layout') != 'standard_3x4_with_r1'
            or segment_map.get('units') != 'uV' or segment_map.get('exportSampleRateHz') != 500
            or segment_map.get('canonicalSampleCount') != 5000 or segment_map.get('endpointConvention') != 'half_open'):
        raise ConversionRefused('unsupported_source_time_publication_contract')
    segments = segment_map.get('segments', [])
    if len(segments) != 12 or {s['lead'] for s in segments} != set(LEADS):
        raise ConversionRefused('source_time_requires_twelve_recorded_segments')
    expected = np.zeros((12, 5000), bool)
    if any(arrays.get(key, np.empty(0)).shape != (12, 5000) for key in ['canonicalUv', 'rowIndices', 'sourcePolarity']):
        raise ValueError('invalid_source_time_capture_shape')
    for li, lead in enumerate(LEADS):
        segment = next(s for s in segments if s['lead'] == lead)
        row, col = (3, 0) if lead == 'II' else next((r, c) for r, names in enumerate(ROWS) for c, name in enumerate(names) if name == lead)
        start, count = (0, 5000) if lead == 'II' else (col * 1250, 1250)
        if (segment['rowIndex'] != row or segment['panelIndex'] != col or segment['canonicalStartSample'] != start
                or segment['sampleCount'] != count or segment.get('polarity', 1) != 1
                or segment['role'] != ('rhythm' if lead == 'II' else 'panel')):
            raise ConversionRefused('source_time_segment_position_or_polarity_mismatch')
        expected[li, start:start + count] = True
        observed = np.isfinite(canonical[li]) if canonical.shape == (12, 5000) else np.zeros(5000, bool)
        if np.any(arrays['rowIndices'][li, observed] != row) or np.any(arrays['sourcePolarity'][li, observed] != 1):
            raise ConversionRefused('captured_source_row_or_polarity_does_not_match_segment')
    if canonical.shape != (12, 5000) or np.isinf(canonical).any() or np.isfinite(canonical[~expected]).any():
        raise ValueError('invalid_published_canonical_shape_or_support')
    finite = np.isfinite(canonical)
    if not np.array_equal(arrays['canonicalUv'][finite], canonical[finite]):
        raise ConversionRefused('publication_values_changed_after_coordinate_capture')
    uncertainty = uncertainty_index(uncertainty_rows, canonical, segments)
    try:
        mapping = _mapping.map_publication(observation['geometry'], arrays, canonical)
    except ValueError as error:
        raise ConversionRefused(str(error)) from error
    paper, connected = mapping['publishedPaperXmm'], mapping['connectedAdjacent']
    if not np.array_equal(np.isfinite(paper), finite):
        raise ConversionRefused('published_samples_have_unsupported_source_grid_coordinates')
    excluded = arrays['canonicalFinite'] & ~finite
    exclusions = []
    for li, lead in enumerate(LEADS):
        for indices in runs(excluded[li], np.ones(4999, bool)):
            lo, hi = indices[0], indices[-1] + 1
            bounds = ([float(paper[li, lo - 1]), float(paper[li, hi])]
                      if lo > 0 and hi < 5000 and np.isfinite(paper[li, [lo - 1, hi]]).all() else None)
            statuses = [uncertainty[(lead, i)]['status'] for i in indices if (lead, i) in uncertainty]
            annotation = any(str(uncertainty[(lead, i)]['annotation_overlap']) == '1' for i in indices if (lead, i) in uncertainty)
            exclusions.append({'lead': lead, 'sourceCanonicalStart': lo, 'sourceCanonicalEnd': hi,
                               'sourceExcludedSamples': len(indices), 'openForbiddenBridgeMm': bounds,
                               'annotation': annotation, 'sourceUncertaintyStatusCounts': {s: statuses.count(s) for s in sorted(set(statuses))}})
    output = {k: np.full((12, 5000), np.nan) for k in ['correctedUv', 'rightWeight', 'leftWeight', 'targetPaperXmm']}
    output.update(sourceLeft=np.full((12, 5000), -1, np.int32), sourceRight=np.full((12, 5000), -1, np.int32),
                  originalFinite=finite.copy(), originalPublicationExcluded=excluded.copy())
    converted_uncertainty = []; lineage = []; origins = []; converted_segments = []
    marks = {m['id']: m for m in observation['origins']['marks']}
    for li, lead in enumerate(LEADS):
        old = next(s for s in segments if s['lead'] == lead); start, count = old['canonicalStartSample'], old['sampleCount']
        mark_id = f"row{old['rowIndex']}:pulse" if old['panelIndex'] == 0 else f"row{old['rowIndex']}:separator{old['panelIndex']}"
        mark = marks[mark_id]; origin = mark['originPaperXmm']; bounds = mark['locus']['paperXmmRange']
        if not math.isfinite(origin) or not bounds[0] <= origin <= bounds[1]: raise ValueError('invalid_observed_source_origin')
        try: result = convert(canonical[li], paper[li], connected[li], origin, count)
        except ValueError as error: raise ConversionRefused(str(error)) from error
        for key, value in [('correctedUv', 'values'), ('sourceLeft', 'left'), ('sourceRight', 'right'), ('rightWeight', 'rightWeight'), ('leftWeight', 'leftWeight'), ('targetPaperXmm', 'targetPaperXmm')]:
            output[key][li, start:start + count] = result[value]
        timing_range = [(origin - bounds[1]) * 40, (origin - bounds[0]) * 40]
        proof = {'sourceOriginMark': mark_id, 'sourceMidpointXY': mark['sourceMidpointXY'], 'originPaperXmm': origin,
                 'originLocusPaperXmmRange': bounds, 'inkFootprintPaperXmmRange': mark['inkFootprint']['paperXmmRange'],
                 'geometricTimingShiftMsRange': timing_range}
        origins.append({'lead': lead, **proof, 'expectedSamples': count, 'returnedSamples': int(np.isfinite(result['values']).sum())})
        segment = copy.deepcopy(old); segment.pop('intervalLineage', None)
        support = []
        for j, value in enumerate(result['values']):
            status = 'returned' if np.isfinite(value) else 'missing'
            if support and support[-1]['status'] == status: support[-1]['endSample'] = j + 1
            else: support.append({'startSample': j, 'endSample': j + 1, 'status': status})
        segment.update(support=support, lineageState='source_time_conversion_recorded', sourceTimeConversion=proof)
        converted_segments.append(segment)
        for j in range(count):
            left, right = int(result['left'][j]), int(result['right'][j])
            w, lw = float(result['rightWeight'][j]), float(result['leftWeight'][j])
            row = {'lead': lead, 'lead_sample': j, 't_s_at_500hz': j / 500, 'canonical_sample': start + j,
                   'value_uv': '', 'review_estimate_uv': '', 'status': 'missing', 'annotation_overlap': 0,
                   'candidate_count': '', 'candidate_spread_uv': '', 'source_left_canonical_sample': '', 'source_right_canonical_sample': '',
                   'source_right_weight': '', 'geometric_time_shift_lower_ms': timing_range[0], 'geometric_time_shift_upper_ms': timing_range[1],
                   'uncertainty_method': 'maximum_required_input_spread_plus_separate_geometric_time_range'}
            if left < 0:
                if any(q['lead'] == lead and q['annotation'] and q['openForbiddenBridgeMm'] and q['openForbiddenBridgeMm'][0] < result['targetPaperXmm'][j] < q['openForbiddenBridgeMm'][1] for q in exclusions):
                    row.update(status='uncertain_annotation', annotation_overlap=1)
            else:
                required = [left] if left == right else [left, right]
                parents = [uncertainty[(lead, k)] for k in required]
                if excluded[li, required].any(): raise ValueError('conversion_uses_excluded_source_sample')
                spreads = [number(q['candidate_spread_uv']) for q in parents]; counts = [number(q['candidate_count']) for q in parents]
                spread = max(spreads) if all(math.isfinite(v) for v in spreads) else ''
                count_value = int(min(counts)) if all(math.isfinite(v) for v in counts) else ''
                status = 'uncertain_candidate_disagreement' if any(q['status'] == 'uncertain_candidate_disagreement' for q in parents) else 'observed'
                if spread == '' or count_value == '' or any(q['status'] == 'uncertainty_unavailable' for q in parents): status = 'uncertainty_unavailable'
                row.update(value_uv=float(result['values'][j]), status=status, candidate_count=count_value, candidate_spread_uv=spread,
                           source_left_canonical_sample=left, source_right_canonical_sample=right, source_right_weight=w)
                native = native_contributors(arrays, li, left, right, lw, w)
                item = {'lead': lead, 'canonical_sample': start + j, 'source_left_canonical_sample': left, 'source_right_canonical_sample': right,
                        'source_left_weight': lw, 'source_right_weight': w, 'source_right_weight_numerator': result['rightWeightRational'][j][0],
                        'source_right_weight_denominator': result['rightWeightRational'][j][1], 'target_paper_x_mm': float(result['targetPaperXmm'][j]),
                        'value_uv': float(result['values'][j]), 'native_contributor_count': len(native)}
                for k, ((native_row, index), weight) in enumerate(sorted(native.items())):
                    x, y = map(float, arrays['nativeSourceXY'][native_row, index])
                    item.update({f'native{k}_row': native_row, f'native{k}_sample': index, f'native{k}_weight': weight,
                                 f'native{k}_source_x': x, f'native{k}_source_y': y})
                lineage.append(item)
            converted_uncertainty.append(row)
    return {'arrays': output, 'mapping': mapping, 'uncertainty': converted_uncertainty, 'lineage': lineage,
            'segments': converted_segments, 'origins': origins, 'sourceExclusions': exclusions}
