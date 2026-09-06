"""Denominators and independent-group uncertainty for a frozen engineering evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg_benchmark.evaluation import denominator_scorecard, grouped_mean_interval


def group_identity(member: dict) -> str:
    for key in ('groupId', 'patientId', 'recordingId'):
        value = member.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError('Every frozen member requires an independent group identity.')


def wilson_interval(successes: int, attempted: int) -> dict:
    """Non-degenerate 95% binomial interval, only for one attempt per independent group."""
    if attempted < 1 or not 0 <= successes <= attempted:
        raise ValueError('Invalid binomial denominator.')
    z = 1.959963984540054
    proportion, denominator = successes / attempted, 1 + z*z / attempted
    center = (proportion + z*z / (2*attempted)) / denominator
    radius = z * math.sqrt(proportion*(1-proportion)/attempted + z*z/(4*attempted**2)) / denominator
    return {'method': 'wilson-score', 'independentAttempts': attempted, 'successes': successes,
            'proportion': proportion, 'interval95': [max(0., center-radius), min(1., center+radius)]}


def summarize(protocol: dict, manifest: dict, report: dict) -> dict:
    prefix = manifest['suiteId'] + '__'
    expected = [prefix + m['caseId'] for m in protocol['membership']]
    if report['denominator'] != len(expected):
        raise ValueError('Evaluation denominator differs from its frozen membership.')
    rows = report['cases']
    card = denominator_scorecard(expected, rows, gates=protocol['gates'])
    members = {prefix + m['caseId']: m for m in protocol['membership']}
    by_id = {r['id']: r for r in rows}
    group_ids = [group_identity(m) for m in protocol['membership']]
    group_count = len(set(group_ids))
    returned_ids = set(card['scoredSignalCaseIds'])
    intervals = {}
    for key in ('globalRmseUv', 'macroMeanCorrelation', 'macroMeanCoverage'):
        values = []
        for row in rows:
            score = (row.get('score') or {}).get('summary', row.get('score') or {})
            value = score.get(key)
            if row['id'] in returned_ids and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                values.append((group_identity(members[row['id']]), value))
        intervals[key] = grouped_mean_interval(values)
    intervals['returnedYield'] = grouped_mean_interval([(group_identity(m), float(prefix + m['caseId'] in returned_ids)) for m in protocol['membership']])
    intervals['referenceQualifiedYield'] = grouped_mean_interval([(group_identity(m), float(card['referenceDecisions'][prefix + m['caseId']]['status'] == 'passed')) for m in protocol['membership']])
    one_per_group = group_count == len(expected)
    binomial_intervals = {'status': 'available' if one_per_group else 'unavailable_repeated_groups'}
    if one_per_group:
        binomial_intervals['returnedYield'] = wilson_interval(len(returned_ids), len(expected))
        binomial_intervals['referenceQualifiedYield'] = wilson_interval(card['referenceQualifiedCount'], len(expected))
    strata = {}
    for key in protocol.get('stratificationKeys', ('layout', 'artifact')):
        strata[key] = {}
        if not all(isinstance(m.get(key), str) and m[key] for m in protocol['membership']):
            raise ValueError('Missing declared stratum in frozen membership.')
        for value in sorted({m[key] for m in protocol['membership']}):
            ids = [prefix + m['caseId'] for m in protocol['membership'] if m[key] == value]
            strata[key][value] = denominator_scorecard(ids, [by_id[i] for i in ids if i in by_id], gates=protocol['gates'])
    return {'version': 2, 'collection': protocol['collection'], 'clinicalValidationUse': False,
            'denominatorScorecard': card, 'independentGroups': group_count,
            'independentUnit': protocol.get('independentUnit', 'patient' if all(m.get('patientId') for m in protocol['membership']) else 'declared group'),
            'groupedIntervals': intervals, 'binomialYieldIntervals': binomial_intervals,
            'strata': strata, 'conditionalPooledMetrics': report.get('summary'),
            'limitations': protocol['limitations'] + [f'Bootstrap intervals resample {group_count} independent groups; small stratum counts do not support narrow precision claims.',
                'Group bootstrap intervals can degenerate at zero or complete yield; Wilson intervals are also reported when there is exactly one attempt per group.',
                'Engineering qualification requires fixed semantic and numeric gates; it is not clinical validation.',
                'All attempted outcomes remain in yield; fidelity is conditional on returned and scored outputs.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'manifest', 'report', 'output'): parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Refusing to overwrite evaluation evidence.')
    inputs = {key: json.loads(getattr(args, key).read_text()) for key in ('protocol', 'manifest', 'report')}
    hashes = {key: hashlib.sha256(getattr(args, key).read_bytes()).hexdigest() for key in inputs}
    if inputs['manifest']['protocolSha256'] != hashes['protocol'] or inputs['report']['evaluationProtocolSha256'] != hashes['protocol']:
        raise ValueError('Evaluation protocol identity mismatch.')
    result = summarize(**inputs)
    result['inputHashes'] = hashes
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['denominatorScorecard']))
