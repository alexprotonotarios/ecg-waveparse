"""Retain compact synthetic CI evidence and compare supported-platform outcomes."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from ecg_benchmark.io import read_leads
from scripts.run_platform_matrix import difference, uncertainty_difference


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def pack(regression: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    report = json.loads((regression/'report.json').read_text())
    manifest = {'version': 1, 'clinicalValidationUse': False, 'reportSha256': sha(regression/'report.json'),
                'denominator': report['denominator'], 'package': report['package'], 'scorer': report['scorer'],
                'platform': report['platform'], 'cases': []}
    if len(report['cases']) != report['denominator'] or len({r['id'] for r in report['cases']}) != report['denominator']: raise ValueError('Incomplete or duplicate CI denominator.')
    for index, case in enumerate(report['cases']):
        row = {key: case.get(key) for key in ('id', 'sourceSha256', 'truthSha256', 'status', 'outcome', 'publicationDecision', 'semanticStatus', 'score')}
        row['assets'] = {}
        if case.get('canonicalSha256'):
            run = json.loads((regression/'workspace/storage/runs'/case['runId']/'metadata.json').read_text())
            if run['sourceIdentity']['sha256'] != case['sourceSha256']: raise ValueError('Run source differs from CI case.')
            directory = output/f'case_{index:02d}'; directory.mkdir()
            for key, name in [('canonicalCsv', 'signal.csv'), ('uncertaintyCsv', 'uncertainty.csv'), ('segmentMapJson', 'segments.json')]:
                asset = run['assets'][key]; source = regression/'workspace'/asset['path']
                if sha(source) != asset['identity']['sha256']: raise ValueError('CI asset identity differs.')
                if key=='canonicalCsv' and sha(source)!=case['canonicalSha256']: raise ValueError('Canonical identity differs from report.')
                target = directory/name; shutil.copyfile(source, target)
                row['assets'][key] = {'path': str(target.relative_to(output)), 'sha256': sha(target)}
        manifest['cases'].append(row)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')


def compare(first: Path, second: Path, limits: dict):
    manifests = [json.loads((p/'manifest.json').read_text()) for p in (first, second)]
    a, b = manifests
    if a['denominator'] != b['denominator'] or a['denominator'] != limits['expectedInputs']: raise ValueError('Platform denominator mismatch.')
    indexed = [{row['id']: row for row in manifest['cases']} for manifest in manifests]
    if any(len(m['cases'])!=limits['expectedInputs'] for m in manifests) or any(len(rows)!=limits['expectedInputs'] for rows in indexed) or set(indexed[0])!=set(indexed[1]): raise ValueError('Platform membership mismatch.')
    result = {'version': 1, 'clinicalValidationUse': False, 'inputs': [sha(p/'manifest.json') for p in (first, second)],
              'samePayload': a['package']['payloadManifestSha256']==b['package']['payloadManifestSha256'],
              'sameRuntime': a['package']['runtimeManifestSha256']==b['package']['runtimeManifestSha256'],
              'sameScorer': a['scorer']==b['scorer'], 'denominator': limits['expectedInputs'], 'cases': []}
    for case_id in indexed[0]:
        x, y = [rows[case_id] for rows in indexed]
        gates = {'sameSource': x['sourceSha256']==y['sourceSha256'], 'sameTruth': x['truthSha256']==y['truthSha256'],
                 'sameStatus': x['status']==y['status'], 'sameOutcome': x['outcome']==y['outcome'],
                 'samePolicy': x['publicationDecision']==y['publicationDecision']}
        item = {'caseId': case_id, 'gates': gates}
        if not x['assets'] and not y['assets']:
            item['comparison'] = 'refusal_only'; gates['expectedRefusal'] = x['outcome']==y['outcome']=='abstention'
        elif not x['assets'] or not y['assets']:
            item['comparison'] = 'output_availability_drift'; gates['sameAvailability'] = False
        else:
            paths = []
            for directory, row in ((first, x), (second, y)):
                files = {}
                for key, asset in row['assets'].items():
                    path = (directory/asset['path']).resolve()
                    if not path.is_relative_to(directory.resolve()) or sha(path)!=asset['sha256']: raise ValueError('Invalid platform evidence asset.')
                    files[key] = path
                paths.append(files)
            signals = [read_leads(p['canonicalCsv']) for p in paths]
            wave = difference(*signals); uncertainty = uncertainty_difference(paths[0]['uncertaintyCsv'], paths[1]['uncertaintyCsv'])
            total_locations = sum(len(v) for v in signals[0].values())
            scores = [r['score'].get('summary', r['score']) for r in (x, y)]
            gates.update(strictSemantics=x['semanticStatus']==y['semanticStatus']=='passed', sameShape=wave['shapeMatches'],
                waveformTolerance=wave['rmseUv'] is not None and wave['rmseUv']<=limits['maximumWaveformRmseUv'],
                missingnessTolerance=wave['missingnessDifferences']/max(total_locations, 1)<=limits['maximumMissingnessDifferenceFraction'],
                truthRmseTolerance=abs(scores[0]['globalRmseUv']-scores[1]['globalRmseUv'])<=limits['maximumAbsoluteTruthRmseDifferenceUv'],
                correlationTolerance=abs(scores[0]['macroMeanCorrelation']-scores[1]['macroMeanCorrelation'])<=limits['maximumAbsoluteCorrelationDifference'],
                coverageTolerance=abs(scores[0]['macroMeanCoverage']-scores[1]['macroMeanCoverage'])<=limits['maximumAbsoluteCoverageDifference'],
                uncertaintyStatusTolerance=uncertainty['statusDifferences']/max(uncertainty['comparedLocations'], 1)<=limits['maximumUncertaintyStatusDifferenceFraction'],
                uncertaintySpreadTolerance=uncertainty['p95AbsoluteSpreadDifferenceUv'] is not None and uncertainty['p95AbsoluteSpreadDifferenceUv']<=limits['maximumP95UncertaintySpreadDifferenceUv'])
            item.update(comparison='waveform_and_uncertainty', waveformDifference=wave, uncertaintyDifference=uncertainty)
        item['passed'] = all(gates.values()); result['cases'].append(item)
    result['passed'] = all(result[key] for key in ('samePayload', 'sameRuntime', 'sameScorer')) and all(r['passed'] for r in result['cases'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('mode', choices=['pack', 'compare'])
    parser.add_argument('--first', type=Path, required=True); parser.add_argument('--second', type=Path)
    parser.add_argument('--limits', type=Path); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode=='pack': pack(args.first, args.output)
    else:
        if not args.second or not args.limits: parser.error('Comparison requires second and limits.')
        result = compare(args.first, args.second, json.loads(args.limits.read_text())); result['limitsSha256'] = sha(args.limits)
        with args.output.open('x') as handle: json.dump(result, handle, indent=2, allow_nan=False)
        print(json.dumps({'denominator': result['denominator'], 'passed': result['passed']})); raise SystemExit(0 if result['passed'] else 1)
