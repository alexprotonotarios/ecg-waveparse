"""Installed-package physical setting/duration experiment using the shared scorer."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from ecg_benchmark.coordinates import contract_for_case
from ecg_benchmark.scoring import score_files


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'consumer', 'runtime', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--reuse-results', type=Path, help='Retain and rescore immutable saved calls from an interrupted harness.')
    args = parser.parse_args(); args.output = args.output.resolve(); args.protocol = args.protocol.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    payload = args.consumer/'node_modules/ecg-waveparse/runtime/payload-manifest.json'
    source_paths = [Path(__file__).resolve(), ROOT/'scripts/platform-inference.mjs', *sorted((ROOT/'ecg_benchmark').glob('*.py'))]
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    report = {'version': 1, 'clinicalValidationUse': False, 'protocolSha256': sha(args.protocol),
              'payloadManifestSha256': sha(payload), 'sourceFiles': source_hashes,
              'attempted': len(protocol['membership']), 'rows': []}
    if args.reuse_results:
        previous = json.loads((args.reuse_results/'report.json').read_text())
        if previous['protocolSha256'] != report['protocolSha256'] or previous['payloadManifestSha256'] != report['payloadManifestSha256']:
            raise ValueError('Previous calls used different inputs or payload.')
        report['reusesCallsFromReportSha256'] = sha(args.reuse_results/'report.json')
        report['harnessCorrection'] = 'Supply the required fixed 40 ms scoring limit; prior call results are unchanged.'
    def save(): (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    save()
    for member in protocol['membership']:
        for key in ('image', 'truth', 'annotations'):
            if sha(args.protocol.parent/member[key]) != member[key+'Sha256']: raise ValueError('Frozen input changed.')
        name = member['caseId']; workspace = args.output/(name+'-workspace'); output = args.output/(name+'.json')
        command = ['node', str(ROOT/'scripts/platform-inference.mjs'), str(args.consumer.resolve()),
                   str(args.protocol.parent/member['image']), str(args.runtime.resolve()), str(workspace), str(output), 'mps']
        prior_output = args.reuse_results/(name+'.json') if args.reuse_results else None
        if prior_output and prior_output.exists():
            output = prior_output.resolve(); workspace = (args.reuse_results/(name+'-workspace')).resolve()
            row = {'caseId': name, 'reusedCompletedCall': True}
        else:
          with (args.output/(name+'.log')).open('w') as log:
            try:
                execution = subprocess.run(['/usr/bin/sandbox-exec', '-p', '(version 1)(allow default)(deny network*)', *command],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=660)
                row = {'caseId': name, 'exitCode': execution.returncode}
            except subprocess.TimeoutExpired:
                row = {'caseId': name, 'status': 'harness_timeout'}
        if output.exists():
            wrapper = json.loads(output.read_text()); run = wrapper.get('run')
            if wrapper['payloadManifestSha256'] != report['payloadManifestSha256']: raise ValueError('Installed payload changed.')
            row.update(resultSha256=sha(output), elapsedMs=wrapper['elapsedMs'], error=wrapper.get('error'))
            if run:
                if run['sourceIdentity']['sha256'] != member['imageSha256']: raise ValueError('Source changed.')
                row.update(status=run['status'], decision=run.get('publicationDecision'), selectedCandidateId=run.get('selectedCandidateId'),
                    sourceInspectionAvailable=wrapper['evidence']['sourceInspectionAvailable'],
                    exportedPaperSpeed=run.get('paperSpeedMmPerSecond'), exportedGain=run.get('gainMmPerMv'))
                if wrapper['reload']['status'] != run['status']: raise ValueError('Reload changed outcome.')
                assets = {key: workspace/value['path'] for key, value in run['assets'].items()}
                for key, value in run['assets'].items():
                    expected = value.get('identity', {}).get('sha256', member['imageSha256'] if key=='input' else None)
                    if expected and sha(assets[key]) != expected: raise ValueError('Retained evidence identity changed.')
                row['quantitativeOutputPresent'] = 'canonicalCsv' in assets
                if 'canonicalCsv' in assets:
                    provenance = json.loads(assets['provenanceJson'].read_text())
                    calibration = provenance['digitization']['calibration']
                    working = provenance['preprocessing']['workingImage']
                    geometry = provenance['digitization']['pipelineEvidence']['layoutDetection']
                    row['calibration'] = calibration
                    row['physicalScaleGate'] = False
                    if geometry['coordinateSpace']=='working' and geometry['inputVariant']=='original':
                        errors = {axis: calibration['pixelsPerMm'+axis]/(member['expectedSource']['pixelsPerMm'+axis]*working['scale'+axis])-1 for axis in ('X','Y')}
                        row['physicalScaleRelativeErrors'] = errors
                        row['physicalScaleGate'] = (all(abs(error)<=protocol['maximumScaleRelativeError'] for error in errors.values())
                            and run.get('paperSpeedMmPerSecond')==member['caseMetadata']['paperSpeedMmPerSecond']
                            and run.get('gainMmPerMv')==member['caseMetadata']['gainMmPerMv'])
                    else: row['physicalScaleEvaluation'] = 'Unavailable for an unexpected transformed fixture frame.'
                    segments = json.loads(assets['segmentMapJson'].read_text())['segments']
                    score = score_files(args.protocol.parent/member['truth'], assets['canonicalCsv'], truth_rate=500, candidate_rate=500,
                        max_alignment_ms=40,
                        annotations_path=args.protocol.parent/member['annotations'], uncertainty_path=assets['uncertaintyCsv'],
                        coordinate_contract=contract_for_case(member['caseMetadata']), candidate_segments=segments, case_id=name)
                    (args.output/(name+'.score.json')).write_text(json.dumps(score, indent=2, allow_nan=False)+'\n')
                    summary = score['summary']; row.update(score=summary, semanticStatus=score['semantics']['status'], segments=segments)
                    row['returnedFidelityGate'] = (score['semantics']['status']=='passed' and summary['globalRmseUv'] is not None
                        and summary['globalRmseUv'] <= protocol['maximumFidelityRmseUv']
                        and summary['macroMeanCorrelation'] >= protocol['minimumCorrelation']
                        and summary['macroMeanCoverage'] >= protocol['minimumCoverage'])
                    row['returnedAllGates'] = row['returnedFidelityGate'] and row['physicalScaleGate']
        row.setdefault('status', 'runtime_failure'); report['rows'].append(row); save()
        print(json.dumps({key: row.get(key) for key in ('caseId', 'status', 'decision', 'quantitativeOutputPresent', 'returnedFidelityGate', 'physicalScaleGate', 'returnedAllGates')}), flush=True)
    for relative, expected in source_hashes.items():
        if sha(ROOT/relative) != expected: raise ValueError('Frozen experiment source changed.')
    report['completed'] = True; save()


if __name__ == '__main__': main()
