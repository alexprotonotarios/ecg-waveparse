"""Profile actual production/exhaustive selection and score every retained candidate.

Candidate process RSS is sampled from owned subprocesses only. Raw command lines
are never retained. Unscheduled candidates and unsampled memory remain explicit.
The scorer receives independent case metadata; only the actual exported winner
receives its own identity/segment map. Candidate identity is never copied from it.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.profile_waveparse import directory_bytes


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_owned(pid: int, workspace: Path) -> dict:
    output = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss=,args='], text=True, timeout=5)
    rows = {}
    for line in output.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4:
            process, parent, rss = map(int, parts[:3]); rows[process] = (parent, rss*1024, parts[3])
    def descendants(root):
        owned = {root}
        while True:
            children = {p for p, (parent, _, _) in rows.items() if parent in owned}
            if children <= owned: return owned
            owned |= children
    owned = descendants(pid)
    pattern = re.compile(re.escape(str(workspace)) + r'/storage/runs/run_[A-Za-z0-9_-]+/candidates/([A-Za-z0-9_-]+)(?:/|\s|$)')
    candidates = {}
    for process in owned:
        if process not in rows: continue
        match = pattern.search(rows[process][2])
        if match:
            candidates[match[1]] = max(candidates.get(match[1], 0), sum(rows[p][1] for p in descendants(process) if p in rows))
    return {'processTreeRssBytes': sum(rows[p][1] for p in owned if p in rows), 'candidateProcessTreeRssBytes': candidates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'runtime', 'resources', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve(); args.protocol = args.protocol.resolve()
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    protocol = json.loads(args.protocol.read_text())
    def verify():
        if sha(args.resources/'payload-manifest.json') != protocol['payloadManifestSha256']: raise ValueError('Payload changed.')
        for relative, expected in protocol['sourceFiles'].items():
            if sha(ROOT/relative) != expected: raise ValueError('Frozen experiment source changed.')
        for relative, expected in json.loads((args.resources/'payload-manifest.json').read_text())['files'].items():
            if sha(args.resources/relative) != expected: raise ValueError('Installed resource changed.')
        for member in protocol['membership']:
            for key in ('image','truth','annotations'):
                if sha((args.protocol.parent/member[key]).resolve()) != member[key+'Sha256']: raise ValueError('Frozen input changed.')
    verify()
    verification_started=time.monotonic()
    subprocess.run(['/usr/bin/sandbox-exec','-p','(version 1)(allow default)(deny network*)',sys.executable,
                    str(args.resources/'runtime_setup.py'),'doctor','--runtime-dir',str(args.runtime.resolve()),
                    '--resource-root',str(args.resources.resolve()),'--device','mps'],check=True,
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    report = {'version': 1, 'clinicalValidationUse': False, 'protocolSha256': sha(args.protocol), 'attemptedInputs': len(protocol['membership']),
              'attemptedRuns': len(protocol['membership'])*2, 'rows': [], 'productionPolicyChanged': False,
              'runtimeVerificationSeconds':time.monotonic()-verification_started,
              'limitations': ['Production and exhaustive profiles are separately executed; benchmark selection is not relabelled production.',
                 'Per-candidate RSS includes its child processes and shared pages; it excludes parent/shared preprocessing and is sampled every 0.5 seconds.',
                 'Standalone candidate identity maps are unavailable; their conditional waveform error never establishes quantitative eligibility.',
                 'These exposed development inputs cannot justify a selector promotion.']}
    save = lambda: (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    save()
    for member in protocol['membership']:
        for profile in ('production', 'benchmark'):
            directory = args.output/(member['caseId']+'-'+profile)
            command = ['/usr/bin/sandbox-exec', '-p', '(version 1)(allow default)(deny network*)',
                       'node', '--import', 'tsx', 'scripts/run-selection-case.mjs', '--protocol', str(args.protocol),
                       '--case', member['caseId'], '--profile', profile, '--output', str(directory),
                       '--runtime', str(args.runtime.resolve()), '--resources', str(args.resources.resolve())]
            samples, errors = [], []
            start = time.monotonic()
            with (args.output/(directory.name+'.log')).open('w') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                while process.poll() is None:
                    try:
                        samples.append({'elapsedSeconds':time.monotonic()-start, **sample_owned(process.pid, directory/'workspace'), 'workspaceBytes':directory_bytes(directory)})
                    except (OSError, subprocess.SubprocessError, ValueError) as error: errors.append(type(error).__name__)
                    time.sleep(.5)
            row = {'caseId': member['caseId'], 'groupId': member['groupId'], 'profile': profile, 'exitCode': process.returncode,
                   'elapsedSeconds':time.monotonic()-start, 'resourceSamples':samples, 'measurementErrors':errors,
                   'peakProcessTreeRssBytes':max((s['processTreeRssBytes'] for s in samples), default=None),
                   'peakWorkspaceBytes':max((s['workspaceBytes'] for s in samples), default=None), 'candidates':[]}
            if (directory/'case-result.json').exists():
                case = json.loads((directory/'case-result.json').read_text())
                row.update(caseResultSha256=sha(directory/'case-result.json'), status=case['status'], publicationDecision=case.get('publicationDecision'),
                           selectedCandidateId=case.get('selectedCandidateId'), stages=case['stages'], pipelineEvidence=case.get('pipelineEvidence'))
                metadata = directory/'case-metadata.json'
                metadata.write_text(json.dumps(member['caseMetadata']))
                candidates = case['candidates'] + ([{'id':'actual_selector','canonicalPath':case['selectedAssets']['canonicalCsv']['path'], 'status':'completed'}]
                                                   if 'canonicalCsv' in case['selectedAssets'] else [{'id':'actual_selector','status':'abstention'}])
                for candidate in candidates:
                    item = {**candidate}
                    memory = [s['candidateProcessTreeRssBytes'][candidate['id']] for s in samples if candidate['id'] in s['candidateProcessTreeRssBytes']]
                    item['peakSampledCandidateProcessTreeRssBytes'] = max(memory, default=None)
                    if candidate.get('canonicalPath'):
                        csv = (directory/candidate['canonicalPath']).resolve()
                        if not csv.is_relative_to(directory): raise ValueError('Candidate path escapes experiment.')
                        score_path = directory/(candidate['id']+'.score.json')
                        command = [sys.executable,str(ROOT/'scripts/score_digitization.py'),'--truth',str((args.protocol.parent/member['truth']).resolve()),
                                   '--candidate',str(csv),'--annotations',str((args.protocol.parent/member['annotations']).resolve()),'--case-metadata',str(metadata),
                                   '--truth-rate',str(member['caseMetadata']['sampleRateHz']),'--candidate-rate','500','--case-id',member['caseId'],'--output',str(score_path)]
                        if candidate['id']=='actual_selector':
                            for asset, flag in [('segmentMapJson','--candidate-segments'),('uncertaintyCsv','--uncertainty')]:
                                if asset in case['selectedAssets']: command.extend([flag,str(directory/case['selectedAssets'][asset]['path'])])
                        started = time.monotonic()
                        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                        item.update(scoringSeconds=time.monotonic()-started, canonicalSha256=sha(csv), scoringExitCode=result.returncode)
                        if result.returncode==0:
                            score=json.loads(score_path.read_text())
                            item.update(score=score['summary'], semanticStatus=score['semantics']['status'],
                                        placementPassed=all(s['placementPassed'] for s in score['semantics']['segments']),scoreSha256=sha(score_path))
                        else: item['scoringFailure']='Retained score command failed; see score-error.log.'; (directory/(candidate['id']+'.score-error.log')).write_text(result.stderr)
                    row['candidates'].append(item)
            else: row['status']='harness_failure'
            report['rows'].append(row);save()
            print(json.dumps({'caseId':member['caseId'],'profile':profile,'status':row['status'],'candidates':len(row['candidates']),'seconds':row['elapsedSeconds']}),flush=True)
    verify(); report['completed']=True; save()


if __name__=='__main__':main()
