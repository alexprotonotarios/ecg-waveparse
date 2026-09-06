"""Same-byte installed-language/device comparison, including uncertainty and policy."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ecg_benchmark.scoring import score_files
from ecg_benchmark.coordinates import contract_for_case
from ecg_benchmark.io import read_leads


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def difference(a,b):
    result={'comparedSamples':0,'missingnessDifferences':0,'maximumAbsoluteDifferenceUv':None,'sumSquaredDifference':0.,'shapeMatches':set(a)==set(b)}
    for lead in sorted(set(a)|set(b)):
        x,y=a.get(lead,np.empty(0)),b.get(lead,np.empty(0))
        if x.shape!=y.shape:result['shapeMatches']=False;continue
        finite=np.isfinite(x)&np.isfinite(y);delta=x[finite]-y[finite]
        result['comparedSamples']+=int(finite.sum());result['missingnessDifferences']+=int(np.count_nonzero(np.isfinite(x)!=np.isfinite(y)))
        if delta.size:
            result['maximumAbsoluteDifferenceUv']=max(result['maximumAbsoluteDifferenceUv'] or 0,float(np.max(np.abs(delta))))
            result['sumSquaredDifference']+=float(np.sum(delta**2))
    result['rmseUv']=float(np.sqrt(result['sumSquaredDifference']/result['comparedSamples'])) if result['comparedSamples'] else None
    del result['sumSquaredDifference'];return result


def uncertainty_difference(a,b):
    def load(path):
        with path.open() as handle:
            reader=csv.DictReader(handle)
            if not {'lead','canonical_sample','status','candidate_count','candidate_spread_uv'}<=set(reader.fieldnames or []):raise ValueError('Incomplete uncertainty schema.')
            rows=list(reader)
        keyed={(r['lead'],int(r['canonical_sample'])):r for r in rows}
        if len(keyed)!=len(rows):raise ValueError('Duplicate uncertainty location.')
        return keyed
    x,y=load(a),load(b);keys=set(x)|set(y)
    result={'comparedLocations':len(keys),'locationDifferences':len(set(x)^set(y)),
            'statusDifferences':sum(x.get(k,{}).get('status')!=y.get(k,{}).get('status') for k in keys),
            'candidateCountDifferences':sum(x.get(k,{}).get('candidate_count')!=y.get(k,{}).get('candidate_count') for k in keys)}
    delta=[]
    for key in set(x)&set(y):
        a,b=x[key].get('candidate_spread_uv'),y[key].get('candidate_spread_uv')
        if a and b and np.isfinite(float(a)) and np.isfinite(float(b)):delta.append(abs(float(a)-float(b)))
    result.update(spreadComparedLocations=len(delta),p95AbsoluteSpreadDifferenceUv=float(np.quantile(delta,.95)) if delta else None,
                  maximumAbsoluteSpreadDifferenceUv=max(delta,default=None))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','consumer','python-consumer','runtime','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.protocol=args.protocol.resolve();args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False,mode=0o700)
    protocol=json.loads(args.protocol.read_text())
    def verify():
        for relative,expected in protocol['sourceFiles'].items():
            if sha(ROOT/relative)!=expected:raise ValueError('Frozen matrix source changed.')
        for member in protocol['membership']:
            for key in ('image','truth','annotations'):
                if sha((args.protocol.parent/member[key]).resolve())!=member[key+'Sha256']:raise ValueError('Frozen matrix input changed.')
    verify()
    report={'version':1,'clinicalValidationUse':False,'protocolSha256':sha(args.protocol),'platform':platform.platform(),
            'plannedInputs':len(protocol['membership']),'plannedRuns':len(protocol['membership'])*len(protocol['devices'])*2,
            'attemptedInputs':0,'attemptedRuns':0,'rows':[],'comparisons':[]}
    def save():(args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save();indexed={}
    for member in protocol['membership']:
        source=(args.protocol.parent/member['image']).resolve()
        report['attemptedInputs']+=1
        for device in protocol['devices']:
            for interface in ('javascript','python'):
                report['attemptedRuns']+=1;save()
                name=member['caseId']+'-'+device+'-'+interface
                output=args.output/(name+'.json');workspace=args.output/(name+'-workspace');profile=args.output/(name+'-resources.json')
                # Resolving the Python executable symlink bypasses its virtualenv.
                command=(['node','scripts/platform-inference.mjs',str(args.consumer.resolve())] if interface=='javascript' else [str(args.python_consumer.absolute()),str(ROOT/'scripts/platform_inference.py')])
                command += [str(source),str(args.runtime.resolve()),str(workspace),str(output),device]
                network_guard=['/usr/bin/sandbox-exec','-p','(version 1)(allow default)(deny network*)'] if sys.platform=='darwin' else []
                with (args.output/(name+'.log')).open('w') as log:
                    execution=subprocess.run([sys.executable,'scripts/profile_waveparse.py','--output',str(profile),'--workspace',str(workspace),'--interval','1','--',*network_guard,*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                row={'caseId':member['caseId'],'device':device,'interface':interface,'exitCode':execution.returncode}
                if profile.exists():
                    resource=json.loads(profile.read_text());row.update(resourceReportSha256=sha(profile),peakSampledRssBytes=resource['peakSampledProcessTreeRssBytes'],peakWorkspaceBytes=resource['peakSampledWorkspaceBytes'])
                if output.exists():
                    wrapper=json.loads(output.read_text());run=wrapper.get('run');row.update(resultSha256=sha(output),elapsedMs=wrapper['elapsedMs'],error=wrapper.get('error'))
                    if wrapper['payloadManifestSha256']!=protocol['payloadManifestSha256']:raise ValueError('Language payload differs from frozen protocol.')
                    if run:
                        if run['sourceIdentity']['sha256']!=member['imageSha256']:raise ValueError('Admitted source differs.')
                        row.update(status=run['status'],publicationDecision=run.get('publicationDecision'),runtimeManifestSha256=run['waveparse']['runtimeManifestSha256'],candidateOutcomes=[{'id':c['id'],'status':c['status'],'runtimeMs':c.get('runtimeMs')} for c in run.get('digitizer',{}).get('candidates',[])],stages=run.get('executionProfile'))
                        if wrapper['reload']['status']!=run['status'] or wrapper['evidence']['sourceInspectionAvailable'] is not True:raise ValueError('Durable source inspection failed.')
                        assets={k:workspace/v['path'] for k,v in run['assets'].items()}
                        for key,asset in run['assets'].items():
                            expected=asset.get('identity',{}).get('sha256',member['imageSha256'] if key=='input' else None)
                            if expected and sha(assets[key])!=expected:raise ValueError('Run asset identity mismatch.')
                        if 'canonicalCsv' in assets:
                            score=score_files((args.protocol.parent/member['truth']).resolve(),assets['canonicalCsv'],truth_rate=member['caseMetadata']['sampleRateHz'],candidate_rate=500,
                                max_alignment_ms=40,
                                annotations_path=(args.protocol.parent/member['annotations']).resolve(),uncertainty_path=assets.get('uncertaintyCsv'),case_id=name,
                                coordinate_contract=contract_for_case(member['caseMetadata']),candidate_segments=json.loads(assets['segmentMapJson'].read_text())['segments'])
                            scorepath=args.output/(name+'.score.json');scorepath.write_text(json.dumps(score,indent=2,allow_nan=False)+'\n')
                            row.update(score=score['summary'],semanticStatus=score['semantics']['status'],scoreSha256=sha(scorepath),canonicalSha256=sha(assets['canonicalCsv']))
                            indexed[(member['caseId'],device,interface)]={'row':row,'signals':read_leads(assets['canonicalCsv']),'uncertainty':assets['uncertaintyCsv']}
                row.setdefault('status','runtime_failure');report['rows'].append(row);save()
                row['resourceGates']={
                    'elapsedBudget':row.get('elapsedMs') is not None and row['elapsedMs']<=protocol['budgets']['maximumCallSeconds']*1000,
                    'sampledRssBudget':row.get('peakSampledRssBytes') is not None and row['peakSampledRssBytes']<=protocol['budgets']['maximumSampledRssGiB'][device]*1024**3,
                    'workspaceBudget':row.get('peakWorkspaceBytes') is not None and row['peakWorkspaceBytes']<=protocol['budgets']['maximumWorkspaceGiB']*1024**3}
                save()
                print(json.dumps({k:row[k] for k in ('caseId','device','interface','status')}),flush=True)
        pairs=[((member['caseId'],d,'javascript'),(member['caseId'],d,'python'),'language') for d in protocol['devices']]
        if 'cpu' in protocol['devices'] and 'mps' in protocol['devices']:pairs.append(((member['caseId'],'mps','javascript'),(member['caseId'],'cpu','javascript'),'device'))
        for first,second,kind in pairs:
            item={'caseId':member['caseId'],'comparison':kind,'reference':list(first[1:]),'candidate':list(second[1:])}
            if first not in indexed or second not in indexed:item.update(status='unavailable_signal',passed=False)
            else:
                a,b=indexed[first],indexed[second];delta=difference(a['signals'],b['signals']);u=uncertainty_difference(a['uncertainty'],b['uncertainty'])
                gates={'sameSourceAndRuntime':a['row']['runtimeManifestSha256']==b['row']['runtimeManifestSha256'],
                    'samePolicy':a['row']['publicationDecision']==b['row']['publicationDecision'],'strictSemantics':a['row']['semanticStatus']==b['row']['semanticStatus']=='passed','sameShape':delta['shapeMatches']}
                if kind=='language':gates.update(waveformTolerance=delta['maximumAbsoluteDifferenceUv'] is not None and delta['maximumAbsoluteDifferenceUv']<=protocol['language']['maximumDifferenceUv'],
                    sameMissingness=delta['missingnessDifferences']==0,sameUncertaintyStatuses=u['statusDifferences']==0,sameCandidateCounts=u['candidateCountDifferences']==0,
                    sameSpread=u['maximumAbsoluteSpreadDifferenceUv'] is not None and u['maximumAbsoluteSpreadDifferenceUv']<=protocol['language']['maximumDifferenceUv'])
                else:
                    ga,gb=a['row']['score'],b['row']['score'];limits=protocol['device']
                    gates.update(rmseNoninferiority=gb['globalRmseUv']-ga['globalRmseUv']<=limits['maximumRmseLossUv'],correlationNoninferiority=ga['macroMeanCorrelation']-gb['macroMeanCorrelation']<=limits['maximumCorrelationLoss'],
                        coverageNoninferiority=ga['macroMeanCoverage']-gb['macroMeanCoverage']<=limits['maximumCoverageLoss'],
                        uncertaintyStatusTolerance=u['statusDifferences']/max(u['comparedLocations'],1)<=limits['maximumUncertaintyStatusDifferenceFraction'],
                        uncertaintySpreadTolerance=u['p95AbsoluteSpreadDifferenceUv'] is not None and u['p95AbsoluteSpreadDifferenceUv']<=limits['maximumP95SpreadDifferenceUv'])
                item.update(status='compared',waveformDifference=delta,uncertaintyDifference=u,gates=gates,passed=all(gates.values()))
            report['comparisons'].append(item);save()
    verify();report['completed']=True;save()


if __name__=='__main__':main()
