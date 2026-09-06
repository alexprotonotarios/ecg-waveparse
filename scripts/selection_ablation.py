"""Offline selection ablation of an existing run. Never writes a selector profile."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ecg_benchmark.evaluation import validate_independence


def finite(value):return type(value) in (float,int) and math.isfinite(value)


def analyze(manifest: dict, run: dict, directory: Path) -> dict:
    validate_independence(manifest['cases'])
    expected=run['caseIds'];by_case={case['caseId']:case for case in manifest['cases']}
    if len(set(expected))!=len(expected) or set(expected)-set(by_case):raise ValueError('Invalid attempted case denominator.')
    grouped=defaultdict(dict);score_hashes={}
    for row in run['results']:
        case_id,candidate=row['caseId'],row['candidateId']
        if case_id not in expected or candidate in grouped[case_id]:raise ValueError('Unexpected or duplicate result.')
        data={**row,'metrics':None}
        if row.get('status')=='completed' and row.get('scorePath'):
            path=(directory/row['scorePath']).resolve()
            if not path.is_relative_to(directory.resolve()):raise ValueError('Score path escapes the run.')
            payload=path.read_bytes();score=json.loads(payload)
            score_hashes[row['scorePath']]=hashlib.sha256(payload).hexdigest()
            summary=score['summary']
            if finite(summary.get('globalRmseUv')):data['metrics']=summary
        grouped[case_id][candidate]=data
    candidates=sorted({candidate for rows in grouped.values() for candidate in rows if candidate!='selector'})
    def summarize(case_ids,choices):
        selected=[choices.get(case_id) for case_id in case_ids]
        returned=[row for row in selected if row and row['metrics']]
        mean=lambda key:sum(row['metrics'][key] for row in returned if finite(row['metrics'].get(key)))/sum(finite(row['metrics'].get(key)) for row in returned) if any(finite(row['metrics'].get(key)) for row in returned) else None
        return {'attempted':len(case_ids),'returned':len(returned),'missingOrFailed':len(case_ids)-len(returned),'returnedFraction':len(returned)/len(case_ids) if case_ids else None,
                'meanCaseRmseUv':mean('globalRmseUv'),'meanCaseCoverage':mean('macroMeanCoverage'),'meanCaseCorrelation':mean('macroMeanCorrelation'),
                'selectedCandidateRuntimeMs':sum(row.get('runtimeMs',0) for row in selected if row),'strictSemanticQualification':'unavailable_in_historical_scorer',
                'referenceQualifiedYield':None}
    development=[case_id for case_id in expected if by_case[case_id]['split']=='development']
    validation=[case_id for case_id in expected if by_case[case_id]['split']=='validation']
    if not development or not validation:raise ValueError('Separate development and validation memberships are required.')
    choices={candidate:{case_id:grouped[case_id].get(candidate) for case_id in expected} for candidate in candidates+['selector']}
    dev_fixed={candidate:summarize(development,choices[candidate]) for candidate in candidates}
    # Lexicographic selection: complete availability first, then conditional RMSE.
    # It is chosen using development scores only and is not a production change.
    best_fixed=min(candidates,key=lambda c:(-dev_fixed[c]['returned'],dev_fixed[c]['meanCaseRmseUv'] if dev_fixed[c]['meanCaseRmseUv'] is not None else math.inf,c))
    baseline_id='default'
    policies={f'fixed:{candidate}':choices[candidate] for candidate in candidates}
    policies['actual_selector']=choices['selector']
    policies['development_chosen_fixed']=choices[best_fixed]
    for fallback in candidates:
        if fallback==baseline_id:continue
        policies[f'default_then_missing_fallback:{fallback}']={case_id:choices.get(baseline_id,{}).get(case_id) if (choices.get(baseline_id,{}).get(case_id) or {}).get('metrics') else choices[fallback].get(case_id) for case_id in expected}
    policies['offline_truth_oracle']={case_id:min((row for candidate,row in grouped[case_id].items() if candidate!='selector' and row['metrics']),key=lambda row:row['metrics']['globalRmseUv'],default=None) for case_id in expected}
    summaries={name:{'development':summarize(development,policy),'validation':summarize(validation,policy),'allAttempted':summarize(expected,policy)} for name,policy in policies.items()}
    common=[case_id for case_id in expected if all((policies[name].get(case_id) or {}).get('metrics') for name in ('actual_selector','development_chosen_fixed','offline_truth_oracle'))]
    return {'version':1,'clinicalValidationUse':False,'extractionSourceIdentity':run.get('gitCommit'),'attempted':len(expected),'development':len(development),'validation':len(validation),
            'bestFixedChosenOnDevelopment':best_fixed,'policies':summaries,'commonReturnedCount':len(common),'commonReturned':{name:summarize(common,policies[name]) for name in ('actual_selector','development_chosen_fixed','offline_truth_oracle')},
            'generatedCandidateRuntimeMs':sum(row.get('runtimeMs',0) for rows in grouped.values() for candidate,row in rows.items() if candidate!='selector'),
            'scoreFileCount':len(score_hashes),'scoreFilesSha256':hashlib.sha256(json.dumps(score_hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'productionPolicyChanged':False,'decision':'Retain current policy; this exposed historical development run lacks current strict scoring, exact extraction source and independent final evaluation.',
            'limitations':['Oracle uses truth only for offline evaluation and cannot be used in production.','Fallback policies here activate only on missing outputs; fidelity-targeted triggers require a separate locked policy experiment.','Selected candidate times exclude preprocessing, shared caches and scheduling; they are not end-to-end latency predictions.','Historical validation was previously exposed; it is not an independent final evaluation.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():parser.error('Refusing to replace an ablation receipt.')
    report=analyze(json.loads(args.manifest.read_text()),json.loads(args.run.read_text()),args.run.parent)
    report['manifestSha256']=hashlib.sha256(args.manifest.read_bytes()).hexdigest();report['runSha256']=hashlib.sha256(args.run.read_bytes()).hexdigest();report['analysisScriptSha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('policies','limitations')},indent=2))
