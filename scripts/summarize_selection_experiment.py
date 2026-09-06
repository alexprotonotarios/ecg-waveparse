"""Current-source fixed-backend, targeted-fallback, selector and oracle accounting."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from ecg_benchmark.evaluation import grouped_mean_interval


def available(row):
    return bool(row and row.get('status')=='completed' and row.get('canonicalPath'))


def scored(row):
    score=(row or {}).get('score') or {}
    return available(row) and type(score.get('comparedSamples')) in (int,float) and score['comparedSamples']>0 and type(score.get('globalRmseUv')) in (int,float) and math.isfinite(score['globalRmseUv'])


def targeted_fallback(base, fallback):
    # Deployment-available trigger only; no score/truth endpoint selects a path.
    if available(base) and base.get('qa',{}).get('passed') is True: return base
    return fallback if available(fallback) else base


def analyze(protocol, report):
    members={m['caseId']:m for m in protocol['membership']}; expected=list(members)
    if len(expected)!=len(protocol['membership']):raise ValueError('Duplicate input membership.')
    runs={}
    for row in report['rows']:
        key=(row['caseId'],row['profile'])
        if key in runs or row['caseId'] not in members or row['profile'] not in ('production','benchmark'):raise ValueError('Unexpected/duplicate run.')
        candidates={c['id']:c for c in row['candidates']}
        if len(candidates)!=len(row['candidates']):raise ValueError('Duplicate candidate.')
        runs[key]=candidates
    groups={m['groupId']:set() for m in members.values()}
    for m in members.values():groups[m['groupId']].add(m['analysisSplit'])
    if any(len(splits)>1 for splits in groups.values()):raise ValueError('Independent group crosses analysis splits.')
    development=[i for i in expected if members[i]['analysisSplit']=='development']
    comparison=[i for i in expected if members[i]['analysisSplit']=='comparison']
    if not development or not comparison:raise ValueError('A separate group comparison is required.')
    def summary(ids, choices):
        returned=[(i,choices.get(i)) for i in ids if scored(choices.get(i))]
        metrics={}
        for key in ('globalRmseUv','macroMeanCoverage','macroMeanCorrelation'):
            values=[(members[i]['groupId'],r['score'][key]) for i,r in returned if type(r['score'].get(key)) in (int,float) and math.isfinite(r['score'][key])]
            metrics[key]={'meanCase':sum(v for _,v in values)/len(values) if values else None,'groupInterval':grouped_mean_interval(values)}
        return {'attempted':len(ids),'comparableSignals':len(returned),'noComparableSignal':len(ids)-len(returned),
                'comparableSignalFraction':len(returned)/len(ids) if ids else None,
                'strictSemanticPassCount':sum(r.get('semanticStatus')=='passed' for _,r in returned),
                'placementPassCount':sum(r.get('placementPassed') is True for _,r in returned),'conditionalMetrics':metrics,
                'selectedCandidateIds':{i:(choices.get(i) or {}).get('id') for i in ids}}
    universe=sorted({c for (i,p),rows in runs.items() if p=='benchmark' for c in rows if c!='actual_selector'})
    fixed={'default':{i:runs.get((i,'benchmark'),{}).get('default') for i in expected},'native_unprepared':{}}
    for i in expected:
        native=[r for r in runs.get((i,'benchmark'),{}).values() if r.get('parameters',{}).get('kind')=='native-grid'
                and not r.get('parameters',{}).get('capabilities',{}).get('preprocessed')
                and r.get('parameters',{}).get('inputVariant')=='original']
        if len(native)>1:raise ValueError('Ambiguous fixed native backend; freeze an explicit selection rule.')
        fixed['native_unprepared'][i]=native[0] if native else None
    dev={name:summary(development,choice) for name,choice in fixed.items()}
    best=min(fixed,key=lambda k:(-dev[k]['comparableSignals'],dev[k]['conditionalMetrics']['globalRmseUv']['meanCase'] if dev[k]['conditionalMetrics']['globalRmseUv']['meanCase'] is not None else math.inf,k))
    policies={f'fixed:{name}':choice for name,choice in fixed.items()}
    policies['development_chosen_fixed']=fixed[best]
    for profile in ('production','benchmark'):
        policies[profile+'_actual_selector']={i:runs.get((i,profile),{}).get('actual_selector') for i in expected}
    for candidate in universe:
        if candidate!='default':policies['default_then_qa_fallback:'+candidate]={i:targeted_fallback(fixed['default'][i],runs.get((i,'benchmark'),{}).get(candidate)) for i in expected}
    for profile in ('production','benchmark'):
        policies[profile+'_truth_oracle']={i:min((r for c,r in runs.get((i,profile),{}).items() if c!='actual_selector' and scored(r)
            and r.get('placementPassed') is True and r['score'].get('macroMeanCoverage',0)>=protocol['oracleMinimumCoverage']),
            key=lambda r:r['score']['globalRmseUv'],default=None) for i in expected}
    summaries={name:{'development':summary(development,choice),'comparison':summary(comparison,choice),'allAttempted':summary(expected,choice)} for name,choice in policies.items()}
    primary=['development_chosen_fixed','production_actual_selector','benchmark_actual_selector','benchmark_truth_oracle']
    common=[i for i in expected if all(scored(policies[name].get(i)) for name in primary)]
    marginal=[]
    for candidate in universe:
        attempts=[(i,runs.get((i,'benchmark'),{}).get(candidate)) for i in expected]
        generated=[(i,r) for i,r in attempts if r]
        paired=[(i,r,fixed['default'][i]) for i,r in generated if scored(r) and scored(fixed['default'][i])]
        gains=[b['score']['globalRmseUv']-r['score']['globalRmseUv'] for _,r,b in paired]
        memory=[r['peakSampledCandidateProcessTreeRssBytes'] for _,r in generated if r.get('peakSampledCandidateProcessTreeRssBytes') is not None]
        marginal.append({'candidateId':candidate,'attemptedInputs':len(expected),'scheduled':len(generated),'unscheduled':len(expected)-len(generated),
            'comparedWithDefault':len(paired),'meanRmseGainVsDefaultUv':sum(gains)/len(gains) if gains else None,
            'addsComparableSignalWhereDefaultMissing':sum(scored(r) and not scored(fixed['default'][i]) for i,r in generated),
            'sumCandidateRuntimeMs':sum(r.get('runtimeMs',0) for _,r in generated),
            'peakSampledCandidateProcessTreeRssBytes':max(memory,default=None),'candidateRssMeasurements':len(memory)})
    return {'version':1,'clinicalValidationUse':False,'attemptedInputs':len(expected),'independentGroups':len(groups),
            'attemptedRuns':2*len(expected),'reportedRuns':len(runs),'bestFixedChosenOnDevelopment':best,
            'policies':summaries,'commonReturnedCount':len(common),'commonReturned':{name:summary(common,policies[name]) for name in primary},
            'marginalCandidateCostsAndBenefits':marginal,'productionPolicyChanged':False,
            'decision':'Retain current selection. Small exposed development/comparison groups do not justify promotion; exhaustive-only candidates remain quarantined.',
            'limitations':report['limitations']+['Oracle uses truth offline with fixed placement and coverage constraints; unavailable candidate identity remains unavailable.',
                'Fallback availability and QA triggers do not inspect truth. These counterfactual pairs share benchmark preprocessing/cache and cannot predict standalone wall time.',
                'Sampled candidate subprocess RSS measures its own extra process footprint; it is not a subtraction of shared caches or the parent process.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','report','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Refusing to replace an ablation receipt.')
    result=analyze(json.loads(args.protocol.read_text()),json.loads(args.report.read_text()))
    result['inputHashes']={name:hashlib.sha256(getattr(args,name).read_bytes()).hexdigest() for name in ('protocol','report')}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('policies','marginalCandidateCostsAndBenefits','limitations')}))
