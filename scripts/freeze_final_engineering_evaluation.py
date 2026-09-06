"""Freeze unexposed PTB patient membership before downloading or viewing waveforms.

This is an engineering holdout relative to known local development groups. Model
training and cross-dataset subject overlap are unknown. The final set must not
be reused for tuning without recording that it has become exposed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STRATA=[('standard_6x2','clean'),('standard_3x4','clean'),('standard_6x2','blur'),
        ('standard_12x1','jpeg_compression'),('standard_6x2','clean'),('standard_3x4','perspective'),
        ('standard_6x2','low_resolution'),('standard_12x1','rotation')]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_members(records: list[str], excluded: set[str], count: int, salt: str) -> list[dict]:
    patients={}
    for record in sorted(records):
        parts=record.split('/')
        if len(parts)!=2 or not parts[0].startswith('patient') or '..' in record:raise ValueError('Invalid public record path.')
        patients.setdefault(parts[0],record)
    eligible=sorted(set(patients)-excluded,key=lambda group:hashlib.sha256((salt+':'+group).encode()).hexdigest())
    if len(eligible)<count:raise ValueError('Insufficient unexposed groups for the declared precision.')
    return [{'caseId':f'ptb_final_{i+1:02d}','patientId':patient,'record':patients[patient],
             'layout':STRATA[i%len(STRATA)][0],'artifact':STRATA[i%len(STRATA)][1],'rhythmLead':None}
            for i,patient in enumerate(eligible[:count])]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('records','checksums','exclusions','snapshot','payload','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--yield-half-width',type=float,default=.2)
    args=parser.parse_args()
    if not .1<=args.yield_half_width<=.25:parser.error('Declare a 0.10–0.25 coarse overall yield precision target.')
    records_sha=sha(args.records);checksums_sha=sha(args.checksums)
    if records_sha!='1bdcd205432b66e98389e095c8ea9f2f642f79265a1874929a9cb6de270852fe' or checksums_sha!='994115d0484fe90daaff0e280ff6eeb1d8b76338f93d1687716efa3acf2cea83':raise ValueError('Public metadata identity changed; review before freezing.')
    exclusions=json.loads(args.exclusions.read_text());snapshot=json.loads(args.snapshot.read_text())
    code_prefixes=('src/lib/','ecg_pipeline/','ecg_benchmark/','config/','packages/runtime/','packages/javascript/src/')
    names={f for f in snapshot['files'] if f.startswith(code_prefixes)}
    names.update(['packages/python/src/ecg_waveparse/__init__.py','scripts/waveparse-regression.mjs','scripts/evaluation-protocol.mjs',
                  'scripts/score_digitization.py','scripts/build_public_engineering_pilot.py','scripts/freeze_final_engineering_evaluation.py',
                  'scripts/summarize_public_engineering_pilot.py','scripts/detect_ecg_layout.py','scripts/prepare_ecg_input.py','output/render_digitized_paper.py'])
    for name in names:
        if sha(ROOT/name)!=snapshot['files'].get(name):raise ValueError('Evaluation code differs from its frozen source snapshot.')
    count=math.ceil(1.96**2*.5*.5/args.yield_half_width**2)
    salt='waveparse-final-engineering-20260906-v1'
    members=choose_members(args.records.read_text().splitlines(),set(exclusions['excludedPatientGroups']),count,salt)
    protocol={'version':1,'createdAt':datetime.now(timezone.utc).isoformat(),'collection':'final_engineering_evaluation',
        'sourceType':'ptb_waveform_render','suiteId':'ptb_final_engineering_20260906','clinicalValidationUse':False,
        'dataset':'PTB Diagnostic ECG Database','datasetVersion':'1.0.0','licence':'Open Data Commons Attribution License v1.0',
        'licenceEvidence':'https://physionet.org/content/ptbdb/1.0.0/','pretrainedTrainingOverlap':'unknown','previouslyExposed':False,
        'recordsSha256':records_sha,'checksumsSha256':checksums_sha,'exclusionAuditSha256':sha(args.exclusions),
        'independentUnit':'PTB patient; one recording and one image per patient','stratificationKeys':['layout','artifact'],
        'sampleSize':{'count':count,'targetApproximateOverallYieldHalfWidth':args.yield_half_width,
            'basis':'ceil(1.96^2 * 0.5 * 0.5 / half_width^2); conservative binomial planning. Report Wilson intervals and group bootstrap, not narrow per-stratum or clinical precision.'},
        'selection':{'salt':salt,'method':'Hash-order patients after excluding every documented locally exposed PTB patient; use first lexical recording before generating variants.'},
        'membership':members,'sourceSampleRateHz':1000,'exportSampleRateHz':500,'intervalSeconds':[10,20],
        'resampling':'One-second padding, resample_poly 1/2, trim, per-lead median centering before independent rendering.',
        'render':{'pixelsPerMm':6,'speedMmPerSecond':25,'gainMmPerMv':10,'gridPalettes':['red','blue','pink','grey'],'traceWidthPixels':2},
        'gates':{'globalRmseUv':{'maximum':100},'macroMeanCoverage':{'minimum':.9},'macroMeanCorrelation':{'minimum':.9}},
        'semanticPassRequired':True,'accessControlReference':'Owner-only local directory, frozen source/gates/membership before waveform download; one-use runner access log.',
        'accessLogFile':'access.jsonl','sourceLock':{'payloadManifestSha256':sha(args.payload),
            'sourceSnapshotManifest':str(args.snapshot.resolve().relative_to(ROOT)),'sourceSnapshotSha256':sha(args.snapshot),
            'scorerFiles':{name:snapshot['files'][name] for name in sorted(names)}},
        'policy':'Evaluate unchanged installed policy v3. Do not tune after observing these outcomes. Log exposure; any reuse is a named exposed-set evaluation.',
        'limitations':['Rendered acquired waveforms; final results do not estimate real scan/photo performance.',
            'Independent of documented local PTB development patients; cross-dataset subject linkage and pretrained training overlap are unknown.',
            'No adjudicated clinical interval labels in PTB; controlled labelled endpoint evidence is reported separately.',
            'One render template and small stratum counts give coarse engineering evidence only.']}
    args.output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    args.output.parent.chmod(0o700)
    with args.output.open('x') as handle:json.dump(protocol,handle,indent=2)
    args.output.chmod(0o600)
    print(json.dumps({'protocolSha256':sha(args.output),'independentPatients':len(members),'excludedPatients':len(exclusions['excludedPatientGroups']),'sourceSnapshotSha256':sha(args.snapshot)}))


if __name__=='__main__':main()
