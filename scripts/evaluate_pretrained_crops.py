"""Paired learned/geometric crop comparison using immutable source images."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from experimental_pretrained_heatmap import PretrainedLeadHeatmap,digest_file
from ecg_pipeline.decoder_backends import TraceCrop,decode_crop


def score(path,truth,ppm):
    observed=np.isfinite(truth);valid=observed&np.isfinite(path)
    adjacent=valid[:-1]&valid[1:]
    return {'rmseUv':float(np.sqrt(np.mean((path[valid]-truth[valid])**2))*100/ppm) if valid.any() else None,
        'coverage':float(valid.sum()/observed.sum()),'comparedSamples':int(valid.sum()),
        'fabricatedGapSamples':int(np.count_nonzero(~observed&np.isfinite(path))),
        'derivativeRmseUvPerPixel':float(np.sqrt(np.mean((np.diff(path)[adjacent]-np.diff(truth)[adjacent])**2))*100/ppm) if adjacent.any() else None}


def main():
    p=argparse.ArgumentParser();p.add_argument('--crops',type=Path,required=True);p.add_argument('--comparison-root',type=Path,required=True)
    p.add_argument('--plans',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    protocol_path=ROOT/'benchmark/protocols/pretrained-crop-comparison.v1.json';protocol=json.loads(protocol_path.read_text())
    membership=json.loads((args.crops/'membership.json').read_text())['cases']
    started=time.perf_counter();model=PretrainedLeadHeatmap(args.comparison_root,args.plans,device='cpu');loaded=time.perf_counter()
    report={'version':1,'clinicalValidationUse':False,'protocolSha256':digest_file(protocol_path),
        'membershipSha256':digest_file(args.crops/'membership.json'),'scriptSha256':digest_file(__file__),
        'model':model.evidence,'loadSeconds':loaded-started,'independentSignalFamilyCount':1,
        'attemptedCrops':len(membership),'rows':[]}
    for case in membership:
        case_id=case['caseId'];image_path=args.crops/case['image']['path'];truth_path=args.crops/case['truth']['path']
        if digest_file(image_path)!=case['image']['sha256'] or digest_file(truth_path)!=case['truth']['sha256']:raise RuntimeError('Immutable crop identity mismatch')
        ppm=int(case_id.rsplit('_',1)[1].removesuffix('ppm'));image=cv2.imread(str(image_path));truth=np.load(truth_path,allow_pickle=False)
        probability=1-np.max(image.astype(float),axis=2)/255;mask=np.max(image,axis=2)<170
        folder=args.output/case_id;folder.mkdir();rows=[]
        for backend in ['waveparse-probability-ridge','experimental-direction-connected-v1','experimental-pretrained-lead-heatmap-v1']:
            begin=time.perf_counter();record={'caseId':case_id,'backend':backend,'sourceSha256':case['image']['sha256']}
            try:
                if backend=='experimental-pretrained-lead-heatmap-v1':
                    heatmaps=model.predict(image);prob=1-heatmaps[0];foreground=prob>=protocol['foregroundThreshold']
                    result=decode_crop(TraceCrop(prob,foreground,case['image']['sha256'],case_id,ppm,ppm),'waveparse-probability-ridge')
                    np.savez_compressed(folder/'learned-foreground.npz',probability=prob.astype(np.float16))
                else:result=decode_crop(TraceCrop(probability,mask,case['image']['sha256'],case_id,ppm,ppm),backend)
                np.save(folder/f'{backend}.npy',result.y_pixels)
                record.update(status='returned',score=score(result.y_pixels,truth,ppm))
            except (ValueError,RuntimeError) as error:record.update(status='failure',errorType=type(error).__name__,reason=str(error))
            record['seconds']=time.perf_counter()-begin;rows.append(record);report['rows'].append(record)
        print(json.dumps({'caseId':case_id,'results':rows}),flush=True)
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    baseline={r['caseId']:r for r in report['rows'] if r['backend']=='waveparse-probability-ridge'}
    candidate=[r for r in report['rows'] if r['backend']=='experimental-pretrained-lead-heatmap-v1']
    paired=[(baseline[r['caseId']],r) for r in candidate if r.get('score',{}).get('rmseUv') is not None]
    deltas=[b['score']['rmseUv']-c['score']['rmseUv'] for b,c in paired]
    full=len(paired)==len(membership)
    checks={'completePairedReturns':full,
        'meanImprovement':bool(full and np.mean(deltas)>=protocol['requiredMeanRmseImprovementUv']),
        'noMaterialRegression':bool(full and min(deltas)>=-protocol['maximumPerCaseRmseRegressionUv']),
        'coverage':all(c.get('score',{}).get('coverage',0)>=baseline[c['caseId']]['score']['coverage']-protocol['maximumCoverageLoss'] for c in candidate),
        'noFabricatedGapSamples':all(c.get('score',{}).get('fabricatedGapSamples',1)<=protocol['maximumFabricatedGapSamples'] for c in candidate),
        'runtime':all(c['seconds']<=protocol['maximumSecondsPerCrop'] for c in candidate)}
    report.update(pairedScoredCount=len(paired),pairedMeanImprovementUv=float(np.mean(deltas)) if deltas else None,
        checks=checks,numericalCriteriaPassed=all(checks.values()),productionChanged=False,
        decision='retain_current_backend' if not all(checks.values()) else 'requires_independent_full_pipeline_and_resource_evaluation')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'rows','model'}}))


if __name__=='__main__':main()
