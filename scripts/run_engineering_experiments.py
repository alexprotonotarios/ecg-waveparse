"""Reproducible crop, geometry and verifier ablations; never enters a final set."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np

from ecg_pipeline.decoder_backends import TraceCrop, decode_crop
from ecg_pipeline.local_grid_warp import VerticalGridWarp
from ecg_pipeline.source_verification import verify_source_trace


def make_crop(family: str, ppm: int):
    width,height=100*ppm,40*ppm
    x=np.arange(width); t=x/(25*ppm)
    triangle=lambda center,half_width,amplitude: amplitude*np.maximum(0,1-np.abs(t-center)/half_width)
    signal=triangle(.8,.07,-2)+triangle(1,.09,10)+triangle(1.15,.08,-3)+triangle(1.55,.25,2)
    if family=="notch":signal+=triangle(1.08,.025,5)
    if family=="negative":signal=-signal
    if family=="broad":signal=triangle(1,.35,8)-triangle(1.5,.25,3)
    if family=="low_amplitude":signal*=.12
    if family=="pacing_mark":signal+=triangle(2.1,.012,6)
    y=height/2-signal*ppm
    source=np.full((height,width,3),255,dtype=np.uint8)
    for xx in range(0,width,ppm):source[:,xx]=(225,225,255)
    for yy in range(0,height,ppm):source[yy,:]=(225,225,255)
    points=np.column_stack((x,np.rint(y))).astype(np.int32)
    cv2.polylines(source,[points],False,(25,25,25),1,cv2.LINE_AA)
    if family=="overlap":cv2.line(source,(int(width*.42),int(height*.30)),(int(width*.65),int(height*.70)),(25,25,25),1)
    if family=="occluded":source[:,int(width*.22):int(width*.25)]=255
    probability=1-np.max(source.astype(float),axis=2)/255
    mask=np.max(source,axis=2)<170
    if family=="occluded":y[int(width*.22):int(width*.25)]=np.nan
    return source,probability,mask,y


def decoder_experiment(output: Path, protocol: dict):
    backends=["upstream-probability-centroid","waveparse-probability-ridge","native-connected-ink","experimental-direction-connected-v1"]
    rows=[];membership=[]
    for family in protocol["families"]:
        for ppm in protocol["pixelsPerMm"]:
            source,probability,mask,truth=make_crop(family,ppm)
            case=f"{family}_{ppm}ppm";case_dir=output/case;case_dir.mkdir()
            cv2.imwrite(str(case_dir/'source.png'),source)
            np.save(case_dir/'truth_y.npy',truth)
            digest=hashlib.sha256((case_dir/'source.png').read_bytes()).hexdigest()
            membership.append({"caseId":case,"groupId":"controlled-piecewise-v1","sourceId":family,"signalFamilyId":"controlled-piecewise-v1","split":"development","image":{"path":f"{case}/source.png","sha256":digest},"truth":{"path":f"{case}/truth_y.npy","sha256":hashlib.sha256((case_dir/'truth_y.npy').read_bytes()).hexdigest()},"evaluationProvenance":{"collection":"development","truthMethod":"piecewise-linear construction at source resolution","licenceEvidence":"repository original synthetic fixture code","pretrainedTrainingOverlap":"none_documented","previouslyExposed":True}})
            for backend in backends:
                started=time.perf_counter()
                record={"caseId":case,"backend":backend,"sourceSha256":digest}
                try:
                    result=decode_crop(TraceCrop(probability,mask,digest,f"{case}:I:panel:0",ppm,ppm),backend)
                    valid=np.isfinite(truth)&np.isfinite(result.y_pixels)
                    record.update(status="returned",rmseUv=float(np.sqrt(np.mean(((result.y_pixels[valid]-truth[valid])*1000/(ppm*10))**2))) if valid.any() else None,
                                  coverage=float(valid.sum()/np.isfinite(truth).sum()),ambiguousColumns=int(result.ambiguous_columns.sum()),missingColumns=int(np.count_nonzero(~np.isfinite(result.y_pixels))))
                    np.save(case_dir/f'{backend}.npy',result.y_pixels)
                except Exception as error:record.update(status="failure",errorType=type(error).__name__,error=str(error))
                record["runtimeSeconds"]=time.perf_counter()-started;rows.append(record)
    baseline={r['caseId']:r for r in rows if r['backend']=="waveparse-probability-ridge"}
    alternative=[r for r in rows if r['backend']=="experimental-direction-connected-v1"]
    complete=all(r['status']=="returned" and baseline[r['caseId']]['status']=="returned" and r.get('rmseUv') is not None for r in alternative)
    deltas=[baseline[r['caseId']]['rmseUv']-r['rmseUv'] for r in alternative] if complete else []
    passed=complete and np.mean(deltas)>=protocol['requiredMeanImprovementUv'] and min(deltas)>=-protocol['maximumPerCaseRmseRegressionUv'] and all(r['coverage']>=baseline[r['caseId']]['coverage']-protocol['maximumCoverageLoss'] and r['runtimeSeconds']<=protocol['maximumSecondsPerCrop'] for r in alternative)
    (output/'membership.json').write_text(json.dumps({"version":1,"cases":membership},indent=2)+'\n')
    return {"attemptedCrops":len(membership),"independentSignalFamilyCount":1,"attemptedBackendRuns":len(rows),"rows":rows,"pairedMeanImprovementUv":float(np.mean(deltas)) if deltas else None,"promotionCriteriaPassed":bool(passed),"productionChanged":False,"decision":"eligible_for_further_full_image_evaluation" if passed else "retain_current_backend; alternative_fails_prespecified_crop_gate"}


def geometry_experiment():
    rows=[]
    nodes=np.linspace(0,1000,21);reference=np.array([0.,100.,200.,300.])
    x=np.linspace(0,1000,1001);y=150+30*np.sin(x/45)
    for amplitude in (0,10,20):
        displacement=lambda value:amplitude*np.sin(np.pi*value/1000)**2
        observed=reference[:,None]+displacement(nodes)[None,:]
        warp=VerticalGridWarp(nodes,observed,reference)
        truth=np.column_stack((x,y));distorted=np.column_stack((x,y+displacement(x)))
        # Best affine vertical fit is the global baseline for this known distortion.
        affine=np.polyval(np.polyfit(nodes,displacement(nodes),1),x)
        corrected=warp.map(distorted)
        rows.append({"amplitudePixels":amplitude,
                     "globalRmsePixels":float(np.sqrt(np.mean((displacement(x)-affine)**2))),
                     "localRmsePixels":float(np.sqrt(np.mean((corrected[:,1]-truth[:,1])**2))),
                     "roundTripMaxPixels":float(np.max(np.abs(warp.map(corrected,inverse=True)-distorted))),
                     "evidence":warp.evidence})
    return {"rows":rows,"geometryOnly":True,"gridNodesIndependentlyDetected":False,"productionChanged":False,"decision":"keep_experimental; independent_grid_detection_and_image_level_comparison_required"}


def verifier_experiment(output: Path):
    width,height=300,150;x=np.arange(width);y=75-45*np.maximum(0,1-np.abs(x-130)/20)
    source=np.full((height,width,3),255,dtype=np.uint8);truth=np.column_stack((x,np.rint(y)))
    cv2.polylines(source,[truth.astype(np.int32)],False,(0,0,0),1)
    region=np.ones((height,width),dtype=bool);masks={key:np.zeros_like(region) for key in ('text','grid','calibration','annotation')}
    paths={'true_irregular':truth,'omitted_apex':np.column_stack((x,np.full(width,75))),'fabricated_notch':truth.copy(),'wrong_row':truth+np.array([0,40]),'text_tracking':truth.copy()}
    paths['fabricated_notch'][195:210,1]-=25
    cv2.putText(source,'TEST',(220,40),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
    masks['text'][20:45,218:265]=True
    paths['text_tracking'][220:260,1]=35
    rows=[]
    cv2.imwrite(str(output/'verifier-source.png'),source)
    for name,points in paths.items():
        result=verify_source_trace(source,points,region=region,excluded=masks)
        cv2.imwrite(str(output/f'verifier-{name}-omissions.png'),result.omitted_mask.astype(np.uint8)*255)
        np.save(output/f'verifier-{name}-distance.npy',result.local_distances)
        rows.append({'caseId':name,**result.summary})
    return {'rows':rows,'productionPolicyChanged':False,'evaluationScope':'independent controlled source pixels and exclusions; no patient validation'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    protocol_path=ROOT/'benchmark/protocols/decoder-experiment.v1.json';protocol=json.loads(protocol_path.read_text())
    report={'version':1,'clinicalValidationUse':False,'protocolSha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),'sourceHashes':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__).resolve(),ROOT/'ecg_pipeline/decoder_backends.py',ROOT/'ecg_pipeline/source_verification.py',ROOT/'ecg_pipeline/local_grid_warp.py',ROOT/'ecg_pipeline/probability_path.py']},'decoder':decoder_experiment(args.output,protocol['cropExperiment']),'geometry':geometry_experiment(),'verifier':verifier_experiment(args.output)}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (args.output/'.keep-artifacts.json').write_text(json.dumps({'version':1,'reason':'Frozen engineering ablation evidence'})+'\n')
    print(json.dumps({'decoder':{k:v for k,v in report['decoder'].items() if k!='rows'},'geometry':report['geometry'],'verifier':[{k:r[k] for k in ('caseId','unsupportedFraction','omittedFraction','reasons')} for r in report['verifier']['rows']]},indent=2))


if __name__=='__main__':main()
