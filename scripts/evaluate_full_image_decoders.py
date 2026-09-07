"""Experimental native candidate pipelines; no production source is patched.

All variants run source-derived geometry, native validity/recovery and export.
Function replacements are scoped to one sequential experiment and restored even
on failure. Missing learned support is removed from CSV and overlay validity.
The production selector and its policy are never altered by this harness.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from ecg_pipeline import native_grid_digitizer as native
from ecg_pipeline.decoder_backends import TraceCrop,decode_crop
from ecg_pipeline.source_verification import verify_source_trace
from ecg_pipeline.domain import LEAD_ORDER
from ecg_benchmark.scoring import score_files
from ecg_benchmark.coordinates import contract_for_case
from experimental_pretrained_heatmap import PretrainedLeadHeatmap,digest_file


@contextmanager
def candidate_backend(image,backend,geometry,probability,source_sha,protocol,captured):
    names=['trace_path','trace_crossing_path','native_path_to_canonical','fidelity_for_path','draw_overlay']
    originals={name:getattr(native,name) for name in names}
    strategy=None
    if backend!='native-connected-ink':
        calibration=geometry.get('calibration') or {}
        if not calibration.get('detected'):
            raise ValueError('Experimental backend requires source-detected pulse/grid calibration.')
        strategy=native.build_native_layout_strategy(geometry,width=image.shape[1],height=image.shape[0],
            paper_speed_mm_per_second=calibration['paperSpeedMmPerSecond'],detected_pixels_per_mm=calibration['pixelsPerMmX'])
        ppmx=calibration['pixelsPerMmX'];ppmy=calibration['pixelsPerMmY']
        source_evidence=native.grid_residual_evidence(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
    def lead_at(columns,row_y):
        row=int(np.argmin(np.abs(np.asarray(strategy.primary_centers)-row_y)))
        middle=float(np.mean([columns[0],columns[-1]]))
        column=int(np.clip(np.searchsorted(strategy.panel_edges,middle,side='right')-1,0,len(strategy.row_leads[row])-1))
        return strategy.row_leads[row][column]
    def availability(columns,path,lead=None):
        if backend=='native-connected-ink':return np.ones(len(columns),dtype=bool)
        y=np.clip(np.rint(path).astype(int),0,image.shape[0]-1)
        x=np.clip(columns.astype(int),0,image.shape[1]-1)
        if backend=='experimental-pretrained-lead-heatmap-v1':
            lead=lead or lead_at(columns,float(np.median(path)))
            return probability[list(LEAD_ORDER).index(lead)+1,y,x]>=protocol['learnedForegroundThreshold']
        return source_evidence[y,x]>=protocol['geometricEvidenceThreshold']
    def trace(evidence,**kwargs):
        x0,x1,y0,y1=(int(kwargs[k]) for k in ['x_start','x_end','y_start','y_end'])
        columns=np.arange(x0,x1,dtype=np.int32);lead=lead_at(columns,kwargs['row_center'])
        if backend=='experimental-pretrained-lead-heatmap-v1':
            crop=probability[list(LEAD_ORDER).index(lead)+1,y0:y1,x0:x1]
            mask=crop>=protocol['learnedForegroundThreshold'];decoder='waveparse-probability-ridge'
        else:
            crop=np.clip(evidence[y0:y1,x0:x1],0,1);mask=crop>=protocol['geometricEvidenceThreshold'];decoder=backend
        result=decode_crop(TraceCrop(crop,mask,source_sha,lead,ppmx,ppmy),decoder)
        # Native paths are integer pixels. Unsupported placeholders are removed
        # by the shared availability filter before quantitative export.
        path=np.where(np.isfinite(result.y_pixels),result.y_pixels+y0,kwargs['row_center'])
        return columns,np.rint(path).astype(np.int32)
    def canonical(columns,path,**kwargs):
        kwargs['validity']=np.asarray(kwargs.get('validity',np.ones(len(columns),bool)))&availability(columns,path)
        return originals['native_path_to_canonical'](columns,path,**kwargs)
    def fidelity(evidence,columns,path,**kwargs):
        kwargs['validity']=np.asarray(kwargs.get('validity',np.ones(len(columns),bool)))&availability(columns,path)
        return originals['fidelity_for_path'](evidence,columns,path,**kwargs)
    def overlay(source,paths,**kwargs):
        adjusted={lead:(x,y,valid&availability(x,y,lead)) for lead,(x,y,valid) in paths.items()}
        captured.update({lead:tuple(a.copy() for a in values) for lead,values in adjusted.items()})
        return originals['draw_overlay'](source,adjusted,**kwargs)
    try:
        native.draw_overlay=overlay
        if backend!='native-connected-ink':
            native.trace_path=trace;native.trace_crossing_path=trace
            native.native_path_to_canonical=canonical;native.fidelity_for_path=fidelity
        yield
    finally:
        for name,function in originals.items():setattr(native,name,function)


def verification(image,captured,folder,score,geometry,protocol):
    results=[];height,width=image.shape[:2];spacing=float(geometry.get('medianRowSpacing') or height/6)
    for lead,(columns,path,valid) in captured.items():
        points=np.column_stack((columns,path)).astype(float);points[~valid]=np.nan
        region=np.zeros((height,width),bool)
        center=float(np.median(path));top=max(0,int(center-spacing*.45));bottom=min(height,int(center+spacing*.45))
        region[top:bottom,max(0,int(columns[0])):min(width,int(columns[-1])+1)]=True
        # Grid colour exclusion is independent of either decoder. Text and
        # annotation attribution is deliberately incomplete, never asserted.
        channels=image.astype(int);grid=(channels.max(axis=2)-channels.min(axis=2))>=45
        result=verify_source_trace(image,points,region=region,excluded={'grid':grid},
            tolerance_pixels=protocol['tolerancePixels'],mask_dependency='independent-original-colour-and-darkness; region from detected row')
        np.savez_compressed(folder/f'{lead}-source-evidence.npz',points=points,omitted=result.omitted_mask,distances=result.local_distances)
        metrics=(score.get('leads') or {}).get(lead,{})
        error=metrics.get('rmseUv')
        summary=result.summary
        flag=(summary['unsupportedFraction'] is not None and summary['unsupportedFraction']>protocol['maximumUnsupportedFraction']) or (
              summary['omittedFraction'] is not None and summary['omittedFraction']>protocol['maximumOmittedFraction'])
        results.append({'lead':lead,'reconstructionRmseUv':error,'flagged':bool(flag),'sourceVerification':summary})
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',action='append',type=Path,required=True)
    parser.add_argument('--comparison-root',type=Path,required=True);parser.add_argument('--plans',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    protocol_path=ROOT/'benchmark/protocols/full-image-decoder-comparison.v1.json';protocol=json.loads(protocol_path.read_text())
    membership=[]
    for manifest in args.manifest:
        data=json.loads(manifest.read_text())
        for case in data['cases']:
            for key in ['image','truth','annotations']:
                if digest_file(manifest.parent/case[key]['path'])!=case[key]['sha256']:raise ValueError('Frozen fixture changed')
            membership.append((f"{data['suiteId']}__{case['caseId']}",case,manifest.parent))
    if len(membership)!=8:raise ValueError('This frozen comparison requires all eight baseline inputs.')
    started=time.perf_counter();model=PretrainedLeadHeatmap(args.comparison_root,args.plans,device='cpu')
    report={'version':1,'clinicalValidationUse':False,'scope':protocol['scope'],'attemptedImages':len(membership),
        'attemptedBackendRuns':len(membership)*len(protocol['backends']),'independentSignalFamilyCount':1,
        'modelLoadSeconds':time.perf_counter()-started,'model':model.evidence,'protocolSha256':digest_file(protocol_path),
        'sourceHashes':{str(p.relative_to(ROOT)):digest_file(p) for p in [Path(__file__),ROOT/'scripts/experimental_pretrained_heatmap.py',ROOT/'ecg_pipeline/native_grid_digitizer.py',ROOT/'scripts/detect_ecg_layout.py',ROOT/'ecg_pipeline/source_verification.py']},
        'rows':[],'productionChanged':False}
    for case_id,case,parent in membership:
        image=cv2.imread(str(parent/case['image']['path']));geometry=native.detect_ecg_layout_geometry(image)
        for backend in protocol['backends']:
            folder=args.output/case_id/backend;folder.mkdir(parents=True);captured={};started=time.perf_counter()
            row={'caseId':case_id,'backend':backend,'sourceSha256':case['image']['sha256'],'expectedLayout':case['layout']}
            try:
                probability=model.predict(image) if backend=='experimental-pretrained-lead-heatmap-v1' else None
                with candidate_backend(image,backend,geometry,probability,case['image']['sha256'],protocol,captured):
                    result=native.digitize_native_grid(image,image,folder)
                score=score_files(parent/case['truth']['path'],Path(result['canonicalPath']),truth_rate=case['sampleRateHz'],candidate_rate=500,
                    max_alignment_ms=40,annotations_path=parent/case['annotations']['path'],case_id=case_id,coordinate_contract=contract_for_case(case))
                (folder/'score.json').write_text(json.dumps(score,indent=2,allow_nan=False)+'\n')
                row.update(status='returned',detectedLayout=result['layout'],score=score['summary'],semanticStatus=score['semantics']['status'],
                    nativeSourceFidelityPassed=result['sourceFidelity']['passed'],canonicalSha256=digest_file(result['canonicalPath']))
                row['sourceVerification']=verification(image,captured,folder,score,geometry,protocol['sourceVerifier'])
                row['quantitativeAdmission']='not_approved_missing_independent_segment_identity'
            except (ValueError,RuntimeError,KeyError,IndexError) as error:
                refusal_prefixes=('Experimental backend requires','Native grid extraction requires',
                    'A source-fidelity path must retain','Comparison accepts BGR')
                row.update(status='abstention' if str(error).startswith(refusal_prefixes) else 'failure',
                           errorType=type(error).__name__,reason=str(error))
            row['seconds']=time.perf_counter()-started;row['withinRuntimeBudget']=row['seconds']<=protocol['maximumSecondsPerImage'];report['rows'].append(row)
            (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k not in {'score','sourceVerification'}}),flush=True)
    print(json.dumps({'attemptedImages':report['attemptedImages'],'attemptedBackendRuns':report['attemptedBackendRuns'],
        'returned':sum(r['status']=='returned' for r in report['rows']),'productionChanged':False}))


if __name__=='__main__':main()
