"""Full raster geometry ablation with independently detected grid positions."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ecg_pipeline.decoder_backends import TraceCrop, decode_crop
from ecg_pipeline.local_grid_warp import detect_vertical_grid_warp, apply_vertical_grid_warp
from prepare_ecg_input import detect_page_geometry, apply_page_geometry


def sheet():
    h, w, ppm = 800, 1200, 4
    image = np.full((h,w,3),255,np.uint8)
    for x in range(0,w,ppm):
        cv2.line(image,(x,80),(x,h-1),(200,200,255),1)
    for y in range(80,h,ppm):
        cv2.line(image,(0,y),(w-1,y),(200,200,255),1)
    cv2.putText(image,"25 mm/s   10 mm/mV",(300,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    truth = []
    for index,name in enumerate(names):
        row,col = index%6,index//6
        left,right,baseline = 70+600*col,570+600*col,140+100*row
        x=np.arange(left,right); t=(x-left)/(25*ppm)
        triangle=lambda center,half,amp:amp*np.maximum(0,1-np.abs(t-center)/half)
        ysignal=triangle(.8,.07,-2)+triangle(1,.09,8)+triangle(1.15,.08,-3)+triangle(1.55,.25,2)
        if row==1: ysignal+=triangle(1.08,.025,3)
        if row==2: ysignal=-ysignal
        if row==3: ysignal=triangle(1,.35,8)-triangle(1.5,.25,3)
        if row==4: ysignal*=.15
        if row==5: ysignal+=triangle(2.1,.012,6)
        y=baseline-ysignal*ppm
        cv2.polylines(image,[np.column_stack((x,np.rint(y))).astype(np.int32)],False,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(image,name,(left-50,baseline-20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
        truth.append({"lead":name,"left":left,"right":right,"baseline":baseline,"y":y})
    cv2.polylines(image,[np.array([(10,140),(20,140),(20,100),(40,100),(40,140),(55,140)])],False,(0,0,0),1)
    return image,truth


def distort(image,amplitude,*,row_dependent=False):
    h,w=image.shape[:2];yy,xx=np.mgrid[:h,:w].astype(np.float32)
    displacement=amplitude*np.sin(np.pi*xx/(w-1))**2
    if row_dependent: displacement*=yy/h
    return cv2.remap(image,xx,yy-displacement,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))


def project(points,transform):
    hom=np.column_stack((points,np.ones(len(points))))@transform.T
    return hom[:,:2]/hom[:,2:]


def trace_scores(image,truth,source_sha,transform=None):
    transform=np.eye(3) if transform is None else np.asarray(transform,dtype=float)
    inverse=np.linalg.inv(transform)
    scores=[]
    for item in truth:
        top,bottom=item['baseline']-45,item['baseline']+46
        corners=project(np.array([[item['left'],top],[item['right']-1,top],
                                  [item['right']-1,bottom-1],[item['left'],bottom-1]]),transform)
        left=max(0,int(np.floor(corners[:,0].min())));right=min(image.shape[1],int(np.ceil(corners[:,0].max()))+1)
        top=max(0,int(np.floor(corners[:,1].min())));bottom=min(image.shape[0],int(np.ceil(corners[:,1].max()))+1)
        crop=image[top:bottom,left:right]
        probability=1-np.max(crop.astype(float),axis=2)/255
        mask=np.max(crop,axis=2)<170
        result=decode_crop(TraceCrop(probability,mask,source_sha,item['lead'],4,4),'waveparse-probability-ridge')
        path=result.y_pixels+top;finite=np.isfinite(path)
        source_points=project(np.column_stack((np.arange(left,right)[finite],path[finite])),inverse)
        inside=(source_points[:,0]>=item['left'])&(source_points[:,0]<=item['right']-1)
        source_points=source_points[inside]
        expected=np.interp(source_points[:,0],np.arange(item['left'],item['right']),item['y'])
        # Coverage is measured in source X bins, so a crop's rotation or resize
        # cannot inflate the retained physical extent.
        coverage=len(np.unique(np.rint(source_points[:,0]).astype(int)))/(item['right']-item['left'])
        scores.append({'lead':item['lead'],'coverage':min(1.,coverage),
            'rmseUv':float(np.sqrt(np.mean((source_points[:,1]-expected)**2))*25) if len(expected) else None})
    return scores


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    protocol_path=ROOT/'benchmark/protocols/detected-grid-experiment.v1.json'
    protocol=json.loads(protocol_path.read_text());source,truth=sheet();rows=[]
    cv2.imwrite(str(args.output/'flat-source.png'),source)
    for amplitude in protocol['amplitudesPixels']:
        image=distort(source,amplitude);case=f'curvature_{amplitude}px'
        path=args.output/f'{case}-original.png';cv2.imwrite(str(path),image)
        sha=hashlib.sha256(path.read_bytes()).hexdigest();started=time.perf_counter()
        geometry=detect_page_geometry(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        global_image=apply_page_geometry(image,geometry)
        record={'caseId':case,'sourceSha256':sha,'amplitudePixels':amplitude,'globalGeometry':geometry}
        try:
            warp=detect_vertical_grid_warp(image)
            local,supported=apply_vertical_grid_warp(image,warp)
            cv2.imwrite(str(args.output/f'{case}-global.png'),global_image)
            cv2.imwrite(str(args.output/f'{case}-local.png'),local)
            cv2.imwrite(str(args.output/f'{case}-support.png'),supported.astype(np.uint8)*255)
            x=np.arange(image.shape[1],dtype=float);shift=amplitude*np.sin(np.pi*x/(len(x)-1))**2
            points=np.column_stack((x,np.full(len(x),350.)+shift));corrected=warp.map(points)
            global_scores=trace_scores(global_image,truth,sha,geometry['transform']);local_scores=trace_scores(local,truth,sha)
            global_rmse=float(np.mean([r['rmseUv'] for r in global_scores if r['rmseUv'] is not None]))
            local_rmse=float(np.mean([r['rmseUv'] for r in local_scores if r['rmseUv'] is not None]))
            record.update(status='returned',gridEvidence=warp.evidence,globalScores=global_scores,localScores=local_scores,
                globalMeanRmseUv=global_rmse,localMeanRmseUv=local_rmse,
                geometryRmsePixels=float(np.sqrt(np.mean((corrected[:,1]-350)**2))),
                globalGeometryRmsePixels=float(np.sqrt(np.mean(shift**2))),
                roundTripMaxPixels=float(np.max(np.abs(warp.map(corrected,inverse=True)-points))))
            elapsed=time.perf_counter()-started
            record['checks']={'geometry':record['geometryRmsePixels']<=protocol['maximumGeometryResidualPixels'],
                'roundTrip':record['roundTripMaxPixels']<=protocol['maximumRoundTripPixels'],
                'coverage':min(r['coverage'] for r in local_scores)>=protocol['minimumSignalCoverage'],
                'runtime':elapsed<=protocol['maximumSecondsPerImage'],
                'waveform': local_rmse-global_rmse<=protocol['maximumFlatRmseRegressionUv'] if amplitude==0 else
                    local_rmse<=global_rmse*(1-protocol['requiredCurvedRmseReductionFraction'])}
            record['passed']=all(record['checks'].values())
        except (ValueError,RuntimeError) as error:
            record.update(status='refused',reason=str(error),passed=False)
        record['seconds']=time.perf_counter()-started;rows.append(record)
        print(json.dumps({k:v for k,v in record.items() if k not in {'globalGeometry','globalScores','localScores','gridEvidence'}}),flush=True)
    refusals=[]
    for name in protocol['refusalCases']:
        image=distort(source,20,row_dependent=name=='row_dependent_distortion')
        if name=='greyscale_grid':image=cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
        if name=='missing_grid_band':image[:,500:550]=255
        cv2.imwrite(str(args.output/f'{name}.png'),image)
        try:
            detect_vertical_grid_warp(image);refusals.append({'caseId':name,'refused':False})
        except ValueError as error:refusals.append({'caseId':name,'refused':True,'reason':str(error)})
    report={'version':1,'collection':'development','clinicalValidationUse':False,'independentSignalFamilies':1,
        'protocolSha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        'sourceHashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'ecg_pipeline/local_grid_warp.py',ROOT/'ecg_pipeline/decoder_backends.py',ROOT/'scripts/prepare_ecg_input.py']},
        'attempted':len(rows),'rows':rows,'refusals':refusals,'productionChanged':False,
        'promotionCriteriaPassed':all(r['passed'] for r in rows) and all(r['refused'] for r in refusals)}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'promotionCriteriaPassed':report['promotionCriteriaPassed'],'refusals':refusals}))
    return 0 if report['promotionCriteriaPassed'] else 1


if __name__=='__main__':raise SystemExit(main())
