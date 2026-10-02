"""Isolated source-local fallback; no fixture or downstream-outcome choices."""
import copy,hashlib,math
import numpy as np
from scripts.detect_ecg_layout import _as_gray,_chromatic_excess_planes
from ecg_pipeline.source_photo._period import grid_period
from ecg_pipeline.source_photo._pulse_fallback import choose,assess_context
from . import _pulse_bottom_refinement as bottom

def local(q,image):
    old=q['candidate'];cy=(q['top']+q['bottom'])/2;h,w=image.shape[:2]
    requested=[math.floor(q['xStart']-old['gridSpacingXPixels']),math.floor(cy-6*old['gridSpacingYPixels']),math.ceil(q['xStart']+11*old['gridSpacingXPixels']),math.ceil(cy+6*old['gridSpacingYPixels'])]
    bounds=[max(0,requested[0]),max(0,requested[1]),min(w,requested[2]),min(h,requested[3])];a,b,d,e=bounds
    proof={'requestedBounds':requested,'bounds':bounds,'records':[],'selectedPlane':None,'accepted':False}
    if d<=a or e<=b:return copy.deepcopy(q),{**proof,'reason':'empty-window'}
    tile=image[b:e,a:d];proof['sourceCropSha256']=hashlib.sha256(tile.tobytes()).hexdigest()
    planes=[('gray',1.-_as_gray(tile).astype(np.float32)/255.),*zip(['red-excess','green-excess','blue-excess'],_chromatic_excess_planes(tile))]
    for name,plane in planes:
        xp=plane.mean(axis=0);yp=plane.mean(axis=1);x,xc=grid_period(xp,96);y,yc=grid_period(yp,96)
        valid=x is not None and y is not None and all(math.isfinite(v) and v>0 for v in [x,y]) and all(math.isfinite(v) for v in [xc,yc])
        dis=abs(x-y)/max(x,y,1e-9) if valid else None
        proof['records'].append({'plane':name,'xProfile':xp.tolist(),'yProfile':yp.tolist(),'xPeriod':x,'xConfidence':xc,'yPeriod':y,'yConfidence':yc,
            'axisRelativeDifference':dis,'eligible':bool(valid and dis<=.15),'selectionConfidence':min(xc,yc)*(1-dis) if valid else None})
    eligible=[r for r in proof['records'] if r['eligible']]
    if not eligible:return copy.deepcopy(q),{**proof,'reason':'no-local-grid-pair'}
    selected=max(eligible,key=lambda r:r['selectionConfidence']);proof['selectedPlane']=selected['plane']
    x,y=selected['xPeriod'],selected['yPeriod'];sx,sy=old['gridScaleMmX'],old['gridScaleMmY']
    xmm,ymm=x/sx,y/sy;hmm=q['pulseHeightPixels']/ymm;wmm=q['pulseWidthPixels']/xmm
    gain=min((5.,10.,20.),key=lambda g:abs(hmm-g));speed=min((25.,50.),key=lambda z:abs(wmm/.2-z))
    ge=abs(hmm-gain)/gain;se=abs(wmm/.2-speed)/speed
    px=q['pulseWidthPixels']/(speed*.2);py=q['pulseHeightPixels']/gain;dis=abs(px-py)/max(px,py,1e-9)
    confidence=min(selected['xConfidence'],selected['yConfidence'])*q['horizontalSupport']*max(0.,1-ge)*max(0.,1-se)
    gates={'sameSettings':gain==old['gainMmPerMv'] and speed==old['paperSpeedMmPerSecond'] and sx==sy and sx in [1.,5.],
        'allowedGridScale':sx in ((1.,5.) if x>=12 else (1.,)) and sy in ((1.,5.) if y>=12 else (1.,)),
        'sourceEdgeAndPhysical':q['acceptedForSelection'] and q['neutralEdgeEvidence']['passed'] and q['physicalConsistencyPassed'],
        'sourceTopSupport':q['horizontalSupport']>=.45,'gainError':ge<=.35,'speedError':se<=.35,'axisPhysicalConsistency':dis<=.15,
        'confidenceThreshold':.35<=confidence<=1.,'confidenceImproved':confidence>old['confidence']}
    v=copy.deepcopy(q);cal=copy.deepcopy(old)
    cal.update(method='grid-period-plus-context-pulse-local-window-v1',gridSpacingXPixels=x,gridSpacingYPixels=y,
        pixelsPerMmX=px,pixelsPerMmY=py,measuredGainMmPerMv=hmm,pulseWidthMm=wmm,confidence=confidence,gridScaleConfidence=confidence)
    v.update(candidate=cal,gainError=ge,speedError=se,refinedDisagreement=dis)
    proof.update(gates=gates,proposedCalibration=cal,gainError=ge,speedError=se,accepted=all(gates.values()),reason=None if all(gates.values()) else 'local-calibration-gate')
    return (v if proof['accepted'] else copy.deepcopy(q)),proof

def replay(baseline,proposals,context,image):
    r=bottom.replay(baseline,proposals,context,image);r['bottomCandidate']=copy.deepcopy(r['candidate']);r['gridFallbacks']=[]
    choice=r['candidate'];index=choice['selectedProposalIndex']
    if index is None or choice['finalResult']['calibration']['confidence']>=.35:return r
    updated=copy.deepcopy(proposals)
    for v in r['refinements']:updated[v['proposalIndex']]=copy.deepcopy(v['refinedProposal'])
    for i,q in enumerate(updated):
        if not (q['acceptedForSelection'] and assess_context(q,context)['confirmed'] and q['candidate']['confidence']<.35):continue
        new,proof=local(q,image);updated[i]=new
        r['gridFallbacks'].append({'proposalIndex':i,'originalProposal':q,'localProposal':new,**proof})
    r['candidate']=choose(baseline,updated,context)
    return r
