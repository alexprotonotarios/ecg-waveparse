"""Isolated conservative replay: refine already eligible pulse edges only."""
import copy
import cv2
import numpy as np
from ecg_pipeline.source_photo._edge_calibration import edge_evidence
from ecg_pipeline.source_photo._pulse_fallback import assess_context,choose

def masks(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    otsu=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[0]
    threshold=min(160.,max(40.,otsu*.75))
    dark=cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY_INV)[1]>0
    neutral=dark & (np.max(image,axis=2)-np.min(image,axis=2)<=28)
    return dark,neutral,otsu,threshold

def refine(q,dark,neutral):
    v=copy.deepcopy(q);lines=[];proof=[]
    for (x,start,end,width),e in zip(q['orderedLines'],q['neutralEdgeEvidence']['verticalEdges'],strict=True):
        a,_,d,_=e['bounds'];support=np.any(neutral[:,a:d],axis=1);seed=q['top']
        if not 0<=seed<len(support) or not support[seed]:
            return v,{'attempted':True,'changed':False,'accepted':q['acceptedForSelection'],'reason':'no-neutral-support-at-recorded-top','edgeStopEvidence':proof}
        hi=seed+1
        while hi<len(support) and support[hi]:hi+=1
        darkAtStop=bool(hi<len(support) and np.any(dark[hi,a:d]))
        evidence={'band':[a,d],'firstUnsupportedY':hi,'darkAtStop':darkAtStop,'insideOriginalCommonHeight':hi<q['bottom']}
        proof.append(evidence)
        lines.append([x,start,min(end,hi),width])
    if not all(e['darkAtStop'] and e['insideOriginalCommonHeight'] for e in proof):
        return v,{'attempted':True,'changed':False,'accepted':q['acceptedForSelection'],'reason':'no-colored-continuation-on-both-edges','edgeStopEvidence':proof}
    top=q['top'];bottom=min(l[2] for l in lines)
    height=bottom-top-max(l[3] for l in lines)
    info={'attempted':True,'changed':True,'edgeStopEvidence':proof,'sourceRuns':lines,'originalHeightPixels':q['pulseHeightPixels'],'heightPixels':height,'top':top,'bottom':bottom}
    if height<=0:
        v['acceptedForSelection']=False
        return v,{**info,'accepted':False,'reason':'nonpositive-height'}
    cal=copy.deepcopy(q['candidate']);ppmx=cal['gridSpacingXPixels']/cal['gridScaleMmX'];ppmy=cal['gridSpacingYPixels']/cal['gridScaleMmY']
    hmm=height/ppmy;wmm=q['pulseWidthPixels']/ppmx
    gain=min((5.,10.,20.),key=lambda x:abs(hmm-x));speed=min((25.,50.),key=lambda x:abs(wmm/.2-x))
    ge=abs(hmm-gain)/gain;se=abs(wmm/.2-speed)/speed
    horizontal=float(np.mean(dark[max(0,top-1):min(len(dark),top+2),q['xStart']:q['xEnd']]))
    px=q['pulseWidthPixels']/(speed*.2);py=height/gain;disagreement=abs(px-py)/max(px,py,1e-9)
    originalFactor=q['horizontalSupport']*(1-q['gainError'])*(1-q['speedError']);assert originalFactor>0
    gridConfidence=q['candidate']['confidence']/originalFactor
    confidence=gridConfidence*horizontal*max(0.,1-ge)*max(0.,1-se)
    cal.update(measuredGainMmPerMv=hmm,pulseWidthMm=wmm,gainMmPerMv=gain,paperSpeedMmPerSecond=speed,
        pixelsPerMmX=px if disagreement<=.15 else ppmx,pixelsPerMmY=py if disagreement<=.15 else ppmy,
        confidence=confidence,gridScaleConfidence=confidence)
    v.update(candidate=cal,orderedLines=lines,linePair=copy.deepcopy(lines),top=top,bottom=bottom,
        pulseHeightPixels=height,horizontalSupport=horizontal,gainError=ge,speedError=se,refinedDisagreement=disagreement)
    v['neutralEdgeEvidence']=edge_evidence(neutral,v)
    v['physicalConsistencyPassed']=bool(cal['gridScaleMmX']==cal['gridScaleMmY'] and disagreement<=.15)
    v['acceptedForSelection']=bool(q['acceptedForSelection'] and horizontal>=.45 and ge<=.35 and se<=.35 and v['neutralEdgeEvidence']['passed'] and v['physicalConsistencyPassed'])
    return v,{**info,'accepted':v['acceptedForSelection'],'reason':None if v['acceptedForSelection'] else 'unchanged-physical-or-edge-gate',
        'globalGridConfidenceUnchanged':gridConfidence,'confidence':confidence,'gainError':ge,'speedError':se,'horizontalSupport':horizontal}

def replay(baseline,proposals,context,image):
    original=choose(baseline,proposals,context)
    if baseline['calibration']['detected'] or not any(q['acceptedForSelection'] and assess_context(q,context)['confirmed'] for q in proposals):
        return {'original':original,'candidate':copy.deepcopy(original),'refinements':[],'maskComputed':False}
    dark,neutral,otsu,threshold=masks(image);updated=copy.deepcopy(proposals);records=[]
    for i,q in enumerate(proposals):
        if not q['acceptedForSelection'] or not assess_context(q,context)['confirmed']:continue
        updated[i],details=refine(q,dark,neutral)
        records.append({'proposalIndex':i,'originalProposal':q,'refinedProposal':updated[i],**details})
    return {'original':original,'candidate':choose(baseline,updated,context),'refinements':records,
        'maskComputed':True,'otsu':otsu,'threshold':threshold}
