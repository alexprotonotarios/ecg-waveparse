"""Isolated image/observation replay with unchanged geometry and timing gates."""
import copy,hashlib,math
import cv2
import numpy as np
from . import _neutral_local_timing as _local_timing

def eligibility(image,cal):
 c=cal.get('sourceContext');v=cal.get('reconciledCalibration',{});recon=v.get('reconciliation') or {};row=cal.get('pulseCalibrationSourceRow')
 identity=bool(c and c.get('sourceIdentityVerifiedForDiagnostic') is True and c.get('sourceContextProvenance',{}).get('sourceRasterSha256')==hashlib.sha256(image.tobytes()).hexdigest())
 confidence=v.get('confidence',0)
 finite=isinstance(confidence,(int,float)) and not isinstance(confidence,bool) and math.isfinite(confidence)
 selection=cal.get('pulseSelection',{});index=selection.get('selectedProposalIndex');assessments=selection.get('candidateAssessments',[])
 matched=assessments[index]['sourceContext'].get('matchedRows',[]) if isinstance(index,int) and 0<=index<len(assessments) else []
 return {'sourceIdentity':identity,'calibrationDetected':v.get('detected') is True,'calibrationConfidence':finite and .35<=confidence<=1,
  'printedCorroboration':recon.get('state')=='corroborated_inference' and recon.get('quantitativeBlocked') is False,
  'samePhysicalGridScale':v.get('gridScaleMmX')==v.get('gridScaleMmY') and v.get('gridScaleMmX') in [1.,5.],
  'sourceRowConfirmed':isinstance(row,int) and not isinstance(row,bool) and 0<=row<4 and matched==[row],
  'existingTimingAvailable':bool(cal.get('existingTiming')),'separatorEvidenceAvailable':bool(cal.get('separatorAttempts'))}

def windows(image,cal):
 gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);otsu=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[0];cutoff=min(160.,max(40.,otsu*.75))
 mask=(gray<=math.floor(cutoff)) & (np.ptp(image.astype(np.int16),axis=2)<=28);attempts=copy.deepcopy(cal['separatorAttempts'])
 for attempt in attempts:
  for window in attempt['windows']:
   a,b,d,e=window['bounds'];profile=mask[b:e,a:d].mean(axis=0);idx=np.flatnonzero(profile>=.8);parts=[p for p in np.split(idx,np.flatnonzero(np.diff(idx)>1)+1) if p.size]
   lo,hi=window['widthRange'];runs=[]
   for part in parts:runs.append({'centerX':float(a+(part[0]+part[-1])/2),'bounds':[int(a+part[0]),b,int(a+part[-1]+1),e],
    'widthPixels':int(part.size),'minimumBlackColumnSupport':float(profile[part].min()),'meanBlackColumnSupport':float(profile[part].mean()),'widthAdmissible':bool(lo<=part.size<=hi)})
   window.update(profileSha256=hashlib.sha256(profile.tobytes()).hexdigest(),maximumColumnSupport=float(profile.max()),supportRuns=runs,candidateCount=sum(q['widthAdmissible'] for q in runs))
   window['neutralColumnProfile']=profile.tolist()
  attempt['candidateCounts']=[w['candidateCount'] for w in attempt['windows']];attempt['allNineUnique']=all(n==1 for n in attempt['candidateCounts']);attempt['originalCountsExact']=None
 return attempts,{'otsu':otsu,'cutoff':cutoff,'maskRasterSha256':hashlib.sha256(mask.tobytes()).hexdigest()}

def case(cal,attempts):
 return {'existingTiming':cal['existingTiming'],'calibration':cal['reconciledCalibration'],'diagnosticContext':cal['sourceContext'],'separatorAttempts':attempts,
  'localGrids':[{k:q[k] for k in ['row','column','bounds','selectedPair']} for q in cal['localGridObservations']], 'pulseCalibrationSourceRow':cal['pulseCalibrationSourceRow']}

def replay(image,cal):
 original=copy.deepcopy(cal.get('timing'));gates=eligibility(image,cal)
 out={'eligibility':gates,'eligible':all(gates.values()),'originalTiming':original,'finalTiming':copy.deepcopy(original),'maskApplied':False,'newTimingAdmission':False}
 if not out['eligible'] or cal['existingTiming'].get('state')=='source_supported':return out
 attempts,mask=windows(image,cal);alternate=_local_timing.propose(image,case(cal,attempts))
 out.update(maskApplied=True,maskEvidence=mask,alternativeAttempts=attempts,alternativeTiming=alternate)
 if original is not None and original.get('candidatePassed') is not True and alternate['candidatePassed']:
  out['finalTiming']=alternate;out['newTimingAdmission']=True
 return out
