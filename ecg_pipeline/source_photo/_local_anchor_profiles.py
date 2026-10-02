"""Unchanged profile observations with explicit first-refusal receipts."""
import hashlib
import cv2
import numpy as np
from ecg_pipeline.source_photo import _masks,_grid_components
shaarray=lambda a:hashlib.sha256(a.tobytes()).hexdigest()

def masks(image,context):
 g=context['grid'];cal=context['sourcePaths']['calibration'];trace=context['sourcePaths']['sourceTrace'];lo,hi=g['sourceInterval'];crop=image[:,lo:hi];channels=crop.astype(np.int16);colour=(channels.max(2)-channels.min(2)>=18).astype(np.uint8);font=cal['labels']['candidate']['diagnosticGeometry']['sourceLabelGrid']['fontHeight'];neutral=_masks.masks(crop,font)['neutral-local-contrast'].astype(bool);limit=max(q['robustUpperLimit'] for q in trace['pulseModels']);old=(neutral & (crop.max(2)<=limit)).astype(np.float32);new,receipt=_grid_components.anchored_dark_components(old,colour,neutral)
 assert shaarray(crop)==g['sourceImageArraySha256'] and shaarray(colour)==g['colourOverrideSha256'] and shaarray(old)==g['occlusionOverrideSha256'] and shaarray(new)==g['darkMaskSha256'] and receipt==g['componentReceipt'] and limit==g['occlusionPulseLimit']
 return colour.astype(np.float32),old,new

def sample(field,darks,node,slope):
 height,width=field.shape;xs=np.arange(max(0,node-32),min(width,node+33),dtype=np.float32);yy=np.arange(height,dtype=np.float32)[:,None]+slope*(xs-node)[None,:];xx=np.broadcast_to(xs,yy.shape).copy();values=cv2.remap(field,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).mean(axis=1);strips=[cv2.remap(dark,xx,yy,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT) for dark in darks]
 return values,strips

def local(values,strips,center,radius):
 lo,hi=max(0,center-radius),min(len(values),center+radius+1);window=values[lo:hi];q={'window':[lo,hi],'maximum':float(window.max()) if len(window) else None,'minimum':float(window.min()) if len(window) else None,'status':None,'band':None,'candidateRow':None,'arms':None}
 if len(window)<3:q['status']='short-profile';return q
 if window.max()<.55:q['status']='low-colour-strength';return q
 if window.max()-window.min()<.25:q['status']='low-profile-contrast';return q
 at=int(np.argmax(window));q['peakIndex']=lo+at
 if at in [0,len(window)-1]:q['status']='peak-at-window-edge';return q
 a,b=at,at+1;cutoff=window.min()+.5*(window.max()-window.min());q['halfHeightCutoff']=float(cutoff)
 while a>0 and window[a-1]>=cutoff:a-=1
 while b<len(window) and window[b]>=cutoff:b+=1
 if a==0 or b==len(window):q['status']='band-touches-window-edge';return q
 candidate=float(np.average(np.arange(lo+a,lo+b),weights=np.maximum(0,window[a:b]-window.min())));arms=[]
 for strip in strips:
  dark=np.any(strip[lo+a:lo+b]>.5,axis=0);fraction=float(dark.mean());arms.append({'darkColumns':dark.astype(int).tolist(),'darkColumnFraction':fraction,'observedRow':None if fraction>.1 else candidate})
 q.update(status='ridge-profile',band=[lo+a,lo+b],candidateRow=candidate,arms=arms);return q

def census(image,context):
 field,old,new=masks(image,context);g=context['grid'];raw=g['componentLimited']['arrays'];slope=g['componentLimited']['evidence']['coarseSlope'];period=g['componentLimited']['evidence']['majorPeriodPixels'];mid=(field.shape[1]-1)//2;radius=max(3,int(np.floor(.4*period)));rows=[];profiles=[];cross={tuple(v['key']):v for v in g['crossingChecks']};receipts={(v['row'],v['node']):v for v in raw['ridgeReceipts']}
 for j,node in enumerate(raw['xNodes']):
  values,strips=sample(field,[old,new],int(node),slope);profiles.append({'nodeIndex':j,'values':values.tolist()})
  expected=np.asarray(raw['seedRows'])+slope*(node-mid)
  for i,location in enumerate(expected):
   q=local(values,strips,int(round(location)),radius);baseline=None if q['arms'] is None else q['arms'][0]['observedRow'];component=None if q['arms'] is None else q['arms'][1]['observedRow'];assert baseline==g['baseline']['arrays']['observedRowsAll'][i][j] and component==raw['observedRowsAll'][i][j]
   if q['arms'] is not None:
    receipt=receipts[i,j];assert q['band']==receipt['bandInNodeCoordinates'] and q['candidateRow']==receipt['candidateRow'] and q['maximum']==receipt['strength'] and q['minimum']==receipt['profileMinimum'] and q['arms'][1]['darkColumns']==receipt['darkColumns'] and q['arms'][1]['darkColumnFraction']==receipt['darkColumnFraction']
   else:assert (i,j) not in receipts
   veto=cross.get((i,j),{}).get('crossesKnownSourcePath',False);guarded=None if veto else component;assert guarded==g['guardedObservedRows'][i][j]
   stage=('baseline-retained' if baseline is not None else 'component-recovery-retained') if guarded is not None else ('crossing-veto' if veto else 'component-dark-occlusion' if component is None and q['status']=='ridge-profile' else q['status'])
   eligible=not(q['arms'] and q['arms'][1]['darkColumnFraction']>.1) and not veto
   rows.append({'row':i,'nodeIndex':j,'sourceX':g['sourceInterval'][0]+node,'expectedRow':float(location),'profile':q,'baselineRow':baseline,'componentRow':component,'guardedRow':guarded,'crossingVeto':veto,'eligibleInNodeDenominator':bool(eligible),'stage':stage})
 supports=[]
 for j in range(len(raw['xNodes'])):
  rr=[q for q in rows if q['nodeIndex']==j];den=sum(v['eligibleInNodeDenominator'] for v in rr);num=sum(v['guardedRow'] is not None for v in rr);fraction=num/max(1,den);assert fraction==g['nodeSupport'][j];supports.append({'nodeIndex':j,'observed':num,'eligible':den,'support':fraction})
 return {'rows':rows,'profiles':profiles,'nodeSupport':supports,'allBaselineComponentGuardedObservationsExact':True,'maskHashesExact':True}
