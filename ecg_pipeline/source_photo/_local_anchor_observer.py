"""Reobserve original profiles from fresh source pixels with local anchors."""
import hashlib
import numpy as np
from . import _grid_crossings, _local_anchor_mask as mask, _local_anchor_inputs as inputs, _local_anchor_profiles as profiles
shaarray=lambda a:hashlib.sha256(a.tobytes()).hexdigest()
def observe(image,context):
 g=context['grid'];raw=g['componentLimited']['arrays'];paths=context['sourcePaths']['sourceTrace']['retainedPaths'];state=inputs.reconstruct(image,context);localized=mask.local_mask(state['original'],state['anchors'],state['labels']);assert np.all(localized<=state['retained']);slope=g['componentLimited']['evidence']['coarseSlope'];period=g['componentLimited']['evidence']['majorPeriodPixels'];radius=max(3,int(np.floor(.4*period)));mid=(localized.shape[1]-1)//2;records=[];observations=[[None]*len(raw['xNodes']) for _ in raw['seedRows']];supports=[];oldcross={tuple(v['key']):v for v in g['crossingChecks']};matched=set()
 for j,node in enumerate(raw['xNodes']):
  values,strips=profiles.sample(state['colour'].astype(np.float32),[state['retained'].astype(np.float32),localized],int(node),slope);num=den=0
  for i,seed in enumerate(raw['seedRows']):
   location=seed+slope*(node-mid);q=profiles.local(values,strips,round(location),radius);oldraw=None if q['arms'] is None else q['arms'][0]['observedRow'];newraw=None if q['arms'] is None else q['arms'][1]['observedRow'];assert oldraw==raw['observedRowsAll'][i][j];veto=None
   if newraw is not None and g['baseline']['arrays']['observedRowsAll'][i][j] is None:
    receipt={'row':i,'node':j,'profileXRange':[max(0,int(node)-32),min(localized.shape[1],int(node)+33)],'bandInNodeCoordinates':q['band']};veto=_grid_crossings.crossings(receipt,paths,g['sourceInterval'][0],raw['xNodes'],slope)
    if (i,j) in oldcross:assert veto==oldcross[i,j];matched.add((i,j))
   guarded=None if veto and veto['crossesKnownSourcePath'] else newraw;oldguard=g['guardedObservedRows'][i][j]
   if oldguard is not None:assert guarded==oldguard
   observations[i][j]=guarded;eligible=not(q['arms'] and q['arms'][1]['darkColumnFraction']>.1) and not(veto and veto['crossesKnownSourcePath']);num+=guarded is not None;den+=bool(eligible)
   records.append({'row':i,'nodeIndex':j,'profile':q,'originalGuardedRow':oldguard,'candidateRawRow':newraw,'candidateGuardedRow':guarded,'crossing':veto,'newGuardedObservation':oldguard is None and guarded is not None})
  supports.append(num/max(1,den))
 assert matched==set(oldcross)
 return {"records":records,"candidateGuardedObservedRows":observations,"candidateNodeSupport":supports,"maskSha256":shaarray(localized),"originalRetainedDarkPixels":int(state["retained"].sum()),"localizedDarkPixels":int(localized.sum()),"neutralAnchorsPreserved":int(state["anchors"].sum()),"oldGuardedObservationsRetained":sum(v["originalGuardedRow"] is not None for v in records),"newGuardedObservations":sum(v["newGuardedObservation"] for v in records),"originalCrossingsExact":len(matched)}
