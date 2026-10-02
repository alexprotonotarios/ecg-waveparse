"""Recompute physical evidence using audited guarded source observations."""
import copy
from ecg_pipeline.source_photo import _grid_census as pairs,_grid_calibration as period,_grid_consensus as consensus
from . import _supported_brackets as brackets, _supported_compose as compose

def rebuild(context,observations,support):
 c=copy.deepcopy(context);g=c['grid'];g['guardedObservedRows']=copy.deepcopy(observations);g['nodeSupport']=list(support);x=g['componentLimited']['arrays']['xNodes'];ids=g['majorIndices'];seed=g['componentLimited']['evidence']['localSeedCalibration'];local=c['sourcePaths']['calibration']['localGridObservations'];checks=[];distributed=[]
 for u in range(len(ids)-1):
  items=[]
  for j,xx in enumerate(x):
   v=pairs.assess_pair(g['sourceInterval'][0]+xx,[observations[u][j],observations[u+1][j]],int(ids[u+1]-ids[u]),local,period.period_at)
   v.update(gapIndex=u,nodeIndex=j,originalSeedIds=[seed['retainedSeeds'][u],seed['retainedSeeds'][u+1]],originalIntegerGapPassed=seed['gapChecks'][u]['passed']);items.append(v)
  checks.extend(items);distributed.append(consensus.consensus(items))
 g['spatialChecks']=checks;g['distributedCalibration']=distributed;br=brackets.census(c)
 inputs={'nodes':x,'observations':observations,'majorIndices':ids,'gapPassed':[v['eligible'] for v in distributed],'nodeSupport':support,'sourceInterval':g['sourceInterval']}
 return {'inputs':inputs,'spatialChecks':checks,'distributedCalibration':distributed,**br,'proposal':compose.propose(inputs,br['brackets'])}
