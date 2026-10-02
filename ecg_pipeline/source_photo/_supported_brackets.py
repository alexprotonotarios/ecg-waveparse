"""Enumerate directly observed indexed grid brackets; no field acceptance change."""
from ecg_pipeline.source_photo import _grid_census as pairs,_grid_calibration as period,_grid_consensus as consensus

def census(context):
 grid=context['grid'];cal=context['sourcePaths']['calibration'];local=cal['localGridObservations'];y=grid['guardedObservedRows'];x=grid['componentLimited']['arrays']['xNodes'];indices=grid['majorIndices'];support=grid['nodeSupport'];offset=grid['sourceInterval'][0];seed=grid['componentLimited']['evidence']['localSeedCalibration'];rows=[]
 for upper in range(len(indices)-1):
  for lower in range(upper+1,len(indices)):
   steps=indices[lower]-indices[upper]
   if steps>4:break
   checks=[]
   for node,position in enumerate(x):
    q=pairs.assess_pair(offset+position,[y[upper][node],y[lower][node]],int(steps),local,period.period_at)
    q.update(gapIndex=upper,nodeIndex=node,originalSeedIds=[seed['retainedSeeds'][upper],seed['retainedSeeds'][lower]],originalIntegerGapPassed=all(q['passed'] for q in seed['gapChecks'][upper:lower]));checks.append(q)
   all_nodes=consensus.consensus(checks);supported=[q for q in checks if support[q['nodeIndex']]>=.8];supported_result=consensus.consensus(supported) if supported else None
   adjacent=lower==upper+1
   if adjacent:
    assert checks==[q for q in grid['spatialChecks'] if q['gapIndex']==upper] and all_nodes==grid['distributedCalibration'][upper]
   inner=grid['distributedCalibration'][upper:lower];failed=[q for q in inner if not q['eligible']];sparse_only=bool(failed) and all(q['measuredCount']<q['minimumMeasuredCount'] and not q['failedMeasuredNodeIds'] for q in failed)
   rows.append({'upperRow':upper,'lowerRow':lower,'majorSteps':int(steps),'checks':checks,'allNodeConsensus':all_nodes,'supportedNodeConsensus':supported_result,'supportedNodeIds':[q['nodeIndex'] for q in supported],'adjacentEvidenceExact':True if adjacent else None,'interveningIneligibleGaps':[q['gapIndex'] for q in failed],'allIneligibleGapsSparseWithoutMeasuredContradiction':sparse_only,'potentialDiagnosticBracket':bool(not adjacent and sparse_only and supported_result and supported_result['eligible'])})
 return {'brackets':rows,'adjacentCount':sum(q['adjacentEvidenceExact'] is True for q in rows),'potentialDiagnosticBrackets':sum(q['potentialDiagnosticBracket'] for q in rows),'fieldChanged':False}
