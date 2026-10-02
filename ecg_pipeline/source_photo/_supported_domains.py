"""Partial grid domains; original cell builder and source observations unchanged."""
import copy
import numpy as np
from ecg_pipeline.source_photo import _grid_field as original

def propose(nodes, observations, major_indices, gap_passed, *, node_support):
 x=np.asarray(nodes,float);y=np.asarray(observations,float);indices=np.asarray(major_indices,int);gaps=np.asarray(gap_passed,bool);support=np.asarray(node_support,float)
 old=original.build_field(x,y,indices,gaps,node_support=support)
 out={'originalField':old,'finalField':copy.deepcopy(old),'fallbackAttempted':False,'newPartialField':False,'domains':[],'excludedNodes':[],'refusalReason':None}
 if old['cells'] or old.get('reason')!='Insufficient original seed/node support.' or len(indices)<8:return out
 out['fallbackAttempted']=True
 if any(np.any(np.diff(column[np.isfinite(column)])<=0) for column in y.T):out['refusalReason']='Observed grid rows cross.';return out
 good=support>=.8;out['excludedNodes']=np.flatnonzero(~good).tolist();positions=np.flatnonzero(good);runs=[q for q in np.split(positions,np.flatnonzero(np.diff(positions)>1)+1) if q.size]
 candidates=[];nodes_out=[];segments=[];cells=[]
 for run in runs:
  left,right=int(run[0]),int(run[-1]+1);receipt={'nodeStart':left,'nodeEndExclusive':right,'nodeIds':run.tolist(),'xDomain':[float(x[left]),float(x[right-1])],'minimumNodeSupport':float(min(support[run])),'eligible':len(run)>=3,'field':None}
  if len(run)>=3:
   field=original.build_field(x[run],y[:,run],indices,gaps,node_support=support[run]);receipt['field']=field
   if field['cells']:
    for key,target,index in [('nodeChecks',nodes_out,'node'),('segmentChecks',segments,'interval'),('candidateCells',candidates,'interval'),('cells',cells,'interval')]:
     for item in field[key]:q=copy.deepcopy(item);q[index]+=left;target.append(q)
  out['domains'].append(receipt)
 if cells:
  final=copy.deepcopy(next(q['field'] for q in out['domains'] if q['field'] and q['field']['cells']))
  final.update(cells=cells,nodeChecks=nodes_out,segmentChecks=segments,candidateCells=candidates,supportedNodeDomains=[{'nodeStart':q['nodeStart'],'nodeEndExclusive':q['nodeEndExclusive'],'xDomain':q['xDomain']} for q in out['domains'] if q['field'] and q['field']['cells']],excludedNodeIndices=out['excludedNodes'],globalSupportScope='each independently supported consecutive node domain')
  out.update(finalField=final,newPartialField=True)
 return out
