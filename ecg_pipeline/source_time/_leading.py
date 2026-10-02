"""Direct observation of one preceding major line; no imputed positions."""
import cv2
import numpy as np
from scipy.signal import find_peaks

def profile(image, node, slope):
    channels=image.astype(np.int16)
    colour=((channels.max(2)-channels.min(2)>=18)&(channels.max(2)>=150)).astype(np.float32)
    dark=(channels.max(2)<150).astype(np.float32)
    xs=np.arange(max(0,node-32),min(image.shape[1],node+33),dtype=np.float32)
    yy=np.arange(image.shape[0],dtype=np.float32)[:,None]+slope*(xs-node)[None,:]
    xx=np.broadcast_to(xs,yy.shape).copy()
    return cv2.remap(colour,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).mean(1),cv2.remap(dark,xx,yy,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)

def ridge(values,dark,expected,period):
    radius=max(3,int(np.floor(.4*period)));center=int(round(expected));lo=max(0,center-radius);hi=min(len(values),center+radius+1)
    local=values[lo:hi];out={'search':[lo,hi],'coordinate':None,'occludedFraction':None,'halfPeakRange':None}
    if len(local)<3 or local.max()<.55 or local.max()-local.min()<.25:return out|{'reason':'weak-or-unavailable-ridge'}
    at=int(np.argmax(local))
    if at==0 or at==len(local)-1:return out|{'reason':'search-edge-maximum'}
    a,b=at,at+1;cutoff=local.min()+.5*(local.max()-local.min())
    while a>0 and local[a-1]>=cutoff:a-=1
    while b<len(local) and local[b]>=cutoff:b+=1
    if a==0 or b==len(local):return out|{'reason':'unbracketed-half-peak'}
    occluded=float(np.any(dark[lo+a:lo+b]>.5,axis=0).mean());out.update(occludedFraction=occluded,halfPeakRange=[lo+a,lo+b])
    if occluded>.1:return out|{'reason':'dark-occlusion'}
    weights=np.maximum(0,local[a:b]-local.min());out.update(coordinate=float(np.average(np.arange(lo+a,lo+b),weights=weights)),reason=None)
    return out

def observe(transposed_full, old, crop_x):
    arrays=old['arrays'];period=old['evidence']['majorPeriodPixels'];slope=old['evidence']['coarseSlope'];nodes=np.asarray(arrays['xNodes']);mid=(transposed_full.shape[1]-1)//2
    first=float(arrays['seedRows'][0]+crop_x);values,_=profile(transposed_full,mid,slope)
    peaks,_=find_peaks(values,height=.55,prominence=.25,distance=max(3,int(.6*period)))
    candidates=[int(x) for x in peaks if .6*period<=first-x<=1.4*period]
    result={'firstOriginalSeedSourceX':first,'candidatePeaks':candidates,'period':period,'slope':slope,'nodes':nodes.tolist(),'newSeedSourceX':None,'observations':[],'seedAdmitted':False}
    if len(candidates)!=1:return result|{'reason':'preceding-seed-missing-or-ambiguous'}
    seed=candidates[0];result['newSeedSourceX']=seed
    if abs(first-seed-period)>2:return result|{'reason':'preceding-seed-spacing-refused'}
    for node in nodes:
        values,dark=profile(transposed_full,int(node),slope);result['observations'].append(ridge(values,dark,seed+slope*(node-mid),period))
    return result|{'seedAdmitted':True,'reason':None}

def identity(observations,old_first,nodes,period):
    new=np.array([q['coordinate'] for q in observations],float);old=np.asarray(old_first,float);nodes=np.asarray(nodes,float)
    valid=np.isfinite(new)&np.isfinite(old);width=old-new
    if not valid.any() or not np.all((width[valid]>0)&(np.rint(width[valid]/period)==1)):
        return {'accepted':False,'reason':'conflicting-or-absent-pair-width','anchorEdges':[],'independentPairs':[]}
    def segments(y):
        residual=np.full(len(y),np.nan)
        for j in range(1,len(y)-1):
            if np.isfinite(y[j-1:j+2]).all() and max(np.diff(nodes[j-1:j+2]))<=64:
                f=(nodes[j]-nodes[j-1])/(nodes[j+1]-nodes[j-1]);residual[j]=abs(y[j]-(y[j-1]*(1-f)+y[j+1]*f))
        out=[]
        for j in range(len(y)-1):
            available=residual[j:j+2][np.isfinite(residual[j:j+2])]
            out.append(bool(np.isfinite(y[j:j+2]).all() and nodes[j+1]-nodes[j]<=64 and len(available)>0 and np.all(available<=1) and abs((y[j+1]-y[j])/(nodes[j+1]-nodes[j]))<=.25))
        return out
    a,b=segments(new),segments(old)
    edges=[j for j in range(len(nodes)-1) if a[j] and b[j] and valid[j:j+2].all() and np.all(np.abs(width[j:j+2]-period)<=2)]
    pairs=[[j,k] for j in edges for k in edges if nodes[k]-nodes[j+1]>64]
    return {'accepted':bool(pairs),'reason':None if pairs else 'missing-separated-identity-anchors','anchorEdges':edges,'independentPairs':pairs,'jointlyObservedNodes':int(valid.sum()),'jointWidthRangePixels':[float(width[valid].min()),float(width[valid].max())]}
