"""Read-only pixel/component and nearest-anchor census."""
import hashlib
import cv2,numpy as np
from scipy.spatial import cKDTree
from ecg_pipeline.source_photo import _masks,_grid_components
shaarray=lambda a:hashlib.sha256(a.tobytes()).hexdigest()
def reconstruct(image,context):
 g=context['grid'];lo,hi=g['sourceInterval'];crop=image[:,lo:hi];channels=crop.astype(np.int16);chroma=channels.max(2)-channels.min(2);colour=chroma>=18;cal=context['sourcePaths']['calibration'];font=cal['labels']['candidate']['diagnosticGeometry']['sourceLabelGrid']['fontHeight'];support=_masks.masks(crop,font)['neutral-local-contrast'].astype(bool);limit=g['occlusionPulseLimit'];old=support & (crop.max(2)<=limit);new,receipt=_grid_components.anchored_dark_components(old.astype(np.float32),colour,support);anchors=old & ~colour;count,labels=cv2.connectedComponents(support.astype(np.uint8),connectivity=8)
 assert receipt==g['componentReceipt'] and count-1==receipt['componentCount'] and shaarray(colour.astype(np.uint8))==g['colourOverrideSha256'] and shaarray(old.astype(np.float32))==g['occlusionOverrideSha256'] and shaarray(new)==g['darkMaskSha256']
 return {'crop':crop,'colour':colour,'support':support,'original':old,'retained':new>.5,'anchors':anchors,'labels':labels,'componentCount':count-1,'offset':lo,'chroma':chroma}
def band_coordinates(receipt,node,slope,height):
 left,right=receipt['profileXRange'];lo,hi=receipt['bandInNodeCoordinates'];xs=np.arange(left,right,dtype=np.float32);yy=np.arange(lo,hi,dtype=np.float32)[:,None]+np.float32(slope)*(xs-node)[None,:];ys=np.rint(yy).astype(int);xx=np.broadcast_to(xs.astype(int),ys.shape);pairs=sorted({(int(y),int(x)) for y,x in zip(ys.ravel(),xx.ravel()) if 0<=y<height});return pairs
def census(state,receipt,node,slope):
 coords=band_coordinates(receipt,node,slope,len(state['crop']));pixels=[];components={};trees={};offset=state['offset']
 for y,x in coords:
  label=int(state['labels'][y,x]);bgr=state['crop'][y,x].astype(int).tolist();chroma=max(bgr)-min(bgr);retained=bool(state['retained'][y,x]);nearest=None;distance=None
  if retained:
   if label not in trees:
    anchors=np.argwhere(state['anchors'] & (state['labels']==label));assert len(anchors)>0;trees[label]=(anchors,cKDTree(anchors));points=np.argwhere(state['labels']==label);components[label]={'label':label,'pixels':len(points),'neutralAnchors':len(anchors),'sourceBounds':[int(points[:,1].min())+offset,int(points[:,0].min()),int(points[:,1].max())+offset+1,int(points[:,0].max())+1]}
   anchors,tree=trees[label];distance,index=tree.query([y,x]);ay,ax=anchors[int(index)];nearest=[int(ax)+offset,int(ay)];distance=float(distance)
  pixels.append({'sourceX':x+offset,'sourceY':y,'bgr':bgr,'absoluteChroma':chroma,'normalizedChroma':chroma/max(1,max(bgr)),'originalDark':bool(state['original'][y,x]),'retainedDark':retained,'neutralAnchor':bool(state['anchors'][y,x]),'contrastSupport':bool(state['support'][y,x]),'componentLabel':label,'nearestSameComponentAnchor':nearest,'euclideanAnchorDistancePixels':distance})
 dark=[v for v in pixels if v['retainedDark']];dist=[v['euclideanAnchorDistancePixels'] for v in dark];columns=sorted({v['sourceX'] for v in dark});fraction=len(columns)/(receipt['profileXRange'][1]-receipt['profileXRange'][0]);assert fraction==receipt['darkColumnFraction']
 return {'pixels':pixels,'components':list(components.values()),'sampledPixels':len(pixels),'retainedDarkPixels':len(dark),'retainedDarkColumns':len(columns),'darkColumnFractionExact':fraction,'anchorPixelsInBand':sum(v['neutralAnchor'] for v in pixels),'anchorDistanceMedian':float(np.median(dist)) if dist else None,'anchorDistanceMaximum':max(dist,default=None),'nearestDistanceIsGeodesic':False}
