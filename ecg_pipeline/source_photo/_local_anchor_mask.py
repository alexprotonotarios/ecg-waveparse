"""Original dark pixels with an original same-component anchor within 16px."""
import numpy as np
from scipy.spatial import cKDTree
def local_mask(original,anchors,labels,radius=16.):
 coordinates=np.argwhere(original);ids=labels[original];order=np.argsort(ids,kind='stable');coordinates=coordinates[order];ids=ids[order];out=np.zeros(original.shape,np.float32)
 groups=np.split(np.arange(len(ids)),np.flatnonzero(np.diff(ids))+1)
 for group in groups:
  if not len(group) or ids[group[0]]==0:continue
  points=coordinates[group];flags=anchors[points[:,0],points[:,1]];source=points[flags]
  if not len(source):continue
  distances,_=cKDTree(source).query(points);keep=points[distances<=radius];out[keep[:,0],keep[:,1]]=1.
 assert np.all(out<=original) and np.all(out[anchors]==1)
 return out
