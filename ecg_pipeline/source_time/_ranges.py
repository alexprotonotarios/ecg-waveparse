"""Range of an unchanged conforming paper field over observed source marks."""
import math
import numpy as np
from shapely.geometry import Polygon,LineString,Point,box
from shapely.ops import unary_union

def build(cells):
 pieces=[];domains=[]
 for ci,c in enumerate(cells):
  u0,u1=c['x'];up0,up1=c['upperY'];dn0,dn1=c['lowerY'];D0,D1=dn0-up0,dn1-up1
  domains.append(Polygon([(u0,up0),(u1,up1),(u1,dn1),(u0,dn0)]))
  knots=sorted({(v-c['upperY'][side])/(c['lowerY'][side]-c['upperY'][side]) for side,profile in enumerate(c['boundaryProfiles']) for v,_ in profile})
  def value(side,f):
   rows=c['boundaryProfiles'][side];v=c['upperY'][side]+f*(c['lowerY'][side]-c['upperY'][side]);return float(np.interp(v,[q[0] for q in rows],[q[1] for q in rows]))
  for lo,hi in zip(knots,knots[1:]):
   A=[];B=[]
   for side in [0,1]:
    lower,upper=value(side,lo),value(side,hi);slope=(upper-lower)/(hi-lo);A.append(lower-slope*lo);B.append(slope)
   polygon=Polygon([(u0,up0+lo*D0),(u1,up1+lo*D1),(u1,up1+hi*D1),(u0,up0+hi*D0)])
   pieces.append({'cellIndex':ci,'component':c['supportedComponent'],'cell':c,'fractionRange':[lo,hi],'A':A,'B':B,'polygon':polygon})
 return pieces,unary_union(domains)

def rational_candidates(n,d):
 """All stationary points plus endpoints for a quadratic/positive-linear ratio."""
 n0,n1,n2=n;d0,d1=d;assert min(d0,d0+d1)>0
 coefs=np.trim_zeros(np.array([n1*d0-n0*d1,2*n2*d0,n2*d1],float),'b');points=[0.,1.]
 if len(coefs)>1:
  for root in np.polynomial.polynomial.polyroots(coefs):
   if abs(complex(root).imag)<1e-10 and 0<float(complex(root).real)<1:points.append(float(complex(root).real))
 return [(s,(n0+n1*s+n2*s*s)/(d0+d1*s)) for s in points]

def edge_range(piece,a,b):
 c=piece['cell'];a=np.asarray(a,float);b=np.asarray(b,float);u0,u1=c['x'];t0=(a[0]-u0)/(u1-u0);td=(b[0]-a[0])/(u1-u0)
 up0,up1=c['upperY'];D0=c['lowerY'][0]-up0;D1=c['lowerY'][1]-up1
 d0=D0+(D1-D0)*t0;d1=(D1-D0)*td;w0=a[1]-up0-(up1-up0)*t0;w1=b[1]-a[1]-(up1-up0)*td
 A0,A1=piece['A'];B0,B1=piece['B'];aa=A0+(A1-A0)*t0;ab=(A1-A0)*td;ba=B0+(B1-B0)*t0;bb=(B1-B0)*td
 n=[aa*d0+ba*w0,aa*d1+ab*d0+ba*w1+bb*w0,ab*d1+bb*w1]
 return [{'paperXmm':value,'sourceUv':(a+(b-a)*s).tolist(),'edgeFraction':s,'cellIndex':piece['cellIndex'],'component':piece['component']} for s,value in rational_candidates(n,[d0,d1])]

def edges(g):
 if g.is_empty:return
 if g.geom_type=='Polygon':
  points=list(g.exterior.coords)
  for a,b in zip(points,points[1:]):yield a,b
 elif g.geom_type in ['LineString','LinearRing']:
  points=list(g.coords)
  for a,b in zip(points,points[1:]):yield a,b
 elif g.geom_type=='Point':yield g.coords[0],g.coords[0]
 elif hasattr(g,'geoms'):
  for part in g.geoms:yield from edges(part)
 else:raise ValueError(g.geom_type)

def shape(bounds,crop):
 l,t,r,b=bounds;u0,u1=t-crop[1],b-crop[1];v0,v1=l-crop[0],r-crop[0]
 assert u1>=u0 and v1>=v0
 if u0==u1 and v0==v1:return Point(u0,v0)
 if u0==u1 or v0==v1:return LineString([(u0,v0),(u1,v1)])
 return box(u0,v0,u1,v1)

def transport(pieces,domain,bounds,crop):
 target=shape(bounds,crop);missing=target.difference(domain);candidates=[];piece_ids=[];components=set()
 for pi,piece in enumerate(pieces):
  if not piece['polygon'].intersects(target):continue
  clipped=piece['polygon'].intersection(target)
  for a,b in edges(clipped):candidates.extend(edge_range(piece,a,b));components.add(piece['component'])
  if not clipped.is_empty:piece_ids.append(pi)
 out={'sourceBounds':bounds,'geometryType':target.geom_type,'completeSupport':bool(domain.covers(target)),'uncoveredAreaPixels2':float(missing.area),'uncoveredLengthPixels':float(missing.length),'components':sorted(components),'pieceIndices':piece_ids,'extremumCandidateCount':len(candidates),'paperXmmRange':None,'minimumWitness':None,'maximumWitness':None,'numericalOutwardAllowanceMm':1e-9}
 if candidates:
  lo=min(candidates,key=lambda q:q['paperXmm']);hi=max(candidates,key=lambda q:q['paperXmm']);out.update(paperXmmRange=[lo['paperXmm']-1e-9,hi['paperXmm']+1e-9],minimumWitness=lo,maximumWitness=hi)
 out['admitted']=out['completeSupport'] and len(components)==1 and out['paperXmmRange'] is not None
 return out
