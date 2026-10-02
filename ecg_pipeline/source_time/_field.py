"""Candidate C0 field from existing accepted boundary vertices only."""
import copy
import numpy as np

def prepare(cells):
    groups={}
    for c in cells:
        comp=c['supportedComponent']
        for side,u in enumerate(c['x']):
            values=groups.setdefault((u,comp),{})
            for key,label in [('upperY',c['paperYmm'][0]),('lowerY',c['paperYmm'][1])]:
                v=c[key][side]
                if v in values and values[v]!=label:raise ValueError('Conflicting accepted vertex labels.')
                values[v]=label
    for values in groups.values():
        ordered=sorted(values.items())
        if any(b[1]<=a[1] for a,b in zip(ordered[:-1],ordered[1:])):raise ValueError('Non-monotone accepted boundary vertices.')
    output=[]
    for old in cells:
        c=copy.deepcopy(old);profiles=[]
        for side,u in enumerate(c['x']):
            lo,hi=c['upperY'][side],c['lowerY'][side]
            rows=[(v,label) for v,label in sorted(groups[(u,c['supportedComponent'])].items()) if lo<=v<=hi]
            assert rows[0]==(lo,c['paperYmm'][0]) and rows[-1]==(hi,c['paperYmm'][1])
            profiles.append([[float(v),float(label)] for v,label in rows])
        c['boundaryProfiles']=profiles;output.append(c)
    return output

def geometry(c,u):
    x0,x1=c['x']
    if not x0<=u<=x1:return None
    t=(u-x0)/(x1-x0)
    top=c['upperY'][0]*(1-t)+c['upperY'][1]*t;bottom=c['lowerY'][0]*(1-t)+c['lowerY'][1]*t
    return t,top,bottom

def boundary_value(c,side,fraction):
    rows=c['boundaryProfiles'][side]
    source=c['upperY'][side]+fraction*(c['lowerY'][side]-c['upperY'][side])
    return float(np.interp(source,[r[0] for r in rows],[r[1] for r in rows]))

def forward(c,u,v):
    g=geometry(c,u)
    if g is None:return None
    t,top,bottom=g
    if not top<=v<=bottom or not bottom>top:return None
    fraction=(v-top)/(bottom-top)
    return boundary_value(c,0,fraction)*(1-t)+boundary_value(c,1,fraction)*t

def inverse(c,u,paper):
    g=geometry(c,u)
    if g is None or not c['paperYmm'][0]<=paper<=c['paperYmm'][1]:return None
    t,top,bottom=g
    knots=sorted({(v-c['upperY'][side])/(c['lowerY'][side]-c['upperY'][side]) for side,profile in enumerate(c['boundaryProfiles']) for v,_ in profile})
    values=[boundary_value(c,0,f)*(1-t)+boundary_value(c,1,f)*t for f in knots]
    if not all(b>a for a,b in zip(values[:-1],values[1:])):raise ValueError('Non-monotone interpolated boundary profile.')
    fraction=float(np.interp(paper,values,knots))
    return top+fraction*(bottom-top)
