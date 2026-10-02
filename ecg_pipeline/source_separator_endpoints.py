"""Bounded separator rectangles witnessed by maximal runs in untouched source ink."""
import copy,hashlib,itertools
import numpy as np
from ecg_pipeline import joint_source_geometry as joint
METHOD='source-endpoint-witnessed-separator-decomposition-v1'
LIMITS={'minimumReferenceMarkers':6,'minimumReferenceMarkersPerRow':2,'maximumLongRunsPerComponent':64,'maximumEndpointPairsPerComponent':256}

def reference_cluster(choices):
    unique=[(i,qs[0]) for i,qs in enumerate(choices) if len(qs)==1]
    admitted=[]
    for count in range(len(unique),LIMITS['minimumReferenceMarkers']-1,-1):
        for group in itertools.combinations(unique,count):
            if any(sum(i//3==row for i,_ in group)<LIMITS['minimumReferenceMarkersPerRow'] for row in range(3)):continue
            heights=[q['height'] for _,q in group];tol=max(2,round(float(np.median(heights))*.05))
            if max(heights)-min(heights)<=tol:admitted.append(group)
        if admitted:break
    return admitted[0] if len(admitted)==1 else None

def decompose(image,region,font,heights):
    minimum=max(3,round(font*.08));maximum=max(4,round(font*.3));retained=[];evidence=[]
    for component in region['components']:
        if not minimum<=component['width']<=maximum:continue
        l,t,r,b=component['bounds'];runs=[]
        for x in range(l,r):
            for a,z in joint.runs(image[t:b,x].max(1)<joint.POLICY['darkExclusive']):
                if z-a>=region['minimumLength']:runs.append({'x':x,'top':t+a,'bottom':t+z})
        if len(runs)>LIMITS['maximumLongRunsPerComponent']:return [],{'reason':'long-run-limit','component':component}
        pairs=[]
        for top,bottom in itertools.product(sorted({q['top'] for q in runs}),sorted({q['bottom'] for q in runs})):
            heights_with=[*heights,bottom-top];tol=max(2,round(float(np.median(heights_with))*.05))
            if bottom>top and max(heights_with)-min(heights_with)<=tol:pairs.append((top,bottom))
        if len(pairs)>LIMITS['maximumEndpointPairsPerComponent']:return [],{'reason':'endpoint-pair-limit','component':component}
        items=[]
        for top,bottom in pairs:
            clipped={**region,'bounds':[l,top,r,bottom]}
            clipped['components']=joint.region_components(image,clipped['bounds'],region['minimumLength'])
            candidates=joint.separator_candidates(image,clipped,font)
            for candidate in candidates:
                cl,ct,cr,cb=candidate['bounds']
                # Neither endpoint may be an arbitrary cut through all source runs.
                upper=[q for q in runs if cl<=q['x']<cr and q['top']==ct]
                lower=[q for q in runs if cl<=q['x']<cr and q['bottom']==cb]
                proposed_heights=[*heights,candidate['height']];tol=max(2,round(float(np.median(proposed_heights))*.05))
                if not upper or not lower or max(proposed_heights)-min(proposed_heights)>tol:continue
                proof={'parentComponent':copy.deepcopy(component),'requestedVerticalBounds':[top,bottom],'upperEndpointWitnesses':upper,'lowerEndpointWitnesses':lower,'parentInkExclusionBounds':copy.deepcopy(component['bounds']),'sourcePixelsAltered':False}
                candidate['strokeDecomposition']=proof
                identity=(tuple(candidate['bounds']),tuple(candidate['observedBounds']))
                if not any((tuple(q['bounds']),tuple(q['observedBounds']))==identity for q in retained):retained.append(candidate)
                items.append({'candidateBounds':candidate['bounds'],'requestedVerticalBounds':[top,bottom]})
        evidence.append({'parentComponent':component,'sourceLongRuns':runs,'endpointPairsEnumerated':[list(x) for x in pairs],'admittedRectangles':items})
    return retained,{'components':evidence}

def propose(image,g,*,checked=True):
    baseline=joint.propose_joint_geometry(image,g) if checked else joint.propose_after_identity(image,g)
    if (baseline.get('state')=='joint_geometry_supported' or baseline.get('failureReason') not in {'separator-components-missing-or-excessive','joint-geometry-missing-or-ambiguous'} or baseline.get('solutionCount',0)>0):return baseline
    choices=baseline.get('separatorChoices',[])
    if len(choices)!=9 or any(len(q)>joint.POLICY['maximumCandidatesPerWindow'] for q in choices):return baseline
    group=reference_cluster(choices)
    if group is None:return {**baseline,'strokeDecomposition':{'method':METHOD,'state':'refused','reason':'no-unique-repeated-height-reference'}}
    indices=[i for i,_ in group];heights=[q['height'] for _,q in group];font=g['sourceLabelGrid']['fontHeight'];replacement=copy.deepcopy(choices);details=[]
    for i,region in enumerate(baseline['regions'][4:]):
        if i in indices:continue
        replacement[i],evidence=decompose(image,region,font,heights)
        details.append({'row':region['row'],'column':region['column'],'evidence':evidence,'candidates':replacement[i]})
    receipt={'method':METHOD,'referenceIndices':indices,'referenceHeights':heights,'limits':LIMITS,'regions':details,'originalFailureReason':baseline.get('failureReason'),'diagnosticOnly':True,'parentInkMustRemainExcluded':True}
    proposed = joint.resolve_observed_candidates(image, g, baseline, replacement)
    return {**proposed,'strokeDecomposition':receipt,'nativeExtractionReady':False,'diagnosticOnly':True}
