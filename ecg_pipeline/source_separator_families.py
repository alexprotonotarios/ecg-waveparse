"""Keep every source endpoint variant; require universal unchanged joint geometry.

The geometry alone is not extraction admission. Versioned timing and source-mask
consumers independently reconstruct this proof before allowing native extraction.
"""
import copy,hashlib,itertools,json,math
from ecg_pipeline import joint_source_geometry as joint
from ecg_pipeline.joint_source_timing import stable
from ecg_pipeline import source_separator_endpoints as endpoints
METHOD='source-horizontal-stroke-endpoint-families-v1'

def require(condition,reason):
    if not condition:raise ValueError(reason)

def source_marker(image,marker):
    l,t,r,b=marker['bounds'];ol,ot,rr,bb=marker['observedBounds'];h,w=image.shape[:2]
    require(0<=ol<=l<r<=rr<=w and 0<=ot==t<b==bb<=h,'invalid-source-core-bounds')
    require(marker['centerX']==(l+r-1)/2 and marker['centerY']==(t+b-1)/2 and marker['height']==b-t,'invalid-source-core-coordinate')
    require(marker['centerIntervalPixels']==sorted([(l+r-1)/2,(ol+rr-1)/2]) and abs((l+r-ol-rr)/2)<=1,'invalid-source-core-uncertainty')
    support=float((image[t:b,l:r].max(2)<160).mean(0).min())
    require(support>=.8 and support==marker['minimumBlackColumnSupport'],'source-core-support-disagrees')
    proof=marker.get('strokeDecomposition')
    if proof:
        parent=proof['parentComponent']['bounds'];pl,pt,pr,pb=parent
        require(0<=pl<=ol<rr<=pr<=w and 0<=pt<=t<b<=pb<=h and proof['parentInkExclusionBounds']==parent,'parent-footprint-disagrees')
        for name in ['upperEndpointWitnesses','lowerEndpointWitnesses']:
            witnesses=proof[name];require(bool(witnesses),'missing-source-endpoint-witness')
            for q in witnesses:
                x,a,z=q['x'],q['top'],q['bottom']
                require(l<=x<r and pt<=a<z<=pb,'endpoint-witness-outside-parent')
                require(bool((image[a:z,x].max(1)<160).all()),'endpoint-witness-missing-ink')
                require(a==pt or image[a-1,x].max()>=160,'unsupported-source-upper-endpoint')
                require(z==pb or image[z,x].max()>=160,'unsupported-source-lower-endpoint')
                require(a==t if name.startswith('upper') else z==b,'endpoint-does-not-bound-core')
    return proof['parentComponent'] if proof else {'id':marker['componentId'],'bounds':marker['observedBounds']}

def families(image,choices):
    require(len(choices)==9 and all(choices),'missing-separator-family')
    grouped=[]
    for index,options in enumerate(choices):
        groups={}
        for marker in options:
            require((marker['row'],marker['column'])==(index//3,index%3+1),'wrong-marker-identity')
            parent=source_marker(image,marker);l,t,r,b=marker['bounds'];ol,ot,rr,bb=marker['observedBounds']
            # Same parent and exact horizontal core/uncertainty; separate strokes never merge.
            key=(l,r,ol,rr,tuple(marker['centerIntervalPixels']),json.dumps(parent,sort_keys=True))
            groups.setdefault(key,[]).append(copy.deepcopy(marker))
        window=[]
        for key,members in groups.items():
            identities=[(tuple(m['bounds']),tuple(m['observedBounds'])) for m in members]
            require(len(set(identities))==len(members),'duplicate-endpoint-member')
            window.append({'row':index//3,'column':index%3+1,'horizontalCore':[key[0],key[1]],'horizontalObserved':[key[2],key[3]],'centerIntervalPixels':list(key[4]),'parentComponent':json.loads(key[5]),'members':members,'memberUnionBounds':[min(m['observedBounds'][0] for m in members),min(m['observedBounds'][1] for m in members),max(m['observedBounds'][2] for m in members),max(m['observedBounds'][3] for m in members)],'topRange':[min(m['bounds'][1] for m in members),max(m['bounds'][1] for m in members)],'bottomRange':[min(m['bounds'][3] for m in members),max(m['bounds'][3] for m in members)]})
        require(len(window)<=joint.POLICY['maximumCandidatesPerWindow'],'too-many-distinct-source-families')
        grouped.append(window)
    check_combinations(grouped)
    return grouped

def check_combinations(grouped):
    require(math.prod(len(q) for q in grouped)<=joint.POLICY['maximumJointCombinations'],'family-combination-limit')
    require(math.prod(sum(len(f['members']) for f in q) for q in grouped)<=joint.POLICY['maximumJointCombinations'],'endpoint-combination-limit')

def resolve(image,g,seed):
    try:groups=families(image,seed['separatorChoices'])
    except (ValueError,KeyError,TypeError,IndexError) as error:
        return {'state':'unresolved','method':METHOD,'failureReason':str(error),'diagnosticOnly':True,'nativeExtractionReady':False}
    outcomes=[];accepted=[]
    for fi,selection in enumerate(itertools.product(*groups)):
        geometries=[];failed=[]
        for vi,members in enumerate(itertools.product(*(q['members'] for q in selection))):
            result = joint.resolve_observed_candidates(image, g, seed,
                                                       [[copy.deepcopy(m)] for m in members])
            if result['state']=='joint_geometry_supported':geometries.append(result)
            else:failed.append({'variantIndex':vi,'failureReason':result.get('failureReason')})
        universal=not failed
        outcomes.append({'familyCombinationIndex':fi,'variantCombinations':len(geometries)+len(failed),'allVariantsPassed':universal,'failures':failed})
        if universal:accepted.append((selection,geometries))
    if len(accepted)!=1:
        return {'state':'unresolved','method':METHOD,'failureReason':'source-family-joint-solution-not-unique','admittedFamilySolutions':len(accepted),'familyChoices':groups,'familyChecks':outcomes,'diagnosticOnly':True,'nativeExtractionReady':False}
    selected,geometries=accepted[0]
    horizontal=[r['rowBoundaries'] for r in geometries[0]['selected']['rows']]
    require(all([r['rowBoundaries'] for r in q['selected']['rows']]==horizontal for q in geometries),'family-horizontal-geometry-disagreement')
    report={'state':'joint_geometry_supported','method':METHOD,'geometryKind':'endpoint_family','diagnosticOnly':True,'nativeExtractionReady':False,'truthUsed':False,'decodedRasterSha256':hashlib.sha256(image.tobytes()).hexdigest(),'imageSize':[image.shape[1],image.shape[0]],'familyChoices':groups,'selectedFamilies':list(selected),'familyChecks':outcomes,'admittedFamilySolutions':1,'variantGeometries':geometries,'allHorizontalBoundariesExact':True,'parentInkMustRemainExcluded':True,'originalRejectedMethod':seed['method'],'originalRejection':seed.get('failureReason'),'sourceEndpointEvidence':seed.get('strokeDecomposition')}
    report['familyProofSha256']=stable(report)
    return report

def propose(image,g,*,checked=True):
    seed=endpoints.propose(image,g,checked=checked)
    if not seed.get('strokeDecomposition',{}).get('regions'):return seed
    return resolve(image,g,seed)

def validate(image,g,report,*,checked=True):
    require(report.get('decodedRasterSha256')==hashlib.sha256(image.tobytes()).hexdigest(),'family-source-raster-mismatch')
    claimed=copy.deepcopy(report);stamp=claimed.pop('familyProofSha256',None)
    require(stamp==stable(claimed),'family-proof-digest-mismatch')
    require(report==propose(image,g,checked=checked),'family-proof-does-not-match-fresh-source')
