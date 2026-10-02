"""Attach explicit source provenance to unchanged, already retained calibration."""
import hashlib,json,math
from ecg_pipeline.source_photo._pulse_fallback import assess_context

def bind(image, observation):
    raster=hashlib.sha256(image.tobytes()).hexdigest()
    out={'version':1,'method':'exact-retained-calibration-source-binding-v1','state':'unresolved','sourceRasterSha256':raster,'sourceRow':None,'proposalIndex':None,'matchingProposalIndices':[],'assessments':[],'calibrationChanged':False,'newCalibration':False}
    def refuse(reason):return out|{'reason':reason}
    if observation.get('existingTiming',{}).get('state')=='source_supported':return out|{'state':'not-required','reason':'preferred-source-timing-retained'}
    context=observation.get('sourceContext');selection=observation.get('pulseSelection',{});raw=observation.get('rawCalibration',{});cal=raw.get('calibration',{})
    if selection.get('branch')!='existing-calibration-retained' or selection.get('newCalibration') is not False or selection.get('selectedProposalIndex') is not None or selection.get('candidateAssessments')!=[] or selection.get('finalResult')!=raw:return refuse('not-an-unchanged-retained-calibration')
    numeric=['confidence','paperSpeedMmPerSecond','gainMmPerMv','pixelsPerMmX','pixelsPerMmY','pulseStartX','pulseEndX']
    if cal.get('detected') is not True or not all(type(cal.get(k)) in [int,float] and math.isfinite(cal[k]) for k in numeric):return refuse('invalid-retained-calibration')
    if not isinstance(context,dict) or context.get('sourceIdentityVerifiedForDiagnostic') is not True or context.get('sourceContextProvenance',{}).get('sourceRasterSha256')!=raster:return refuse('source-identity-or-raster-unconfirmed')
    out['retainedCalibrationSha256']=hashlib.sha256(json.dumps(cal,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    proposals=observation.get('pulseProposals',[])
    if not isinstance(proposals,list):return refuse('invalid-pulse-proposals')
    height,width=image.shape[:2]
    for i,q in enumerate(proposals):
        exact=q.get('candidate')==cal
        edges=q.get('acceptedForSelection') is True and q.get('physicalConsistencyPassed') is True and q.get('neutralEdgeEvidence',{}).get('passed') is True
        bounds=[q.get(k) for k in ['xStart','top','xEnd','bottom']]
        bounded=all(type(v) in [int,float] and math.isfinite(v) for v in bounds) and 0<=bounds[0]<bounds[2]<=width and 0<=bounds[1]<bounds[3]<=height
        assessment=assess_context(q,context) if bounded else {'confirmed':False,'matchedRows':[],'failureReason':'invalid-pulse-bounds'}
        matched=assessment.get('matchedRows',[])
        valid_row=assessment.get('confirmed') is True and len(matched)==1 and type(matched[0]) is int and 0<=matched[0]<4
        eligible=exact and edges and bounded and valid_row
        out['assessments'].append({'proposalIndex':i,'calibrationExact':exact,'edgeAndPhysicalSupported':edges,'sourceBoundsValid':bounded,'sourceContext':assessment,'eligible':eligible})
        # Ambiguous duplicate calibration witnesses are refused even if one
        # happens to fail a downstream edge/row check.
        if exact:out['matchingProposalIndices'].append(i)
    if len(out['matchingProposalIndices'])!=1:return refuse('missing-or-ambiguous-exact-calibration-witness')
    i=out['matchingProposalIndices'][0];a=out['assessments'][i]
    if not a['eligible']:return refuse('matching-pulse-source-evidence-unconfirmed')
    return out|{'state':'confirmed','reason':None,'proposalIndex':i,'sourceRow':a['sourceContext']['matchedRows'][0]}
