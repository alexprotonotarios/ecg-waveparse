"""A source-context fallback for previously undetected raw calibration only."""
import copy, math

def assess_context(q, context):
    if not isinstance(context, dict) or context.get('sourceIdentityVerifiedForDiagnostic') is not True:
        return {'confirmed': False, 'failureReason': 'source-identity-context-unconfirmed', 'rows': []}
    font = context.get('fontHeight')
    rows = context.get('firstColumnRows')
    if not isinstance(font, (float, int)) or isinstance(font, bool) or (not math.isfinite(font)) or (font <= 0) or (not isinstance(rows, list)) or (len(rows) != 4):
        return {'confirmed': False, 'failureReason': 'invalid-source-context', 'rows': []}
    checks = []
    for row in rows:
        x, y = (row.get('labelX'), row.get('waveformCenterY'))
        if not all((isinstance(v, (int, float)) and (not isinstance(v, bool)) and math.isfinite(v) for v in [x, y])):
            return {'confirmed': False, 'failureReason': 'invalid-source-context', 'rows': []}
        xe = abs(q['xEnd'] - x)
        ye = abs(q['bottom'] - y)
        checks.append({'row': row['row'], 'lead': row['lead'], 'labelX': x, 'waveformCenterY': y, 'pulseEndXError': xe, 'pulseBottomYError': ye, 'tolerancePixels': font, 'matched': xe <= font and ye <= font})
    matches = [r['row'] for r in checks if r['matched']]
    return {'confirmed': len(matches) == 1, 'rows': checks, 'matchedRows': matches}

def choose(baseline, proposals, context):
    if baseline['calibration']['detected']:
        return {'branch': 'existing-calibration-retained', 'finalResult': copy.deepcopy(baseline), 'selectedProposalIndex': None, 'candidateAssessments': [], 'newCalibration': False}
    assessments = []
    for i, q in enumerate(proposals):
        evidence = assess_context(q, context)
        eligible = bool(q['acceptedForSelection'] and evidence['confirmed'])
        assessments.append({'proposalIndex': i, 'priorEdgeAndPhysicalPassed': q['acceptedForSelection'], 'sourceContext': evidence, 'eligible': eligible})
    eligible = [r['proposalIndex'] for r in assessments if r['eligible']]
    settings = {(proposals[i]['candidate']['paperSpeedMmPerSecond'], proposals[i]['candidate']['gainMmPerMv'], proposals[i]['candidate']['gridScaleMmX'], proposals[i]['candidate']['gridScaleMmY']) for i in eligible}
    if not eligible or len(settings) != 1:
        return {'branch': 'fallback-unresolved', 'failureReason': 'no-context-bound-pulse' if not eligible else 'conflicting-context-bound-settings', 'finalResult': copy.deepcopy(baseline), 'selectedProposalIndex': None, 'candidateAssessments': assessments, 'newCalibration': False}
    selected = max(eligible, key=lambda i: proposals[i]['candidate']['confidence'])
    return {'branch': 'source-context-fallback', 'finalResult': {'calibration': copy.deepcopy(proposals[selected]['candidate'])}, 'selectedProposalIndex': selected, 'candidateAssessments': assessments, 'newCalibration': True}
