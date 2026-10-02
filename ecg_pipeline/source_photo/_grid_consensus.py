"""New development eligibility from repeated measured positions; missingness is explicit."""

def consensus(checks, *, minimum_count=8, minimum_fraction=0.9):
    assert minimum_count >= 1 and 0 < minimum_fraction <= 1 and checks
    assert len({q['nodeIndex'] for q in checks}) == len(checks)
    assert len({q['sourceX'] for q in checks}) == len(checks)
    assert len({q['gapIndex'] for q in checks}) == 1
    assert len({q['expectedMajorSteps'] for q in checks}) == 1
    assert all((q['status'] in ['measured', 'missing-paired-observation', 'outside-model-domain'] for q in checks))
    measured = [q for q in checks if q['status'] == 'measured']
    assert all((isinstance(q['passed'], bool) for q in measured))
    passed = [q for q in measured if q['passed']]
    fraction = len(passed) / len(measured) if measured else None
    return {'gapIndex': checks[0]['gapIndex'], 'majorSteps': checks[0]['expectedMajorSteps'], 'allPositions': len(checks), 'measuredNodeIds': [q['nodeIndex'] for q in measured], 'passingNodeIds': [q['nodeIndex'] for q in passed], 'failedMeasuredNodeIds': [q['nodeIndex'] for q in measured if not q['passed']], 'missingPairedNodeIds': [q['nodeIndex'] for q in checks if q['status'] == 'missing-paired-observation'], 'outsideDomainNodeIds': [q['nodeIndex'] for q in checks if q['status'] == 'outside-model-domain'], 'measuredCount': len(measured), 'passingCount': len(passed), 'agreementFraction': fraction, 'minimumMeasuredCount': minimum_count, 'minimumAgreementFraction': minimum_fraction, 'eligible': bool(len(measured) >= minimum_count and fraction >= minimum_fraction), 'independentAccuracyEstimate': False}
