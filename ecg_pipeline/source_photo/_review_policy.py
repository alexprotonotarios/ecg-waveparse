"""Preserve connected source samples; narrow morphology is a review flag."""
from copy import deepcopy

def retain_with_review_flags(path):
    old, current = (path['oldValid'], path['valid'])
    assert len(old) == len(current) == len(path['pathY'])
    assert all((type(v) in (bool, int) and v in (0, 1) for v in old + current))
    assert not any((now and (not was) for was, now in zip(old, current)))
    removed = {i for i, (was, now) in enumerate(zip(old, current)) if was and (not now)}
    flagged = set()
    flags = []
    for decision in path['decisions']:
        if not decision['rejectAsUncorroboratedNarrowExcursion']:
            continue
        left, right = decision['gapIndexRange']
        assert isinstance(left, int) and isinstance(right, int) and (0 <= left < right <= len(old))
        assert all(old[left:right])
        flagged.update(range(left, right))
        flags.append({'sourceIndexRangeHalfOpen': [left, right], 'sourceXRangeHalfOpen': [path['range'][0] + left, path['range'][0] + right], 'reason': 'narrow_unpaired_source_excursion', 'disposition': 'retain_source_samples_for_review', 'artifactIdentity': 'unclassified', 'clinicalMeaning': 'unclassified', 'historicalDecision': deepcopy(decision)})
    assert flagged == removed and len(removed) == path['newlyRejectedColumns']
    result = {k: deepcopy(path[k]) for k in ['lead', 'row', 'column', 'range', 'pathY']}
    result.update(valid=old.copy(), reviewFlags=flags, sourcePolicy='morphology_flags_without_source_sample_deletion_v1', earlierSourceGapsRetained=sum((not v for v in old)), morphologyOnlyColumnsRetained=len(removed))
    return result

def canonical_review_flags(path, converted):
    flagged = set((i for f in path['reviewFlags'] for i in range(*f['sourceIndexRangeHalfOpen'])))
    intervals = [(converted['sourceTimesSeconds'][i], converted['sourceTimesSeconds'][i + 1]) for i, connected in enumerate(converted['connectionValid']) if connected and (i in flagged or i + 1 in flagged)]
    return [value is not None and any((a <= k / converted['canonicalSampleRate'] <= b for a, b in intervals)) for k, value in enumerate(converted['canonicalMv'])]
