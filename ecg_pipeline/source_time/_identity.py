"""Diagnostic identity candidate; never supplies unavailable grid coordinates."""
def identity_candidate(pair, nodes, period):
    original=pair['originalSeedGapCheck'];radius=max(3,int(.4*period))
    def refuse(reason):return {'candidate':False,'reason':reason,'step':None,'independentAnchorPairs':[]}
    if original['pixels']<=2*radius:return refuse('overlapping-seed-search-windows')
    anchors=pair['anchors'];steps={q['step'] for q in anchors}
    if len(steps)!=1:return refuse('missing-or-conflicting-anchor-steps')
    step=next(iter(steps))
    if not 1<=step<=4 or step!=original['roundedMajorSteps']:return refuse('anchor-step-disagrees-with-seed')
    for q in pair['nodeChecks']:
        if q['width'] is not None and (q['width']<=0 or q['step']!=step):return refuse('conflicting-observed-interval')
    separated=[]
    for a in anchors:
        for b in anchors:
            if b['edge']>a['edge'] and nodes[b['edge']]-nodes[a['edge']+1]>64:
                separated.append([a['edge'],b['edge']])
    if not separated:return refuse('no-disjoint-anchor-profile-bands')
    return {'candidate':True,'reason':None,'step':step,'independentAnchorPairs':separated}
