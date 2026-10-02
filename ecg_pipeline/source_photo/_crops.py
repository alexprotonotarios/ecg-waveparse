"""Isolated recognition-crop expansion; no final acceptance rule changes."""
import copy
import math
RULE = 'local-recognition-source-context-one-half-font-v1'

def expand(proposal, width, height):
    if proposal.get('state') != 'crop_proposal':
        return proposal
    font = proposal['fit']['fontHeight']
    assert math.isfinite(font) and font > 0
    result = copy.deepcopy(proposal)
    for crop in result['crops']:
        left, top, right, bottom = crop['bounds']
        assert 0 <= left < right <= width and 0 <= top < bottom <= height
        crop['originalBounds'] = list(crop['bounds'])
        crop['bounds'] = [max(0, round(left - font)), max(0, round(top - font * 0.5)), min(width, round(right + font * 0.5)), min(height, round(bottom + font * 0.5))]
    result['recognitionContextRule'] = RULE
    return result

def install(module, image, initial_tokens):
    """Freeze only the previously observed first anchor call; read local crops anew."""
    originals = {'read_label_anchors': module.read_label_anchors, 'propose_local_label_crops': module.propose_local_label_crops}
    calls = {'initialReused': 0, 'localCalls': 0}

    def read(raster, ocr_runner=None):
        if calls['initialReused'] == 0:
            assert raster is image and ocr_runner is None
            calls['initialReused'] += 1
            return copy.deepcopy(initial_tokens)
        calls['localCalls'] += 1
        return originals['read_label_anchors'](raster, ocr_runner)

    def crops(raster, tokens):
        return expand(originals['propose_local_label_crops'](raster, tokens), raster.shape[1], raster.shape[0])
    module.read_label_anchors = read
    module.propose_local_label_crops = crops
    return (originals, calls)

def restore(module, originals):
    for name, function in originals.items():
        setattr(module, name, function)
