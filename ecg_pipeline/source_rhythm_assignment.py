"""Source-supported rhythm assignment before canonicalization, in one candidate.

The coordinate recorder observes this policy's actual assignment. Neither source
label identity nor row association establishes timing, gain or source-ink fidelity.
Unsupported evidence leaves the upstream assignment unchanged, with a refusal.
"""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from threading import Lock
from unittest.mock import patch

import cv2
import numpy as np
import torch

from ecg_pipeline.decoder_coordinates import project
from ecg_pipeline.lead_label_identity import recognise_rhythm_lead_identity
from ecg_pipeline.source_label_geometry import (
    _translate_local_label_observations, propose_label_grid,
    supported_trace_rows, valid_tiled_label_grid,
)

_LOCK = Lock()
METHOD = 'source-confirmed-decoder-rhythm-assignment-v1'
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _grid_matches_pixels(image, grid):
    if grid.get('tiledRecognition') is not None:
        return valid_tiled_label_grid(image, grid)
    local = grid.get('localRecognition')
    if local is None:
        tokens = grid.get('anchors', [])
    else:
        if (local.get('version') != 1 or local.get('method') != 'sparse-source-proposal-image-only-local-ocr-v1'
                or local.get('sourceRasterSha256') != digest(image.tobytes())
                or local.get('expectedNamesProvidedToRecognizer') is not False
                or local.get('strictGridRequired') is not True or local.get('truthUsed') is not False):
            return False
        tokens = list(local['originalAnchors'])
        for crop in local['crops']:
            left, top, right, bottom = crop['bounds']
            if not (0 <= left < right <= image.shape[1] and 0 <= top < bottom <= image.shape[0]):
                return False
            if digest(image[top:bottom, left:right].tobytes()) != crop['rasterSha256']:
                return False
            translated = _translate_local_label_observations(crop['localTokens'], crop['bounds'])
            if translated != crop['sourceTokens']:
                return False
            tokens.extend(translated)
    proposed = propose_label_grid(image, tokens)
    return proposed.get('sourceLabelGrid') == {k:v for k,v in grid.items() if k != 'localRecognition'}


def validate_source_context(context, image_tensor):
    """Bind source observations and preparation to the actual uint8 model input."""
    require(isinstance(context, dict) and context.get('version') == 1 and context.get('method') == METHOD, 'unsupported_source_rhythm_context')
    blobs = {}
    for key in ['originalSource', 'workingSource', 'preparedInput']:
        item = context[key]
        blobs[key] = Path(item['path']).read_bytes()
        require(digest(blobs[key]) == item['sha256'], key + '_identity_mismatch')
    source = cv2.imdecode(np.frombuffer(blobs['workingSource'], np.uint8), cv2.IMREAD_COLOR)
    require(source is not None, 'source_decode_failed')
    geometry = context['geometry']; grid = geometry['sourceLabelGrid']
    require(geometry.get('coordinateSpace') == 'working' and geometry.get('detectedInputVariant') == 'original'
            and geometry.get('layoutHint') == 'standard_3x4_with_r1'
            and grid.get('imageSize') == list(source.shape[1::-1]), 'unsupported_source_rhythm_frame')
    primary = geometry.get('leadLabelValidation', {})
    recognition = primary.get('semanticRecognition', {})
    require(primary.get('semanticIdentityConfirmed') is True and recognition.get('passed') is True
            and recognition.get('semanticIdentityConfirmed') is True and recognition.get('failureReasons') == []
            and recognition.get('expectedLabels') == recognition.get('recognizedLabels') == LEADS,
            'primary_source_labels_unconfirmed')
    rhythm = geometry.get('rhythmLeadValidation', {})
    slot = grid.get('rhythmSlot', {})
    require(rhythm.get('passed') is True and rhythm.get('semanticIdentityConfirmed') is True
            and rhythm.get('lead') == slot.get('lead') == 'II' and rhythm.get('failureReasons') == []
            and (slot.get('row'), slot.get('column')) == (3, 0), 'source_rhythm_label_unconfirmed')
    require(_grid_matches_pixels(source, grid), 'source_grid_pixel_binding_failed')
    proof = recognise_rhythm_lead_identity(source, geometry, ocr_runner=lambda _: {'observations': []},
                                          primary_identity_confirmed=True)
    require(proof.get('passed') is True and proof.get('lead') == 'II'
            and proof.get('validationSource') == 'source-derived-primary-ii-glyph-match', 'source_rhythm_glyph_failed')
    centers = supported_trace_rows(source, grid)
    require(centers == geometry.get('rowCenters') and len(centers) == 4, 'source_rows_unconfirmed')
    preparation = context['preparation']
    require(preparation.get('preparedSha256') == context['preparedInput']['sha256']
            and preparation.get('inputPreserved') is True and preparation.get('cropBox') is None,
            'unsupported_rhythm_preparation')
    require(image_tensor.dtype == torch.uint8 and tuple(image_tensor.shape) ==
            (1, 3, *preparation['preparedSizeWh'][::-1]), 'actual_decoder_input_shape_mismatch')
    actual = image_tensor[0].detach().cpu().permute(1, 2, 0).contiguous().numpy()
    require(digest(actual.tobytes()) == preparation['preparedRasterRgbSha256'], 'actual_decoder_input_raster_mismatch')
    matrix = np.asarray(context['workingToCandidateEdges'], float)
    require(matrix.shape == (3, 3) and np.isfinite(matrix).all()
            and abs(np.linalg.det(matrix)) > 1e-12, 'invalid_rhythm_source_transform')
    # Every admitted input variant shares the unrectified working-source frame.
    expected_matrix = [[preparation['preparedSizeWh'][0]/source.shape[1], 0, 0],
                       [0, preparation['preparedSizeWh'][1]/source.shape[0], 0], [0, 0, 1]]
    require(preparation['inputSizeWh'] == list(source.shape[1::-1])
            and np.array_equal(matrix, np.asarray(preparation['inputToPreparedEdges'], float))
            and np.array_equal(matrix, expected_matrix),
            'rhythm_preparation_transform_mismatch')
    return {'sourceRowCenters': centers, 'sourceRasterSha256': digest(source.tobytes()),
            'sourceGlyphValidation': proof, 'workingToCandidateEdges': matrix.tolist()}


def bind_decoder_rows(capture, match, lines, proof):
    require(match.get('layout') == 'standard_3x4_with_r1' and match.get('flip') is False
            and tuple(lines.shape) == (4, 5000), 'unsupported_rhythm_decoder_layout')
    layout = capture.identifier.layouts[match['layout']]
    require(layout['layout'] == {'rows':3, 'cols':4}
            and layout['leads'] == [['I','aVR','V1','V4'],['II','aVL','V2','V5'],['III','aVF','V3','V6']]
            and len(layout.get('rhythm_leads', [])) == 1, 'unsupported_rhythm_decoder_definition')
    data = capture.data
    rotation = data.get('rotation')
    require((rotation is None and not capture.wrapper.rotate_on_resample)
            or (rotation is not None and rotation.get('clockwiseQuarterTurns') == 0),
            'rotated_rhythm_decoder_unsupported')
    merged = data['mergedPixelRows']
    require(merged.ndim == 2 and merged.shape[0] == 4, 'decoder_rows_incomplete')
    xy = np.stack((np.broadcast_to(np.arange(merged.shape[1]) + data['extractorCrop']['first'], merged.shape), merged), axis=-1)
    candidate = capture.aligned_to_candidate(xy)
    inverse = np.linalg.inv(np.asarray(proof['workingToCandidateEdges'], float))
    source = project(candidate + .5, inverse) - .5
    centers = np.asarray(proof['sourceRowCenters'], float)
    medians = np.nanmedian(source[...,1], axis=1); spacing = np.min(np.diff(centers))
    require(np.isfinite(medians).all() and spacing > 0
            and np.array_equal(np.argmin(abs(medians[:,None] - centers), axis=1), np.arange(4))
            and np.all(abs(medians-centers) < spacing/4), 'decoder_source_row_mismatch')
    return {'decoderSourceRowMedianY':medians.tolist(), 'sourceRowCenters':centers.tolist(),
            'maximumMedianRowDistancePixels':float(max(abs(medians-centers))), 'rowDistanceLimitPixels':float(spacing/4)}


class SourceRhythmAssignment:
    """Scoped upstream hooks, installed before the observational capture hooks."""
    def __init__(self, capture, context_path, image):
        self.capture = capture
        self.active = False
        self.proof = None
        self.context_json = None
        self.evidence = {'version':1, 'method':METHOD, 'state':'refused', 'reason':None,
                         'physicalTimingVerified':False, 'sourceInkVerified':False,
                         'canonicalAssignmentPolicyChanged':False}
        try:
            raw = Path(context_path).read_bytes()
            self.context_json = raw.decode('utf-8')
            self.evidence['contextSha256'] = digest(raw)
            context = json.loads(raw)
            self.evidence['sourceSha256'] = context['originalSource']['sha256']
            self.evidence['workingSourceSha256'] = context['workingSource']['sha256']
            self.evidence['preparedInputSha256'] = context['preparedInput']['sha256']
            self.proof = validate_source_context(context, image)
            self.evidence.update(self.proof, state='pending')
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
            self.evidence['reason'] = str(error)

    def __enter__(self):
        import src.model.lead_identifier as upstream
        require(_LOCK.acquire(blocking=False), 'concurrent_source_rhythm_assignment')
        self.stack = ExitStack(); self.stack.callback(_LOCK.release)
        try:
            original = self.capture.identifier._canonicalize_lines
            solver = upstream.linear_sum_assignment
            def canonicalize(lines, match):
                if self.proof is not None:
                    try:
                        self.evidence.update(bind_decoder_rows(self.capture, match, lines, self.proof))
                        self.active = True
                    except (ValueError, KeyError, TypeError, IndexError) as error:
                        self.evidence.update(state='refused', reason=str(error))
                try:
                    return original(lines, match)
                finally:
                    self.active = False
            def assignment(cost):
                if not self.active:
                    return solver(cost)
                require(cost.shape == (1,12), 'unexpected_source_rhythm_assignment_shape')
                try:
                    old = solver(cost)
                    previous = {'rhythm':old[0].tolist(), 'canonical':old[1].tolist()}
                except ValueError:
                    previous = None
                self.evidence.update(state='applied', reason=None, lead='II', rhythmRow=3,
                                     canonicalIndex=1, previousAssignment=previous,
                                     canonicalAssignmentPolicyChanged=True)
                return np.array([0]), np.array([1])
            self.stack.enter_context(patch.object(self.capture.identifier, '_canonicalize_lines', canonicalize))
            self.stack.enter_context(patch.object(upstream, 'linear_sum_assignment', assignment))
        except BaseException:
            self.stack.close(); raise
        return self

    def __exit__(self, *exc):
        if self.evidence['state'] == 'pending':
            self.evidence.update(state='refused', reason='canonical_assignment_not_observed')
        return self.stack.__exit__(*exc)
