"""Transport captured prepared-raster coordinates with observed preparation geometry."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def transport(arrays: dict, metadata: dict, chain: dict, raw_canonical: np.ndarray) -> tuple[dict, dict]:
    preparation = chain['preparation']
    matrix = np.asarray(chain['candidateToOriginal'], float)
    forward = np.asarray(chain['originalToCandidate'], float)
    size = chain['candidateSize']; original = chain['sourceSize']
    if (metadata.get('coordinateScope') != 'prepared_decoder_input_raster'
            or metadata.get('coordinateConvention') != 'pixel_centers_zero_based'
            or chain.get('convention') != 'pixel_edges'
            or metadata.get('preparedInputSizeWh') != [size['width'], size['height']]
            or preparation.get('preparedSizeWh') != [size['width'], size['height']]
            or not all(type(v) is int and v > 0 for v in [size['width'], size['height'], original['width'], original['height']])
            or matrix.shape != (3, 3) or forward.shape != (3, 3)
            or not np.isfinite(matrix).all() or not np.isfinite(forward).all()
            or not np.allclose(matrix @ forward, np.eye(3), rtol=0, atol=1e-10)):
        raise ValueError('invalid_decoder_preparation_correspondence')
    # Use the same numerical inversion as decoder-coordinate transport. Retain
    # the actual evaluated matrix separately from the caller's inverse receipt.
    matrix = np.linalg.inv(forward)
    canonical = arrays['canonicalUv']; finite = arrays['canonicalFinite']
    if (canonical.ndim != 2 or canonical.shape[0] != 12 or canonical.shape != raw_canonical.shape
            or finite.dtype != bool or not np.array_equal(finite, np.isfinite(canonical))
            or not np.array_equal(canonical, raw_canonical, equal_nan=True)):
        raise ValueError('captured_canonical_does_not_match_actual_candidate')
    result = {k: v.copy() for k, v in arrays.items()}
    for key, name in [('candidateLeftXY', 'sourceLeftXY'), ('candidateRightXY', 'sourceRightXY'), ('nativeCandidateXY', 'nativeSourceXY')]:
        xy = np.asarray(arrays[key], float)
        if xy.ndim != 3 or xy.shape[-1] != 2 or (key != 'nativeCandidateXY' and xy.shape[:2] != canonical.shape):
            raise ValueError('invalid_captured_coordinate_shape')
        edge = xy + .5
        homogeneous = np.concatenate([edge, np.ones((*edge.shape[:-1], 1))], axis=-1)
        mapped = homogeneous @ matrix.T
        with np.errstate(invalid='ignore', divide='ignore'):
            points = mapped[..., :2] / mapped[..., 2:] - .5
        if not np.array_equal(np.isfinite(points).all(-1), np.isfinite(xy).all(-1)):
            raise ValueError('source_transport_changes_coordinate_missingness')
        result[name] = points
    left, right = result['sourceLeftXY'], result['sourceRightXY']
    inside = lambda xy: (xy[..., 0] >= 0) & (xy[..., 0] <= original['width'] - 1) & (xy[..., 1] >= 0) & (xy[..., 1] <= original['height'] - 1)
    result['bothRequiredNodesInsideSource'] = finite & inside(left) & ((arrays['rightWeights'] == 0) | inside(right))
    assignment = metadata.get('metadata', {}).get('sourceRhythmAssignment')
    if assignment is not None:
        context_json = metadata['metadata'].get('sourceRhythmContextJson')
        if (not isinstance(context_json, str)
                or hashlib.sha256(context_json.encode('utf-8')).hexdigest() != assignment.get('contextSha256')):
            raise ValueError('source_rhythm_context_identity_mismatch')
        context = json.loads(context_json)
        if (context.get('version') != 1 or context.get('method') != assignment.get('method')
                or context.get('originalSource', {}).get('sha256') != assignment.get('sourceSha256')
                or context.get('workingSource', {}).get('sha256') != assignment.get('workingSourceSha256')
                or context.get('preparedInput', {}).get('sha256') != assignment.get('preparedInputSha256')):
            raise ValueError('source_rhythm_context_binding_mismatch')
        if (assignment.get('version') != 1 or assignment.get('method') != 'source-confirmed-decoder-rhythm-assignment-v1'
                or assignment.get('state') not in ('applied', 'refused')
                or assignment.get('sourceSha256') != chain['sourceSha256']
                or assignment.get('preparedInputSha256') != preparation['preparedSha256']
                or assignment.get('physicalTimingVerified') is not False or assignment.get('sourceInkVerified') is not False):
            raise ValueError('invalid_source_rhythm_assignment_binding')
        if assignment['state'] == 'applied':
            if (assignment.get('lead') != 'II' or assignment.get('canonicalIndex') != 1 or assignment.get('rhythmRow') != 3
                    or assignment.get('canonicalAssignmentPolicyChanged') is not True or assignment.get('reason') is not None
                    or metadata['metadata'].get('rhythmAssignments') != [{'rhythm':[0], 'canonical':[1]}]
                    or not np.all(arrays['rowIndices'][1] == 3)):
                raise ValueError('source_rhythm_assignment_route_mismatch')
        elif assignment.get('canonicalAssignmentPolicyChanged') is not False or not assignment.get('reason'):
            raise ValueError('invalid_source_rhythm_assignment_refusal')
    output_metadata = {**metadata, 'coordinateScope': 'original_source_raster', 'preparationCorrespondenceVerified': True,
                       'sourceTransform': chain, 'sourceInkVerified': False, 'physicalTimingVerified': False,
                       'transportMatrixCandidateToOriginal': matrix.tolist(),
                       'productionSelectionLineageTransported': False, 'waveformChanged': False,
                       'sourceLabelIdentityPromoted': False}
    return result, output_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['capture', 'transform', 'canonical', 'prepared-input', 'original-source', 'output']:
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    chain = json.loads(args.transform.read_text())
    if sha(args.prepared_input) != chain['preparation']['preparedSha256'] or sha(args.original_source) != chain['sourceSha256']:
        raise ValueError('decoder_source_file_identity_mismatch')
    with np.load(args.capture, allow_pickle=False) as payload:
        metadata = json.loads(payload['metadataJsonUtf8'].tobytes())
        arrays = {k: payload[k].copy() for k in payload.files if k != 'metadataJsonUtf8'}
    canonical = np.genfromtxt(args.canonical, delimiter=',', skip_header=1).T
    arrays, metadata = transport(arrays, metadata, chain, canonical)
    metadata.update(captureSha256=sha(args.capture), rawCanonicalSha256=sha(args.canonical), sourceSha256=chain['sourceSha256'])
    arrays['metadataJsonUtf8'] = np.frombuffer(json.dumps(metadata, sort_keys=True, allow_nan=False).encode(), dtype=np.uint8)
    with args.output.open('xb') as f:np.savez_compressed(f, **arrays)
    with np.load(args.output, allow_pickle=False) as saved:
        for k, value in arrays.items():
            if not np.array_equal(saved[k], value, equal_nan=True):raise ValueError('source_correspondence_write_mismatch')
    assignment = metadata.get('metadata', {}).get('sourceRhythmAssignment')
    print(json.dumps({'version': 1, 'method': 'observed-decoder-preparation-correspondence-v1',
                      'sha256': sha(args.output), 'sourceSha256': chain['sourceSha256'], 'rawCanonicalSha256': sha(args.canonical),
                      'captureSha256': sha(args.capture), 'coordinateScope': 'original_source_raster',
                      'preparationCorrespondenceVerified': True, 'sourceInkVerified': False, 'physicalTimingVerified': False,
                      'productionSelectionLineageTransported': False, 'waveformChanged': False,
                      **({'sourceRhythmAssignment': assignment} if assignment is not None else {})}, allow_nan=False))


if __name__ == '__main__':main()
