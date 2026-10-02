"""Persist the complete time-conversion evidence before publishing its outputs."""
import copy
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ecg_pipeline.source_photo._grid_observer import clean, pack
from .publication import LEADS, METHOD, ConversionRefused, convert_publication

sha = lambda data: hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with path.open('x') as f: json.dump(clean(value), f, indent=2, allow_nan=False); f.write('\n')


def write_rows(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def source_overlay(image, lineage, output):
    original = Image.fromarray(image[:, :, ::-1].copy())
    canvas = Image.new('RGB', (original.width, original.height + 44), 'white'); canvas.paste(original, (0, 44))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), 'Source timing remap | Review required | Gaps remain absent', fill='black', font=ImageFont.load_default(size=20))
    previous = {}
    for row in lineage:
        xy = [sum(row[f'native{k}_weight'] * row[f'native{k}_source_{axis}'] for k in range(row['native_contributor_count'])) for axis in ['x', 'y']]
        point = (xy[0], xy[1] + 44); lead = row['lead']; sample = row['canonical_sample']
        old = previous.get(lead)
        if old and old[0] + 1 == sample: draw.line([old[1], point], fill=(0, 140, 140), width=2)
        else: draw.point(point, fill=(0, 140, 140))
        previous[lead] = (sample, point)
    canvas.save(output)


def write_publication_bundle(image, observation, coordinate_bytes, canonical_bytes, uncertainty_bytes, segment_bytes, output):
    """The caller supplies fresh observations of the same untouched source."""
    output = Path(output); output.mkdir(exist_ok=False)
    inputs = {'input-canonical.csv': canonical_bytes, 'input-uncertainty.csv': uncertainty_bytes, 'input-segments.json': segment_bytes}
    for name, data in inputs.items(): (output / name).write_bytes(data)
    write_json(output / 'source-observation.json', pack(observation))
    segments = json.loads(segment_bytes)
    if segments.get('canonicalSha256') != sha(canonical_bytes): raise ValueError('source_time_input_canonical_hash_mismatch')
    if canonical_bytes.splitlines()[0].decode() != ','.join(LEADS): raise ValueError('unexpected_canonical_lead_order')
    canonical = np.genfromtxt(io.BytesIO(canonical_bytes), delimiter=',', skip_header=1).T
    uncertainty = list(csv.DictReader(io.StringIO(uncertainty_bytes.decode())))
    with np.load(io.BytesIO(coordinate_bytes), allow_pickle=False) as z:
        metadata = json.loads(z['metadataJsonUtf8'].tobytes())
        arrays = {k: z[k].copy() for k in z.files if k != 'metadataJsonUtf8'}
    if observation['sourceSha256'] != metadata.get('sourceSha256') or observation['sourceSha256'] != segments['sourceSha256']:
        raise ValueError('source_time_input_identity_mismatch')
    if observation['imageSize'] != [image.shape[1], image.shape[0]] or observation['decodedRasterSha256'] != sha(image.tobytes()):
        raise ValueError('source_time_observation_raster_mismatch')
    manifest = {'version': 1, 'method': METHOD, 'state': 'refused', 'sourceSha256': observation['sourceSha256'],
                'coordinateSha256': sha(coordinate_bytes), 'inputCanonicalSha256': sha(canonical_bytes),
                'inputUncertaintySha256': sha(uncertainty_bytes), 'inputSegmentMapSha256': sha(segment_bytes),
                'sourceLabelIdentityPromoted': False, 'sourceInkVerified': False, 'requiresReview': True,
                'originRangesAreGeometricNotStatistical': True, 'waveformConverted': False}
    try:
        converted = convert_publication(observation, arrays, metadata, canonical, uncertainty, segments)
    except ConversionRefused as error:
        manifest['reason'] = str(error)
    else:
        with (output / 'canonical.csv').open('x', newline='') as f:
            writer = csv.writer(f); writer.writerow(LEADS)
            writer.writerows([[repr(float(v)) if np.isfinite(v) else '' for v in row] for row in converted['arrays']['correctedUv'].T])
        write_rows(output / 'uncertainty.csv', converted['uncertainty'])
        write_json(output / 'uncertainty.json', converted['uncertainty'])
        write_rows(output / 'lineage.csv', converted['lineage'])
        np.savez_compressed(output / 'mapping.npz', **converted['mapping'])
        np.savez_compressed(output / 'conversion.npz', **converted['arrays'])
        updated = copy.deepcopy(segments); updated['canonicalSha256'] = sha((output / 'canonical.csv').read_bytes())
        updated['segments'] = converted['segments']
        proof = {'version': 1, 'method': METHOD, 'evidenceAsset': 'sourceTimeEvidence', 'lineageFile': 'lineage.csv',
                 'lineageSha256': sha((output / 'lineage.csv').read_bytes()), 'coordinateSha256': sha(coordinate_bytes),
                 'inputCanonicalSha256': sha(canonical_bytes), 'inputUncertaintySha256': sha(uncertainty_bytes),
                 'inputSegmentMapSha256': sha(segment_bytes)}
        for segment in updated['segments']:
            segment['immutableId'] = sha(f"{updated['runId']}:{updated['sourceSha256']}:{updated['canonicalSha256']}:{segment['segmentId']}".encode())
            segment['sourceTimeConversion'].update(proof)
        for segment in updated['omittedSourceSegments']:
            if 'segmentId' in segment:
                segment['immutableId'] = sha(f"{updated['runId']}:{updated['sourceSha256']}:{updated['canonicalSha256']}:{segment['segmentId']}".encode())
        write_json(output / 'segments.json', updated)
        source_overlay(image, converted['lineage'], output / 'source-overlay.png')
        # Read back values, gaps and all uncertainty rows before the manifest seals them.
        reloaded = np.genfromtxt(output / 'canonical.csv', delimiter=',', skip_header=1).T
        np.testing.assert_array_equal(reloaded, converted['arrays']['correctedUv'])
        for name, values in [('mapping.npz', converted['mapping']), ('conversion.npz', converted['arrays'])]:
            with np.load(output / name, allow_pickle=False) as z:
                for key in values: np.testing.assert_array_equal(z[key], values[key])
        manifest.update(state='converted', waveformConverted=True, outputCanonicalSha256=updated['canonicalSha256'],
                        expectedSamples=len(converted['uncertainty']), returnedSamples=len(converted['lineage']),
                        sourcePublicationExclusions=int(converted['arrays']['originalPublicationExcluded'].sum()),
                        sourceAnnotationExclusions=converted['sourceExclusions'], origins=converted['origins'],
                        uncertaintyMethod='maximum_required_input_spread_plus_separate_geometric_time_range')
    files = sorted(output.iterdir())
    manifest['files'] = {p.name: {'sha256': sha(p.read_bytes()), 'bytes': p.stat().st_size} for p in files}
    write_json(output / 'manifest.json', manifest)
    bundle = output / 'evidence.zip'
    with zipfile.ZipFile(bundle, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*files, output / 'manifest.json']: archive.write(path, path.name)
    with zipfile.ZipFile(bundle) as archive:
        if sorted(archive.namelist()) != sorted([*manifest['files'], 'manifest.json']): raise ValueError('source_time_archive_members_mismatch')
        for name in archive.namelist():
            if archive.read(name) != (output / name).read_bytes(): raise ValueError('source_time_archive_content_mismatch')
    return manifest
