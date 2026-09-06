"""Build a frozen PTB waveform pilot locally; never select cases from outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg_benchmark.io import LEADS, write_leads_csv
from ecg_benchmark.paired_render import canonical_visible_truth, render_clinical_page, apply_artifact, layout_column_count
from ecg_benchmark.wfdb_source import download_verified, parse_checksum_index
from ecg_benchmark.evaluation import validate_independence


def read_record(header: str, data: bytes) -> np.ndarray:
    lines = header.splitlines()
    identity = lines[0].split()
    if len(identity) < 4 or identity[1:3] != ['15', '1000']:
        raise ValueError('Unsupported PTB channel/rate contract.')
    count = int(identity[3])
    if not 21_000 <= count <= 1_000_000 or len(data) != count * 12 * 2:
        raise ValueError('Invalid PTB record length.')
    channels = [line.split() for line in lines[1:13]]
    if any(len(c) < 9 for c in channels) or [c[-1].lower() for c in channels] != [lead.lower() for lead in LEADS]:
        raise ValueError('Unsupported PTB lead order.')
    if len({c[0] for c in channels}) != 1 or any(c[1] != '16' for c in channels):
        raise ValueError('Unsupported PTB signal encoding.')
    gains = np.array([float(c[2]) for c in channels])
    zeros = np.array([int(c[4]) for c in channels])
    if not np.isfinite(gains).all() or np.any(gains <= 0):
        raise ValueError('Invalid PTB physical gain.')
    digital = np.frombuffer(data, dtype='<i2').reshape(count, 12)
    checksums = np.array([int(c[6]) for c in channels]) % 65536
    if not np.array_equal(digital.sum(axis=0, dtype=np.int64) % 65536, checksums):
        raise ValueError('PTB channel checksum mismatch.')
    uv = (digital.astype(float) - zeros) * (1000 / gains)
    padded = resample_poly(uv[9000:21000], 1, 2, axis=0, padtype='line')
    segment = padded[500:5500]
    return segment - np.median(segment, axis=0, keepdims=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(protocol_path: Path, checksum_path: Path, output: Path, cache: Path, offline: bool) -> dict:
    protocol = json.loads(protocol_path.read_text())
    if protocol['collection'] not in {'frozen_public_engineering_pilot', 'final_engineering_evaluation'} or protocol['clinicalValidationUse'] is not False:
        raise ValueError('A separately frozen engineering protocol is required.')
    if sha(checksum_path) != protocol['checksumsSha256']:
        raise ValueError('Pinned PTB checksum index mismatch.')
    is_final = protocol['collection'] == 'final_engineering_evaluation'
    if is_final and (protocol.get('previouslyExposed') is not False or not protocol.get('sourceLock') or not protocol.get('accessControlReference')):
        raise ValueError('Final engineering membership requires an unexposed, source-locked access protocol.')
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    checksums = parse_checksum_index(checksum_path)
    cases, failures = [], []
    for index, member in enumerate(protocol['membership']):
        record = member['record']
        if not record.startswith(member['patientId'] + '/') or len(Path(record).parts) != 2 or '..' in record:
            raise ValueError('Invalid PTB membership path.')
        try:
            paths = {suffix: cache / (record + suffix) for suffix in ('.hea', '.dat')}
            for suffix, target in paths.items():
                download_verified(target, url=f'https://physionet.org/files/ptbdb/1.0.0/{record}{suffix}',
                                  expected_sha256=checksums[record + suffix], offline=offline)
            values = read_record(paths['.hea'].read_text(), paths['.dat'].read_bytes())
            leads = {lead: values[:, i] for i, lead in enumerate(LEADS)}
            layout = member['layout']
            image = render_clinical_page(leads, layout=layout, rhythm_lead=None,
                                         grid_palette=protocol['render']['gridPalettes'][index % 4])
            params = {'blur': {'radiusPixels': 1.2}, 'jpeg_compression': {'quality': 28},
                      'perspective': {'insetFraction': .035}, 'low_resolution': {'targetWidthPixels': 900},
                      'rotation': {'degrees': 3.0}}.get(member['artifact'], {})
            image = apply_artifact(image, kind=member['artifact'], parameters=params, seed=index)
            case_dir = output / member['caseId']; case_dir.mkdir()
            image.save(case_dir / 'image.png')
            write_leads_csv(case_dir / 'truth.csv', canonical_visible_truth(leads, layout=layout), sample_rate_hz=500)
            (case_dir / 'annotations.json').write_text(json.dumps({'version': 1, 'caseId': member['caseId'],
                                                                  'highErrorThresholdUv': 100, 'leads': {}}) + '\n')
            artifacts = {key: {'path': f"{member['caseId']}/{name}", 'sha256': sha(case_dir / name)}
                         for key, name in [('image', 'image.png'), ('truth', 'truth.csv'), ('annotations', 'annotations.json')]}
            cases.append({'caseId': member['caseId'], 'sourceId': record, 'groupId': member['patientId'],
                          'split': 'final_evaluation' if is_final else 'frozen_public_pilot', 'layout': layout, 'sampleRateHz': 500,
                          'segmentDurationSeconds': 10 / layout_column_count(layout),
                          'truthCoordinateFrame': 'canonical_display',
                          'degradation': {'kind': member['artifact'], 'parameters': params},
                          'strata': {'truthSupport': 'layout_observed_panels', 'dataset': protocol['dataset'],
                                     'gridPalette': protocol['render']['gridPalettes'][index % 4], 'acquisition': 'rendered_public_waveform'},
                          'provenance': {'clinicalValidationUse': False, 'quantitativeTruth': True,
                                         'waveformOrigin': 'public acquired signal', 'acquisition': 'synthetic_render_from_acquired_waveform',
                                         'sourceFileHashes': {suffix: sha(p) for suffix, p in paths.items()},
                                         'licenceEvidence': protocol['licenceEvidence'], 'pretrainedTrainingOverlap': 'unknown'},
                          'evaluationProvenance': {'collection': 'final_evaluation' if is_final else 'development',
                                                  'previouslyExposed': False if is_final else True,
                                                  'truthMethod': 'Hash-verified acquired WFDB waveform, independently rendered and mapped to visible panels.',
                                                  'pretrainedTrainingOverlap': 'unknown', 'licenceEvidence': protocol['licenceEvidence'],
                                                  'accessControlReference': protocol.get('accessControlReference')}, **artifacts})
        except Exception as error:
            failures.append({'caseId': member['caseId'], 'errorType': type(error).__name__, 'outcome': 'source_preparation_failure'})
    manifest = {'version': 1, 'suiteId': protocol.get('suiteId', 'ptb_public_pilot_20260906'), 'protocolSha256': sha(protocol_path),
                'expectedCaseIds': [m['caseId'] for m in protocol['membership']], 'cases': cases, 'preparationFailures': failures,
                'grouping': validate_independence(cases, require_provenance=True)}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / '.keep-artifacts.json').write_text('{"version":1,"reason":"Frozen public engineering pilot; retain input, truth and protocol."}\n')
    return {'attempted': len(protocol['membership']), 'prepared': len(cases), 'preparationFailures': len(failures),
            'manifestSha256': sha(output / 'manifest.json'), 'independentPtbPatientGroups': len({c['groupId'] for c in cases})}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'checksums', 'output', 'cache'): parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--offline', action='store_true'); args = parser.parse_args()
    print(json.dumps(build(args.protocol, args.checksums, args.output, args.cache, args.offline)))
