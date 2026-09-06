"""Full-page development fixtures for supported settings and unsupported duration.

The renderer's layout duration is changed only inside this disposable builder;
the production domain and shared default renderer are left unchanged. Every
source coordinate is declared independently of the eventual reconstruction.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg_benchmark.generate import LAYOUTS, engineering_waveforms, render_waveforms
from ecg_benchmark.io import write_leads_csv


def build(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    settings = [(speed, gain, 5.) for speed in (25, 50) for gain in (5, 10, 20)]
    settings += [(25, 10, 2.5), (25, 10, 10.)]
    protocol = {'version': 1, 'clinicalValidationUse': False, 'purpose': 'Development only; never final tuning evidence',
                'maximumScaleRelativeError': .02, 'maximumFidelityRmseUv': 75, 'minimumCorrelation': .9,
                'minimumCoverage': .9, 'maximumCaseSeconds': 600,
                'expected': 'Recover declared scales and segment timing or explicitly refuse quantitative output; refusals are not fidelity passes.',
                'sourceFiles': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (Path(__file__).resolve(), ROOT/'ecg_benchmark/generate.py')}, 'membership': []}
    for speed, gain, duration in settings:
        case_id = f'speed_{speed}_gain_{gain}_segment_{duration:g}'
        directory = output/case_id; directory.mkdir()
        leads, _ = engineering_waveforms(sample_rate_hz=500, duration_seconds=duration)
        original_duration = LAYOUTS['standard_6x2']['segmentSeconds']
        try:
            LAYOUTS['standard_6x2']['segmentSeconds'] = duration
            image, geometry = render_waveforms(leads, sample_rate_hz=500, layout_name='standard_6x2',
                paper_speed_mm_per_second=speed, gain_mm_per_mv=gain, pixels_per_mm=6, trace_width_pixels=2)
        finally:
            LAYOUTS['standard_6x2']['segmentSeconds'] = original_duration
        image.save(directory/'image.png'); write_leads_csv(directory/'truth.csv', leads, sample_rate_hz=500)
        (directory/'source-geometry.json').write_text(json.dumps(geometry, indent=2)+'\n')
        annotations = {'version': 1, 'caseId': case_id, 'highErrorThresholdUv': 75,
                       'leads': {lead: {'events': [], 'occludedRanges': []} for lead in leads}}
        (directory/'annotations.json').write_text(json.dumps(annotations, indent=2)+'\n')
        member = {'caseId': case_id, 'groupId': 'engineering_morphology_fixture_v1',
                  'caseMetadata': {'layout': 'standard_6x2', 'sampleRateHz': 500, 'segmentDurationSeconds': duration,
                                   'truthCoordinateFrame': 'lead_local', 'paperSpeedMmPerSecond': speed, 'gainMmPerMv': gain},
                  'expectedSource': {'width': image.width, 'height': image.height, 'pixelsPerMmX': 6, 'pixelsPerMmY': 6,
                                     'pixelsPerSecond': speed*6, 'pixelsPerMv': gain*6}}
        for key in ('image', 'truth', 'annotations'):
            name = {'image': 'image.png', 'truth': 'truth.csv', 'annotations': 'annotations.json'}[key]
            member[key] = f'{case_id}/{name}'; member[key+'Sha256'] = hashlib.sha256((directory/name).read_bytes()).hexdigest()
        protocol['membership'].append(member)
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(json.dumps({'cases': len(settings), 'protocolSha256': hashlib.sha256((output/'protocol.json').read_bytes()).hexdigest()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
