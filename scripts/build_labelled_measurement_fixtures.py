"""Build controlled research endpoints with visibility at the admitted resolution."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg_benchmark.generate import render_waveforms, validate_annotations
from ecg_benchmark.io import LEADS, write_leads_csv
from ecg_benchmark.measurements import source_feature_visibility


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def build(protocol_path: Path, output: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    if protocol['version'] != 1 or protocol['clinicalValidationUse'] is not False:
        raise ValueError('An engineering-only frozen protocol is required.')
    output.mkdir(parents=True, exist_ok=False)
    rate = protocol['sampleRateHz']; time = np.arange(rate*5)/rate
    # Rise/flat/fall widths and amplitudes define the plotted source, not a
    # diagnostic class or a model of a patient population.
    profiles = [(900., .024, .0), (-900., .024, .0), (800., .08, .04),
                (60., .05, .02), (600., .024, .045), (350., .002, .0)]
    leads, omitted, labels, shape_geometry = {}, {}, {}, {}
    for index, lead in enumerate(LEADS):
        amplitude, ramp, plateau = profiles[index % len(profiles)]
        signal = np.zeros(time.size); first_pulse = np.zeros(time.size)
        for beat in (.6, 1.5, 2.4, 3.3, 4.2):
            pulse = np.interp(time, [0, beat-ramp, beat, beat+plateau, beat+plateau+ramp, 5],
                              [0, 0, amplitude, amplitude, 0, 0])
            signal += pulse
            if beat == 1.5: first_pulse = pulse
            # Separate low-amplitude P/T-like construction; no overlap in the
            # fixed first-pulse measurement/baseline windows is assumed exact.
            signal += .07*abs(amplitude)*np.exp(-.5*((time-(beat-.19))/.025)**2)
            signal += .18*abs(amplitude)*np.exp(-.5*((time-(beat+.29))/.05)**2)
        leads[lead] = signal; omitted[lead] = signal-first_pulse
        baseline = {'startSample': 545, 'endSample': 560}  # 1.09–1.12 s, away from constructed P/T peaks.
        common = {'labelMethod': 'controlled', 'visibility': 'visible', 'polarity': 1 if amplitude > 0 else -1,
                  'baselineRange': baseline}
        mid = 1.5 + plateau/2
        labels[lead] = [dict(common, id=f'{lead}_half_height_duration', kind='interval',
                            referenceValue=(ramp+plateau)*1000, levelUv=abs(amplitude)/2,
                            startRange={'startSample': (1.5-ramp-.01)*rate, 'endSample': mid*rate+1},
                            endRange={'startSample': mid*rate, 'endSample': (1.5+plateau+ramp+.01)*rate}),
                        dict(common, id=f'{lead}_signed_amplitude', kind='amplitude', referenceValue=amplitude,
                             peakRange={'startSample': (1.5-ramp-.01)*rate, 'endSample': (1.5+plateau+ramp+.01)*rate})]
        shape_geometry[lead] = (ramp+plateau, abs(amplitude))
    cases = []
    for member in protocol['membership']:
        directory = output/member['caseId']; directory.mkdir()
        ppm = member['pixelsPerMm']
        options = dict(sample_rate_hz=rate, layout_name=protocol['layout'], paper_speed_mm_per_second=25,
                       gain_mm_per_mv=10, pixels_per_mm=ppm, trace_width_pixels=max(1, round(ppm/3)))
        source, geometry = render_waveforms(leads, **options)
        source.save(directory/'image.png'); write_leads_csv(directory/'truth.csv', leads, sample_rate_hz=rate)
        annotations = {'version': 1, 'caseId': member['caseId'], 'highErrorThresholdUv': 75, 'leads': {}}
        visibility = {}
        for index, lead in enumerate(LEADS):
            counterfactual, _ = render_waveforms({**leads, lead: omitted[lead]}, **options)
            counterfactual.save(directory/f'counterfactual-{lead}.png')
            g = geometry[lead]; duration, amplitude = shape_geometry[lead]
            x0 = max(0, int(g['x0Pixels']+1.25*g['pixelsPerSecond']))
            x1 = min(source.width, int(g['x0Pixels']+1.75*g['pixelsPerSecond'])+1)
            y0 = max(0, int(g['baselineYPixels']-1.2*g['pixelsPerMv']))
            y1 = min(source.height, int(g['baselineYPixels']+1.2*g['pixelsPerMv'])+1)
            assessment = source_feature_visibility(np.array(source), np.array(counterfactual), region=(x0,y0,x1,y1),
                                                  horizontal_extent_pixels=duration*g['pixelsPerSecond'],
                                                  vertical_extent_pixels=amplitude/1000*g['pixelsPerMv'])
            visibility[lead] = {**assessment, 'sourceRegion': [x0,y0,x1,y1], 'shape': protocol['shapeStrata'][index%6],
                                'counterfactualSha256': digest(directory/f'counterfactual-{lead}.png')}
            annotations['leads'][lead] = {'events': [], 'occludedRanges': [],
                                          'measurements': [dict(m, visibility=assessment['state']) for m in labels[lead]]}
        validate_annotations(annotations, ROOT/'benchmark/schemas/annotations.schema.json')
        (directory/'annotations.json').write_text(json.dumps(annotations, indent=2)+'\n')
        (directory/'recoverability.json').write_text(json.dumps(visibility, indent=2)+'\n')
        artifacts = {key: {'path': f"{member['caseId']}/{name}", 'sha256': digest(directory/name)}
                     for key, name in [('image','image.png'),('truth','truth.csv'),('annotations','annotations.json')]}
        cases.append({'caseId': member['caseId'], 'sourceId': protocol['family'], 'groupId': protocol['family'],
                      'split': 'development', 'sampleRateHz': rate, 'layout': protocol['layout'], 'segmentDurationSeconds': 5,
                      'truthCoordinateFrame': 'lead_local', 'degradation': {'kind': 'resolution', 'parameters': {'pixelsPerMm': ppm}},
                      'strata': {'acquisition': 'controlled_synthetic', 'pixelsPerMm': ppm, 'template': protocol['family']},
                      'provenance': {'clinicalValidationUse': False, 'quantitativeTruth': True,
                                     'waveformOrigin': 'deterministic engineering fixture', 'acquisition': 'synthetic_render'}, **artifacts})
    manifest = {'version': 1, 'suiteId': 'labelled_measurements_v1', 'protocolSha256': digest(protocol_path), 'cases': cases}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'.keep-artifacts.json').write_text('{"version":1,"reason":"Controlled measurement and raster recoverability evidence."}\n')
    return {'cases': len(cases), 'manifestSha256': digest(output/'manifest.json'),
            'sourceHashes': {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__).resolve(), ROOT/'ecg_benchmark/generate.py', ROOT/'ecg_benchmark/measurements.py']}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(build(args.protocol, args.output), indent=2))
