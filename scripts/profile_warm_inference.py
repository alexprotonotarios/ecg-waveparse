"""Actual first/repeated model forwards, without enabling a persistent worker.

The installed runtime is verified before loading. Every input has a new model;
its first forward is compared with two repeated forwards on the same model.
This measures a backend stage, not application policy, OCR or end-to-end yield.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime', 'resources', 'config', 'protocol', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu','mps'), required=True)
    args = parser.parse_args()
    for name in ('runtime', 'resources', 'config', 'protocol', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    if protocol['clinicalValidationUse'] is not False or protocol['version'] != 1:
        raise ValueError('A frozen engineering protocol is required.')
    payload_path = args.resources/'payload-manifest.json'; payload = json.loads(payload_path.read_text())
    for relative, digest in payload['files'].items():
        if sha(args.resources/relative) != digest: raise ValueError('Installed payload changed.')
    subprocess.run([sys.executable, str(args.resources/'runtime_setup.py'), 'doctor', '--runtime-dir', str(args.runtime),
                    '--resource-root', str(args.resources), '--device', args.device], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sys.path[:0] = [str(args.resources), str(args.runtime/'engine')]
    os.chdir(args.runtime/'engine')
    import torch
    import numpy as np
    from torchvision.io import decode_image
    from src.config.default import get_cfg
    from src.utils import import_class_from_path
    torch.set_num_threads(protocol['cpuThreads'])
    config = get_cfg(); config.merge_from_file(str(args.config))
    config.MODEL.KWARGS.device = args.device
    config.MODEL.KWARGS.config.LAYOUT_IDENTIFIER.KWARGS.device = args.device
    config.MODEL.KWARGS.enable_timing = False
    wrapper = import_class_from_path(config.MODEL.class_path)
    def sync():
        if args.device == 'mps': torch.mps.synchronize()
    report = {'version': 1, 'clinicalValidationUse': False, 'device': args.device,
              'protocolSha256': sha(args.protocol), 'configSha256': sha(args.config),
              'payloadManifestSha256': sha(payload_path), 'attemptedInputs': len(protocol['membership']), 'rows': [],
              'definition': 'New-model first forward, then two forwards on the same model; OS/filesystem caches are not flushed.',
              'applicationPolicyAndUncertainty': 'not evaluated by this backend-stage profile', 'productionChanged': False,
              'stageTimingLimit': 'Upstream stage brackets are host wall times and can overlap asynchronous MPS work; whole-forward timings synchronize the device.'}
    def save():
        temporary = args.output/'report.json.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n'); temporary.replace(args.output/'report.json')
    save()
    for case in protocol['membership']:
        image_path = (args.protocol.parent/case['image']).resolve()
        if sha(image_path) != case['sha256']: raise ValueError('Input hash mismatch.')
        image = decode_image(str(image_path), mode='RGB').float().unsqueeze(0)/255
        started = time.perf_counter(); model = wrapper(**config.MODEL.KWARGS); sync()
        load_seconds = time.perf_counter()-started
        reference = None; first_layout = None; first_spacing = None
        for repetition in range(3):
            row = {'caseId': case['caseId'], 'phase': 'new_model_first_forward' if repetition == 0 else f'warm_forward_{repetition}',
                   'modelConstructionSeconds': load_seconds if repetition == 0 else 0., 'sourceSha256': case['sha256']}
            sync(); started = time.perf_counter()
            try:
                output = model(image.clone(), layout_should_include_substring=None); sync()
                row['forwardSeconds'] = time.perf_counter()-started
                spacing = {k: float(v) for k,v in output['pixel_spacing_mm'].items()}
                if not all(np.isfinite(v) and v > 0 for v in spacing.values()):
                    raise ValueError('Invalid inferred pixel spacing.')
                row['layout'] = output['layout_name']; row['pixelSpacingMm'] = spacing
                row['stagesSeconds'] = {k: float(v) for k,v in model.times.items()}
                values = output['signal']['canonical_lines']
                if values is None:
                    row['outcome'] = 'no_canonical_candidate'
                else:
                    values = values.detach().cpu().numpy()
                    if np.isinf(values).any(): raise ValueError('Infinite canonical value.')
                    filename = f"{case['caseId']}-{repetition}.npy"
                    np.save(args.output/filename, values, allow_pickle=False)
                    row.update(outcome='candidate_returned' if np.isfinite(values).any() else 'empty_canonical_array', shape=list(values.shape), units='uV',
                               arraySha256=sha(args.output/filename), finiteSamples=int(np.isfinite(values).sum()))
                    if repetition == 0:
                        reference = values.copy(); first_layout = row['layout']; first_spacing = row['pixelSpacingMm']
                    elif reference is not None and values.shape == reference.shape:
                        mask = np.isfinite(values)&np.isfinite(reference)
                        error = values[mask]-reference[mask]
                        row['repeatComparison'] = {'layoutMatches': row['layout']==first_layout,
                            'missingnessDifferences': int(np.count_nonzero(np.isfinite(values)!=np.isfinite(reference))),
                            'maximumAbsoluteDifferenceUv': float(np.max(np.abs(error))) if error.size else None,
                            'rmseUv': float(np.sqrt(np.mean(error**2))) if error.size else None,
                            'maximumSpacingDifferenceMm': max(abs(float(row['pixelSpacingMm'][k])-float(first_spacing[k])) for k in first_spacing)}
                    else:
                        row['repeatComparison'] = {'status': 'first_result_missing_or_shape_changed'}
                del output
            except Exception as error:
                row['forwardSeconds'] = time.perf_counter()-started; row.update(outcome='backend_failure', errorType=type(error).__name__)
            row['withinRuntimeBudget'] = row['forwardSeconds'] <= protocol['maximumSecondsPerForward']
            report['rows'].append(row); save()
            print(json.dumps({k:row[k] for k in ('caseId','phase','outcome','forwardSeconds')}), flush=True)
        del model, image; gc.collect()
        if args.device == 'mps': torch.mps.empty_cache()
    for relative, digest in payload['files'].items():
        if sha(args.resources/relative) != digest: raise ValueError('Installed payload changed during profiling.')


if __name__ == '__main__': main()
