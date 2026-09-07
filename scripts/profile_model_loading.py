"""Profile existing model construction without modifying the installed engine."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('runtime', 'resources', 'config', 'output'): parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    for key in ('runtime', 'resources', 'config', 'output'): setattr(args, key, getattr(args, key).resolve())
    if args.output.exists(): parser.error('Refusing to overwrite profiling evidence.')
    sys.path[:0] = [str(args.resources.resolve()), str((args.runtime / 'engine').resolve())]
    os.chdir(args.runtime / 'engine')
    from src.config.default import get_cfg
    from src.utils import import_class_from_path
    config = get_cfg(); config.merge_from_file(str(args.config.resolve()))
    if config.MODEL.KWARGS.device != 'cpu': raise ValueError('This construction profile is CPU-only.')
    wrapper = import_class_from_path(config.MODEL.class_path)
    stages = {}
    measured_methods = {}
    for name in ('_load_segmentation_model', '_load_layout_identifier', '_load_signal_extractor', '_load_perspective_detector'):
        original = getattr(wrapper, name)
        def measured(self, *positional, _original=original, _name=name, **keywords):
            started = time.perf_counter()
            try:
                return _original(self, *positional, **keywords)
            finally:
                stages[_name] = stages.get(_name, 0) + time.perf_counter() - started
        measured_methods[name] = measured
    measured_wrapper = type('MeasuredWrapper', (wrapper,), measured_methods)
    rows = []
    for label in ('first_construction_in_process', 'repeat_construction_in_same_process'):
        stages = {}; started = time.perf_counter()
        model = measured_wrapper(**config.MODEL.KWARGS)
        elapsed = time.perf_counter() - started
        if set(stages) != set(measured_methods): raise RuntimeError('An expected model-loading stage was not observed.')
        rows.append({'phase': label, 'constructionSeconds': elapsed, 'cumulativeStageSeconds': stages.copy()})
        del model; gc.collect()
    result = {'version': 1, 'configSha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
              'rows': rows, 'clinicalValidationUse': False,
              'limitations': ['Construction includes weight loading and object initialization, without waveform inference.',
                             'The filesystem cache may already be warm; this is not a disk-cold benchmark.',
                             'A local timing subclass forwards each original method unchanged; cumulative stages are within total construction time.']}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
