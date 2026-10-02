"""Image-derived geometry and timing evidence for neural waveform conversion.

Source evidence does not qualify a waveform or verify source-label identities in
an export. Decoder correspondence and publication lineage are separate inputs.
"""
from __future__ import annotations
import hashlib
import re
from types import SimpleNamespace
import numpy as np


def observe_source_time(image: np.ndarray, *, source_sha256: str | None = None) -> dict:
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or min(image.shape[:2]) < 200 or image.shape[0] * image.shape[1] > 12_000_000):
        raise ValueError('Expected a bounded untouched BGR uint8 source raster.')
    if source_sha256 is not None and (not isinstance(source_sha256, str) or re.fullmatch('[0-9a-f]{64}', source_sha256) is None):
        raise ValueError('Source identity must be a lowercase SHA-256 digest.')
    from ecg_pipeline import source_label_geometry, lead_label_identity, printed_calibration, source_panel_timing
    from ecg_pipeline.source_photo import _calibration_observer
    from ecg_pipeline.source_photo._dependencies import PARTS
    from scripts import detect_ecg_layout
    from . import _grid, _timing, _origins

    digest = hashlib.sha256(image.tobytes()).hexdigest()
    names = ['crops', 'constraints', 'composite', 'label_observer', 'edge_calibration', 'pulse_fallback', 'period', 'windows', 'local_timing']
    # Native joint timing is available to native extraction, but has not been
    # applied to this neural waveform. Keep the original source-window policy
    # here so an available native fallback cannot suppress neural conversion.
    neural_timing = SimpleNamespace(
        propose_source_panel_timing=source_panel_timing._propose_source_window_timing,
        _window_counts=source_panel_timing._window_counts,
    )
    observation = _calibration_observer.observe(image, source_label_geometry, lead_label_identity,
        **{name: PARTS[name] for name in names}, layout=detect_ecg_layout, printed_calibration=printed_calibration,
        maintained_timing=neural_timing, source_sha=source_sha256 or digest)
    resolved = _timing.resolve(image, observation)
    result = {'version': 1, 'method': 'observed_source_grid_time_v1', 'state': 'unresolved',
              'sourceSha256': source_sha256 or digest, 'decodedRasterSha256': digest,
              'imageSize': [image.shape[1], image.shape[0]], 'calibrationObservation': observation,
              'timingResolution': resolved, 'geometry': None, 'origins': None,
              'waveformConverted': False, 'sourceIdentityPromoted': False}
    if observation.get('existingTiming', {}).get('state') == 'source_supported':
        result['state'] = 'preferred_timing_retained'
    elif resolved['finalTiming'] and resolved['finalTiming'].get('candidatePassed'):
        geometry, _ = _calibration_observer.source_context(observation['labels'], image)
        cal = observation['reconciledCalibration']
        if cal['paperSpeedMmPerSecond'] == 25 and cal['gainMmPerMv'] == 10:
            try:
                grid = _grid.observe(image, geometry, cal)
                origins = _origins.observe(grid, resolved['finalTiming']['candidate'])
                result.update(geometry=grid, origins=origins,
                              state='source_time_supported' if origins['admitted'] else 'unresolved')
            except ValueError as error:
                result['refusalReason'] = str(error)
        else:
            result['refusalReason'] = 'source_time_method_requires_25_mm_per_second_and_10_mm_per_mv'
    if hashlib.sha256(image.tobytes()).hexdigest() != digest:
        raise RuntimeError('Source time observer changed the input raster.')
    return result
