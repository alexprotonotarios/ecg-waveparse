"""Bind a retained calibration to its source row without reselecting calibration."""
import copy
from ecg_pipeline.source_photo import _retained_pulse_binding, _local_timing
from ecg_pipeline.source_photo import _neutral_timing_replay, _neutral_local_timing


def resolve(image, observation):
    proof = _retained_pulse_binding.bind(image, observation)
    final = copy.deepcopy(observation.get('timing'))
    ordinary = neutral = None
    if proof['state'] == 'confirmed':
        updated = copy.deepcopy(observation)
        updated['pulseCalibrationSourceRow'] = proof['sourceRow']
        updated['retainedCalibrationSourceBinding'] = proof
        ordinary = _local_timing.propose(image, _neutral_timing_replay.case(updated, updated['separatorAttempts']))
        final = ordinary
        if not ordinary['candidatePassed']:
            gates = _neutral_timing_replay.eligibility(image, observation)
            candidate_gates = dict(gates, sourceRowConfirmed=True)
            neutral = {'originalEligibility': gates, 'candidateEligibility': candidate_gates,
                       'eligible': all(candidate_gates.values()), 'newTimingAdmission': False}
            if neutral['eligible']:
                attempts, mask = _neutral_timing_replay.windows(image, updated)
                alternate = _neutral_local_timing.propose(image, _neutral_timing_replay.case(updated, attempts))
                neutral.update(maskEvidence=mask, alternativeAttempts=attempts, alternativeTiming=alternate,
                               newTimingAdmission=alternate['candidatePassed'])
                if alternate['candidatePassed']: final = alternate
    return {'binding': proof, 'ordinaryTiming': ordinary, 'neutralReplay': neutral, 'finalTiming': final}
