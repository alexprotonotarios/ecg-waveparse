// Distribution projections preserve operational settings and omit unused history.
export function distributionResource(relative, bytes) {
  if (relative === 'config/candidate-selector-calibration.v2.json') {
    const value = JSON.parse(bytes);
    const { pipelineContractVersion, pipelineCommit, heldoutUsed, reproducibility } = value.training;
    value.training = { pipelineContractVersion, pipelineCommit, heldoutUsed,
      ...(reproducibility ? { reproducibility } : {}) };
    return JSON.stringify(value, null, 2) + '\n';
  }
  if (/^config\/digitizer-policy\.v[23]\.json$/.test(relative)) {
    const value = JSON.parse(bytes);
    value.evidenceBasis = 'Engineering thresholds; not clinically calibrated. Inspect source, calibration, missingness and uncertainty before quantitative use.';
    value.rationale.selectorCalibration = 'Versioned layout and artifact-context risk priors apply only after structural gates. Incompatible parameter contracts and unsupported contexts disable the priors. Priors cannot make an unsafe candidate publishable.';
    return JSON.stringify(value, null, 2) + '\n';
  }
  return bytes;
}

export const sourceScripts = {
  'waveparse:build': 'node scripts/build-waveparse.mjs',
  'waveparse:verify': 'node scripts/verify-waveparse-packages.mjs',
};
