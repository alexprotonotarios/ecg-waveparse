import policyDocument from "../../../config/digitizer-policy.v3.json"

type DigitizerPolicyThresholds = typeof policyDocument.thresholds

function validatePolicy(thresholds: DigitizerPolicyThresholds) {
  if (
    policyDocument.version !== 3 ||
    !policyDocument.profileId ||
    !Number.isInteger(policyDocument.selectorCalibrationContractVersion) ||
    policyDocument.selectorCalibrationContractVersion < 1
  ) {
    throw new Error("Unsupported or unnamed digitizer policy profile.")
  }
  for (const [name, value] of Object.entries(thresholds)) {
    if (Array.isArray(value)) {
      if (value.length === 0 || value.some((entry) => !Number.isFinite(entry))) {
        throw new Error(`Digitizer policy ${name} must contain finite values.`)
      }
    } else if (!Number.isFinite(value)) {
      throw new Error(`Digitizer policy ${name} must be finite.`)
    }
  }
}

validatePolicy(policyDocument.thresholds)

export const DIGITIZER_POLICY_VERSION = policyDocument.version
export const DIGITIZER_POLICY_ID = policyDocument.profileId
export const DIGITIZER_POLICY_EVIDENCE_BASIS = policyDocument.evidenceBasis
export const DIGITIZER_SELECTOR_CALIBRATION_CONTRACT_VERSION =
  policyDocument.selectorCalibrationContractVersion
export const DIGITIZER_POLICY = Object.freeze(policyDocument.thresholds)
