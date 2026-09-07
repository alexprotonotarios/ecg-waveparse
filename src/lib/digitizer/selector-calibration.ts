import calibrationDocument from "../../../config/candidate-selector-calibration.v2.json"

import { candidateKind, type CandidateCapabilityInput } from "@/lib/digitizer/candidate-capabilities"
import {
  DIGITIZER_POLICY,
  DIGITIZER_SELECTOR_CALIBRATION_CONTRACT_VERSION,
} from "@/lib/digitizer/policy"
import type { PreprocessingReport } from "@/lib/digitizer/contracts"
import { normalizeSupportedEcgLayout } from "@/lib/ecg-layouts"
import type { DigitizerInputVariant, DigitizerVectorizer } from "@/lib/runs"
import { profileParameterCompatibility, type ProfileParameterIdentity } from "@/lib/digitizer/selector-feature-identity"

type CalibrationEntry = {
  caseCount: number
  medianRmseUv: number
  p75RmseUv: number
  shrunkRmseUv: number
  riskRmseUv: number
}

type CalibrationBucket = {
  referenceMedianRmseUv: number
  referenceRiskRmseUv: number
  scoreableCandidateCount: number
  candidates: Record<string, CalibrationEntry>
  families: Record<string, CalibrationEntry>
  selection?: {
    enabled: boolean
    validationCaseCount: number
    layoutPriorMeanRmseUv: number
    contextPriorMeanRmseUv: number
    meanImprovementUv: number
  }
}

type LayoutCalibration = CalibrationBucket & {
  contexts?: Record<string, CalibrationBucket>
}

type SelectorCalibrationInput = CandidateCapabilityInput & {
  id: string
  inputVariant: DigitizerInputVariant
  vectorizer: DigitizerVectorizer
}

export type SelectorCalibrationEvidence = {
  parameterCompatibility: "compatible" | "incompatible" | "historical_unverified"
  profileId: string
  layout?: string
  artifactContext?: string
  contextSelectionEnabled?: boolean
  source:
    | "context-candidate"
    | "context-family"
    | "candidate"
    | "family"
    | "layout-reference"
    | "unavailable"
  priorRmseUv: number
  bestLayoutPriorRmseUv: number
  penalty: number
  trainingCaseCount: number
  trainingPipelineCommit: string
  calibrationContractVersion: number
  requiredCalibrationContractVersion: number
  priorsApplied: boolean
  heldoutUsed: boolean
  artifactContextSupported: boolean
  exactCandidateCalibrated: boolean
  calibrationConfidence:
    | "validation-activated-context"
    | "layout-only"
    | "uncalibrated"
  outOfDomainReasons: string[]
}

const layouts = calibrationDocument.layouts as Record<string, LayoutCalibration>

export function selectorCalibrationEvidence(
  layout: string | undefined,
  candidate: SelectorCalibrationInput,
  preprocessing?: PreprocessingReport
): SelectorCalibrationEvidence {
  const calibrationContractVersion =
    calibrationDocument.training.pipelineContractVersion
  const calibrationContractCompatible =
    calibrationContractVersion ===
    DIGITIZER_SELECTOR_CALIBRATION_CONTRACT_VERSION
  const normalizedLayout = normalizeSupportedEcgLayout(layout)
  const identity = (calibrationDocument.training as typeof calibrationDocument.training & { reproducibility?: ProfileParameterIdentity }).reproducibility
  const parameterCompatibility = profileParameterCompatibility(identity, normalizedLayout, candidate)
  const calibration = normalizedLayout ? layouts[normalizedLayout] : undefined
  const artifactContext = preprocessing
    ? selectorArtifactContext(preprocessing)
    : undefined
  const contextCalibration = artifactContext
    ? calibration?.contexts?.[artifactContext]
    : undefined
  if (!calibrationContractCompatible || !normalizedLayout || !calibration || parameterCompatibility === "incompatible") {
    return {
      parameterCompatibility,
      profileId: calibrationDocument.profileId,
      ...(normalizedLayout ? { layout: normalizedLayout } : {}),
      ...(artifactContext ? { artifactContext } : {}),
      ...(artifactContext
        ? {
            contextSelectionEnabled:
              contextCalibration?.selection?.enabled === true,
          }
        : {}),
      source: "unavailable",
      priorRmseUv: 0,
      bestLayoutPriorRmseUv: 0,
      penalty: 0,
      trainingCaseCount: 0,
      trainingPipelineCommit: calibrationDocument.training.pipelineCommit,
      calibrationContractVersion,
      requiredCalibrationContractVersion:
        DIGITIZER_SELECTOR_CALIBRATION_CONTRACT_VERSION,
      priorsApplied: false,
      heldoutUsed: calibrationDocument.training.heldoutUsed,
      artifactContextSupported: Boolean(contextCalibration),
      exactCandidateCalibrated: false,
      calibrationConfidence: "uncalibrated",
      outOfDomainReasons: [
        ...(parameterCompatibility === "incompatible" ? ["candidate-parameter-contract-mismatch"] : []),
        ...(!calibrationContractCompatible
          ? ["pipeline-calibration-contract-mismatch"]
          : []),
        ...(!normalizedLayout || !calibration
          ? ["layout-not-present-in-calibration-profile"]
          : []),
        ...(artifactContext && !contextCalibration
          ? [`artifact-context-${artifactContext}-not-calibrated`]
          : []),
      ],
    }
  }

  const family = candidateFamily(candidate)
  const contextCandidateEntry = contextCalibration?.candidates[candidate.id]
  const contextFamilyEntry = contextCalibration?.families[family]
  const contextSelectionEnabled =
    contextCalibration?.selection?.enabled === true
  const useContext = Boolean(
    contextSelectionEnabled &&
      (contextCandidateEntry ?? contextFamilyEntry)
  )
  const activeCalibration = useContext
    ? (contextCalibration as CalibrationBucket)
    : calibration
  const candidateEntry = useContext
    ? contextCandidateEntry
    : calibration.candidates[candidate.id]
  const familyEntry = useContext
    ? contextFamilyEntry
    : calibration.families[family]
  const entry = candidateEntry ?? familyEntry
  const source = useContext
    ? candidateEntry
      ? "context-candidate"
      : "context-family"
    : candidateEntry
      ? "candidate"
      : familyEntry
        ? "family"
        : "layout-reference"
  const priorRmseUv =
    entry?.riskRmseUv ?? activeCalibration.referenceRiskRmseUv
  const adequatelySupportedCandidatePriors = Object.values(
    activeCalibration.candidates
  ).filter(
    (candidatePrior) =>
      candidatePrior.caseCount >= DIGITIZER_POLICY.minimumSelectorCalibrationCases
  )
  const bestLayoutPriorRmseUv = Math.min(
    activeCalibration.referenceRiskRmseUv,
    ...adequatelySupportedCandidatePriors.map(
      (candidatePrior) => candidatePrior.riskRmseUv
    )
  )
  const trainingCaseCount = entry?.caseCount ?? 0
  const outOfDomainReasons: string[] = []
  if (artifactContext && !contextCalibration) {
    outOfDomainReasons.push(`artifact-context-${artifactContext}-not-calibrated`)
  } else if (
    artifactContext &&
    contextSelectionEnabled &&
    !contextCandidateEntry &&
    !contextFamilyEntry
  ) {
    outOfDomainReasons.push("artifact-context-candidate-family-not-calibrated")
  }
  if (!entry) outOfDomainReasons.push("candidate-family-not-calibrated")
  if (entry && trainingCaseCount < DIGITIZER_POLICY.minimumSelectorCalibrationCases) {
    outOfDomainReasons.push("calibration-bucket-too-small")
  }
  const basePenalty = Math.round(
    Math.max(0, priorRmseUv - bestLayoutPriorRmseUv) *
      DIGITIZER_POLICY.selectorCalibrationPenaltyPerUv
  )
  const priorsApplied = outOfDomainReasons.length === 0
  const penalty = priorsApplied ? basePenalty : 0
  const calibrationConfidence =
    outOfDomainReasons.length > 0
      ? "uncalibrated"
      : useContext
        ? "validation-activated-context"
        : "layout-only"

  return {
    parameterCompatibility,
    profileId: calibrationDocument.profileId,
    layout: normalizedLayout,
    ...(artifactContext ? { artifactContext } : {}),
    ...(artifactContext
      ? { contextSelectionEnabled }
      : {}),
    source,
    priorRmseUv,
    bestLayoutPriorRmseUv,
    penalty,
    trainingCaseCount,
    trainingPipelineCommit: calibrationDocument.training.pipelineCommit,
    calibrationContractVersion,
    requiredCalibrationContractVersion:
      DIGITIZER_SELECTOR_CALIBRATION_CONTRACT_VERSION,
    priorsApplied,
    heldoutUsed: calibrationDocument.training.heldoutUsed,
    artifactContextSupported: Boolean(contextCalibration),
    exactCandidateCalibrated: Boolean(
      useContext ? contextCandidateEntry : calibration.candidates[candidate.id]
    ),
    calibrationConfidence,
    outOfDomainReasons,
  }
}

export function selectorArtifactContext(
  preprocessing: PreprocessingReport
) {
  const artifact = preprocessing.artifactPreprocessing
  const adaptive = preprocessing.adaptivePreprocessing
  if (artifact?.screenArtifactLikely) return "screen"
  if (preprocessing.geometryCorrection?.applied) return "geometry"
  if (adaptive?.lowResolution) return "low-resolution"
  if (preprocessing.annotationMask.maskedPixels > 0) return "annotation"
  if (adaptive?.belowSharpnessFloor) return "blur"
  if (adaptive?.foregroundGridDegraded) return "degraded-raster"
  return "clean"
}

function candidateFamily(candidate: SelectorCalibrationInput) {
  return [
    candidateKind(candidate),
    candidate.vectorizer,
    candidate.inputVariant,
  ].join(":")
}

export const SELECTOR_CALIBRATION_PROFILE_ID = calibrationDocument.profileId
export const SELECTOR_CALIBRATION_TRAINING = calibrationDocument.training
