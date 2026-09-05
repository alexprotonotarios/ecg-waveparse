import type {
  DigitizerCandidateCapabilities,
  DigitizerCandidateKind,
} from "@/lib/digitizer/domain"
import type {
  DigitizerInputVariant,
  DigitizerVectorizer,
} from "@/lib/runs"

export type CandidateCapabilityInput = {
  kind?: DigitizerCandidateKind
  inputVariant?: DigitizerInputVariant
  vectorizer?: DigitizerVectorizer
  layoutConstraint?: string
  cropBox?: unknown
  selectionEligible?: boolean
  darkInkEnhancement?: boolean
  capabilities?: DigitizerCandidateCapabilities
}

export function candidateKind(
  candidate: CandidateCapabilityInput
): DigitizerCandidateKind {
  if (candidate.kind) return candidate.kind
  if (candidate.vectorizer === "native-grid-path") return "native-grid"
  if (candidate.cropBox) return "crop-proposal"
  if (candidate.layoutConstraint) return "geometry-constrained"
  if (candidate.inputVariant === "preprocessed") return "adaptive-preprocessed"
  return "source-model"
}

export function candidateCapabilities(
  candidate: CandidateCapabilityInput
): DigitizerCandidateCapabilities {
  if (candidate.capabilities) return candidate.capabilities
  const kind = candidateKind(candidate)
  const nativeGrid = candidate.vectorizer === "native-grid-path"
  const derived =
    kind === "panel-composite" ||
    kind === "peer-gap-repair" ||
    kind === "lead-fusion"
  const preprocessed =
    candidate.inputVariant === "preprocessed" ||
    candidate.inputVariant === "geometry-corrected" ||
    candidate.inputVariant === "artifact-preprocessed"
  return {
    derived,
    nativeGrid,
    preprocessed,
    geometryConstrained: Boolean(candidate.layoutConstraint),
    sourcePixelEvidence:
      nativeGrid ||
      (candidate.vectorizer === "dynamic-path" &&
        candidate.darkInkEnhancement === true),
    selectionEligible: candidate.selectionEligible !== false,
  }
}

export function isBaselineSourceModelCandidate(
  candidate: CandidateCapabilityInput & {
    resampleSize?: number
    labelThresh?: number
  }
) {
  return (
    candidateKind(candidate) === "source-model" &&
    candidate.inputVariant === "original" &&
    candidate.vectorizer === "probability-centroid" &&
    candidate.resampleSize === 1500 &&
    candidate.labelThresh === undefined &&
    !candidate.layoutConstraint &&
    !candidate.cropBox
  )
}
