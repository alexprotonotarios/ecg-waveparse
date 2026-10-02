import type { LayoutGeometryReport } from "./contracts"
import { canCaptureDecoderCoordinates } from "./candidate-preparation"
import { EvidenceContractError } from "./physical-contracts"

export const SOURCE_RHYTHM_ASSIGNMENT_METHOD = "source-confirmed-decoder-rhythm-assignment-v1"

/** Only unrectified, complete source geometry shares this preparation frame. */
export function canUseSourceRhythmAssignment(
  candidate: Parameters<typeof canCaptureDecoderCoordinates>[0] & { inputVariant?: string },
  geometry?: LayoutGeometryReport,
) {
  return canCaptureDecoderCoordinates(candidate) &&
    ["original", "annotation-masked", "preprocessed"].includes(candidate.inputVariant ?? "original") &&
    geometry?.layoutHint === "standard_3x4_with_r1" &&
    geometry.coordinateSpace === "working" && geometry.detectedInputVariant === "original" &&
    geometry.sourceLabelGrid?.method === "source-value-grid-below-trace-v1" &&
    geometry.leadLabelValidation?.semanticIdentityConfirmed === true &&
    geometry.rhythmLeadValidation?.passed === true &&
    geometry.rhythmLeadValidation.semanticIdentityConfirmed === true &&
    geometry.rhythmLeadValidation.lead === "II"
}

export type SourceRhythmAssignmentEvidence = {
  version: 1
  method: typeof SOURCE_RHYTHM_ASSIGNMENT_METHOD
  state: "applied" | "refused"
  reason: string | null
  contextSha256: string
  sourceSha256: string
  workingSourceSha256: string
  preparedInputSha256: string
  canonicalAssignmentPolicyChanged: boolean
  physicalTimingVerified: false
  sourceInkVerified: false
  lead?: "II"
  canonicalIndex?: 1
  rhythmRow?: 3
}

export function validateSourceRhythmAssignment(value: unknown, expected: {
  contextSha256: string; sourceSha256: string; workingSourceSha256: string; preparedInputSha256: string
}): asserts value is SourceRhythmAssignmentEvidence {
  const r = value as SourceRhythmAssignmentEvidence | undefined
  if (!r || r.version !== 1 || r.method !== SOURCE_RHYTHM_ASSIGNMENT_METHOD ||
      !["applied", "refused"].includes(r.state) || r.physicalTimingVerified !== false || r.sourceInkVerified !== false ||
      Object.entries(expected).some(([key, hash]) => !/^[a-f0-9]{64}$/.test(hash) || r[key as keyof typeof expected] !== hash) ||
      (r.state === "applied" ? r.reason !== null || r.lead !== "II" || r.canonicalIndex !== 1 || r.rhythmRow !== 3 ||
        r.canonicalAssignmentPolicyChanged !== true : !r.reason || r.canonicalAssignmentPolicyChanged !== false)) {
    throw new EvidenceContractError("source_rhythm_assignment_mismatch", "Rhythm assignment does not match this source and candidate.")
  }
}
