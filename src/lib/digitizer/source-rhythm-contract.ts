import { EvidenceContractError } from "./physical-contracts"

export type SourceStrokeTransitionEvidence = {
  eligible: boolean
  certified: boolean
  sourcePixels: number[][]
  displacementPixels: number
  sourceInterval: number[]
  evidenceThreshold: 0.24
  horizontalMarginPixels: 3
  endpointRadiusPixels: 1
  pathOrValidityChanged: false
  bounds?: number[]
  verticalSupportFraction?: number
  sharedComponentCount?: number
  completeConnectedComponentCount?: number
  binarySha256?: string
  certifiedComponentSha256?: string
}

export type SourceRhythmContinuityEvidence = {
  version: 1
  method: "complete-connected-source-rhythm-strokes-v1"
  decodedRasterSha256: string
  imageSize: number[]
  sourceInterval: number[]
  evidenceEncoding: "float32-le-row-major-v1"
  evidenceSha256: string
  pathEncoding: "int32-le-y-by-source-column-v1"
  columnsSha256: string
  sourcePathSha256: string
  validitySha256: string
  recordedValiditySha256: string
  retainedRecordedColumns: number
  measuredPairCount: number
  rowSpacingPixels: number
  rawMaximumJumpPixels: number
  originalMaximumAllowedJumpPixels: number
  originalMaximumJumpPassed: boolean
  otherRhythmChecksPassed: boolean
  legacyRhythmTracePassed: boolean
  overLimitMeasuredPairCount: number
  overLimitTransitions: SourceStrokeTransitionEvidence[]
  allOverLimitMeasuredPairsCertified: boolean
  rhythmTracePassed: boolean
  sourcePathChanged: false
  sourceEvidenceChanged: false
  sourceTimingChanged: false
  validityChanged: false
  gapsRecovered: 0
  truthUsed: false
}

const fail = (): never => {
  throw new EvidenceContractError("invalid_source_rhythm_continuity", "Steep rhythm transitions require complete source-stroke evidence and their original QA measurements.")
}
function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return fail()
  return value as Record<string, unknown>
}
function numeric(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return fail()
  return value
}
function integer(value: unknown, low: number, high: number): number {
  const result = numeric(value)
  if (!Number.isSafeInteger(result) || result < low || result > high) return fail()
  return result
}
function numbers(value: unknown, count: number): number[] {
  if (!Array.isArray(value) || value.length !== count) return fail()
  return value.map(numeric)
}
function equal(a: number[], b: number[]) {
  return a.length === b.length && a.every((value, i) => value === b[i])
}
function hash(value: unknown) {
  return typeof value === "string" && /^[a-f0-9]{64}$/.test(value)
}

/** Validate the new rule separately; legacy source QA retains its old method. */
export function validateSourceRhythmContinuity(
  fidelity: Record<string, unknown>,
  timing: Record<string, unknown>,
  size: number[]
) {
  const proof = object(fidelity.sourceRhythmContinuity)
  const interval = numbers(timing.rhythmRange, 2)
  const metric = object(object(fidelity.leadMetrics).II)
  const refinement = object(fidelity.sourceRhythmRefinement)
  if (!["source-label-separator-local-grid-supported-ink-v2", "source-label-separator-local-grid-supported-ink-v3", "source-label-joint-row-grid-supported-ink-v4", "source-label-endpoint-family-row-grid-supported-ink-v5"].includes(String(fidelity.method)) ||
      proof.version !== 1 || proof.method !== "complete-connected-source-rhythm-strokes-v1" ||
      !hash(proof.decodedRasterSha256) || proof.decodedRasterSha256 !== timing.decodedRasterSha256 ||
      !equal(numbers(proof.imageSize, 2), size) || !equal(numbers(proof.sourceInterval, 2), interval) ||
      proof.evidenceEncoding !== "float32-le-row-major-v1" ||
      proof.pathEncoding !== "int32-le-y-by-source-column-v1" ||
      !["evidenceSha256", "columnsSha256", "sourcePathSha256", "validitySha256", "recordedValiditySha256"].every(k => hash(proof[k])) ||
      ["sourcePathChanged", "sourceEvidenceChanged", "sourceTimingChanged", "validityChanged", "truthUsed"].some(k => proof[k] !== false) ||
      proof.gapsRecovered !== 0 || proof.columnsSha256 !== refinement.columnsSha256 ||
      proof.sourcePathSha256 !== refinement.refinedPathSha256) fail()
  const width = interval[1] - interval[0]
  const retained = integer(proof.retainedRecordedColumns, 1, width)
  const pairs = integer(proof.measuredPairCount, Math.max(0, 2 * retained - width - 1), retained - 1)
  if (retained !== refinement.refinedValidColumns) fail()
  if (numeric(metric.coverage) !== retained / width) fail()
  if (!Array.isArray(timing.rowGridMeasurements) || timing.rowGridMeasurements.length !== 4) fail()
  const centers = (timing.rowGridMeasurements as unknown[]).map(row => numeric(object(row).rowCenter))
  if (centers.some((y, i) => y < 0 || y >= size[1] || (i > 0 && y <= centers[i - 1]))) fail()
  const spacings = centers.slice(1).map((y, i) => y - centers[i]).sort((a, b) => a - b)
  const spacing = numeric(proof.rowSpacingPixels)
  if (spacing <= 0 || spacing !== spacings[1]) fail()
  const raw = numeric(proof.rawMaximumJumpPixels), bound = spacing * .35
  if (raw < 0 || raw !== metric.maxNativeJumpPixels || proof.originalMaximumAllowedJumpPixels !== bound) fail()
  const originalPassed = raw <= bound
  if (proof.originalMaximumJumpPassed !== originalPassed) fail()
  if (fidelity.sourceGridConversion !== undefined) {
    const conversion = object(fidelity.sourceGridConversion)
    if (conversion.state === "applied" && ["columnsSha256", "sourcePathSha256", "validitySha256"].some(k => proof[k] !== conversion[k])) fail()
  }
  const count = integer(proof.overLimitMeasuredPairCount, 0, pairs)
  if (!Array.isArray(proof.overLimitTransitions) || proof.overLimitTransitions.length !== count ||
      (count === 0) !== originalPassed) fail()
  let previousColumn = -1
  const transitions = (proof.overLimitTransitions as unknown[]).map(rawTransition => {
    const transition = object(rawTransition)
    if (!Array.isArray(transition.sourcePixels) || transition.sourcePixels.length !== 2) fail()
    const [from, to] = (transition.sourcePixels as unknown[]).map(p => numbers(p, 2))
    const [x0, y0] = from, [x1, y1] = to
    for (const [x, y] of [from, to]) {
      integer(x, interval[0], interval[1] - 1)
      integer(y, 0, size[1] - 1)
    }
    const displacement = Math.abs(y1 - y0), eligible = x1 === x0 + 1
    if (x0 <= previousColumn || x1 <= x0 || displacement <= bound || displacement > raw ||
        transition.displacementPixels !== displacement || transition.eligible !== eligible ||
        !equal(numbers(transition.sourceInterval, 2), interval) ||
        transition.evidenceThreshold !== .24 || transition.horizontalMarginPixels !== 3 ||
        transition.endpointRadiusPixels !== 1 || transition.pathOrValidityChanged !== false) fail()
    previousColumn = x0
    if (!eligible) {
      if (transition.certified !== false || ["bounds", "verticalSupportFraction", "sharedComponentCount", "completeConnectedComponentCount", "binarySha256", "certifiedComponentSha256"].some(k => transition[k] !== undefined)) fail()
      return transition
    }
    const bounds = [Math.max(interval[0], x0 - 3), Math.min(y0, y1), Math.min(interval[1], x1 + 4), Math.max(y0, y1) + 1]
    if (!equal(numbers(transition.bounds, 4), bounds) || !hash(transition.binarySha256)) fail()
    const pixels = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    const shared = integer(transition.sharedComponentCount, 0, pixels)
    const complete = integer(transition.completeConnectedComponentCount, 0, shared)
    const support = numeric(transition.verticalSupportFraction)
    if (support < 0 || support > 1 || transition.certified !== (complete === 1)) fail()
    if (complete === 1) {
      if (support !== 1 || !hash(transition.certifiedComponentSha256)) fail()
    } else if (transition.certifiedComponentSha256 !== undefined) fail()
    return transition
  })
  if (count > 0 && Math.max(...transitions.map(t => numeric(t.displacementPixels))) !== raw) fail()
  const allCertified = count > 0 && transitions.every(t => t.certified === true)
  const p10 = numeric(metric.evidenceP10), home = numeric(metric.homeRowFraction)
  const p95 = numeric(metric.p95NativeJumpPixels)
  if (p10 < 0 || p10 > 1 || home < 0 || home > 1 || p95 < 0 || p95 > raw) fail()
  const otherPassed = retained / width >= .90 && p10 >= .45 && home >= .95 && p95 <= spacing * .20
  const passed = otherPassed && (originalPassed || allCertified)
  if (proof.allOverLimitMeasuredPairsCertified !== allCertified || proof.otherRhythmChecksPassed !== otherPassed ||
      proof.legacyRhythmTracePassed !== (otherPassed && originalPassed) ||
      proof.rhythmTracePassed !== passed || fidelity.rhythmTracePassed !== passed ||
      (fidelity.passed === true && !passed)) fail()
}
