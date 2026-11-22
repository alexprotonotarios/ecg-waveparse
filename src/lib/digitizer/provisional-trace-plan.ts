import type { DigitizerCandidateConfig } from "./candidate-plan"
import type { LayoutGeometryReport, PreprocessingReport } from "./contracts"
import { ECG_LEADS, PARTIAL_LAYOUT_LEADS } from "./domain"
import type { DigitizerComputeDevice } from "../runs"

const FULL_LAYOUTS = new Set(["standard_3x4", "standard_3x4_with_r1", "standard_3x4_with_r2",
  "standard_3x4_with_r3", "standard_6x2", "standard_6x2_with_r1_ignored", "standard_12x1", "cabrera_12x1"])
const PARTIAL_CONSTRAINTS: Readonly<Record<string, string>> = {
  standard_6x1_limb: "standard_6x1_limb_constrained",
  cabrera_6x1_limb: "cabrera_6x1_limb_constrained",
  precordial_6x1: "precordial_6x1_constrained",
}

/** Planning hints do not establish identity, calibration, or source fidelity. */
export function expectedProvisionalLeads(geometry: LayoutGeometryReport): readonly string[] {
  const partial = PARTIAL_LAYOUT_LEADS[geometry.layoutHint ?? ""]
  const evidence = geometry.semanticLeadEvidence
  if (partial && evidence?.passed && evidence.visibleLeads.length === partial.length &&
      new Set(evidence.visibleLeads).size === partial.length && partial.every(lead => evidence.visibleLeads.includes(lead))) {
    return partial
  }
  return ECG_LEADS
}

export function buildProvisionalTracePlan(preprocessing: PreprocessingReport, geometry: LayoutGeometryReport,
  device: DigitizerComputeDevice): DigitizerCandidateConfig[] {
  const raster = preprocessing.workingImage ?? preprocessing.source
  const preserveOrientation = Number.isFinite(raster?.width) && Number.isFinite(raster?.height) &&
    raster.width > 0 && raster.height > raster.width
  const defaultCandidate: DigitizerCandidateConfig = { kind: "source-model", id: "trace-default", label: "Provisional source tracing",
    resampleSize: 1500, inputVariant: "original", vectorizer: "probability-centroid", device }
  // Portrait frames can contain upright horizontal ECGs. Keep their source
  // orientation first, with the historical automatic rotation as an alternative.
  const plan: DigitizerCandidateConfig[] = preserveOrientation
    ? [{ ...defaultCandidate, id: "trace-source-orientation", rotateOnResample: false },
      { ...defaultCandidate, rotateOnResample: true }]
    : [defaultCandidate]
  const hint = geometry.layoutHint ?? ""
  const constraint = FULL_LAYOUTS.has(hint) ? hint : PARTIAL_CONSTRAINTS[hint]
  if (constraint && geometry.semanticLeadEvidence?.passed) {
    plan.push({ kind: "geometry-constrained", id: `trace-constrained-${constraint}`, label: "Provisional label-guided tracing",
      resampleSize: 1500, layoutConstraint: constraint, maximumLayoutCost: .75,
      inputVariant: geometry.detectedInputVariant === "geometry-corrected" ? "geometry-corrected" :
        preprocessing.annotationMask.maskedPixels > 0 ? "annotation-masked" : "original",
      ...(preserveOrientation ? { rotateOnResample: false } : {}),
      vectorizer: "probability-centroid", device })
  } else if (preserveOrientation) {
    // A tracing hypothesis must not become label or geometry evidence. Keep
    // each short panel available even when the neural labels are unreadable.
    plan.push({ ...defaultCandidate, id: "trace-provisional-3x4-r1",
      label: "Provisional short-panel layout hypothesis", rotateOnResample: false,
      layoutConstraint: "standard_3x4_with_r1", provisionalLayoutHint: true,
      provisionalSourceInk: true,
      selectionEligible: false })
  }
  const variant = preprocessing.artifactPreprocessing?.eligible ? "artifact-preprocessed" :
    preprocessing.annotationMask.maskedPixels > 0 ? "annotation-masked" :
      preprocessing.geometryCorrection?.applied ? "geometry-corrected" :
        preprocessing.adaptivePreprocessing?.eligible ? "preprocessed" : undefined
  if (variant) {
    plan.push({ kind: "adaptive-preprocessed", id: "trace-cleaned", label: "Provisional deterministic-cleanup tracing",
      resampleSize: 1500, inputVariant: variant, vectorizer: "probability-centroid", device,
      ...(preserveOrientation ? { rotateOnResample: false } : {}) })
  }
  return plan.slice(0, 3)
}

export type ProvisionalNumericCandidate = {
  status: string
  layout?: string
  canonical?: { leads: string[]; rows: number[][] }
  panelObservationPath?: string
}

/** Numeric support permits early return; actual physical coverage still needs review. */
export function hasExpectedProvisionalSupport(candidate: ProvisionalNumericCandidate, expected: readonly string[]) {
  if (candidate.status !== "completed" || !candidate.canonical || !expected.length) return false
  const { leads, rows } = candidate.canonical
  if (!rows.length || new Set(leads).size !== leads.length || rows.some(row => row.length !== leads.length)) return false
  const counts = leads.map((_, index) => rows.reduce((count, row) => count + Number(Number.isFinite(row[index])), 0))
  if (!expected.every(lead => { const index = leads.indexOf(lead); return index >= 0 && counts[index] >= 250 })) return false
  const overwrite = /^standard_3x4_with_r[123]$/.test(candidate.layout ?? "") &&
    leads.some((lead, index) => lead !== "II" && counts[index] > rows.length / 4 + 250)
  if (overwrite && (candidate.layout !== "standard_3x4_with_r1" || !candidate.panelObservationPath)) return false
  return true
}

export async function collectProvisionalTraceCandidates<T extends ProvisionalNumericCandidate>(
  plan: DigitizerCandidateConfig[], execute: (config: DigitizerCandidateConfig) => Promise<T>,
  expected: readonly string[], signal?: AbortSignal): Promise<T[]> {
  const results: T[] = []
  for (const config of plan.slice(0, 3)) {
    signal?.throwIfAborted()
    const result = await execute(config)
    signal?.throwIfAborted()
    results.push(result)
    if (hasExpectedProvisionalSupport(result, expected)) break
  }
  return results
}
