import type { LayoutGeometryReport, PreprocessingReport, RasterCropBox } from "@/lib/digitizer/contracts"
import type { DigitizerCandidateKind } from "@/lib/digitizer/domain"
import { DIGITIZER_POLICY } from "@/lib/digitizer/policy"
import type {
  DigitizerComputeDevice,
  DigitizerInputVariant,
  DigitizerVectorizer,
} from "@/lib/runs"

const MODEL_SCALE_CENTROID_DIMENSION = 2200

export type DigitizerCandidateConfig = {
  kind: DigitizerCandidateKind
  id: string
  label: string
  resampleSize: number
  upscaleToMaxDimension?: number
  darkInkEnhancement?: boolean
  darkInkSupportRadius?: number
  gainMmPerMv?: number
  labelThresh?: number
  layoutConstraint?: string
  geometryConfirmedLayout?: boolean
  geometryLayoutConfidence?: number
  semanticLeadIdentityConfirmed?: boolean
  semanticLeadIdentityMethod?: string
  semanticLeadIdentityOrder?: "standard" | "cabrera"
  semanticLeadIdentityLayout?: string
  adaptivePreprocessingEligible?: boolean
  artifactPreprocessingEligible?: boolean
  maximumLayoutCost?: number
  cropBox?: RasterCropBox
  selectionEligible?: boolean
  inputVariant: DigitizerInputVariant
  vectorizer: DigitizerVectorizer
  device: DigitizerComputeDevice
  planningPhase?: CandidatePlanningPhase
  scheduleReason?: string
}

export type NativeGridLayoutConstraint =
  | "standard_6x2"
  | "standard_6x2_with_r1_ignored"
  | "standard_3x4"
  | "standard_3x4_with_r1"
  | "standard_12x1"
  | "row_local_compound_12lead"
  | "row_local_labeled_6x2"

export type CandidatePlanningContext = {
  adaptivePreprocessingEligible: boolean
  artifactPreprocessingEligible: boolean
  geometryCorrectionApplied: boolean
  screenArtifactLikely: boolean
  sourceMaxDimension: number
  lowResolutionNativeUpscaleTarget?: number
  plainThreeByFourLayout: boolean
  plainSixByTwoLayout: boolean
  constrainedInputVariant: DigitizerInputVariant
  nativeGridLayout?: NativeGridLayoutConstraint
  nativeGridInputVariant: DigitizerInputVariant
}

export function analyzeCandidatePlanningContext(
  preprocessing: PreprocessingReport,
  geometry: LayoutGeometryReport
): CandidatePlanningContext {
  const adaptivePreprocessingEligible =
    preprocessing.adaptivePreprocessing?.eligible === true
  const artifactPreprocessingEligible =
    preprocessing.artifactPreprocessing?.eligible === true
  const sourceMaxDimension = Math.max(
    preprocessing.source.width,
    preprocessing.source.height
  )
  const constrainedInputVariant: DigitizerInputVariant =
    geometry.detectedInputVariant === "geometry-corrected"
      ? "geometry-corrected"
      : preprocessing.annotationMask.maskedPixels > 0
      ? "annotation-masked"
      : "original"
  const nativeGridLayout = selectNativeGridLayout(geometry)
  const nativeGridRequiresOriginalRaster =
    nativeGridLayout === "row_local_compound_12lead" ||
    nativeGridLayout === "row_local_labeled_6x2" ||
    nativeGridLayout === "standard_12x1"
  return {
    adaptivePreprocessingEligible,
    artifactPreprocessingEligible,
    geometryCorrectionApplied:
      preprocessing.geometryCorrection?.applied === true,
    screenArtifactLikely:
      preprocessing.artifactPreprocessing?.screenArtifactLikely === true,
    sourceMaxDimension,
    ...(sourceMaxDimension <= DIGITIZER_POLICY.lowResolutionNativeMaxDimension
      ? { lowResolutionNativeUpscaleTarget: 1200 }
      : {}),
    plainThreeByFourLayout:
      geometry.layoutHint === "standard_3x4" &&
      geometry.semanticLeadEvidence?.passed === true &&
      (geometry.confidence ?? 0) >= DIGITIZER_POLICY.minimumGeometry3x4Confidence,
    plainSixByTwoLayout:
      geometry.layoutHint === "standard_6x2" &&
      (geometry.semanticLeadEvidence?.passed === true ||
        nativeGridLayout === "standard_6x2") &&
      (geometry.confidence ?? 0) >= DIGITIZER_POLICY.minimumGeometry6x2Confidence,
    constrainedInputVariant,
    ...(nativeGridLayout ? { nativeGridLayout } : {}),
    nativeGridInputVariant:
      geometry.detectedInputVariant === "geometry-corrected"
        ? "geometry-corrected"
        : nativeGridRequiresOriginalRaster
        ? "original"
        : constrainedInputVariant,
  }
}

export function buildPrimaryCandidatePlan({
  context,
  geometry,
  exhaustiveMode,
  device,
  annotationMasked,
}: {
  context: CandidatePlanningContext
  geometry: LayoutGeometryReport
  exhaustiveMode: boolean
  device: DigitizerComputeDevice
  annotationMasked: boolean
}): DigitizerCandidateConfig[] {
  const plan: DigitizerCandidateConfig[] = []
  const semanticLeadIdentity = semanticLeadIdentityParameters(geometry)
  const geometryCorrectedTwelveRow = Boolean(
    context.nativeGridLayout === "standard_12x1" &&
      context.geometryCorrectionApplied &&
      context.constrainedInputVariant === "geometry-corrected"
  )
  // A source-supported deterministic trace is still only one extraction path.
  // Always schedule a bounded neural core so morphology is independently
  // corroborated before publication; recovery candidates execute only when the
  // core does not agree.
  const shouldRunModel = true
  if (shouldRunModel) {
    plan.push({
      kind: "source-model",
      id: "default",
      label: "Default extraction",
      resampleSize: 1500,
      inputVariant: "original",
      vectorizer: "probability-centroid",
      device,
    })
  }
  if (
    context.plainThreeByFourLayout &&
    shouldRunModel
  ) {
    plan.push(
      {
        kind: "geometry-constrained",
        id: "geometry-standard-3x4-centroid-1500",
        label: "Label-confirmed 3 × 4 centroid extraction",
        resampleSize: 1500,
        layoutConstraint: "standard_3x4",
        geometryConfirmedLayout: true,
        geometryLayoutConfidence: geometry.confidence,
        ...semanticLeadIdentity,
        inputVariant: context.constrainedInputVariant,
        vectorizer: "probability-centroid",
        device,
      },
      {
        kind: "geometry-constrained",
        id: "geometry-standard-3x4-path-2200",
        label: "Label-confirmed 3 × 4 source-ink extraction",
        resampleSize: 2200,
        upscaleToMaxDimension: 2200,
        darkInkEnhancement: true,
        layoutConstraint: "standard_3x4",
        geometryConfirmedLayout: true,
        geometryLayoutConfidence: geometry.confidence,
        ...semanticLeadIdentity,
        inputVariant: context.constrainedInputVariant,
        vectorizer: "dynamic-path",
        device,
        planningPhase: "core",
        scheduleReason: "3 × 4 label-grid corroboration",
      }
    )
  }
  if (context.plainSixByTwoLayout && shouldRunModel) {
    plan.push(
      {
        kind: "geometry-constrained",
        id: "geometry-standard-6x2-centroid-1500",
        label: "Geometry-confirmed 6 × 2 centroid extraction",
        resampleSize: 1500,
        upscaleToMaxDimension: 1500,
        layoutConstraint: "standard_6x2",
        geometryConfirmedLayout: true,
        geometryLayoutConfidence: geometry.confidence,
        ...semanticLeadIdentity,
        inputVariant: context.constrainedInputVariant,
        vectorizer: "probability-centroid",
        device,
      },
      {
        kind: "geometry-constrained",
        id: "geometry-standard-6x2-path-2200",
        label: "Geometry-confirmed 6 × 2 source-ink extraction",
        resampleSize: 2200,
        upscaleToMaxDimension: 2200,
        darkInkEnhancement: true,
        layoutConstraint: "standard_6x2",
        geometryConfirmedLayout: true,
        geometryLayoutConfidence: geometry.confidence,
        ...semanticLeadIdentity,
        inputVariant: context.constrainedInputVariant,
        vectorizer: "dynamic-path",
        device,
        planningPhase: "core",
        scheduleReason: "6 × 2 constrained path corroboration",
      }
    )
  }
  if (
    context.nativeGridLayout === "standard_6x2_with_r1_ignored" &&
    shouldRunModel
  ) {
    plan.push({
      kind: "geometry-constrained",
      id: "geometry-standard-6x2-rhythm-centroid-2200",
      label: "Geometry-confirmed 6 x 2 + rhythm centroid extraction",
      resampleSize: 2200,
      upscaleToMaxDimension: 2200,
      layoutConstraint: "standard_6x2_with_r1_ignored",
      geometryConfirmedLayout: true,
      geometryLayoutConfidence: geometry.confidence,
      ...semanticLeadIdentity,
      inputVariant: context.constrainedInputVariant,
      vectorizer: "probability-centroid",
      device,
      planningPhase: "core",
      scheduleReason: "6 x 2 + rhythm corrected-layout corroboration",
    })
  }
  if (context.nativeGridLayout === "standard_12x1" && shouldRunModel) {
    plan.push(
      {
        kind: "geometry-constrained",
        id: "geometry-standard-12x1-centroid-2200",
        label: "Label-anchored 12 × 1 centroid extraction",
        resampleSize: 2200,
        upscaleToMaxDimension: 2200,
        layoutConstraint: "standard_12x1",
        geometryConfirmedLayout: true,
        geometryLayoutConfidence: geometry.confidence,
        ...semanticLeadIdentity,
        inputVariant: context.constrainedInputVariant,
        vectorizer: "probability-centroid",
        device,
      },
      {
        kind: "geometry-constrained",
        id: "geometry-standard-12x1-path-2200",
        label: "Label-anchored 12 × 1 source-ink extraction",
        resampleSize: 2200,
        upscaleToMaxDimension: 2200,
        darkInkEnhancement: true,
        layoutConstraint: "standard_12x1",
        geometryConfirmedLayout: true,
        geometryLayoutConfidence: geometry.confidence,
        ...semanticLeadIdentity,
        inputVariant: context.constrainedInputVariant,
        vectorizer: "dynamic-path",
        device,
        planningPhase: "core",
        scheduleReason: "12 × 1 constrained path corroboration",
      }
    )
  }
  if (shouldRunModel && (exhaustiveMode || annotationMasked)) {
    plan.push({
      kind: "source-model",
      id: "annotation-masked",
      label: "Annotation-suppressed extraction",
      resampleSize: 1500,
      inputVariant: "annotation-masked",
      vectorizer: "probability-centroid",
      device,
    })
  }
  if (shouldRunModel && !geometryCorrectedTwelveRow) {
    plan.push({
      kind: "source-model",
      id: "ink-path-2200",
      label: "Source-ink fidelity extraction",
      resampleSize: 2200,
      upscaleToMaxDimension: 2200,
      darkInkEnhancement: true,
      inputVariant: annotationMasked ? "annotation-masked" : "original",
      vectorizer: "dynamic-path",
      device,
      planningPhase: "core",
      scheduleReason: "source-ink morphology corroboration",
    })
  }
  if (
    (exhaustiveMode ||
      context.adaptivePreprocessingEligible ||
      context.geometryCorrectionApplied) &&
    !geometryCorrectedTwelveRow
  ) {
    const geometryRecoveryLayout =
      context.geometryCorrectionApplied &&
      (context.nativeGridLayout === "standard_3x4" ||
        context.nativeGridLayout === "standard_6x2" ||
        context.nativeGridLayout === "standard_6x2_with_r1_ignored")
        ? context.nativeGridLayout
        : undefined
    plan.push({
      kind: "adaptive-preprocessed",
      id: "preprocessed-path-2200",
      label: "Adaptive colour/grid path extraction",
      resampleSize: 2200,
      ...(context.plainThreeByFourLayout || geometryRecoveryLayout
        ? {
            layoutConstraint:
              geometryRecoveryLayout ?? "standard_3x4",
            geometryConfirmedLayout: true,
            geometryLayoutConfidence: geometry.confidence,
            ...(!context.plainThreeByFourLayout && geometryRecoveryLayout
              ? { upscaleToMaxDimension: 2200 }
              : {}),
          }
        : { upscaleToMaxDimension: 2200 }),
      adaptivePreprocessingEligible: context.adaptivePreprocessingEligible,
      inputVariant: "preprocessed",
      vectorizer: "dynamic-path",
      device,
    })
  }
  if (context.artifactPreprocessingEligible) {
    const specialistLabel = context.screenArtifactLikely
      ? "Screen/moire corrected"
      : context.geometryCorrectionApplied
        ? "Perspective-corrected"
        : "Glare/background corrected"
    plan.push({
      kind: "adaptive-preprocessed",
      id: "artifact-preprocessed-centroid-2200",
      label: `${specialistLabel} centroid extraction`,
      resampleSize: 2200,
      upscaleToMaxDimension: 2200,
      artifactPreprocessingEligible: true,
      inputVariant: "artifact-preprocessed",
      vectorizer: "probability-centroid",
      device,
      planningPhase: "core",
      scheduleReason: context.screenArtifactLikely
        ? "screen/moire corrected core corroboration"
        : "geometry/background corrected core corroboration",
    })
    if (!geometryCorrectedTwelveRow) {
      plan.push({
        kind: "adaptive-preprocessed",
        id: "artifact-preprocessed-path-2200",
        label: `${specialistLabel} source-ink extraction`,
        resampleSize: 2200,
        upscaleToMaxDimension: 2200,
        darkInkEnhancement: true,
        artifactPreprocessingEligible: true,
        inputVariant: "artifact-preprocessed",
        vectorizer: "dynamic-path",
        device,
        planningPhase: "recovery",
        scheduleReason: context.screenArtifactLikely
          ? "screen/moire corrected path recovery"
          : "geometry/background corrected path recovery",
      })
    }
  }
  if (
    (exhaustiveMode ||
      context.sourceMaxDimension >=
        DIGITIZER_POLICY.minimumNativeModelInputDimension) &&
    !geometryCorrectedTwelveRow
  ) {
    plan.push(
      {
        kind: "source-model",
        id: "morphology-2200",
        label: "High-resolution path extraction",
        resampleSize: 2200,
        inputVariant: annotationMasked ? "annotation-masked" : "original",
        vectorizer: "dynamic-path",
        device,
      },
      {
        kind: "source-model",
        id: "centroid-2200",
        label: "High-resolution centroid extraction",
        resampleSize: 2200,
        inputVariant: annotationMasked ? "annotation-masked" : "original",
        vectorizer: "probability-centroid",
        device,
      }
    )
  }
  if (
    exhaustiveMode ||
    (context.sourceMaxDimension < MODEL_SCALE_CENTROID_DIMENSION &&
      !geometryCorrectedTwelveRow)
  ) {
    plan.push({
      kind: "source-model",
      id: "upscaled-centroid-2200",
      label: "Model-scale centroid extraction",
      resampleSize: MODEL_SCALE_CENTROID_DIMENSION,
      upscaleToMaxDimension: MODEL_SCALE_CENTROID_DIMENSION,
      inputVariant: annotationMasked ? "annotation-masked" : "original",
      vectorizer: "probability-centroid",
      device,
    })
  }
  if (exhaustiveMode) {
    plan.push(
      {
        kind: "source-model",
        id: "path-1500",
        label: "Ablation: standard-resolution path extraction",
        resampleSize: 1500,
        inputVariant: annotationMasked ? "annotation-masked" : "original",
        vectorizer: "dynamic-path",
        device,
      },
      {
        kind: "source-model",
        id: "label-thresh-002",
        label: "Ablation: 0.02 signal threshold",
        resampleSize: 1500,
        labelThresh: 0.02,
        inputVariant: "original",
        vectorizer: "probability-centroid",
        device,
      },
      {
        kind: "source-model",
        id: "label-thresh-001",
        label: "Ablation: 0.01 signal threshold",
        resampleSize: 1500,
        labelThresh: 0.01,
        inputVariant: "original",
        vectorizer: "probability-centroid",
        device,
      }
    )
  }
  if (!semanticLeadIdentity.semanticLeadIdentityConfirmed) return plan
  // The recognition proof belongs to the full source page, not to one
  // vectorizer. Propagate it to every full-page candidate; publication later
  // requires the candidate's detected layout to exactly match this recorded
  // proof layout before treating the lead identities as explicit.
  return plan.map((candidate) => ({
    ...candidate,
    ...semanticLeadIdentity,
  }))
}

export function semanticLeadIdentityParameters(
  geometry: LayoutGeometryReport
): Pick<
  DigitizerCandidateConfig,
  | "semanticLeadIdentityConfirmed"
  | "semanticLeadIdentityMethod"
  | "semanticLeadIdentityOrder"
  | "semanticLeadIdentityLayout"
> {
  if (geometry.semanticLeadEvidence?.passed !== true) return {}
  const recognition = [
    geometry.leadLabelValidation?.semanticRecognition,
    geometry.limbLabelValidation?.semanticRecognition,
    geometry.precordialLabelValidation?.semanticRecognition,
  ].find(
    (report) =>
      report?.passed === true &&
      report.semanticIdentityConfirmed === true &&
      (report.order === "standard" || report.order === "cabrera")
  )
  if (!recognition) return {}
  return {
    semanticLeadIdentityConfirmed: true,
    semanticLeadIdentityMethod: recognition.method,
    semanticLeadIdentityOrder: recognition.order as "standard" | "cabrera",
    semanticLeadIdentityLayout: geometry.layoutHint ?? undefined,
  }
}

export type CandidatePlanningPhase = "core" | "recovery" | "exhaustive"

export function candidatePlanningPhase(
  candidate: DigitizerCandidateConfig
): CandidatePlanningPhase {
  if (candidate.planningPhase) return candidate.planningPhase
  if (
    candidate.id === "path-1500" ||
    candidate.id === "label-thresh-002" ||
    candidate.id === "label-thresh-001"
  ) {
    return "exhaustive"
  }
  if (
    candidate.id.includes("path-2200") ||
    candidate.id === "morphology-2200" ||
    candidate.inputVariant === "preprocessed" ||
    candidate.inputVariant === "artifact-preprocessed"
  ) {
    return "recovery"
  }
  return "core"
}

export function candidateScheduleReason(
  candidate: DigitizerCandidateConfig,
  context: CandidatePlanningContext
) {
  if (candidate.scheduleReason) return candidate.scheduleReason
  if (candidatePlanningPhase(candidate) === "exhaustive") {
    return "exhaustive candidate search"
  }
  if (candidate.inputVariant === "artifact-preprocessed") {
    return context.screenArtifactLikely
      ? "screen/moire artifact specialist"
      : context.geometryCorrectionApplied
        ? "page geometry correction specialist"
        : "glare/background specialist"
  }
  if (candidate.layoutConstraint === "standard_12x1") {
    return "12 × 1 label-anchored specialist"
  }
  if (candidate.layoutConstraint?.includes("6x2")) {
    return "6 × 2 geometry specialist"
  }
  if (candidate.inputVariant === "annotation-masked") {
    return "annotation-safe corroboration"
  }
  if (candidatePlanningPhase(candidate) === "core") {
    return "bounded independent morphology corroboration"
  }
  return "expanded recovery after insufficient core agreement"
}

function selectNativeGridLayout(
  geometry: LayoutGeometryReport
): NativeGridLayoutConstraint | undefined {
  if (
    geometry.layoutHint === "standard_3x4" &&
    geometry.semanticLeadEvidence?.passed === true &&
    geometry.leadLabelValidation?.passed === true &&
    geometry.leadLabelValidation.order === "standard" &&
    (geometry.confidence ?? 0) >= DIGITIZER_POLICY.minimumGeometry3x4Confidence
  ) {
    return "standard_3x4"
  }
  if (
    geometry.layoutHint === "standard_3x4" &&
    geometry.rowCenters?.length === 3 &&
    (geometry.confidence ?? 0) >= DIGITIZER_POLICY.minimumGeometry3x4Confidence
  ) {
    // Geometry alone may schedule a deterministic diagnostic extraction, but
    // it cannot establish the printed lead values or order. Quantitative
    // publication remains blocked by the separate semantic-identity gate.
    return "standard_3x4"
  }
  if (
    geometry.compoundPanels?.method ===
      "separate-grid-panels-plus-label-geometry-v1" &&
    geometry.compoundPanels.first.rowCenters?.length === 6 &&
    geometry.compoundPanels.second.rowCenters?.length === 6
  ) {
    return "row_local_compound_12lead"
  }
  if (
    geometry.layoutHint === "standard_6x2" &&
    geometry.method === "paired-standard-limb-and-v-label-geometry-v1" &&
    geometry.semanticLeadEvidence?.passed === true
  ) {
    return "row_local_labeled_6x2"
  }
  if (
    geometry.layoutHint === "standard_6x2" &&
    geometry.method === "paired-standard-limb-and-v-label-geometry-v1" &&
    geometry.precordialLabelValidation?.passed === true &&
    geometry.rowCenters?.length === 6 &&
    (geometry.confidence ?? 0) >= 0.8
  ) {
    // A calibration pulse can merge with the first limb label and make the
    // pre-extraction Roman-label width sequence inconclusive.  Scheduling the
    // native extractor is still safe here: it must independently validate the
    // standard limb order from simultaneous lead algebra, re-confirm V1-V6,
    // and pass source-pixel fidelity before it can establish semantic identity
    // or corroborate any quantitative output.
    return "standard_6x2"
  }
  if (
    geometry.layoutHint === "standard_6x2" &&
    geometry.rowCenters?.length === 6 &&
    (geometry.confidence ?? 0) >=
      DIGITIZER_POLICY.minimumReviewLayoutConfidence
  ) {
    // Schedule a provisional standard-layout extraction when the six trace
    // rows are clear but small labels are incomplete. Quantitative retention
    // later remains limited to conservative source-backed traces in all 12
    // positions and carries a mandatory manual lead-label warning.
    return "standard_6x2"
  }
  if (
    geometry.layoutHint === "standard_6x2_with_r1_ignored" &&
    geometry.precordialLabelValidation?.passed === true &&
    geometry.rowCenters?.length === 7 &&
    (geometry.confidence ?? 0) >= 0.8
  ) {
    // A confidently detected seventh full-width row establishes the rhythm
    // layout, while V1-V6 labels establish the right-panel order. The native
    // candidate must still validate standard limb order from simultaneous
    // lead algebra and pass source-pixel fidelity before it can corroborate
    // any quantitative output.
    return "standard_6x2_with_r1_ignored"
  }
  if (
    geometry.semanticLeadEvidence?.passed === true &&
    (geometry.layoutHint === "standard_6x2_with_r1_ignored" ||
      geometry.layoutHint === "standard_3x4_with_r1" ||
      geometry.layoutHint === "standard_6x2" ||
      geometry.layoutHint === "standard_12x1")
  ) {
    return geometry.layoutHint
  }
  if (
    geometry.layoutHint === "standard_12x1" &&
    geometry.rowCenters?.length === 12 &&
    (geometry.confidence ?? 0) >=
      DIGITIZER_POLICY.minimumReviewLayoutConfidence
  ) {
    // A clean twelve-row geometry can support provisional row-to-lead
    // assignment even when small printed labels defeat OCR. Publication still
    // requires a conservative source-backed trace in every row and records
    // that lead identities must be checked manually against the source.
    return "standard_12x1"
  }
  if (
    geometry.compoundPanels &&
    geometry.semanticLeadEvidence?.passed === true &&
    geometry.compoundPanels.confidence >= 0.08 &&
    geometry.compoundPanels.first.rowCenters?.length === 6 &&
    geometry.compoundPanels.second.rowCenters?.length === 6
  ) {
    return "row_local_compound_12lead"
  }
  return undefined
}
