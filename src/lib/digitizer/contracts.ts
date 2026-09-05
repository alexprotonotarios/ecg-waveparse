import type {
  AnnotationComponent,
  DigitizerSourceFidelitySummary,
} from "@/lib/runs"

export type RasterCropBox = {
  left: number
  top: number
  right: number
  bottom: number
}

export type PreprocessingReport = {
  version: number
  sourceSha256: string
  source: { width: number; height: number; format?: string }
  workingImage?: {
    width: number
    height: number
    scaleX: number
    scaleY: number
    method: string
    maxLongEdgePixels: number
  }
  annotationMask: {
    maskedPixels: number
    maskedFraction: number
    components: AnnotationComponent[]
  }
  preparedImage: {
    method: string
    maskDilationPixels: number
    fillRingPixels?: number
    fillPercentile?: number
    medianWindowPixels?: number
    morphologyReconstructed: false
  }
  evidenceMaps?: {
    method: string
    backgroundKernelPixels: number
    backgroundKernelShortEdgeFraction: number
    gridProfileQuantile: number
    traceProbabilityP95: number
    gridProbabilityP95: number
    excludedPixels: number
    morphologyReconstructed: false
  }
  adaptivePreprocessing?: {
    method: string
    eligible: boolean
    reasons: string[]
    longEdgePixels: number
    sharpnessLaplacianVariance: number
    minimumSharpnessLaplacianVariance: number
    lowResolution: boolean
    lowResolutionLongEdgeThresholdPixels: number
    foregroundGridDegraded: boolean
    foregroundContaminationTraceP95Threshold: number
    weakGridProbabilityP95Threshold: number
    belowSharpnessFloor?: boolean
    traceDominantSmoothSource?: boolean
    traceDominantSmoothSourceP95Threshold?: number
    traceDominantSmoothSourceMaxGridP95?: number
    blurRejected: boolean
  }
  geometryCorrection?: {
    applied: boolean
    method: string
    confidence: number
    perspectiveStrength: number
    rotationDegrees: number
    pageAreaFraction?: number
    pageBoundaryCrop?: boolean
    sourceCorners?: number[][] | null
    outputWidth: number
    outputHeight: number
    transform: number[][]
    morphologyReconstructed: false
  }
  artifactPreprocessing?: {
    method: string
    eligible: boolean
    screenArtifactLikely: boolean
    chromaMoireScore: number
    chromaMoireThreshold: number
    chromaMoireCoverage: number
    chromaMoireMinimumCoverage: number
    glareLikely: boolean
    glareFraction: number
    glareFractionThreshold: number
    reasons: string[]
    backgroundMoireSuppressed: boolean
    geometryCorrected: boolean
    traceProbabilityP95: number
    gridProbabilityP95: number
    morphologyReconstructed: false
  }
  inputQuality?: {
    method: string
    outcome: "acceptable" | "review" | "insufficient"
    quantitativeEligible: boolean
    reasons: string[]
    originalLongEdgePixels: number
    originalShortEdgePixels: number
    minimumQuantitativeLongEdgePixels: number
    minimumQuantitativeShortEdgePixels: number
    workingScale: number
  }
  execution?: {
    attempts: Array<{
      maxWorkingLongEdgePixels: number
      runtimeMs: number
      succeeded: boolean
      error?: string
    }>
    preprocessingPermits: number
    sourceMegapixels: number
  }
}

export type PreparedRunInput = {
  sourcePath: string
  workingSourcePath: string
  preparedPath: string
  enhancedPath: string
  backgroundPath: string
  traceProbabilityPath: string
  gridProbabilityPath: string
  exclusionMaskPath: string
  geometryCorrectedPath: string
  artifactPreprocessedPath: string
  maskPath: string
  reportPath: string
  report: PreprocessingReport
}

export type SemanticLeadRecognitionReport = {
  passed: boolean
  order?: string | null
  confidence: number
  method: string
  semanticIdentityConfirmed: boolean
  expectedLabels?: string[]
  recognizedLabels?: string[]
  failureReasons: string[]
  engine?: {
    name?: string | null
    revision?: number | null
    platform?: string | null
    platformVersion?: string | null
  }
}

type SemanticLabelValidation = {
  passed?: boolean
  order?: string | null
  confidence?: number
  method?: string
  semanticIdentityConfirmed?: boolean
  semanticRecognition?: SemanticLeadRecognitionReport
}

export type LayoutGeometryReport = {
  detectedInputVariant?:
    | "original"
    | "annotation-masked"
    | "geometry-corrected"
  coordinateSpace?: "working" | "geometry-corrected"
  detectionFailure?: { method: string; message: string }
  layoutHint?: string | null
  confidence?: number
  method?: string | null
  leadLabelValidation?: SemanticLabelValidation
  precordialLabelValidation?: SemanticLabelValidation
  limbLabelValidation?: SemanticLabelValidation & {
    labelLeftEdges?: number[]
  }
  rhythmLeadValidation?: SemanticLabelValidation & {
    lead?: string | null
    expectedLabel?: string
    recognizedLabels?: string[]
    failureReasons?: string[]
  }
  semanticLeadEvidence?: {
    passed: boolean
    confidence: number
    method: string
    visibleLeads: string[]
    reasons: string[]
  }
  rowCenters?: number[]
  contentBox?: RasterCropBox | null
  contentConfidence?: number
  contentRetainedFraction?: number
  compoundPanels?: {
    orientation: "side_by_side" | "stacked"
    method?: string
    first: RasterCropBox & {
      rowCenters?: number[]
      rowMethod?: string | null
      leadOrderHint?: {
        passed?: boolean
        order?: string | null
        confidence?: number
        method?: string
        labelLeftEdges?: number[]
        semanticIdentityConfirmed?: boolean
        semanticRecognition?: SemanticLeadRecognitionReport
      }
    }
    second: RasterCropBox & {
      rowCenters?: number[]
      rowMethod?: string | null
      leadOrderHint?: {
        passed?: boolean
        order?: string | null
        confidence?: number
        method?: string
        labelLeftEdges?: number[]
        semanticIdentityConfirmed?: boolean
        semanticRecognition?: SemanticLeadRecognitionReport
      }
    }
    confidence: number
  } | null
  calibration?: {
    method: string
    detected: boolean
    gridScaleDetected?: boolean
    gridScaleAmbiguous?: boolean
    gridScaleConfidence?: number
    gridScaleMm?: number
    gridScaleMmX?: number
    gridScaleMmY?: number
    gridSpacingXPixels?: number
    gridSpacingYPixels?: number
    paperSpeedMmPerSecond?: number
    gainMmPerMv?: number
    pixelsPerMmX?: number
    pixelsPerMmY?: number
    rowPixelsPerMmX?: number[]
    pulseStartX?: number
    pulseEndX?: number
    confidence: number
  }
}

export type NativeExtractionReport = {
  layout: string
  layoutCost: number
  effectiveSampleRateHz: number
  sourceFidelity: DigitizerSourceFidelitySummary
}

function object(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be a JSON object.`)
  }
  return value as Record<string, unknown>
}

function finite(value: unknown) {
  return typeof value === "number" && Number.isFinite(value)
}

function positiveInteger(value: unknown) {
  return finite(value) && Number.isInteger(value) && Number(value) > 0
}

function finiteArray(value: unknown) {
  return Array.isArray(value) && value.every(finite)
}

function finiteMatrix3x3(value: unknown) {
  return (
    Array.isArray(value) &&
    value.length === 3 &&
    value.every((row) => Array.isArray(row) && row.length === 3 && row.every(finite))
  )
}

export function parseJsonObject(text: string, label: string) {
  let value: unknown
  try {
    value = JSON.parse(text)
  } catch {
    throw new Error(`${label} returned malformed JSON.`)
  }
  return object(value, label)
}

export function parsePreprocessingReport(text: string): PreprocessingReport {
  const value = parseJsonObject(text, "Input preparation")
  const source = object(value.source, "Input preparation source")
  const annotationMask = object(
    value.annotationMask,
    "Input preparation annotation mask"
  )
  const evidenceMaps = object(
    value.evidenceMaps,
    "Input preparation evidence maps"
  )
  const adaptive = object(
    value.adaptivePreprocessing,
    "Input preparation adaptive classification"
  )
  if (
    (value.version !== 5 && value.version !== 6 && value.version !== 7) ||
    typeof value.sourceSha256 !== "string" ||
    value.sourceSha256.length !== 64 ||
    !positiveInteger(source.width) ||
    !positiveInteger(source.height) ||
    !Array.isArray(annotationMask.components) ||
    typeof evidenceMaps.method !== "string" ||
    !finite(evidenceMaps.backgroundKernelPixels) ||
    typeof adaptive.method !== "string" ||
    typeof adaptive.eligible !== "boolean" ||
    !Array.isArray(adaptive.reasons) ||
    !adaptive.reasons.every((reason) => typeof reason === "string")
  ) {
    throw new Error("Input preparation returned invalid provenance metadata.")
  }
  if (value.version === 7) {
    const workingImage = object(
      value.workingImage,
      "Input preparation working image"
    )
    const inputQuality = object(
      value.inputQuality,
      "Input preparation quality gate"
    )
    const qualityOutcome = String(inputQuality.outcome)
    const scaleX = Number(workingImage.scaleX)
    const scaleY = Number(workingImage.scaleY)
    const expectedScaleX = Number(workingImage.width) / Number(source.width)
    const expectedScaleY = Number(workingImage.height) / Number(source.height)
    const sourceLongEdge = Math.max(Number(source.width), Number(source.height))
    const sourceShortEdge = Math.min(Number(source.width), Number(source.height))
    if (
      !positiveInteger(workingImage.width) ||
      !positiveInteger(workingImage.height) ||
      !finite(workingImage.scaleX) ||
      !finite(workingImage.scaleY) ||
      scaleX <= 0 ||
      scaleY <= 0 ||
      Math.abs(scaleX - expectedScaleX) > 1e-6 ||
      Math.abs(scaleY - expectedScaleY) > 1e-6 ||
      typeof workingImage.method !== "string" ||
      !positiveInteger(workingImage.maxLongEdgePixels) ||
      Math.max(Number(workingImage.width), Number(workingImage.height)) >
        Number(workingImage.maxLongEdgePixels) ||
      !["acceptable", "review", "insufficient"].includes(
        qualityOutcome
      ) ||
      typeof inputQuality.quantitativeEligible !== "boolean" ||
      (qualityOutcome === "insufficient") ===
        Boolean(inputQuality.quantitativeEligible) ||
      !finite(inputQuality.originalLongEdgePixels) ||
      !finite(inputQuality.originalShortEdgePixels) ||
      Number(inputQuality.originalLongEdgePixels) !== sourceLongEdge ||
      Number(inputQuality.originalShortEdgePixels) !== sourceShortEdge ||
      !finite(inputQuality.minimumQuantitativeLongEdgePixels) ||
      !finite(inputQuality.minimumQuantitativeShortEdgePixels) ||
      !finite(inputQuality.workingScale) ||
      Number(inputQuality.workingScale) <= 0 ||
      Math.abs(Number(inputQuality.workingScale) - Math.min(scaleX, scaleY)) >
        1e-6 ||
      !Array.isArray(inputQuality.reasons) ||
      !inputQuality.reasons.every((reason) => typeof reason === "string")
    ) {
      throw new Error("Input preparation returned invalid quality metadata.")
    }
  }
  if (value.geometryCorrection !== undefined) {
    const geometry = object(
      value.geometryCorrection,
      "Input preparation geometry correction"
    )
    if (
      typeof geometry.applied !== "boolean" ||
      typeof geometry.method !== "string" ||
      !finite(geometry.confidence) ||
      !positiveInteger(geometry.outputWidth) ||
      !positiveInteger(geometry.outputHeight) ||
      !finiteMatrix3x3(geometry.transform)
    ) {
      throw new Error("Input preparation returned invalid geometry metadata.")
    }
  }
  if (value.artifactPreprocessing !== undefined) {
    const artifact = object(
      value.artifactPreprocessing,
      "Input preparation artifact classification"
    )
    if (
      typeof artifact.eligible !== "boolean" ||
      typeof artifact.screenArtifactLikely !== "boolean" ||
      typeof artifact.glareLikely !== "boolean" ||
      !Array.isArray(artifact.reasons)
    ) {
      throw new Error("Input preparation returned invalid artifact metadata.")
    }
  }
  return value as unknown as PreprocessingReport
}

function validateCropBox(value: unknown, label: string) {
  const crop = object(value, label)
  if (
    ![crop.left, crop.top, crop.right, crop.bottom].every(finite) ||
    Number(crop.left) < 0 ||
    Number(crop.top) < 0 ||
    Number(crop.right) <= Number(crop.left) ||
    Number(crop.bottom) <= Number(crop.top)
  ) {
    throw new Error(`${label} contains invalid coordinates.`)
  }
}

function stringArray(value: unknown) {
  return Array.isArray(value) && value.every((item) => typeof item === "string")
}

function validateSemanticRecognition(value: unknown, label: string) {
  const report = object(value, label)
  const expectedLabels = report.expectedLabels as string[] | undefined
  const recognizedLabels = report.recognizedLabels as string[] | undefined
  if (
    typeof report.passed !== "boolean" ||
    typeof report.semanticIdentityConfirmed !== "boolean" ||
    report.passed !== report.semanticIdentityConfirmed ||
    typeof report.method !== "string" ||
    report.method.length === 0 ||
    !finite(report.confidence) ||
    Number(report.confidence) < 0 ||
    Number(report.confidence) > 1 ||
    !stringArray(report.failureReasons) ||
    (report.expectedLabels !== undefined &&
      !stringArray(report.expectedLabels)) ||
    (report.recognizedLabels !== undefined &&
      !stringArray(report.recognizedLabels))
  ) {
    throw new Error("Layout detector returned invalid semantic-recognition metadata.")
  }
  if (
    report.passed === true &&
    (report.order !== "standard" ||
      Number(report.confidence) <= 0 ||
      !Array.isArray(expectedLabels) ||
      !Array.isArray(recognizedLabels) ||
      expectedLabels.length !== recognizedLabels.length ||
      expectedLabels.some(
        (lead, index) => lead !== recognizedLabels[index]
      ) ||
      (report.failureReasons as unknown[]).length !== 0)
  ) {
    throw new Error("Layout detector returned incomplete semantic-recognition proof.")
  }
}

function validateLabelRecognition(value: unknown, label: string) {
  if (value === undefined) return
  const validation = object(value, label)
  if (validation.semanticRecognition !== undefined) {
    validateSemanticRecognition(
      validation.semanticRecognition,
      `${label} semantic recognition`
    )
  }
  if (
    validation.semanticIdentityConfirmed === true &&
    validation.semanticRecognition === undefined
  ) {
    throw new Error("Layout detector returned an unsubstantiated semantic-identity flag.")
  }
}

export function parseLayoutGeometryReport(text: string): LayoutGeometryReport {
  const value = parseJsonObject(text, "Layout detector")
  if (value.rowCenters !== undefined && !finiteArray(value.rowCenters)) {
    throw new Error("Layout detector returned invalid row centers.")
  }
  if (value.confidence !== undefined && !finite(value.confidence)) {
    throw new Error("Layout detector returned invalid confidence.")
  }
  validateLabelRecognition(
    value.leadLabelValidation,
    "Lead-label validation"
  )
  validateLabelRecognition(
    value.precordialLabelValidation,
    "Precordial-label validation"
  )
  validateLabelRecognition(
    value.limbLabelValidation,
    "Limb-label validation"
  )
  validateLabelRecognition(
    value.rhythmLeadValidation,
    "Rhythm-lead validation"
  )
  if (value.contentBox) validateCropBox(value.contentBox, "Layout content box")
  if (value.compoundPanels) {
    const panels = object(value.compoundPanels, "Compound panels")
    if (
      !["side_by_side", "stacked"].includes(String(panels.orientation)) ||
      !finite(panels.confidence)
    ) {
      throw new Error("Layout detector returned invalid compound-panel metadata.")
    }
    validateCropBox(panels.first, "First compound panel")
    validateCropBox(panels.second, "Second compound panel")
  }
  return value as unknown as LayoutGeometryReport
}

export function parseNativeExtractionReport(
  text: string,
  expectedLayout: string
): NativeExtractionReport {
  const value = parseJsonObject(text, "Native grid extraction")
  const fidelity = object(value.sourceFidelity, "Native source fidelity")
  if (
    value.layout !== expectedLayout ||
    !finite(value.layoutCost) ||
    !finite(value.effectiveSampleRateHz) ||
    typeof fidelity.passed !== "boolean" ||
    typeof fidelity.method !== "string" ||
    !fidelity.leadMetrics ||
    typeof fidelity.leadMetrics !== "object"
  ) {
    throw new Error("Native grid extraction returned invalid fidelity metadata.")
  }
  return value as unknown as NativeExtractionReport
}
