import type {
  AnnotationComponent,
  DigitizerSourceFidelitySummary,
} from "@/lib/runs"
import { ECG_LEADS } from "@/lib/digitizer/domain"
import { validatePrintedCalibration, type PrintedSettingsReport, type CalibrationReconciliation, type PrintedSpeedResolution } from "@/lib/digitizer/calibration-settings"
import { EvidenceContractError, invertibleTransform, nonnegativePhysicalValue, positivePhysicalValue, unitFraction, type EvidenceState } from "@/lib/digitizer/physical-contracts"
import { validateSourceGridEvidence, type SourceGridIdentity } from "./source-grid-contract"

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

export type SourceLabelSlot = { lead: string; row: number; column: number; box: number[] }
export type SourceLabelGrid = {
  version: 1
  method: "source-value-grid-below-trace-v1"
  imageSize: number[]
  rowSpacing: number
  columnSpacing: number
  fontHeight: number
  labelRowCenters: number[]
  xFit: number[]
  yFit: number[]
  maximumXResidual: number
  maximumYResidual: number
  anchors: Array<{ lead: string; confidence: number; x: number; y: number; width: number; height: number }>
  slots: SourceLabelSlot[]
  rhythmSlot: SourceLabelSlot
  maskMethod: "maximum-channel-below-160-v1"
  localRecognition?: LocalLabelRecognition
  tiledRecognition?: TiledLabelRecognition
}

export type LayoutGeometryReport = {
  version?: 1
  image?: { width: number; height: number }
  evidenceStatus?: { calibration: EvidenceState; coordinateBounds: EvidenceState }
  detectedInputVariant?:
    | "original"
    | "annotation-masked"
    | "geometry-corrected"
  coordinateSpace?: "working" | "geometry-corrected"
  detectionFailure?: { method: string; message: string }
  layoutHint?: string | null
  confidence?: number
  method?: string | null
  sourceLabelGrid?: SourceLabelGrid
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
    printedSettings?: PrintedSettingsReport
    reconciliation?: CalibrationReconciliation
    speedResolution?: PrintedSpeedResolution
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
    measuredGainMmPerMv?: number
    pulseWidthMm?: number
    pixelsPerMmX?: number
    pixelsPerMmY?: number
    rowPixelsPerMmX?: number[]
    horizontalScaleEvidence?: {
      version: 1; method: "consistent-row-grid-median-v1"; priorPixelsPerMmX: number
      rowCount: number; relativeRange: number
    }
    pulseStartX?: number
    pulseEndX?: number
    confidence: number
  }
}

export type NativeExtractionReport = {
  version?: 1
  layout: string
  layoutCost: number
  effectiveSampleRateHz: number
  sourceFidelity: DigitizerSourceFidelitySummary
}

function object(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new EvidenceContractError("invalid_object", `${label} must be a JSON object.`)
  }
  return value as Record<string, unknown>
}

function finite(value: unknown) {
  return typeof value === "number" && Number.isFinite(value)
}

function positiveInteger(value: unknown) {
  return finite(value) && Number.isInteger(value) && Number(value) > 0
}

function finiteArray(value: unknown): value is number[] {
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
    throw new EvidenceContractError("malformed_json", `${label} returned malformed JSON.`)
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
    !/^[a-f0-9]{64}$/.test(value.sourceSha256) ||
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
    throw new EvidenceContractError("invalid_preprocessing_provenance", "Input preparation returned invalid provenance metadata.")
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
      throw new EvidenceContractError("invalid_input_quality", "Input preparation returned invalid quality metadata.")
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
      !unitFraction(geometry.confidence) ||
      !positiveInteger(geometry.outputWidth) ||
      !positiveInteger(geometry.outputHeight) ||
      !finiteMatrix3x3(geometry.transform) ||
      !invertibleTransform(geometry.transform as number[][]) ||
      geometry.morphologyReconstructed !== false
    ) {
      throw new EvidenceContractError("invalid_transform", "Input preparation returned invalid geometry metadata.")
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
      throw new EvidenceContractError("invalid_artifact_metadata", "Input preparation returned invalid artifact metadata.")
    }
  }
  // v7 deliberately reports annotation components/counts in ORIGINAL source
  // coordinates, even when the detector operated on a bounded working raster.
  const width = Number(source.width)
  const height = Number(source.height)
  if (!Number.isSafeInteger(annotationMask.maskedPixels) || Number(annotationMask.maskedPixels) < 0 ||
      Number(annotationMask.maskedPixels) > width * height || !unitFraction(annotationMask.maskedFraction) ||
      Math.abs(Number(annotationMask.maskedFraction) - Number(annotationMask.maskedPixels) / (width * height)) > 1e-6 + 1 / (width * height)) {
    throw new EvidenceContractError("invalid_mask_counts", "Input preparation returned inconsistent annotation counts.")
  }
  for (const raw of annotationMask.components as unknown[]) {
    const component = object(raw, "Annotation component")
    validateCropBox({ left: component.x0, top: component.y0, right: component.x1, bottom: component.y1 }, "Annotation component", { width, height })
    if (!Number.isSafeInteger(component.pixels) || Number(component.pixels) < 0 ||
        Number(component.pixels) > (Number(component.x1) - Number(component.x0)) * (Number(component.y1) - Number(component.y0)) ||
        !unitFraction(component.fillFraction) || !["red", "blue"].includes(String(component.dominantColor))) {
      throw new EvidenceContractError("invalid_annotation_component", "Input preparation returned invalid annotation component evidence.")
    }
  }
  return value as unknown as PreprocessingReport
}

function validateCropBox(value: unknown, label: string, image?: { width: number; height: number }) {
  const crop = object(value, label)
  if (
    ![crop.left, crop.top, crop.right, crop.bottom].every(finite) ||
    Number(crop.left) < 0 ||
    Number(crop.top) < 0 ||
    Number(crop.right) <= Number(crop.left) ||
    Number(crop.bottom) <= Number(crop.top) ||
    (image && (Number(crop.right) > image.width || Number(crop.bottom) > image.height))
  ) {
    throw new EvidenceContractError("invalid_crop", `${label} contains invalid coordinates.`)
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
    throw new EvidenceContractError("invalid_semantic_recognition", "Layout detector returned invalid semantic-recognition metadata.")
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
    throw new EvidenceContractError("incomplete_semantic_proof", "Layout detector returned incomplete semantic-recognition proof.")
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
    throw new EvidenceContractError("unsubstantiated_semantic_flag", "Layout detector returned an unsubstantiated semantic-identity flag.")
  }
}

export function parseLayoutGeometryReport(text: string): LayoutGeometryReport {
  const value = parseJsonObject(text, "Layout detector")
  if (value.version !== undefined && value.version !== 1) throw new EvidenceContractError("unsupported_layout_version", "Layout detector returned an unsupported report version.")
  const image = value.image === undefined ? undefined : object(value.image, "Layout image") as { width: number; height: number }
  if (image && (!positiveInteger(image.width) || !positiveInteger(image.height))) throw new EvidenceContractError("invalid_image_dimensions", "Layout detector returned invalid image dimensions.")
  if (value.rowCenters !== undefined && !finiteArray(value.rowCenters)) {
    throw new EvidenceContractError("invalid_row_centers", "Layout detector returned invalid row centers.")
  }
  if (value.rowCenters && (value.rowCenters as number[]).some((y) => y < 0 || (image && y >= image.height))) {
    throw new EvidenceContractError("invalid_row_centers", "Layout detector returned invalid row centers.")
  }
  if (value.confidence !== undefined && !unitFraction(value.confidence)) {
    throw new EvidenceContractError("invalid_confidence", "Layout detector returned invalid confidence.")
  }
  if (value.sourceLabelGrid !== undefined) {
    const grid = object(value.sourceLabelGrid, "Source-label grid")
    const invalid = () => { throw new EvidenceContractError("invalid_source_label_grid", "Source-label evidence must match its image frame, ordered crops and supported grid.") }
    if (!image || value.layoutHint !== "standard_3x4_with_r1" ||
        grid.version !== 1 || grid.method !== "source-value-grid-below-trace-v1" ||
        grid.maskMethod !== "maximum-channel-below-160-v1" ||
        !finiteArray(grid.imageSize) || grid.imageSize.length !== 2 || grid.imageSize[0] !== image.width || grid.imageSize[1] !== image.height ||
        !positivePhysicalValue(grid.rowSpacing) || grid.rowSpacing <= .09*image.height || grid.rowSpacing >= .30*image.height ||
        !positivePhysicalValue(grid.columnSpacing) || grid.columnSpacing <= .12*image.width || grid.columnSpacing >= .30*image.width ||
        !positivePhysicalValue(grid.fontHeight) || grid.fontHeight <= .02*grid.rowSpacing || grid.fontHeight >= .18*grid.rowSpacing ||
        !finiteArray(grid.xFit) || grid.xFit.length !== 3 || !finiteArray(grid.yFit) || grid.yFit.length !== 3 ||
        !finiteArray(grid.labelRowCenters) || grid.labelRowCenters.length !== 4 ||
        grid.labelRowCenters.some((y, i, rows) => y < 0 || y >= image.height || (i > 0 && y <= rows[i-1])) ||
        !nonnegativePhysicalValue(grid.maximumXResidual) || !nonnegativePhysicalValue(grid.maximumYResidual) ||
        !Array.isArray(grid.slots) || grid.slots.length !== 12 || !Array.isArray(grid.anchors) || grid.anchors.length < 6 || grid.anchors.length > 9) invalid()
    const names = ["I", "aVR", "V1", "V4", "II", "aVL", "V2", "V5", "III", "aVF", "V3", "V6"]
    const checkSlot = (raw: unknown, lead: string, row: number, column: number) => {
      const slot = object(raw, "Source-label crop")
      if (slot.lead !== lead || slot.row !== row || slot.column !== column || !finiteArray(slot.box) || slot.box.length !== 4 || !slot.box.every(Number.isInteger)) invalid()
      const [left, top, right, bottom] = slot.box as number[]
      validateCropBox({left, top, right, bottom}, "Source-label crop", image)
    }
    ;(grid.slots as unknown[]).forEach((slot, i) => checkSlot(slot, names[i], Math.floor(i/4), i%4))
    checkSlot(grid.rhythmSlot, "II", 3, 0)
    const anchors = grid.anchors as Array<Record<string, unknown>>
    const allowed = names.filter((_, i) => i%4 !== 0)
    if (new Set(anchors.map(anchor => anchor.lead)).size !== anchors.length) invalid()
    for (const anchor of anchors) {
      if (!allowed.includes(String(anchor.lead)) || !unitFraction(anchor.confidence) || Number(anchor.confidence) < .65 ||
          !nonnegativePhysicalValue(anchor.x) || !nonnegativePhysicalValue(anchor.y) ||
          !positivePhysicalValue(anchor.width) || !positivePhysicalValue(anchor.height) ||
          Number(anchor.x)+Number(anchor.width) > image!.width || Number(anchor.y) >= image!.height) invalid()
    }
    validateLocalLabelRecognition(grid, image!)
    validateTiledLabelRecognition(grid, image!)
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
  if (value.contentBox) validateCropBox(value.contentBox, "Layout content box", image)
  if (value.compoundPanels) {
    const panels = object(value.compoundPanels, "Compound panels")
    if (
      !["side_by_side", "stacked"].includes(String(panels.orientation)) ||
      !unitFraction(panels.confidence)
    ) {
      throw new EvidenceContractError("invalid_compound_panels", "Layout detector returned invalid compound-panel metadata.")
    }
    validateCropBox(panels.first, "First compound panel", image)
    validateCropBox(panels.second, "Second compound panel", image)
  }
  let calibrationState: EvidenceState = "unresolved"
  if (value.calibration !== undefined) {
    const calibration = object(value.calibration, "Calibration")
    if (typeof calibration.detected !== "boolean" || typeof calibration.method !== "string" || !calibration.method || !unitFraction(calibration.confidence)) {
      throw new EvidenceContractError("invalid_calibration", "Layout detector returned invalid calibration evidence.")
    }
    for (const key of ["gridScaleMm", "gridScaleMmX", "gridScaleMmY", "gridSpacingXPixels", "gridSpacingYPixels", "paperSpeedMmPerSecond", "gainMmPerMv", "pixelsPerMmX", "pixelsPerMmY"]) {
      if (calibration[key] !== undefined && calibration[key] !== null && !positivePhysicalValue(calibration[key])) {
        throw new EvidenceContractError("invalid_physical_scale", `Calibration ${key} must be finite and positive.`)
      }
    }
    if (calibration.rowPixelsPerMmX !== undefined && (!Array.isArray(calibration.rowPixelsPerMmX) || !calibration.rowPixelsPerMmX.every(positivePhysicalValue))) {
      throw new EvidenceContractError("invalid_row_scales", "Calibration row scales must be finite and positive.")
    }
    if (calibration.horizontalScaleEvidence !== undefined) {
      const evidence = object(calibration.horizontalScaleEvidence, "Horizontal scale evidence")
      const rows = (calibration.rowPixelsPerMmX as number[] | undefined)?.slice().sort((a, b) => a-b) ?? []
      const center = rows.length ? (rows[Math.floor((rows.length-1)/2)] + rows[Math.floor(rows.length/2)])/2 : NaN
      const spread = rows.length ? (rows.at(-1)!-rows[0])/center : NaN
      if (evidence.version !== 1 || evidence.method !== "consistent-row-grid-median-v1" ||
          !positivePhysicalValue(evidence.priorPixelsPerMmX) || ![6,12].includes(rows.length) ||
          rows.some(value => value < 1 || value > 40) ||
          evidence.rowCount !== rows.length || !nonnegativePhysicalValue(evidence.relativeRange) ||
          Math.abs(Number(evidence.relativeRange)-spread) > 1e-9 || spread > .02 ||
          Math.abs(Number(calibration.pixelsPerMmX)-center) > 1e-9 ||
          Math.abs(center-Number(evidence.priorPixelsPerMmX))/Math.max(center,Number(evidence.priorPixelsPerMmX)) > .15 ||
          calibration.detected !== true || Number(calibration.confidence) < .35 ||
          ![1,5].includes(Number(calibration.gridScaleMmX ?? calibration.gridScaleMm))) {
        throw new EvidenceContractError("invalid_horizontal_scale_evidence", "Horizontal scale must follow its consistent, calibrated row-grid evidence.")
      }
    }
    if (calibration.detected) {
      if (!["pixelsPerMmX", "pixelsPerMmY", "paperSpeedMmPerSecond", "gainMmPerMv"].every(key => positivePhysicalValue(calibration[key])) || Number(calibration.confidence) <= 0 || calibration.gridScaleAmbiguous === true) {
        throw new EvidenceContractError("unsubstantiated_calibration", "Detected calibration lacks positive scales or has unresolved grid ambiguity.")
      }
      // The rectangular pulse is detected; its 200 ms / 1 mV convention is inferred.
      calibrationState = "inferred"
    } else if (calibration.gridScaleDetected === true) calibrationState = "inferred"
    validatePrintedCalibration(calibration)
    if (calibration.reconciliation && (calibration.reconciliation as CalibrationReconciliation).quantitativeBlocked) calibrationState = "unresolved"
  }
  value.evidenceStatus = { calibration: calibrationState, coordinateBounds: image ? "verified" : "unresolved" }
  return value as unknown as LayoutGeometryReport
}

export function parseNativeExtractionReport(
  text: string,
  expectedLayout: string,
  expectedSourceIdentity?: SourceGridIdentity
): NativeExtractionReport {
  const value = parseJsonObject(text, "Native grid extraction")
  const fidelity = object(value.sourceFidelity, "Native source fidelity")
  if (
    value.layout !== expectedLayout ||
    (value.version !== undefined && value.version !== 1) ||
    !nonnegativePhysicalValue(value.layoutCost) ||
    !positivePhysicalValue(value.effectiveSampleRateHz) ||
    typeof fidelity.passed !== "boolean" ||
    typeof fidelity.method !== "string" ||
    !fidelity.leadMetrics ||
    typeof fidelity.leadMetrics !== "object" || Array.isArray(fidelity.leadMetrics)
  ) {
    throw new EvidenceContractError("invalid_native_fidelity", "Native grid extraction returned invalid fidelity metadata.")
  }
  if (!positivePhysicalValue(fidelity.pixelsPerMm)) throw new EvidenceContractError("invalid_physical_scale", "Native pixelsPerMm must be finite and positive.")
  validateSourceGridEvidence(fidelity, expectedSourceIdentity)
  if (fidelity.sourceTimingInference !== undefined && fidelity.sourceTimingInference !== null) {
    const timing = object(fidelity.sourceTimingInference, "Native source timing")
    if (timing.quantitativeCalibrationConfirmed !== undefined && typeof timing.quantitativeCalibrationConfirmed !== "boolean") {
      throw new EvidenceContractError("invalid_timing_confirmation", "Source timing confirmation must be explicit boolean evidence.")
    }
    if (timing.quantitativeCalibrationConfirmed === true &&
        (!positivePhysicalValue(timing.calibrationPixelsPerMmX) || !positivePhysicalValue(timing.calibrationPixelsPerMmY) ||
         !unitFraction(timing.calibrationConfidence) || timing.calibrationConfidence <= 0 ||
         typeof timing.quantitativeCalibrationMethod !== "string" || !timing.quantitativeCalibrationMethod)) {
      throw new EvidenceContractError("unsubstantiated_timing_confirmation", "Source timing confirmation lacks calibration evidence.")
    }
  }
  for (const key of ["layoutConfidence", "evidenceMedian", "evidenceP10", "minimumCoverage"]) {
    if (!unitFraction(fidelity[key])) throw new EvidenceContractError("invalid_fidelity_fraction", `Native ${key} must be in [0, 1].`)
  }
  const leads = object(fidelity.leadMetrics, "Native lead metrics")
  if (!Object.keys(leads).length) throw new EvidenceContractError("missing_lead_metrics", "Native source fidelity has no lead metrics.")
  for (const [lead, raw] of Object.entries(leads)) {
    if (!(ECG_LEADS as readonly string[]).includes(lead)) throw new EvidenceContractError("invalid_lead_id", `Native source fidelity has an unknown lead: ${lead}.`)
    const metrics = object(raw, `Native ${lead}`)
    for (const key of ["evidenceMedian", "evidenceP10", "coverage"]) {
      if (!unitFraction(metrics[key])) throw new EvidenceContractError("invalid_lead_metric", `Native ${lead} ${key} must be in [0, 1].`)
    }
    for (const key of ["maxNativeJumpPixels", "p95NativeJumpPixels", "sourceStartPixel", "sourceEndPixel"]) {
      if (!nonnegativePhysicalValue(metrics[key])) throw new EvidenceContractError("invalid_lead_metric", `Native ${lead} ${key} must be finite and nonnegative.`)
    }
    if (Number(metrics.sourceEndPixel) <= Number(metrics.sourceStartPixel) || Number(metrics.p95NativeJumpPixels) > Number(metrics.maxNativeJumpPixels)) {
      throw new EvidenceContractError("inconsistent_lead_metrics", `Native ${lead} has inconsistent source support or jump metrics.`)
    }
    for (const [key, count] of Object.entries(metrics)) {
      if (key.endsWith("Count") && (!Number.isSafeInteger(count) || Number(count) < 0)) throw new EvidenceContractError("invalid_metric_count", `Native ${lead} ${key} must be a nonnegative integer.`)
    }
  }
  return value as unknown as NativeExtractionReport
}


export type LocalLabelToken = {lead: string; x: number; y: number; width: number; height: number; confidence: number}
export type TiledLabelRecognition = {
  version: 1
  method: "fixed-half-page-source-tiles-image-only-ocr-v1"
  sourceRasterSha256: string
  originalFailureReason: string
  originalAnchors: LocalLabelToken[]
  crops: {tileRow: number; tileColumn: number; bounds: number[]; rasterSha256: string
    localTokens: LocalLabelToken[]; sourceTokens: LocalLabelToken[]}[]
  expectedNamesProvidedToRecognizer: false
  strictGridRequired: true
  truthUsed: false
}
export type LocalLabelRecognition = {
  version: 1
  method: "sparse-source-proposal-image-only-local-ocr-v1"
  sourceRasterSha256: string
  originalFailureReason: string
  originalAnchors: LocalLabelToken[]
  proposalFit: {
    xFit: number[]; yFit: number[]; rowSpacing: number; columnSpacing: number; fontHeight: number
    maximumXResidual: number; maximumYResidual: number; rank: 3
    rowCounts: number[]; columnsObserved: number[]; numericChecks: Record<string, boolean>
  }
  crops: {row: number; column: number; role: "panel" | "rhythm"; bounds: number[]; rasterSha256: string
    localTokens: LocalLabelToken[]; sourceTokens: LocalLabelToken[]}[]
  expectedNamesProvidedToRecognizer: false
  strictGridRequired: true
  truthUsed: false
}

function invalidLocalLabels(): never {
  throw new EvidenceContractError("invalid_source_label_grid", "Local source labels must retain their observed crop coordinates and strict grid evidence.")
}
function labelObject(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return invalidLocalLabels()
  return value as Record<string,unknown>
}
function labelNumbers(value: unknown, count: number): number[] {
  if (!Array.isArray(value) || value.length !== count || value.some(x => typeof x !== "number" || !Number.isFinite(x))) return invalidLocalLabels()
  return value
}
function labelArray(value: unknown): unknown[] {
  if (!Array.isArray(value)) return invalidLocalLabels()
  return value
}
function labelHash(value: unknown) { return typeof value === "string" && /^[a-f0-9]{64}$/.test(value) }
function labelEqual(a: number[], b: number[]) { return a.length === b.length && a.every((x,i) => x === b[i]) }
function roundLabelPixel(x: number) {
  const floor = Math.floor(x), fraction = x-floor
  return fraction > .5 || (fraction === .5 && floor%2 !== 0) ? floor+1 : floor
}
const labelNames = ["I","aVR","V1","V4","II","aVL","V2","V5","III","aVF","V3","V6"]
function checkedLabelToken(value: unknown, width: number, height: number): LocalLabelToken {
  const token = labelObject(value)
  const [x,y,w,h] = labelNumbers([token.x, token.y, token.width, token.height],4)
  if (!labelNames.includes(String(token.lead)) || !unitFraction(token.confidence) || token.confidence < .65 ||
      !(0 <= x && x < x+w && x+w <= width && 0 <= y-h/2 && y-h/2 < y+h/2 && y+h/2 <= height)) return invalidLocalLabels()
  return token as LocalLabelToken
}
function sameLabelToken(a: LocalLabelToken, b: LocalLabelToken) {
  return a.lead === b.lead && a.confidence === b.confidence && a.x === b.x && a.y === b.y && a.width === b.width && a.height === b.height
}

export function validateTiledLabelRecognition(grid: Record<string, unknown>, image: {width:number;height:number}) {
  if (grid.tiledRecognition === undefined) return
  const proof = labelObject(grid.tiledRecognition)
  if (grid.localRecognition !== undefined || proof.version !== 1 ||
      proof.method !== "fixed-half-page-source-tiles-image-only-ocr-v1" ||
      !labelHash(proof.sourceRasterSha256) || typeof proof.originalFailureReason !== "string" || !proof.originalFailureReason ||
      proof.expectedNamesProvidedToRecognizer !== false || proof.strictGridRequired !== true || proof.truthUsed !== false) invalidLocalLabels()
  const observations = labelArray(proof.originalAnchors).map(t=>checkedLabelToken(t,image.width,image.height))
  const crops = labelArray(proof.crops)
  if (crops.length !== 9) invalidLocalLabels()
  crops.forEach((raw,i)=>{
    const crop = labelObject(raw), row = Math.floor(i/3), column = i%3
    const bounds = labelNumbers(crop.bounds,4), [left,top,right,bottom] = bounds
    const expected = [column*image.width/4,row*image.height/4,(column+2)*image.width/4,(row+2)*image.height/4].map(roundLabelPixel)
    if (crop.tileRow !== row || crop.tileColumn !== column || !labelEqual(bounds,expected) || !labelHash(crop.rasterSha256) ||
        !(0 <= left && left < right && right <= image.width && 0 <= top && top < bottom && bottom <= image.height)) invalidLocalLabels()
    const local = labelArray(crop.localTokens).map(t=>checkedLabelToken(t,right-left,bottom-top))
    const translated = labelArray(crop.sourceTokens).map(t=>checkedLabelToken(t,image.width,image.height))
    if (local.length !== translated.length || local.some((t,i)=>!sameLabelToken({...t,x:t.x+left,y:t.y+top},translated[i]))) invalidLocalLabels()
    observations.push(...translated)
  })
  // Retain every observed conflict and the exact highest-confidence selection.
  const selected = new Map<string,LocalLabelToken>()
  for (const token of observations) {
    if (labelNames.indexOf(token.lead)%4 === 0) continue
    const prior = selected.get(token.lead)
    if (prior && (Math.abs(token.x-prior.x)>image.width*.025 || Math.abs(token.y-prior.y)>image.height*.025)) invalidLocalLabels()
    if (!prior || token.confidence > prior.confidence) selected.set(token.lead,token)
  }
  const anchors = [...selected.values()], finalAnchors = labelArray(grid.anchors).map(t=>checkedLabelToken(t,image.width,image.height))
  const rowCounts = [0,0,0], columns = new Set<number>()
  for (const token of anchors) {const i=labelNames.indexOf(token.lead);rowCounts[Math.floor(i/4)]++;columns.add(i%4)}
  if (anchors.length < 6 || rowCounts.some(n=>n<2) || columns.size !== 3 || finalAnchors.length !== anchors.length ||
      anchors.some((t,i)=>!sameLabelToken(t,finalAnchors[i]))) invalidLocalLabels()
  const x=labelNumbers(grid.xFit,3), y=labelNumbers(grid.yFit,3)
  const [spacing,columnSpacing,font,maxX,maxY] = labelNumbers([grid.rowSpacing,grid.columnSpacing,grid.fontHeight,grid.maximumXResidual,grid.maximumYResidual],5)
  const predicted = (fit:number[],row:number,column:number)=>fit[0]+row*fit[1]+column*fit[2]
  const residual = (fit:number[],key:"x"|"y")=>Math.max(...anchors.map(t=>{const i=labelNames.indexOf(t.lead);return Math.abs(predicted(fit,Math.floor(i/4),i%4)-t[key])}))
  const heights=anchors.map(t=>t.height).sort((a,b)=>a-b)
  if (spacing !== y[1] || columnSpacing !== x[2] ||
      Math.abs(x[1]) >= columnSpacing*.06 || Math.abs(y[2]) >= spacing*.06 ||
      maxX >= columnSpacing*.04 || maxY >= Math.max(font*.6,spacing*.04) ||
      Math.abs(residual(x,"x")-maxX)>1e-8 || Math.abs(residual(y,"y")-maxY)>1e-8 ||
      font !== (heights[Math.floor((heights.length-1)/2)]+heights[Math.floor(heights.length/2)])/2) invalidLocalLabels()
  const slots=labelArray(grid.slots)
  if (slots.length !== 12) invalidLocalLabels()
  slots.forEach((raw,i)=>{
    const px=predicted(x,Math.floor(i/4),i%4), py=predicted(y,Math.floor(i/4),i%4)
    if (!labelEqual(labelNumbers(labelObject(raw).box,4),[px-font*.65,py-font*.6,px+font*3.4,py+font*.6].map(roundLabelPixel))) invalidLocalLabels()
  })
  const rhythm=observations.filter(t=>t.lead === "II" &&
    Math.abs(t.x-predicted(x,3,0))<columnSpacing*.12 && Math.abs(t.y-predicted(y,3,0))<spacing*.25)
  if (!rhythm.length) invalidLocalLabels()
  const anchor=rhythm.reduce((a,b)=>b.confidence>a.confidence?b:a)
  const rows=labelNumbers(grid.labelRowCenters,4)
  if (rhythm.some(t=>Math.abs(t.y-anchor.y)>font) || rows[3] !== anchor.y ||
      rows.slice(0,3).some((v,i)=>Math.abs(v-predicted(y,i,1.5))>1e-8) ||
      !labelEqual(labelNumbers(labelObject(grid.rhythmSlot).box,4),
        [anchor.x-font*.65,anchor.y-font*.6,anchor.x+font*3.4,anchor.y+font*.6].map(roundLabelPixel))) invalidLocalLabels()
}

export function validateLocalLabelRecognition(grid: Record<string, unknown>, image: {width:number;height:number}) {
  if (grid.localRecognition === undefined) return
  const proof = labelObject(grid.localRecognition), fit = labelObject(proof.proposalFit)
  if (proof.version !== 1 || proof.method !== "sparse-source-proposal-image-only-local-ocr-v1" ||
      !labelHash(proof.sourceRasterSha256) || typeof proof.originalFailureReason !== "string" || !proof.originalFailureReason ||
      proof.expectedNamesProvidedToRecognizer !== false || proof.strictGridRequired !== true || proof.truthUsed !== false) invalidLocalLabels()
  const original = labelArray(proof.originalAnchors).map(t => checkedLabelToken(t,image.width,image.height))
  const selected = new Map<string,LocalLabelToken>()
  for (const token of original) {
    if (labelNames.indexOf(token.lead)%4 === 0) continue
    const prior = selected.get(token.lead)
    if (prior && (Math.abs(token.x-prior.x) > image.width*.025 || Math.abs(token.y-prior.y) > image.height*.025)) invalidLocalLabels()
    if (!prior || token.confidence > prior.confidence) selected.set(token.lead,token)
  }
  const anchors = [...selected.values()], rowCounts = [0,0,0], columns = new Set<number>()
  for (const token of anchors) {const i = labelNames.indexOf(token.lead); rowCounts[Math.floor(i/4)]++; columns.add(i%4)}
  if (anchors.length < 6 || rowCounts.some(n => n < 1) || columns.size < 2 || fit.rank !== 3 ||
      !labelEqual(labelNumbers(fit.rowCounts,3),rowCounts) ||
      !labelEqual(labelNumbers(fit.columnsObserved,columns.size),[...columns].sort())) invalidLocalLabels()
  const x = labelNumbers(fit.xFit,3), y = labelNumbers(fit.yFit,3)
  const [spacing, columnSpacing, font, maxX, maxY] = labelNumbers([fit.rowSpacing,fit.columnSpacing,fit.fontHeight,fit.maximumXResidual,fit.maximumYResidual],5)
  const check = labelObject(fit.numericChecks)
  const checks = {rowSpacing:.09*image.height < spacing && spacing < .30*image.height,
    columnSpacing:.12*image.width < columnSpacing && columnSpacing < .30*image.width,
    horizontalSkew:Math.abs(x[1]) < columnSpacing*.06, verticalSkew:Math.abs(y[2]) < spacing*.06,
    horizontalResidual:0 <= maxX && maxX < columnSpacing*.04, verticalResidual:0 <= maxY && maxY < Math.max(font*.6,spacing*.04),
    fontHeight:.02*spacing < font && font < .18*spacing}
  if (spacing !== y[1] || columnSpacing !== x[2] || Object.entries(checks).some(([k,v]) => !v || check[k] !== true)) invalidLocalLabels()
  const predicted = (fit: number[], row:number, column:number) => fit[0]+row*fit[1]+column*fit[2]
  const residual = (fit: number[], key: "x" | "y") => Math.max(...anchors.map(a => {const i=labelNames.indexOf(a.lead); return Math.abs(predicted(fit,Math.floor(i/4),i%4)-a[key])}))
  if (Math.abs(residual(x,"x")-maxX)>1e-8 || Math.abs(residual(y,"y")-maxY)>1e-8) invalidLocalLabels()
  const heights = anchors.map(a => a.height).sort((a,b)=>a-b)
  if (font !== (heights[Math.floor((heights.length-1)/2)]+heights[Math.floor(heights.length/2)])/2) invalidLocalLabels()
  const crops = labelArray(proof.crops), observations = [...original]
  if (crops.length !== 13) invalidLocalLabels()
  crops.forEach((raw,i) => {
    const crop = labelObject(raw), row = i === 12 ? 3 : Math.floor(i/4), column = i === 12 ? 0 : i%4
    const box = labelNumbers(crop.bounds,4), [left,top,right,bottom] = box
    const px = predicted(x,row,column), py = predicted(y,row,column)
    const expected = [px-font*.65,py-font*.6,px+font*3.4,py+font*.6].map(roundLabelPixel)
    if (crop.row !== row || crop.column !== column || crop.role !== (i===12 ? "rhythm":"panel") || !labelHash(crop.rasterSha256) ||
        !labelEqual(box,expected) || !(0<=left && left<right && right<=image.width && 0<=top && top<bottom && bottom<=image.height)) invalidLocalLabels()
    const local = labelArray(crop.localTokens).map(t=>checkedLabelToken(t,right-left,bottom-top))
    const translated = labelArray(crop.sourceTokens).map(t=>checkedLabelToken(t,image.width,image.height))
    if (local.length !== translated.length || local.some((t,i)=>!sameLabelToken({...t,x:t.x+left,y:t.y+top},translated[i]))) invalidLocalLabels()
    observations.push(...translated)
  })
  const finalAnchors = labelArray(grid.anchors).map(t=>checkedLabelToken(t,image.width,image.height))
  for (const token of finalAnchors) {
    if (!observations.some(observed=>sameLabelToken(token,observed))) invalidLocalLabels()
  }
  // The strict grid's independently named rhythm crop must also have an actual
  // observed II token. Preliminary crop positions cannot manufacture it.
  const finalX = labelNumbers(grid.xFit,3), finalY = labelNumbers(grid.yFit,3)
  const [finalColumns,finalSpacing,finalFont] = labelNumbers([grid.columnSpacing,grid.rowSpacing,grid.fontHeight],3)
  const rhythm = observations.filter(t=>t.lead === "II" &&
    Math.abs(t.x-predicted(finalX,3,0)) < finalColumns*.12 &&
    Math.abs(t.y-predicted(finalY,3,0)) < finalSpacing*.25)
  if (!rhythm.length) invalidLocalLabels()
  const anchor = rhythm.reduce((a,b)=>b.confidence>a.confidence ? b:a)
  if (rhythm.some(t=>Math.abs(t.y-anchor.y)>finalFont) ||
      !labelEqual(labelNumbers(labelObject(grid.rhythmSlot).box,4),
        [anchor.x-finalFont*.65,anchor.y-finalFont*.6,anchor.x+finalFont*3.4,anchor.y+finalFont*.6].map(roundLabelPixel))) invalidLocalLabels()
}
