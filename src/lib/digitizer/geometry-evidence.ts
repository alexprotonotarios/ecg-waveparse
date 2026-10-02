import type { LayoutGeometryReport, PreprocessingReport, RasterCropBox } from "@/lib/digitizer/contracts"
import { EvidenceContractError, invertibleTransform } from "@/lib/digitizer/physical-contracts"
import { calibrationIsUsable, calibrationPublicationBlock } from "@/lib/digitizer/calibration-settings"
import { parseCandidatePreparation, type CandidatePreparationReceipt } from "./candidate-preparation"

function hasConfirmedSourceGrid(geometry: LayoutGeometryReport) {
  return geometry.sourceLabelGrid?.method === "source-value-grid-below-trace-v1" &&
    geometry.layoutHint === "standard_3x4_with_r1" && geometry.semanticLeadEvidence?.passed === true &&
    geometry.rhythmLeadValidation?.semanticIdentityConfirmed === true
}

/** Source timing is admitted only on the unmodified, unscaled working raster. */
export function shouldUseSourceGridNativeMode(geometry: LayoutGeometryReport, inputVariant: string,
  usePreparedEvidence: boolean, upscaleToMaxDimension?: number) {
  return hasConfirmedSourceGrid(geometry) && geometry.coordinateSpace === "working" &&
    inputVariant === "original" && !usePreparedEvidence && !upscaleToMaxDimension
}

/** Recognized panels can still lack original-raster grid and rhythm evidence. */
export function shouldInspectUnmaskedSourceGeometry(current: LayoutGeometryReport) {
  return current.semanticLeadEvidence?.passed !== true ||
    (current.layoutHint === "standard_3x4_with_r1" &&
      (current.detectedInputVariant === "annotation-masked" || !hasConfirmedSourceGrid(current)))
}

/** Use complete unmasked source proof only in the same working-image frame. */
export function preferUnmaskedSourceGeometry(current: LayoutGeometryReport, source: LayoutGeometryReport) {
  return shouldInspectUnmaskedSourceGeometry(current) && hasConfirmedSourceGrid(source) &&
    current.image?.width === source.image?.width && current.image?.height === source.image?.height &&
    source.image !== undefined && source.coordinateSpace === "working"
}

/** Rectified calibration cannot replace a usable original-raster native grid. */
export function preserveCalibratedSourceGeometry(source: LayoutGeometryReport, corrected: LayoutGeometryReport) {
  return source.detectedInputVariant === "original" &&
    shouldUseSourceGridNativeMode(source, "original", false) &&
    calibrationIsUsable(source.calibration) && !calibrationPublicationBlock(source.calibration) &&
    corrected.detectedInputVariant === "geometry-corrected" && corrected.coordinateSpace === "geometry-corrected"
}

export type Point = { x: number; y: number }
type Matrix = number[][]
const identity = (): Matrix => [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

export function composeTransforms(after: Matrix, before: Matrix): Matrix {
  return after.map(row => before[0].map((_, column) => row.reduce((sum, value, i) => sum + value * before[i][column], 0)))
}

export function inverseTransform(m: Matrix): Matrix {
  if (!invertibleTransform(m)) throw new EvidenceContractError("singular_transform", "Source mapping is not invertible.")
  const [[a, b, c], [d, e, f], [g, h, i]] = m
  const adjugate = [[e*i-f*h, c*h-b*i, b*f-c*e], [f*g-d*i, a*i-c*g, c*d-a*f], [d*h-e*g, b*g-a*h, a*e-b*d]]
  const determinant = a*adjugate[0][0] + b*adjugate[1][0] + c*adjugate[2][0]
  const inverse = adjugate.map(row => row.map(value => value / determinant))
  if (!Number.isFinite(determinant) || determinant === 0 || !inverse.flat().every(Number.isFinite)) {
    throw new EvidenceContractError("unrepresentable_inverse_transform", "Source mapping cannot be inverted with finite numerical precision.")
  }
  return inverse
}

export function transformPoint(matrix: Matrix, point: Point): Point {
  const denominator = matrix[2][0]*point.x + matrix[2][1]*point.y + matrix[2][2]
  if (!Number.isFinite(denominator) || Math.abs(denominator) < 1e-12) throw new EvidenceContractError("point_at_projective_infinity", "Source point is outside a valid projective mapping.")
  const mapped = { x: (matrix[0][0]*point.x + matrix[0][1]*point.y + matrix[0][2]) / denominator,
    y: (matrix[1][0]*point.x + matrix[1][1]*point.y + matrix[1][2]) / denominator }
  if (![mapped.x, mapped.y].every(Number.isFinite)) throw new EvidenceContractError("invalid_mapped_point", "Source mapping returned nonfinite coordinates.")
  return mapped
}

/** Pixel-edge coordinates; transforms describe geometry, never another waveform resampling. */
export function sourceTransformChain(preprocessing: PreprocessingReport, parameters?: { inputVariant?: string; cropBox?: RasterCropBox }, preparation?: CandidatePreparationReceipt) {
  const source = preprocessing.source
  const working = preprocessing.workingImage ?? source
  const corrected = ["geometry-corrected", "artifact-preprocessed"].includes(parameters?.inputVariant ?? "") && preprocessing.geometryCorrection?.applied === true
  const correction = corrected ? preprocessing.geometryCorrection! : undefined
  const crop = parameters?.cropBox
  const observed = preparation ? parseCandidatePreparation(preparation) : undefined
  const inputSize = [correction?.outputWidth ?? working.width, correction?.outputHeight ?? working.height]
  if (observed && (observed.inputSizeWh.some((v, i) => v !== inputSize[i]) ||
      JSON.stringify(observed.cropBox) !== JSON.stringify(crop ? [crop.left, crop.top, crop.right, crop.bottom] : null))) {
    throw new EvidenceContractError("candidate_preparation_frame_mismatch", "Observed candidate preparation does not match the source frame.")
  }
  const steps = [
    { from: "original", to: "working", matrix: [[working.width/source.width, 0, 0], [0, working.height/source.height, 0], [0, 0, 1]] },
    { from: "working", to: "corrected", matrix: correction?.transform ?? identity() },
    { from: "corrected", to: "candidate", matrix: observed?.inputToPreparedEdges ?? [[1, 0, -(crop?.left ?? 0)], [0, 1, -(crop?.top ?? 0)], [0, 0, 1]] },
  ]
  const originalToCandidate = steps.reduce((matrix, step) => composeTransforms(step.matrix, matrix), identity())
  return { version: 1 as const, convention: "pixel_edges" as const, sourceSha256: preprocessing.sourceSha256,
    sourceSize: { width: source.width, height: source.height },
    candidateSize: { width: observed?.preparedSizeWh[0] ?? (crop ? crop.right-crop.left : correction?.outputWidth ?? working.width),
      height: observed?.preparedSizeWh[1] ?? (crop ? crop.bottom-crop.top : correction?.outputHeight ?? working.height) },
    ...(observed ? { preparation: observed } : {}),
    steps, originalToCandidate, candidateToOriginal: inverseTransform(originalToCandidate),
    waveformResamplingPerformed: false as const,
  }
}

export type SourceRegion = { state: "unresolved"; reason: string } | {
  state: "inferred"; coordinateSpace: "original"; polygon: Point[]; bounds: RasterCropBox;
  method: "layout-band-inverse-source-transform-v1" | "source-timing-band-inverse-source-transform-v1"; traceSupportVerified: false;
}

/** A review-navigation region, not a claim that every pixel in the band is waveform. */
export function segmentSourceRegion(preprocessing: PreprocessingReport, geometry: LayoutGeometryReport, row: number, panel: number, panelCount: number, rhythm: boolean, sourceTimingRange?: readonly number[]): SourceRegion {
  const centers = geometry.rowCenters
  if (!geometry.image || !centers?.length || geometry.compoundPanels || row < 0 || panel < 0 || panel >= panelCount) {
    return { state: "unresolved", reason: "No unambiguous rectangular source layout band is available for this segment." }
  }
  const index = rhythm ? centers.length-1 : row
  if (index >= centers.length) return { state: "unresolved", reason: "Source row is absent from the layout evidence." }
  const chain = sourceTransformChain(preprocessing, { inputVariant: geometry.coordinateSpace === "geometry-corrected" ? "geometry-corrected" : "original" })
  const width = geometry.image.width, height = geometry.image.height
  if (width !== chain.candidateSize.width || height !== chain.candidateSize.height) return { state: "unresolved", reason: "Layout dimensions do not match the retained transform frame." }
  const top = index ? (centers[index-1]+centers[index])/2 : 0
  const bottom = index+1 < centers.length ? (centers[index]+centers[index+1])/2 : height
  if (sourceTimingRange && (sourceTimingRange.length !== 2 || !sourceTimingRange.every(Number.isFinite) ||
      sourceTimingRange[0] < 0 || sourceTimingRange[1] > width || sourceTimingRange[1] <= sourceTimingRange[0])) {
    return { state: "unresolved", reason: "Source timing bounds are outside the retained image frame." }
  }
  const left = sourceTimingRange?.[0] ?? (rhythm ? 0 : panel*width/panelCount)
  const right = sourceTimingRange?.[1] ?? (rhythm ? width : (panel+1)*width/panelCount)
  const polygon = [{x:left,y:top}, {x:right,y:top}, {x:right,y:bottom}, {x:left,y:bottom}]
    .map(point => transformPoint(chain.candidateToOriginal, point))
  return { state: "inferred", coordinateSpace: "original", polygon,
    bounds: { left: Math.max(0, Math.min(...polygon.map(p=>p.x))), top: Math.max(0, Math.min(...polygon.map(p=>p.y))),
      right: Math.min(preprocessing.source.width, Math.max(...polygon.map(p=>p.x))), bottom: Math.min(preprocessing.source.height, Math.max(...polygon.map(p=>p.y))) },
    method: sourceTimingRange ? "source-timing-band-inverse-source-transform-v1" : "layout-band-inverse-source-transform-v1", traceSupportVerified: false }
}

export function calibrationEvidence(calibration: LayoutGeometryReport["calibration"]) {
  const detected = calibration?.detected === true
  return { version: 1, physicalUnitsState: detected && !calibration?.reconciliation?.quantitativeBlocked ? "inferred" : "unresolved",
    speed: { value: calibration?.paperSpeedMmPerSecond ?? null, state: detected ? "inferred" : "unresolved", method: detected ? calibration?.speedResolution?.method ?? "rectangular-pulse-assuming-200-ms" : "not-detected" },
    ...(calibration?.speedResolution ? { speedResolution: calibration.speedResolution } : {}),
    gain: { value: calibration?.gainMmPerMv ?? null, state: detected ? "inferred" : "unresolved", method: detected ? "rectangular-pulse-assuming-1-mV" : "not-detected" },
    printedSettings: calibration?.printedSettings ?? { state: "unresolved", reason: "Printed settings were not retained in this historical report." },
    reconciliation: calibration?.reconciliation ?? { state: "unresolved", reason: "No printed-setting reconciliation was recorded." },
    grid: { pixelsPerMmX: calibration?.pixelsPerMmX ?? null, pixelsPerMmY: calibration?.pixelsPerMmY ?? null,
      ambiguous: calibration?.gridScaleAmbiguous ?? null },
    acquisitionSampleRateHz: null, requiresSourceReview: true,
  }
}
