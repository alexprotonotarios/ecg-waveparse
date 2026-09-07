import type { LayoutGeometryReport, PreprocessingReport, RasterCropBox } from "@/lib/digitizer/contracts"
import { EvidenceContractError, invertibleTransform } from "@/lib/digitizer/physical-contracts"

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
export function sourceTransformChain(preprocessing: PreprocessingReport, parameters?: { inputVariant?: string; cropBox?: RasterCropBox }) {
  const source = preprocessing.source
  const working = preprocessing.workingImage ?? source
  const corrected = ["geometry-corrected", "artifact-preprocessed"].includes(parameters?.inputVariant ?? "") && preprocessing.geometryCorrection?.applied === true
  const correction = corrected ? preprocessing.geometryCorrection! : undefined
  const crop = parameters?.cropBox
  const steps = [
    { from: "original", to: "working", matrix: [[working.width/source.width, 0, 0], [0, working.height/source.height, 0], [0, 0, 1]] },
    { from: "working", to: "corrected", matrix: correction?.transform ?? identity() },
    { from: "corrected", to: "candidate", matrix: [[1, 0, -(crop?.left ?? 0)], [0, 1, -(crop?.top ?? 0)], [0, 0, 1]] },
  ]
  const originalToCandidate = steps.reduce((matrix, step) => composeTransforms(step.matrix, matrix), identity())
  return { version: 1 as const, convention: "pixel_edges" as const, sourceSha256: preprocessing.sourceSha256,
    sourceSize: { width: source.width, height: source.height },
    candidateSize: { width: crop ? crop.right-crop.left : correction?.outputWidth ?? working.width,
      height: crop ? crop.bottom-crop.top : correction?.outputHeight ?? working.height },
    steps, originalToCandidate, candidateToOriginal: inverseTransform(originalToCandidate),
    waveformResamplingPerformed: false as const,
  }
}

export type SourceRegion = { state: "unresolved"; reason: string } | {
  state: "inferred"; coordinateSpace: "original"; polygon: Point[]; bounds: RasterCropBox;
  method: "layout-band-inverse-source-transform-v1"; traceSupportVerified: false;
}

/** A review-navigation region, not a claim that every pixel in the band is waveform. */
export function segmentSourceRegion(preprocessing: PreprocessingReport, geometry: LayoutGeometryReport, row: number, panel: number, panelCount: number, rhythm: boolean): SourceRegion {
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
  const left = rhythm ? 0 : panel*width/panelCount, right = rhythm ? width : (panel+1)*width/panelCount
  const polygon = [{x:left,y:top}, {x:right,y:top}, {x:right,y:bottom}, {x:left,y:bottom}]
    .map(point => transformPoint(chain.candidateToOriginal, point))
  return { state: "inferred", coordinateSpace: "original", polygon,
    bounds: { left: Math.max(0, Math.min(...polygon.map(p=>p.x))), top: Math.max(0, Math.min(...polygon.map(p=>p.y))),
      right: Math.min(preprocessing.source.width, Math.max(...polygon.map(p=>p.x))), bottom: Math.min(preprocessing.source.height, Math.max(...polygon.map(p=>p.y))) },
    method: "layout-band-inverse-source-transform-v1", traceSupportVerified: false }
}

export function calibrationEvidence(calibration: LayoutGeometryReport["calibration"]) {
  const detected = calibration?.detected === true
  return { version: 1, physicalUnitsState: detected && !calibration?.reconciliation?.quantitativeBlocked ? "inferred" : "unresolved",
    speed: { value: calibration?.paperSpeedMmPerSecond ?? null, state: detected ? "inferred" : "unresolved", method: detected ? "rectangular-pulse-assuming-200-ms" : "not-detected" },
    gain: { value: calibration?.gainMmPerMv ?? null, state: detected ? "inferred" : "unresolved", method: detected ? "rectangular-pulse-assuming-1-mV" : "not-detected" },
    printedSettings: calibration?.printedSettings ?? { state: "unresolved", reason: "Printed settings were not retained in this historical report." },
    reconciliation: calibration?.reconciliation ?? { state: "unresolved", reason: "No printed-setting reconciliation was recorded." },
    grid: { pixelsPerMmX: calibration?.pixelsPerMmX ?? null, pixelsPerMmY: calibration?.pixelsPerMmY ?? null,
      ambiguous: calibration?.gridScaleAmbiguous ?? null },
    acquisitionSampleRateHz: null, requiresSourceReview: true,
  }
}
