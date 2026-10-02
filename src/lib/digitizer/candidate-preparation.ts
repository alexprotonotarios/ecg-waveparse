import { EvidenceContractError } from "./physical-contracts"
import type { RasterCropBox } from "./contracts"

export type CandidatePreparationReceipt = {
  version: 1
  method: "observed-candidate-preparation-v1"
  coordinateConvention: "pixel_edges"
  mode: "copy" | "png"
  inputSha256: string
  preparedSha256: string
  inputRasterRgbSha256: string
  preparedRasterRgbSha256: string
  inputSizeWh: [number, number]
  croppedSizeWh: [number, number]
  preparedSizeWh: [number, number]
  cropBox: [number, number, number, number] | null
  requestedMaxDimension: number
  resampling: "none" | "pillow_lanczos"
  inputToPreparedEdges: number[][]
  inputPreserved: true
  photometricFilterWeightsExported: false
  waveformResamplingPerformed: false
}

const fail = (): never => { throw new EvidenceContractError("invalid_candidate_preparation", "Candidate preparation does not match its observed image and transform.") }
const hash = (v: unknown): v is string => typeof v === "string" && /^[a-f0-9]{64}$/.test(v)
const size = (v: unknown): v is [number, number] => Array.isArray(v) && v.length === 2 && v.every(n => Number.isSafeInteger(n) && n > 0)

export function parseCandidatePreparation(value: unknown): CandidatePreparationReceipt {
  if (!value || typeof value !== "object" || Array.isArray(value)) return fail()
  const r = value as Record<string, unknown>
  if (r.version !== 1 || r.method !== "observed-candidate-preparation-v1" || r.coordinateConvention !== "pixel_edges" ||
      !["copy", "png"].includes(String(r.mode)) || r.inputPreserved !== true || r.photometricFilterWeightsExported !== false ||
      r.waveformResamplingPerformed !== false || ![r.inputSha256, r.preparedSha256, r.inputRasterRgbSha256, r.preparedRasterRgbSha256].every(hash) ||
      !size(r.inputSizeWh) || !size(r.croppedSizeWh) || !size(r.preparedSizeWh) ||
      !Number.isSafeInteger(r.requestedMaxDimension) || Number(r.requestedMaxDimension) < 0) return fail()
  const crop = r.cropBox
  if (crop !== null && (!Array.isArray(crop) || crop.length !== 4 || !crop.every(Number.isSafeInteger) ||
      !(0 <= crop[0] && crop[0] < crop[2] && crop[2] <= r.inputSizeWh[0] && 0 <= crop[1] && crop[1] < crop[3] && crop[3] <= r.inputSizeWh[1]))) return fail()
  const [left, top, right, bottom] = crop ?? [0, 0, ...r.inputSizeWh]
  const croppedSize = r.croppedSizeWh
  if (r.croppedSizeWh[0] !== right - left || r.croppedSizeWh[1] !== bottom - top) return fail()
  const resized = Number(r.requestedMaxDimension) > Math.max(...r.croppedSizeWh)
  if (r.resampling !== (resized ? "pillow_lanczos" : "none") ||
      (!resized && r.preparedSizeWh.some((n, i) => n !== croppedSize[i]))) return fail()
  if (resized) {
    // Validate the observed rounded result without replacing it with a guessed
    // aspect ratio. Pillow's Python round uses ties-to-even.
    const scale = Number(r.requestedMaxDimension) / Math.max(...r.croppedSizeWh)
    for (let i = 0; i < 2; i += 1) {
      const raw = r.croppedSizeWh[i] * scale, lower = Math.floor(raw)
      const rounded = raw - lower === .5 ? (lower % 2 === 0 ? lower : lower + 1) : Math.round(raw)
      if (r.preparedSizeWh[i] !== Math.max(1, rounded)) return fail()
    }
  }
  if (r.mode === "copy" && (crop !== null || r.requestedMaxDimension !== 0 || r.inputSha256 !== r.preparedSha256 || r.inputRasterRgbSha256 !== r.preparedRasterRgbSha256)) return fail()
  const sx = r.preparedSizeWh[0] / r.croppedSizeWh[0], sy = r.preparedSizeWh[1] / r.croppedSizeWh[1]
  const matrix = [[sx, 0, -sx * left], [0, sy, -sy * top], [0, 0, 1]]
  if (!Array.isArray(r.inputToPreparedEdges) || r.inputToPreparedEdges.length !== 3 ||
      r.inputToPreparedEdges.some((row, i) => !Array.isArray(row) || row.length !== 3 || row.some((n, j) => n !== matrix[i][j]))) return fail()
  return value as CandidatePreparationReceipt
}

export function candidatePreparationMatchesRequest(receipt: CandidatePreparationReceipt, target?: number, crop?: RasterCropBox) {
  return receipt.requestedMaxDimension === (target ?? 0) &&
    JSON.stringify(receipt.cropBox) === JSON.stringify(crop ? [crop.left, crop.top, crop.right, crop.bottom] : null)
}

/** This capture contract covers the full four-row reliable path, including rhythm. */
export function canCaptureDecoderCoordinates(candidate: { vectorizer?: string; layoutConstraint?: string; cropBox?: RasterCropBox }) {
  return candidate.vectorizer === "dynamic-path" && candidate.layoutConstraint === "standard_3x4_with_r1" && !candidate.cropBox
}
