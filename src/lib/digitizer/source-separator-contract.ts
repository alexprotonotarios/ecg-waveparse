import { isDeepStrictEqual } from "node:util"
import { EvidenceContractError } from "./physical-contracts"

export type SourceSeparatorPathEvidence = {
  row: number
  priorValidityRule: "verified-source-transition-gaps-v1"
  pathEncoding: "int32-le-y-by-source-column-v1"
  validityEncoding: "uint8-by-source-column-v1"
  columnsSha256: string
  sourcePathSha256: string
  priorValiditySha256: string
  publishedValiditySha256: string
  sourceColumnCount: number
  priorValidColumnCount: number
  publishedValidColumnCount: number
  removedSourcePixels: number[][]
  oldGapsRecovered: 0
  sourcePathChanged: false
  truthUsed: false
}

export type SourceSeparatorPublicationEvidence = {
  version: 1
  method: "observed-separator-column-publication-gaps-v1"
  decodedRasterSha256: string
  imageSize: number[]
  sourceTimingVersion: number
  sourceTimingMethod: string
  rowCenters: number[]
  rowBounds: number[]
  sourceSeparators: {
    row: number
    column: number
    observedMark: Record<string, unknown>
    existingExclusion: Record<string, unknown>
    publicationBounds: number[]
  }[]
  horizontalMarginPixels: 1
  verticalExtent: "adjacent-source-row-midpoints"
  publicationValidityOnly: true
  oldConnectedSourceValidityAppliedFirst: true
  sourceSupportChanged: false
  sourceImageChanged: false
  sourceEvidenceChanged: false
  sourcePathChanged: false
  timeCoordinatesChanged: false
  oldGapsRecovered: 0
  sourceSupportSha256: string
  sourceSupportOutsideMaskSha256: string
  publicationMaskSha256: string
  maskPixelCount: number
  newlyExcludedSupportedPixelCount: number
  separateRhythmRowUnchanged: true
  truthUsed: false
  paths: Record<string, SourceSeparatorPathEvidence>
}

const nativeMethods = ["source-label-separator-local-grid-supported-ink-v3", "source-label-joint-row-grid-supported-ink-v4", "source-label-endpoint-family-row-grid-supported-ink-v5"]
const fail = (): never => {
  throw new EvidenceContractError("invalid_source_grid_evidence", "Separator publication gaps require matching source marks, unchanged paths and complete validity evidence.")
}
function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return fail()
  return value as Record<string, unknown>
}
function numbers(value: unknown, length: number): number[] {
  if (!Array.isArray(value) || value.length !== length ||
      value.some(x => typeof x !== "number" || !Number.isFinite(x))) return fail()
  return value
}
function integer(value: unknown, low: number, high: number): number {
  if (!Number.isSafeInteger(value) || Number(value) < low || Number(value) > high) return fail()
  return Number(value)
}
function hash(value: unknown) {
  return typeof value === "string" && /^[a-f0-9]{64}$/.test(value)
}

/** Called after the original timing, ink and transition contracts are validated. */
export function validateSourceSeparatorPublication(
  fidelity: Record<string, unknown>, timing: Record<string, unknown>,
  ink: Record<string, unknown>, size: number[]
) {
  if (fidelity.sourceSeparatorPublication === undefined && !nativeMethods.includes(String(fidelity.method))) return
  const proof = object(fidelity.sourceSeparatorPublication)
  if (!nativeMethods.includes(String(fidelity.method)) || proof.version !== 1 ||
      proof.method !== "observed-separator-column-publication-gaps-v1" ||
      proof.decodedRasterSha256 !== timing.decodedRasterSha256 ||
      !isDeepStrictEqual(numbers(proof.imageSize, 2), size) ||
      proof.sourceTimingVersion !== timing.version || proof.sourceTimingMethod !== timing.method ||
      proof.horizontalMarginPixels !== 1 || proof.verticalExtent !== "adjacent-source-row-midpoints" ||
      proof.publicationValidityOnly !== true || proof.oldConnectedSourceValidityAppliedFirst !== true ||
      proof.separateRhythmRowUnchanged !== true || proof.oldGapsRecovered !== 0 ||
      ["sourceSupportChanged", "sourceImageChanged", "sourceEvidenceChanged", "sourcePathChanged", "timeCoordinatesChanged", "truthUsed"].some(k => proof[k] !== false) ||
      !["sourceSupportSha256", "sourceSupportOutsideMaskSha256", "publicationMaskSha256"].every(k => hash(proof[k]))) fail()
  if((timing.version===4 || timing.version===5) && proof.sourceSupportSha256!==ink.sourceSupportSha256) fail()
  const centers = numbers(proof.rowCenters, 4)
  const measurements = timing.rowGridMeasurements as Record<string, unknown>[]
  if (centers.some((y, i) => y < 0 || y >= size[1] || y !== measurements[i].rowCenter || (i > 0 && y <= centers[i - 1]))) fail()
  const rowBounds = [0, ...centers.slice(0, 3).map((y, i) => Math.ceil((y + centers[i + 1]) / 2)), size[1]]
  if (!isDeepStrictEqual(numbers(proof.rowBounds, 5), rowBounds) ||
      !Array.isArray(proof.sourceSeparators) || proof.sourceSeparators.length !== 9) fail()
  const family = timing.version===5
  const marks = (family ? timing.sourceSeparatorFamilies : timing.sourceSeparators) as Record<string, unknown>[]
  const exclusions = ink.exclusionBoxes as Record<string, unknown>[]
  let pixels = 0
  const bands = (proof.sourceSeparators as unknown[]).map((value, i) => {
    const entry = object(value), row = Math.floor(i / 3)
    const mark = marks[i], exclusion = exclusions[i + 13]
    const bounds = numbers(exclusion.bounds, 4), observed = numbers(family ? object(mark.parentComponent).bounds : mark.bounds, 4)
    const band = [bounds[0], rowBounds[row], bounds[2], rowBounds[row + 1]]
    if (entry.row !== row || entry.column !== i % 3 + 1 ||
        !isDeepStrictEqual(entry.observedMark, mark) || !isDeepStrictEqual(entry.existingExclusion, exclusion) ||
        !isDeepStrictEqual(numbers(entry.publicationBounds, 4), band) ||
        bounds[1] < rowBounds[row] || bounds[3] > rowBounds[row + 1] ||
        observed[1] < rowBounds[row] || observed[3] > rowBounds[row + 1] ||
        (!family && mark.centerX !== (observed[0] + observed[2] - 1) / 2) ||
        (i % 3 > 0 && bounds[0] < numbers(exclusions[i + 12].bounds, 4)[2])) fail()
    pixels += (band[2] - band[0]) * (band[3] - band[1])
    return band
  })
  if (proof.maskPixelCount !== pixels) fail()
  const excluded = integer(proof.newlyExcludedSupportedPixelCount, 0, pixels)
  if ((excluded === 0) !== (proof.sourceSupportSha256 === proof.sourceSupportOutsideMaskSha256)) fail()

  const paths = object(proof.paths), transitions = object(object(fidelity.sourceTransitionEvidence).paths)
  const rows: Record<string, number> = {I: 0, aVR: 0, V1: 0, V4: 0, II: 1, aVL: 1, V2: 1, V5: 1, III: 2, aVF: 2, V3: 2, V6: 2, "rhythm II": 3}
  if (Object.keys(paths).length !== 13 || Object.keys(rows).some(name => !Object.hasOwn(paths, name))) fail()
  for (const [name, row] of Object.entries(rows)) {
    const path = object(paths[name]), transition = object(transitions[name])
    const count = integer(path.sourceColumnCount, 1, size[0])
    const prior = integer(path.priorValidColumnCount, 0, count)
    const published = integer(path.publishedValidColumnCount, 0, prior)
    if (path.row !== row || path.priorValidityRule !== transition.method ||
        path.pathEncoding !== "int32-le-y-by-source-column-v1" || path.validityEncoding !== "uint8-by-source-column-v1" ||
        path.oldGapsRecovered !== 0 || path.sourcePathChanged !== false || path.truthUsed !== false ||
        !["columnsSha256", "sourcePathSha256", "priorValiditySha256", "publishedValiditySha256"].every(k => hash(path[k])) ||
        !Array.isArray(path.removedSourcePixels) || path.removedSourcePixels.length !== prior - published ||
        (prior === published) !== (path.priorValiditySha256 === path.publishedValiditySha256)) fail()
    let previous = -1
    for (const pixel of path.removedSourcePixels as unknown[]) {
      const [x, y] = numbers(pixel, 2)
      integer(x, 0, size[0] - 1); integer(y, rowBounds[row], rowBounds[row + 1] - 1)
      if (x <= previous || row === 3 ||
          !bands.slice(row * 3, row * 3 + 3).some(b => x >= b[0] && x < b[2]) ||
          (transition.removedSourceColumns as number[]).includes(x)) fail()
      previous = x
    }
    if (row === 3) {
      const rhythm = object(fidelity.sourceRhythmContinuity)
      if (prior !== published || path.columnsSha256 !== rhythm.columnsSha256 ||
          path.sourcePathSha256 !== rhythm.sourcePathSha256 ||
          path.publishedValiditySha256 !== rhythm.validitySha256) fail()
    }
  }
}
