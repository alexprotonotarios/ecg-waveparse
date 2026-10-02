import { EvidenceContractError, positivePhysicalValue, unitFraction } from "./physical-contracts"
import { validateSourceRhythmContinuity } from "./source-rhythm-contract"
import { validateSourceSeparatorPublication } from "./source-separator-contract"
import { validateFamilySourceTiming, validateFamilyExclusion } from "./source-family-timing-contract"
import { validateJointSourceTiming } from "./source-joint-timing-contract"
export type { SourceRhythmContinuityEvidence } from "./source-rhythm-contract"
export type { SourceSeparatorPublicationEvidence } from "./source-separator-contract"

export type SourceGridIdentity = {
  sourceRasterFileSha256: string
  calibrationSourceSha256: string
  imageSize: number[]
  annotationMaskFileSha256?: string
}

type LegacySourcePanelTimingEvidence = {
  version: 2 | 3
  method: "source-separator-local-grid-timing-v2" | "source-separator-window-consensus-timing-v3"
  separatorWindowConsensus?: Record<string, unknown>
  state: "source_supported"
  imageSize: number[]
  decodedRasterSha256: string
  sourceIdentity: Omit<SourceGridIdentity, "imageSize"> & {coordinateSpace: "working"}
  panelDurationSeconds: 2.5
  paperSpeedMmPerSecond: number
  panelRanges: number[][]
  rhythmRange: number[]
  rhythmMapping: "four-contiguous-panels-common-baseline-v1"
  truthUsed: false
  physicalCalibration: Record<string, unknown>
  rowGridMeasurements: Record<string, unknown>[]
  sourceSeparators: Record<string, unknown>[]
  sourcePulseEdges: Record<string, unknown>[]
  sourceInkSupport: Record<string, unknown>[]
  finalPanelLocalGrid: Record<string, unknown>
}

export type JointSourcePanelTimingEvidence = Omit<LegacySourcePanelTimingEvidence,
  "version" | "method" | "panelRanges" | "rhythmMapping" | "finalPanelLocalGrid" | "separatorWindowConsensus"> & {
  version: 4
  method: "source-joint-corresponding-grid-row-timing-v4"
  nativeExtractionReady: true
  rowPanelRanges: number[][][]
  rowRecordedRanges: number[][]
  rowPhysicalBoundariesX: number[][]
  rhythmPanelRanges: number[][]
  rhythmPhysicalBoundariesX: number[]
  rhythmMapping: "four-local-grid-quarters-shared-baseline-v2"
  rhythmOrigin: "observed-rhythm-pulse-falling-stroke-right-edge-v1"
  rounding: "nearest-integer-ties-to-even-v1"
  jointGeometry: Record<string, unknown>
  geometryProofSha256: string
  geometryProofEncoding?: "typed-json-finite-f64be-v1"
  sourceGridContext: Record<string, unknown>
  primaryQuarterMeasurements: Record<string, unknown>[][]
  rhythmQuarterMeasurements: Record<string, unknown>[]
  finalPanelMeasurements: Record<string, unknown>[]
  correspondingColumnConsistency: Record<string, unknown>[]
}
export type FamilySourcePanelTimingEvidence = Omit<JointSourcePanelTimingEvidence, "version" | "method" | "sourceSeparators"> & {
  version: 5
  method: "source-endpoint-family-corresponding-grid-row-timing-v5"
  sourceSeparatorFamilies: Record<string, unknown>[]
  allVariantPhysicalTimingExact: true
  variantCount: number
  variantTimingDigests: string[]
}
export type SourcePanelTimingEvidence = LegacySourcePanelTimingEvidence | JointSourcePanelTimingEvidence | FamilySourcePanelTimingEvidence

export type SourceInkEvidence = {
  version: 1 | 2 | 3 | 4 | 5
  method: "neutral-source-ink-with-named-token-exclusions-v2" | "neutral-source-ink-with-observed-calibration-exclusions-v3" | "neutral-source-ink-with-separator-consensus-exclusions-v4" | "neutral-source-ink-with-joint-full-footprint-exclusions-v5" | "neutral-source-ink-with-endpoint-family-parent-exclusions-v6"
  excludedMaskSha256?: string
  sourceSupportSha256?: string
  sourceEvidenceSha256?: string
  imageSize: number[]
  decodedRasterSha256: string
  thresholdMaximumChannelBelow: 160
  supportRadiusPixels: 1
  removedBlackPixels: number
  exclusionBoxes: Record<string, unknown>[]
  truthUsed: false
  gapsPreserved: true
  annotationExclusion?: {
    method: "source-pixel-mask-before-vectorization-v1" | "source-colour-seeds-before-vectorization-v2"
    maskFileSha256: string
    decodedMaskSha256: string
    imageSize: number[]
    coordinateSpace: "working"
    excludedPixels: number
    removedBlackPixels: number
    pixelExclusionApplied: true
    gapsPreserved: true
    inputExcludedPixels?: number
    appliedMaskSha256?: string
    colourRule?: "red-blue-source-pixel-rule-v1"
  }
}

export type SourceTransitionPathEvidence = {
  version: 1
  method: "verified-source-transition-gaps-v1"
  sourceInterval: number[]
  supportRadiusPixels: 1
  minimumDisplacementExclusivePixels: 3
  evidenceThreshold: 0.24
  minimumVerticalSupportFraction: 0.35
  horizontalMarginPixels: 3
  unsupportedTransitions: {
    fromSourcePixel: number[]
    toSourcePixel: number[]
    evidenceWindowBounds: number[]
    verticalSupportFraction: number
    displacementPixels: number
  }[]
  removedSourceColumns: number[]
  pathModified: false
  existingGapsPreserved: true
  truthUsed: false
}

export type SourceTransitionEvidence = {
  version: 1
  method: "verified-source-transition-gaps-v1"
  decodedRasterSha256: string
  imageSize: number[]
  paths: Record<string, SourceTransitionPathEvidence>
}

export type SourceRhythmRefinementEvidence = {
  version: 1
  method: "source-connected-rhythm-refinement-v1"
  decodedRasterSha256: string
  imageSize: number[]
  sourceInterval: number[]
  traceBounds: number[]
  supportThreshold: 0.24
  supportRadiusPixels: 1
  retainedBlurMarginPixels: 1
  transitionScale: 2
  displacementRewardPerPixel: 0.03
  componentCount: number
  seededComponentCount: number
  removedPreferencePixels: number
  pathEncoding: "int32-le-y-by-source-column-v1"
  columnsSha256: string
  originalPathSha256: string
  refinedPathSha256: string
  changedRecordedColumns: number
  originalValidColumns: number
  refinedValidColumns: number
  newlyLostColumns: number[]
  oldGapsRecovered: 0
  sourceEvidenceChanged: false
  primaryAnchorsChanged: false
  sourceTimingChanged: false
  truthUsed: false
}

type SourceGridConversionBase = {
  version: 1
  method: "source-grid-rhythm-coordinate-conversion-v1"
  decodedRasterSha256: string
  imageSize: number[]
  sourceInterval: number[]
  sourcePathChanged: false
  sourceTimingChanged: false
  validityChanged: false
  sourceImageChanged: false
  extrapolationUsed: false
  truthUsed: false
}

export type SourceGridConversionEvidence = SourceGridConversionBase & (
  { state: "unavailable", reason: string } |
  {
    state: "applied"
    grid: Record<string, unknown> & {
      xNodes: number[]
      observedGridY: number[][]
      referenceGridY: number[]
      displacementAtNodes: number[]
      eligibleRowsPerNode: number[]
      observedRowsPerNode: number[]
    }
    mapping: "paper_y = source_y - displacement(source_x)"
    interpolation: "piecewise-linear-no-extrapolation"
    pathEncoding: "float64-le-paper-y-by-source-column-v1"
    columnsSha256: string
    sourcePathSha256: string
    paperPathSha256: string
    validitySha256: string
    mappedSourceColumns: number
    unmappedValidOffCanvasColumns: number
    originalMedianPixels: number
    paperMedianPixels: number
    sourceInverseMaximumErrorPixels: number
  }
)

const fail = (): never => {
  throw new EvidenceContractError("invalid_source_grid_evidence", "Source-grid timing and ink must retain matching source identities, physical support and explicit gaps.")
}
function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return fail()
  return value as Record<string, unknown>
}
function numbers(value: unknown, count: number): number[] {
  if (!Array.isArray(value) || value.length !== count ||
      value.some(v => typeof v !== "number" || !Number.isFinite(v))) return fail()
  return value
}
function objects(value: unknown, count: number): Record<string, unknown>[] {
  if (!Array.isArray(value) || value.length !== count) return fail()
  return value.map(object)
}
function hash(value: unknown) {
  return typeof value === "string" && /^[a-f0-9]{64}$/.test(value)
}
function equal(a: number[], b: number[]) {
  return a.length === b.length && a.every((v, i) => v === b[i])
}

function validateGridPositionMethod(grid: Record<string, unknown>, height: number,
  seedCount: number, nodeCount: number, observedCount: number) {
  const legacy = "unoccluded-major-grid-ridges-v1"
  const fractional = "unoccluded-major-grid-fractional-seed-ridges-v2"
  const chromatic = "unoccluded-binary-support-chromatic-grid-ridges-v1"
  const seedMethod = "bounded-half-height-weighted-source-profile-v1"
  const positionMethod = "positive-source-channel-range-within-unchanged-half-height-support-v1"
  if (![legacy, fractional, chromatic].includes(String(grid.method))) fail()
  const refined = grid.method === fractional || (grid.method === chromatic && grid.priorMethod === fractional)
  if (refined) {
    const seed = object(grid.seedObservation)
    if (grid.seedObservationMethod !== seedMethod || grid.integerSpacingRefused !== true ||
        seed.version !== 1 || seed.method !== seedMethod || seed.integerSpacingRefused !== true ||
        seed.sourceProfileLength !== height || !hash(seed.sourceProfileSha256)) fail()
    const original = numbers(seed.originalRows, seedCount), centers = numbers(seed.refinedRows, seedCount)
    if (!Array.isArray(seed.supportBounds) || seed.supportBounds.length !== seedCount) fail()
    const supports = (seed.supportBounds as unknown[]).map(v => numbers(v, 2))
    const period = Number(grid.majorPeriodPixels)
    for (let i=0;i<seedCount;i++) {
      const [lo,hi] = supports[i]
      if (!Number.isSafeInteger(lo) || !Number.isSafeInteger(hi) || lo<=0 || hi>=height ||
          hi<=lo || hi-lo>.4*period || !Number.isSafeInteger(original[i]) ||
          original[i]<lo || original[i]>=hi || centers[i]<lo || centers[i]>hi-1 ||
          (i>0 && (original[i]<=original[i-1] || centers[i]<=centers[i-1]))) fail()
    }
    const consistent = (rows: number[]) => rows.slice(1).filter((v,i) => {
      const gap=v-rows[i], ratio=gap/period, lower=Math.floor(ratio)
      // Match NumPy's nearest-even rounding in the source observer.
      const multiple=ratio-lower===.5 ? lower+(lower%2) : Math.round(ratio)
      return multiple>=1 && multiple<=4 && Math.abs(gap-multiple*period)<=2
    }).length/(seedCount-1)
    if (consistent(original)>=.9 || consistent(centers)<.9 ||
        !equal(numbers(grid.seedReferenceBoundsY, 2), [centers[0], centers.at(-1)!])) fail()
  } else if (["seedObservation", "seedObservationMethod", "integerSpacingRefused"].some(k=>grid[k]!==undefined)) fail()
  if (grid.method === chromatic) {
    const position = object(grid.ridgePositionEvidence)
    if (![legacy,fractional].includes(String(grid.priorMethod)) || grid.priorState!=="unresolved" ||
        grid.priorReason!=="Rows disagree with shared displacement by more than one pixel." ||
        grid.positionMethod!==positionMethod || grid.binarySupportUnchanged!==true ||
        position.version!==1 || position.method!==positionMethod || position.binarySupportUnchanged!==true ||
        position.sourceWindowWidth!==65 || position.channelRangeMinimum!==18 || position.brightnessMinimum!==150 ||
        position.weightNormalization!=="subtract-local-profile-minimum" ||
        position.observedPositionCount!==observedCount ||
        !equal(numbers(position.supportShape,3),[seedCount,nodeCount,2]) ||
        !["supportBoundsSha256","windowBoundsSha256","sourceProfilesSha256","binaryPositionsSha256",
          "positionsSha256","finiteMaskSha256"].every(k=>hash(position[k]))) fail()
  } else if (["ridgePositionEvidence","positionMethod","binarySupportUnchanged","priorMethod","priorState","priorReason"]
    .some(k=>grid[k]!==undefined)) fail()
}

function validateSourceGridConversion(fidelity: Record<string, unknown>, timing: Record<string, unknown>, size: number[]) {
  const proof = object(fidelity.sourceGridConversion)
  const interval = numbers(proof.sourceInterval, 2), expectedInterval = numbers(timing.rhythmRange, 2)
  if (proof.version !== 1 || proof.method !== "source-grid-rhythm-coordinate-conversion-v1" ||
      proof.decodedRasterSha256 !== timing.decodedRasterSha256 ||
      !equal(numbers(proof.imageSize, 2), size) || !equal(interval, expectedInterval) ||
      ["sourcePathChanged", "sourceTimingChanged", "validityChanged", "sourceImageChanged", "extrapolationUsed", "truthUsed"].some(k => proof[k] !== false)) fail()
  if (proof.state === "unavailable") {
    if (typeof proof.reason !== "string" || !proof.reason.trim() || proof.reason.length > 300 ||
        ["grid", "paperPathSha256", "mappedSourceColumns"].some(k => proof[k] !== undefined)) fail()
    return
  }
  if (proof.state !== "applied" || proof.reason !== undefined ||
      proof.mapping !== "paper_y = source_y - displacement(source_x)" ||
      proof.interpolation !== "piecewise-linear-no-extrapolation" ||
      proof.pathEncoding !== "float64-le-paper-y-by-source-column-v1" ||
      !["columnsSha256", "sourcePathSha256", "paperPathSha256", "validitySha256"].every(k => hash(proof[k]))) fail()
  const width = interval[1]-interval[0]
  const integer = (value: unknown, minimum: number, maximum: number) => {
    if (!Number.isSafeInteger(value) || Number(value) < minimum || Number(value) > maximum) fail()
    return Number(value)
  }
  const mapped = integer(proof.mappedSourceColumns, 1, width)
  integer(proof.unmappedValidOffCanvasColumns, 0, size[0]-mapped)
  for (const key of ["originalMedianPixels", "paperMedianPixels"]) {
    if (typeof proof[key] !== "number" || !Number.isFinite(proof[key]) || Number(proof[key]) < 0 || Number(proof[key]) >= size[1]) fail()
  }
  if (typeof proof.sourceInverseMaximumErrorPixels !== "number" || !Number.isFinite(proof.sourceInverseMaximumErrorPixels) ||
      proof.sourceInverseMaximumErrorPixels < 0 || proof.sourceInverseMaximumErrorPixels > 1e-12) fail()
  if (fidelity.sourceRhythmRefinement !== undefined) {
    const refinement = object(fidelity.sourceRhythmRefinement)
    if (proof.sourcePathSha256 !== refinement.refinedPathSha256 || proof.columnsSha256 !== refinement.columnsSha256 ||
        mapped !== refinement.refinedValidColumns) fail()
  }
  const grid = object(proof.grid)
  const nodeCount = integer(grid.nodeCount, 5, Math.ceil(width/64)+1)
  const seedCount = integer(grid.seedRowCount, 8, size[1])
  const rowCount = integer(grid.completeRowCount, 8, seedCount)
  const x = numbers(grid.xNodes, nodeCount), displacement = numbers(grid.displacementAtNodes, nodeCount)
  const reference = numbers(grid.referenceGridY, rowCount), seedBounds = numbers(grid.seedReferenceBoundsY, 2)
  if (grid.state !== "supported" || grid.reason !== null ||
      grid.candidateSignalOrTruthUsed !== false || grid.maximumAllowedOccludedColumnFraction !== .1 ||
      grid.supportDenominator !== "Seeds not directly observed to be dark-occluded; failed or unavailable ridge observations remain in denominator" ||
      !positivePhysicalValue(grid.majorPeriodPixels) || Math.abs(grid.majorPeriodPixels-5*Number(fidelity.pixelsPerMm)) > 1e-9 ||
      !equal(numbers(grid.boundsX, 2), [0, width-1]) || x[0] !== interval[0] || x.at(-1) !== interval[1]-1 ||
      x.some((v,i) => !Number.isSafeInteger(v) || (i>0 && (v<=x[i-1] || v-x[i-1]>64))) ||
      reference.some((v,i) => v<0 || v>=size[1] || (i>0 && v<=reference[i-1])) ||
      !equal(numbers(grid.referenceBoundsY, 2), [reference[0],reference.at(-1)!]) ||
      seedBounds[0]<0 || seedBounds[1]>=size[1] || seedBounds[1]<=seedBounds[0] ||
      reference.at(-1)!-reference[0] < .75*(seedBounds[1]-seedBounds[0])) fail()
  const observed = numbers(grid.observedRowsPerNode, nodeCount), eligible = numbers(grid.eligibleRowsPerNode, nodeCount)
  for (let j=0;j<nodeCount;j++) {
    integer(eligible[j],rowCount,seedCount); integer(observed[j],rowCount,eligible[j])
  }
  const conditional = Math.min(...observed.map((v,i)=>v/eligible[i]))
  const allSeeds = Math.min(...observed.map(v=>v/seedCount))
  if (conditional<.8 || grid.minimumNodeSupportFraction !== conditional || grid.minimumAllSeedSupportFraction !== allSeeds) fail()
  validateGridPositionMethod(grid, size[1], seedCount, nodeCount, observed.reduce((sum,v)=>sum+v,0))
  if (!Array.isArray(grid.observedGridY) || grid.observedGridY.length !== rowCount) fail()
  const rows = (grid.observedGridY as unknown[]).map(row=>numbers(row,nodeCount))
  let residual = 0
  for (let j=0;j<nodeCount;j++) {
    const delta = rows.map((row,i)=> {
      if (row[j]<0 || row[j]>=size[1] || row[0]!==reference[i] || (i>0 && row[j]<=rows[i-1][j])) fail()
      return row[j]-reference[i]
    }).sort((a,b)=>a-b)
    const median = rowCount%2 ? delta[Math.floor(rowCount/2)] : (delta[rowCount/2-1]+delta[rowCount/2])/2
    if (Math.abs(median-displacement[j])>1e-10 ||
        (j>0 && Math.abs((displacement[j]-displacement[j-1])/(x[j]-x[j-1]))>.25)) fail()
    residual = Math.max(residual,...delta.map(v=>Math.abs(v-displacement[j])))
  }
  if (residual>1 || typeof grid.maxRowResidualPixels!=="number" || !Number.isFinite(grid.maxRowResidualPixels) ||
      Math.abs(grid.maxRowResidualPixels-residual)>1e-10 ||
      grid.endpointDisplacementPixels!==displacement.at(-1)) fail()
}

/** Validate new proofs without changing any pre-existing native report contract. */
export function validateSourceGridEvidence(fidelity: Record<string, unknown>, expected?: SourceGridIdentity) {
  if (fidelity.sourcePanelTiming === undefined && fidelity.sourceInkEvidence === undefined &&
      fidelity.rhythmFidelityDomain === undefined && fidelity.sourceTransitionEvidence === undefined &&
      fidelity.sourceRhythmRefinement === undefined && fidelity.sourceGridConversion === undefined &&
      fidelity.sourceRhythmContinuity === undefined && fidelity.sourceSeparatorPublication === undefined &&
      !["source-label-separator-local-grid-supported-ink-v2", "source-label-separator-local-grid-supported-ink-v3", "source-label-joint-row-grid-supported-ink-v4", "source-label-endpoint-family-row-grid-supported-ink-v5"].includes(String(fidelity.method)) && !expected) return
  const timing = object(fidelity.sourcePanelTiming), ink = object(fidelity.sourceInkEvidence)
  const consensusTiming = timing.version === 3 && timing.method === "source-separator-window-consensus-timing-v3"
  const consensusInk = ink.version === 3 && ink.method === "neutral-source-ink-with-separator-consensus-exclusions-v4"
  const familyTiming = timing.version === 5 && timing.method === "source-endpoint-family-corresponding-grid-row-timing-v5"
  const familyInk = ink.version === 5 && ink.method === "neutral-source-ink-with-endpoint-family-parent-exclusions-v6"
  const jointTiming = familyTiming || (timing.version === 4 && timing.method === "source-joint-corresponding-grid-row-timing-v4")
  const jointInk = familyInk || (ink.version === 4 && ink.method === "neutral-source-ink-with-joint-full-footprint-exclusions-v5")
  const pulseExcluded = jointInk || consensusInk || (ink.version === 2 && ink.method === "neutral-source-ink-with-observed-calibration-exclusions-v3")
  const identity = object(timing.sourceIdentity)
  const size = numbers(timing.imageSize, 2)
  if (size.some(v => !Number.isSafeInteger(v) || v <= 0) ||
      !equal(size, numbers(ink.imageSize, 2)) || !hash(timing.decodedRasterSha256) ||
      ink.decodedRasterSha256 !== timing.decodedRasterSha256 ||
      !hash(identity.sourceRasterFileSha256) || !hash(identity.calibrationSourceSha256) ||
      identity.coordinateSpace !== "working" || consensusTiming !== consensusInk || jointTiming !== jointInk ||
      familyTiming !== familyInk ||
      (jointTiming !== (["source-label-joint-row-grid-supported-ink-v4", "source-label-endpoint-family-row-grid-supported-ink-v5"].includes(String(fidelity.method)))) ||
      (familyTiming !== (fidelity.method === "source-label-endpoint-family-row-grid-supported-ink-v5")) ||
      (!consensusTiming && !jointTiming && (timing.version !== 2 || timing.method !== "source-separator-local-grid-timing-v2" || timing.separatorWindowConsensus !== undefined)) ||
      timing.state !== "source_supported" ||
      timing.panelDurationSeconds !== 2.5 || timing.truthUsed !== false ||
      timing.rhythmMapping !== (jointTiming ? "four-local-grid-quarters-shared-baseline-v2" : "four-contiguous-panels-common-baseline-v1") ||
      !positivePhysicalValue(timing.paperSpeedMmPerSecond) ||
      (!pulseExcluded && !(ink.version === 1 && ink.method === "neutral-source-ink-with-named-token-exclusions-v2")) ||
      ink.thresholdMaximumChannelBelow !== 160 || ink.supportRadiusPixels !== 1 ||
      ink.truthUsed !== false || ink.gapsPreserved !== true ||
      !Number.isSafeInteger(ink.removedBlackPixels) || Number(ink.removedBlackPixels) < 0 ||
      Number(ink.removedBlackPixels) > size[0]*size[1]) fail()
  if (jointInk && !["excludedMaskSha256", "sourceSupportSha256", "sourceEvidenceSha256"].every(k=>hash(ink[k]))) fail()
  if (expected && (!equal(size, expected.imageSize) ||
      identity.sourceRasterFileSha256 !== expected.sourceRasterFileSha256 ||
      identity.calibrationSourceSha256 !== expected.calibrationSourceSha256)) fail()
  if (ink.annotationExclusion !== undefined || identity.annotationMaskFileSha256 !== undefined || expected?.annotationMaskFileSha256 !== undefined) {
    const exclusion = object(ink.annotationExclusion)
    if (!["source-pixel-mask-before-vectorization-v1", "source-colour-seeds-before-vectorization-v2"].includes(String(exclusion.method)) ||
        !hash(exclusion.maskFileSha256) || !hash(exclusion.decodedMaskSha256) ||
        exclusion.maskFileSha256 !== identity.annotationMaskFileSha256 ||
        (expected?.annotationMaskFileSha256 !== undefined && exclusion.maskFileSha256 !== expected.annotationMaskFileSha256) ||
        !equal(size, numbers(exclusion.imageSize, 2)) || exclusion.coordinateSpace !== "working" ||
        exclusion.pixelExclusionApplied !== true || exclusion.gapsPreserved !== true ||
        !Number.isSafeInteger(exclusion.excludedPixels) || Number(exclusion.excludedPixels) < 0 || Number(exclusion.excludedPixels) > size[0]*size[1] ||
        !Number.isSafeInteger(exclusion.removedBlackPixels) || Number(exclusion.removedBlackPixels) < 0 ||
        Number(exclusion.removedBlackPixels) > Number(exclusion.excludedPixels) ||
        Number(exclusion.removedBlackPixels) > Number(ink.removedBlackPixels)) fail()
    if (exclusion.method === "source-colour-seeds-before-vectorization-v2" &&
        (!Number.isSafeInteger(exclusion.inputExcludedPixels) ||
         Number(exclusion.inputExcludedPixels) < Number(exclusion.excludedPixels) ||
         Number(exclusion.inputExcludedPixels) > size[0]*size[1] ||
         !hash(exclusion.appliedMaskSha256) || exclusion.colourRule !== "red-blue-source-pixel-rule-v1")) fail()
  }
  const calibration = object(timing.physicalCalibration)
  const reconciliation = object(calibration.reconciliation)
  if (calibration.detected !== true || !unitFraction(calibration.confidence) || calibration.confidence < .35 ||
      !["pixelsPerMmX", "pixelsPerMmY", "gainMmPerMv"].every(k => positivePhysicalValue(calibration[k])) ||
      calibration.paperSpeedMmPerSecond !== timing.paperSpeedMmPerSecond ||
      reconciliation.state !== "corroborated_inference" || reconciliation.quantitativeBlocked !== false ||
      fidelity.sourcePanelTimingDetected !== true || fidelity.rhythmSourceTimingDetected !== true ||
      object(fidelity.leadLabelValidation).semanticIdentityConfirmed !== true ||
      object(fidelity.rhythmLeadValidation).semanticIdentityConfirmed !== true) fail()
  const bounds = (raw: unknown) => {
    const box = numbers(raw, 4)
    if (!box.every(Number.isSafeInteger) || box[0] < 0 || box[1] < 0 ||
        box[2] <= box[0] || box[3] <= box[1] || box[2] > size[0] || box[3] > size[1]) fail()
    return box
  }
  if (jointTiming) {
    if (familyTiming) validateFamilySourceTiming(timing,size,bounds)
    else validateJointSourceTiming(timing,size,bounds)
    const metrics=object(fidelity.leadMetrics)
    const leadRows=[["I","aVR","V1","V4"],["II","aVL","V2","V5"],["III","aVF","V3","V6"]]
    for (const [row,names] of leadRows.entries()) for (const [column,name] of names.entries()) {
      const metric=object(metrics[name])
      const interval=name==="II" ? numbers(timing.rhythmRange,2)
        : numbers(((timing.rowPanelRanges as unknown[][])[row])[column],2)
      if (!Number.isSafeInteger(metric.sourceStartPixel) || !Number.isSafeInteger(metric.sourceEndPixel) ||
          Number(metric.sourceStartPixel)<interval[0] || Number(metric.sourceEndPixel)>interval[1] ||
          Number(metric.sourceStartPixel)>=Number(metric.sourceEndPixel)) fail()
    }
  }
  const rawRanges = jointTiming ? timing.rhythmPanelRanges : timing.panelRanges
  if (!Array.isArray(rawRanges) || rawRanges.length !== 4) fail()
  const ranges = (rawRanges as unknown[]).map(v => numbers(v, 2))
  if (ranges.some(([a,b], i) => !Number.isSafeInteger(a) || !Number.isSafeInteger(b) ||
      a < 0 || b <= a || b > size[0] || (i > 0 && a !== ranges[i-1][1])) ||
      !equal(numbers(timing.rhythmRange, 2), [ranges[0][0], ranges[3][1]])) fail()
  if (fidelity.rhythmFidelityDomain !== undefined) {
    const domain = object(fidelity.rhythmFidelityDomain)
    const interval = numbers(domain.sourceInterval, 2)
    const rhythm = object(object(fidelity.leadMetrics).II)
    if (domain.method !== "source-confirmed-recorded-interval-v1" ||
        !equal(interval, numbers(timing.rhythmRange, 2)) ||
        domain.expectedColumns !== interval[1] - interval[0] ||
        domain.missingColumnsCountAgainstCoverage !== true ||
        !unitFraction(rhythm.coverage) ||
        !Number.isSafeInteger(rhythm.sourceStartPixel) ||
        !Number.isSafeInteger(rhythm.sourceEndPixel) ||
        Number(rhythm.sourceStartPixel) < interval[0] ||
        Number(rhythm.sourceEndPixel) > interval[1] ||
        Number(rhythm.sourceEndPixel) <= Number(rhythm.sourceStartPixel)) fail()
  }
  if (fidelity.sourceRhythmContinuity !== undefined ||
      ["source-label-separator-local-grid-supported-ink-v2", "source-label-separator-local-grid-supported-ink-v3", "source-label-joint-row-grid-supported-ink-v4", "source-label-endpoint-family-row-grid-supported-ink-v5"].includes(String(fidelity.method))) {
    validateSourceRhythmContinuity(fidelity, timing, size)
  }
  if (fidelity.sourceTransitionEvidence !== undefined || fidelity.method === "source-label-separator-local-grid-supported-ink-v3" || jointTiming) {
    validateSourceTransitions(fidelity.sourceTransitionEvidence, timing, size, bounds)
  }
  if (fidelity.sourceGridConversion !== undefined) {
    validateSourceGridConversion(fidelity, timing, size)
  }
  if (fidelity.sourceRhythmRefinement !== undefined) {
    const refinement = object(fidelity.sourceRhythmRefinement)
    const interval = numbers(refinement.sourceInterval, 2)
    const box = bounds(refinement.traceBounds)
    const count = (key: string, limit: number) => {
      const value = refinement[key]
      if (!Number.isSafeInteger(value) || Number(value) < 0 || Number(value) > limit) fail()
      return Number(value)
    }
    if (refinement.version !== 1 || refinement.method !== "source-connected-rhythm-refinement-v1" ||
        refinement.decodedRasterSha256 !== timing.decodedRasterSha256 ||
        !equal(numbers(refinement.imageSize, 2), size) ||
        !equal(interval, numbers(timing.rhythmRange, 2)) || box[0] > interval[0] || box[2] < interval[1] ||
        refinement.supportThreshold !== .24 || refinement.supportRadiusPixels !== 1 ||
        refinement.retainedBlurMarginPixels !== 1 || refinement.transitionScale !== 2 ||
        refinement.displacementRewardPerPixel !== .03 ||
        refinement.pathEncoding !== "int32-le-y-by-source-column-v1" ||
        !["columnsSha256", "originalPathSha256", "refinedPathSha256"].every(key => hash(refinement[key])) ||
        refinement.oldGapsRecovered !== 0 || refinement.sourceEvidenceChanged !== false ||
        refinement.primaryAnchorsChanged !== false || refinement.sourceTimingChanged !== false ||
        refinement.truthUsed !== false) fail()
    const pixels = (box[2] - box[0]) * (box[3] - box[1])
    const components = count("componentCount", pixels + 1)
    if (components < 1) fail()
    count("seededComponentCount", components - 1)
    count("removedPreferencePixels", pixels)
    count("changedRecordedColumns", interval[1] - interval[0])
    const original = count("originalValidColumns", interval[1] - interval[0])
    const refined = count("refinedValidColumns", original)
    const lost = numbers(refinement.newlyLostColumns, original - refined)
    if (lost.some((value, index) => !Number.isSafeInteger(value) || value < interval[0] || value >= interval[1] ||
        (index > 0 && value <= lost[index - 1]))) fail()
  }
  const gridScale = (measurements: unknown) => {
    const rows = objects(measurements, 4)
    const scales = rows.map(m => {
      if (!positivePhysicalValue(m.periodPixels) || ![1,5].includes(Number(m.periodMm)) ||
          !unitFraction(m.confidence) || m.confidence < .18) fail()
      const scale = Number(m.periodPixels)/Number(m.periodMm)
      if (scale < 1 || scale > 40) fail()
      return scale
    }).sort((a,b) => a-b)
    const median = (scales[1]+scales[2])/2
    if ((scales[3]-scales[0])/median > .02) fail()
    return median
  }
  const scale = gridScale(timing.rowGridMeasurements)
  if (!positivePhysicalValue(timing.gridPixelsPerMmX) ||
      Math.abs(scale-timing.gridPixelsPerMmX) > 1e-9 ||
      Math.abs(scale-Number(calibration.pixelsPerMmX))/scale > .15) fail()
  if (!jointTiming) {
    const local = object(timing.finalPanelLocalGrid), localScale = gridScale(local.measurements)
    if (local.method !== "four-row-local-grid-with-calibrated-speed-v1" ||
        !positivePhysicalValue(local.pixelsPerMm) || Math.abs(localScale-local.pixelsPerMm) > 1e-9 ||
        !positivePhysicalValue(local.physicalEndX) || Math.round(local.physicalEndX) !== ranges[3][1] ||
        !positivePhysicalValue(local.nominalExtrapolatedEndX) ||
        Math.abs(local.physicalEndX-local.nominalExtrapolatedEndX)/
          (local.nominalExtrapolatedEndX-ranges[3][0]) > .0301 ||
        Math.abs((ranges[3][1]-ranges[3][0])-localScale*Number(timing.paperSpeedMmPerSecond)*2.5) > 1) fail()
  }
  const families = familyTiming ? objects(timing.sourceSeparatorFamilies,9) : undefined
  const separators = familyTiming ? [] : objects(timing.sourceSeparators, 9)
  separators.forEach((mark, i) => {
    const box = bounds(mark.bounds)
    if (mark.row !== Math.floor(i/3) || mark.column !== 1+i%3 ||
        !unitFraction(mark.minimumBlackColumnSupport) || mark.minimumBlackColumnSupport < .8 ||
        !positivePhysicalValue(mark.centerX) || mark.centerX < box[0] || mark.centerX >= box[2]) fail()
  })
  const pulses = objects(timing.sourcePulseEdges, 4)
  pulses.forEach((mark,i) => {
    const box = bounds(mark.bounds)
    if (mark.row !== i || !unitFraction(mark.minimumBlackColumnSupport) || mark.minimumBlackColumnSupport < .65 ||
        !positivePhysicalValue(mark.centerX) || mark.centerX < box[0] || mark.centerX >= box[2]) fail()
  })
  const consensusSeparators = consensusTiming
    ? validateSeparatorConsensus(timing, size, bounds)
    : undefined
  const pulseRowBounds: number[] = []
  if (pulseExcluded) {
    const centers = numbers(objects(timing.rowGridMeasurements, 4).map(m => m.rowCenter), 4)
    if (centers[0] < 0 || centers[3] >= size[1] || centers.some((v,i) => i > 0 && v <= centers[i-1])) fail()
    pulseRowBounds.push(0, ...centers.slice(0,3).map((v,i) => Math.ceil((v+centers[i+1])/2)), size[1])
  }
  objects(timing.sourceInkSupport, 16).forEach((m,i) => {
    if (m.row !== Math.floor(i/4) || m.column !== i%4 || !unitFraction(m.fraction) || m.fraction < .2) fail()
  })
  const names = ["I","aVR","V1","V4","II","aVL","V2","V5","III","aVF","V3","V6","II"]
  objects(ink.exclusionBoxes, pulseExcluded ? 26 : 22).forEach((entry,i) => {
    const box = bounds(entry.bounds)
    if (i < 13) {
      const original = bounds(entry.originalVerificationCrop)
      if (entry.kind !== "verified-label-crop" || entry.lead !== names[i] ||
          typeof entry.tightenedToObservedNamedOcrBox !== "boolean" ||
          box[0] < original[0] || box[1] < original[1] || box[2] > original[2] || box[3] > original[3]) fail()
      if (entry.tightenedToObservedNamedOcrBox) {
        const anchor = object(entry.sourceAnchor)
        if (anchor.lead !== entry.lead || !unitFraction(anchor.confidence) || anchor.confidence < .65 ||
            !["x","y","width","height"].every(k => typeof anchor[k] === "number" && Number.isFinite(anchor[k])) ||
            !positivePhysicalValue(anchor.width) || !positivePhysicalValue(anchor.height)) fail()
        const expectedBox = [Math.max(original[0],Math.floor(Number(anchor.x))-2),
          Math.max(original[1],Math.floor(Number(anchor.y)-Number(anchor.height)/2)-2),
          Math.min(original[2],Math.ceil(Number(anchor.x)+Number(anchor.width))+2),
          Math.min(original[3],Math.ceil(Number(anchor.y)+Number(anchor.height)/2)+2)]
        if (!equal(box, expectedBox)) fail()
      } else if (!equal(box, original)) fail()
    } else if (i < 22) {
      if (families) {
        validateFamilyExclusion(entry,families[i-13],size,bounds)
        return
      }
      const original = bounds(separators[i-13].bounds)
      const group = consensusSeparators?.[i-13]
      if (group) {
        const union = bounds(group.unionBounds)
        if (entry.kind !== "observed-separator-consensus-union" ||
            !equal(bounds(entry.representativeBounds), original) ||
            entry.sourceObservationCount !== (group.observations as unknown[]).length ||
            !equal(numbers(entry.centerIntervalPixels, 2), numbers(group.centerIntervalPixels, 2)) ||
            !equal(box, [Math.max(0,union[0]-1),union[1],Math.min(size[0],union[2]+1),union[3]])) fail()
      } else if (jointTiming) {
        const footprint=bounds(separators[i-13].observedBounds)
        if(entry.kind!=="observed-separator-joint-full-footprint" ||
            !equal(bounds(entry.representativeBounds),original) || !equal(bounds(entry.observedBounds),footprint) ||
            !equal(numbers(entry.centerIntervalPixels,2),numbers(separators[i-13].centerIntervalPixels,2)) ||
            !equal(box,[Math.max(0,footprint[0]-1),footprint[1],Math.min(size[0],footprint[2]+1),footprint[3]])) fail()
      } else if (entry.kind !== "observed-separator" || !equal(box,
        [Math.max(0,original[0]-1),original[1],Math.min(size[0],original[2]+1),original[3]])) fail()
    } else {
      const row = i-22, pulse = pulses[row], core = bounds(pulse.bounds)
      const original=jointTiming?bounds(pulse.observedBounds):core
      if (entry.kind !== (jointTiming?"observed-calibration-full-edge-columns":"observed-calibration-edge-columns") || entry.row !== row ||
          (jointTiming && !equal(bounds(entry.coreBounds),core)) ||
          !equal(bounds(entry.observedBounds), original) || entry.observedCenterX !== pulse.centerX ||
          pulse.centerX !== (core[0]+core[2]-1)/2 ||
          entry.minimumBlackColumnSupport !== pulse.minimumBlackColumnSupport ||
          original[1] < pulseRowBounds[row] || original[3] > pulseRowBounds[row+1] ||
          entry.horizontalMarginPixels !== 1 || entry.verticalExtent !== "adjacent-source-row-midpoints" ||
          entry.reason !== "calibration-edge-is-not-waveform" || entry.originalTimeCoordinatesPreserved !== true ||
          !equal(box, [Math.max(0,original[0]-1),pulseRowBounds[row],
            Math.min(size[0],original[2]+1),pulseRowBounds[row+1]])) fail()
    }
  })
  validateSourceSeparatorPublication(fidelity, timing, ink, size)
}

function validateSourceTransitions(
  value: unknown, timing: Record<string, unknown>, size: number[], bounds: (value: unknown) => number[]
) {
  const receipt = object(value), paths = object(receipt.paths)
  const names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6","rhythm II"]
  if (receipt.version !== 1 || receipt.method !== "verified-source-transition-gaps-v1" ||
      receipt.decodedRasterSha256 !== timing.decodedRasterSha256 ||
      !equal(numbers(receipt.imageSize, 2), size) || Object.keys(paths).length !== names.length ||
      names.some(name => !Object.hasOwn(paths, name))) fail()
  const rows: Record<string,number> = {I:0,aVR:0,V1:0,V4:0,II:1,aVL:1,V2:1,V5:1,III:2,aVF:2,V3:2,V6:2}
  for (const name of names) {
    const interval = (timing.version===4 || timing.version===5) && name!=="rhythm II"
      ? numbers((timing.rowRecordedRanges as unknown[])[rows[name]],2)
      : numbers(timing.rhythmRange,2)
    const [lo,hi]=interval
    const proof = object(paths[name])
    if (proof.version !== 1 || proof.method !== receipt.method ||
        !equal(numbers(proof.sourceInterval, 2), interval) || proof.supportRadiusPixels !== 1 ||
        proof.minimumDisplacementExclusivePixels !== 3 || proof.evidenceThreshold !== .24 ||
        proof.minimumVerticalSupportFraction !== .35 || proof.horizontalMarginPixels !== 3 ||
        proof.pathModified !== false || proof.existingGapsPreserved !== true || proof.truthUsed !== false ||
        !Array.isArray(proof.unsupportedTransitions) || !Array.isArray(proof.removedSourceColumns)) fail()
    const removed = new Set<number>()
    let previousColumn = -1
    for (const raw of proof.unsupportedTransitions as unknown[]) {
      const transition = object(raw)
      const from = numbers(transition.fromSourcePixel, 2), to = numbers(transition.toSourcePixel, 2)
      const support = transition.verticalSupportFraction, jump = Math.abs(to[1] - from[1])
      if ([...from, ...to].some(v => !Number.isSafeInteger(v)) || from[0] < lo || to[0] >= hi ||
          to[0] !== from[0] + 1 || from[0] <= previousColumn ||
          from[1] < 0 || to[1] < 0 || from[1] >= size[1] || to[1] >= size[1] ||
          jump <= 3 || transition.displacementPixels !== jump || !unitFraction(support) || support >= .35 ||
          !equal(bounds(transition.evidenceWindowBounds), [Math.max(lo, from[0]-3), Math.min(from[1],to[1]),
            Math.min(hi, to[0]+4), Math.max(from[1],to[1])+1])) fail()
      previousColumn = from[0]
      removed.add(from[0]); removed.add(to[0])
    }
    const columns = numbers(proof.removedSourceColumns, (proof.removedSourceColumns as unknown[]).length)
    if (!equal(columns, [...removed].sort((a,b) => a-b))) fail()
  }
}

function sameEvidence(a: unknown, b: unknown): boolean {
  if (a === b) return true
  if (Array.isArray(a) || Array.isArray(b)) {
    return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v,i) => sameEvidence(v,b[i]))
  }
  if (!a || !b || typeof a !== "object" || typeof b !== "object") return false
  const left = a as Record<string, unknown>, right = b as Record<string, unknown>
  const keys = Object.keys(left)
  return keys.length === Object.keys(right).length && keys.every(k => Object.hasOwn(right,k) && sameEvidence(left[k],right[k]))
}

// Python's source-window rounding uses ties to even, including half-pixel edges.
function roundSourcePixel(value: number) {
  const floor = Math.floor(value), fraction = value-floor
  return fraction > .5 || (fraction === .5 && floor % 2 !== 0) ? floor+1 : floor
}

function validateSeparatorConsensus(
  timing: Record<string, unknown>, size: number[], bounds: (value: unknown) => number[]
): Record<string, unknown>[] {
  const receipt = object(timing.separatorWindowConsensus)
  const radius = Number(receipt.radiusPixels), font = Number(receipt.fontHeightPixels)
  if (receipt.version !== 1 || receipt.method !== "bounded-source-separator-window-consensus-v1" ||
      !Number.isSafeInteger(receipt.radiusPixels) || radius < 1 || radius > 64 ||
      !positivePhysicalValue(receipt.fontHeightPixels) || radius !== roundSourcePixel(font*.25) ||
      receipt.windowRadiusFraction !== .25 || receipt.truthUsed !== false ||
      receipt.selectionRule !== "minimum-absolute-offset-negative-first" ||
      receipt.allPublishedTimingAndPulsesIdentical !== true || receipt.maximumCenterIntervalPixels !== 1) fail()
  const offsets = numbers(receipt.offsetsAttempted, radius*2+1)
  if (offsets.some((v,i) => v !== i-radius) || !Array.isArray(receipt.successfulOffsets)) fail()
  const successful = numbers(receipt.successfulOffsets, (receipt.successfulOffsets as unknown[]).length)
  if (successful.length < 2 || successful.some((v,i) => !Number.isSafeInteger(v) || v < -radius || v > radius ||
      v === 0 || (i > 0 && v <= successful[i-1])) || !successful.some((v,i) => i > 0 && v-successful[i-1] === 1)) fail()
  const selected = [...successful].sort((a,b) => Math.abs(a)-Math.abs(b) || a-b)[0]
  if (receipt.selectedOffsetPixels !== selected) fail()
  const attempts = objects(receipt.attempts, offsets.length)
  attempts.forEach((attempt,i) => {
    const accepted = successful.includes(offsets[i]), counts = numbers(attempt.windowCandidateCounts, 9)
    if (attempt.offsetPixels !== offsets[i] || counts.some(v => v !== 0 && v !== 1) ||
        attempt.state !== (accepted ? "source_supported" : "unresolved") ||
        (accepted ? attempt.failureReason !== null || counts.some(v => v !== 1)
          : typeof attempt.failureReason !== "string" || attempt.failureReason.length === 0)) fail()
  })
  if (attempts[radius].failureReason !== "source-separator-missing-or-ambiguous") fail()
  const proposals = objects(receipt.timingProposals, successful.length)
  const boundaryProposals = proposals.map((proposal,i) => {
    if (proposal.offsetPixels !== successful[i] || !sameEvidence(proposal.panelRanges,timing.panelRanges) ||
        !sameEvidence(proposal.rhythmRange,timing.rhythmRange) || !sameEvidence(proposal.sourcePulseEdges,timing.sourcePulseEdges)) fail()
    if (!Array.isArray(proposal.rowBoundariesX) || proposal.rowBoundariesX.length !== 3) fail()
    const rows = (proposal.rowBoundariesX as unknown[]).map(v => numbers(v,5))
    if (successful[i] === selected && !sameEvidence(rows,timing.rowBoundariesX)) fail()
    return rows
  })
  if (!Array.isArray(receipt.rowBoundaryIntervalsPixels) || receipt.rowBoundaryIntervalsPixels.length !== 3) fail()
  ;(receipt.rowBoundaryIntervalsPixels as unknown[]).forEach((row,r) => {
    if (!Array.isArray(row) || row.length !== 5) fail()
    ;(row as unknown[]).forEach((interval,c) => {
      const values = boundaryProposals.map(p => p[r][c])
      if (!equal(numbers(interval,2), [Math.min(...values),Math.max(...values)])) fail()
    })
  })
  const sourceRows = numbers(objects(timing.rowGridMeasurements,4).map(m => m.rowCenter),4)
  const representatives = objects(timing.sourceSeparators,9)
  const groups = objects(receipt.sourceSeparators,9)
  groups.forEach((group,i) => {
    if (group.row !== Math.floor(i/3) || group.column !== i%3+1) fail()
    const observations = objects(group.observations,successful.length)
    const rectangles = observations.map((observation,j) => {
      const box = bounds(observation.bounds), offset = successful[j], center = sourceRows[Math.floor(i/3)]
      if (observation.offsetPixels !== offset || observation.row !== group.row || observation.column !== group.column ||
          observation.centerX !== (box[0]+box[2]-1)/2 ||
          !unitFraction(observation.minimumBlackColumnSupport) || observation.minimumBlackColumnSupport < .8 ||
          box[2]-box[0] < Math.max(3,roundSourcePixel(font*.08)) ||
          box[2]-box[0] > Math.max(4,roundSourcePixel(font*.30)) ||
          box[1] !== Math.max(0,roundSourcePixel(center-font*.8)+offset) ||
          box[3] !== Math.min(size[1],roundSourcePixel(center+font*.55)+offset)) fail()
      if (offset === selected && !sameEvidence(
        Object.fromEntries(Object.entries(observation).filter(([key]) => key !== "offsetPixels")), representatives[i])) fail()
      return box
    })
    const centers = observations.map(m => Number(m.centerX))
    const interval = [Math.min(...centers),Math.max(...centers)]
    const intersection = [Math.max(...rectangles.map(b => b[0])),Math.min(...rectangles.map(b => b[2]))]
    const union = [Math.min(...rectangles.map(b => b[0])),Math.min(...rectangles.map(b => b[1])),
      Math.max(...rectangles.map(b => b[2])),Math.max(...rectangles.map(b => b[3]))]
    if (interval[1]-interval[0] > 1 || !equal(numbers(group.centerIntervalPixels,2),interval) ||
        intersection[1] <= intersection[0] || !equal(numbers(group.horizontalIntersection,2),intersection) ||
        !equal(bounds(group.unionBounds),union)) fail()
  })
  return groups
}
