import { EvidenceContractError, unitFraction } from "@/lib/digitizer/physical-contracts"

export type PrintedSettingsReport = {
  version: 1
  state: "unresolved" | "recognized" | "conflict"
  method: "local-setting-token-ocr-v1"
  coordinateSpace: "original"
  sourceSize: { width: number; height: number }
  sourceSha256?: string
  confidenceMinimum: number
  engine?: string
  reason?: string
  values: { speed: number[]; gain: number[] }
  observations: Array<{
    kind: "speed" | "gain"; value: number; units: "mm/s" | "mm/mV"; confidence: number
    sourceBox: { left: number; top: number; right: number; bottom: number }
  }>
  recognizerFallback?: {
    version: 1
    method: "agreement-gated-missing-setting-v1"
    outcome: "completed" | "conflict" | "no_agreed_completion"
    primary: Omit<PrintedSettingsReport, "recognizerFallback">
    secondary: Omit<PrintedSettingsReport, "recognizerFallback">
  }
}
export type CalibrationReconciliation = {
  version: 1 | 2
  state: "unsupported" | "conflict" | "corroborated_inference" | "partially_corroborated" | "unresolved"
  reasons: string[]
  compared: Array<"speed" | "gain">
  quantitativeBlocked: boolean
  pulseAssumptions: { durationSeconds: 0.2 | null; amplitudeMv: 1 }
  acquisitionSampleRateHz: null
}
export type PrintedSpeedResolution = {
  version: 1
  method: "printed-speed-with-row-grid-v1"
  previousInference: {
    method: "grid-period-plus-rectangular-pulse-v1"
    paperSpeedMmPerSecond: 25
    assumedPulseDurationSeconds: 0.2
  }
  printedSpeedMmPerSecond: 50
  derivedPulseDurationSeconds: number
}

const fail = (): never => { throw new EvidenceContractError("invalid_calibration_reconciliation", "Printed calibration evidence is malformed or disagrees with its physical-unit decision.") }
function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return fail()
  return value as Record<string, unknown>
}
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value)

/** Recheck the independent grid evidence; a new method string grants no authority. */
function validateSpeedResolution(calibration: Record<string, unknown>, values: { speed: Set<number>; gain: Set<number> }) {
  const resolution = record(calibration.speedResolution)
  const previous = record(resolution.previousInference)
  const rows = calibration.rowPixelsPerMmX
  const evidence = record(calibration.horizontalScaleEvidence)
  const gain = calibration.gainMmPerMv
  if (resolution.version !== 1 || resolution.method !== "printed-speed-with-row-grid-v1" ||
      calibration.method !== resolution.method || previous.method !== "grid-period-plus-rectangular-pulse-v1" ||
      previous.paperSpeedMmPerSecond !== 25 || previous.assumedPulseDurationSeconds !== .2 ||
      resolution.printedSpeedMmPerSecond !== 50 || calibration.paperSpeedMmPerSecond !== 50 ||
      calibration.detected !== true || calibration.gridScaleDetected !== true || calibration.gridScaleAmbiguous !== false ||
      calibration.gridScaleMmX !== 1 || calibration.gridScaleMmY !== 1 ||
      values.speed.size !== 1 || !values.speed.has(50) ||
      !finite(gain) || ![5,10,20].includes(gain) || values.gain.size !== 1 || !values.gain.has(gain) ||
      !finite(calibration.confidence) || calibration.confidence < .35 || calibration.confidence > 1 ||
      !Array.isArray(rows) || ![6,12].includes(rows.length) || !rows.every(v => finite(v) && v >= 1 && v <= 40)) fail()
  const ordered = (rows as number[]).slice().sort((a,b) => a-b)
  const center = (ordered[Math.floor((ordered.length-1)/2)] + ordered[Math.floor(ordered.length/2)])/2
  const spread = (ordered.at(-1)!-ordered[0])/center
  const x = calibration.pixelsPerMmX, y = calibration.pixelsPerMmY, prior = evidence.priorPixelsPerMmX
  const gx = calibration.gridSpacingXPixels, gy = calibration.gridSpacingYPixels
  const measuredGain = calibration.measuredGainMmPerMv, width = calibration.pulseWidthMm
  if (evidence.version !== 1 || evidence.method !== "consistent-row-grid-median-v1" || evidence.rowCount !== ordered.length ||
      !finite(evidence.relativeRange) || Math.abs(evidence.relativeRange-spread) > 1e-9 || spread > .02 ||
      !finite(prior) || prior <= 0 || Math.abs(center-prior)/Math.max(center,prior) > .15 ||
      !finite(x) || Math.abs(x-center) > 1e-9 || !finite(y) || y <= 0 || Math.abs(x-y)/Math.max(x,y) > .05 ||
      !finite(gx) || Math.abs(gx-x)/x > .05 || !finite(gy) || Math.abs(gy-y)/y > .05 ||
      !finite(measuredGain) || Math.abs(measuredGain-Number(gain))/Number(gain) > .1 ||
      !finite(width) || width < 4.5 || width > 5.5 ||
      !finite(resolution.derivedPulseDurationSeconds) || Math.abs(resolution.derivedPulseDurationSeconds-width/50) > 1e-12) fail()
}

/** Recompute cross-field decisions; a serialized flag cannot hide a conflict. */
export function validatePrintedCalibration(calibration: Record<string, unknown>) {
  if (calibration.printedSettings === undefined && calibration.reconciliation === undefined &&
      calibration.speedResolution === undefined && calibration.method !== "printed-speed-with-row-grid-v1") return
  const printed = record(calibration.printedSettings), reconciliation = record(calibration.reconciliation)
  const size = record(printed.sourceSize), values = record(printed.values)
  if (printed.version !== 1 || printed.method !== "local-setting-token-ocr-v1" ||
      printed.coordinateSpace !== "original" || printed.confidenceMinimum !== .85 ||
      !finite(size.width) || !Number.isInteger(size.width) || size.width <= 0 ||
      !finite(size.height) || !Number.isInteger(size.height) || size.height <= 0 ||
      (printed.sourceSha256 !== undefined && (typeof printed.sourceSha256 !== "string" || !/^[a-f0-9]{64}$/.test(printed.sourceSha256))) ||
      !Array.isArray(printed.observations)) fail()
  const derived = { speed: new Set<number>(), gain: new Set<number>() }
  for (const raw of printed.observations as unknown[]) {
    const observation = record(raw), box = record(observation.sourceBox)
    if ((observation.kind !== "speed" && observation.kind !== "gain") || !finite(observation.value) ||
        !unitFraction(observation.confidence) || observation.confidence < .85 ||
        observation.units !== (observation.kind === "speed" ? "mm/s" : "mm/mV") ||
        ![box.left, box.top, box.right, box.bottom].every(finite) ||
        Number(box.left) < 0 || Number(box.top) < 0 || Number(box.right) > Number(size.width) || Number(box.bottom) > Number(size.height) ||
        Number(box.right) <= Number(box.left) || Number(box.bottom) <= Number(box.top)) fail()
    derived[observation.kind as "speed" | "gain"].add(observation.value as number)
  }
  for (const kind of ["speed", "gain"] as const) {
    if (!Array.isArray(values[kind]) || !(values[kind] as unknown[]).every(finite) ||
        JSON.stringify(values[kind]) !== JSON.stringify([...derived[kind]].sort((a,b) => a-b))) fail()
  }
  const conflict = derived.speed.size > 1 || derived.gain.size > 1
  const printedState = conflict ? "conflict" : derived.speed.size || derived.gain.size ? "recognized" : "unresolved"
  if (printed.state !== printedState) fail()
  const resolvedSpeed = calibration.speedResolution !== undefined
  if (resolvedSpeed) validateSpeedResolution(calibration, derived)
  else if (calibration.method === "printed-speed-with-row-grid-v1") fail()
  const reasons: string[] = conflict ? ["conflicting-printed-settings"] : []
  const unsupported = [...derived.speed].some(v => ![25,50].includes(v)) || [...derived.gain].some(v => ![5,10,20].includes(v))
  if (unsupported) reasons.push("unsupported-printed-setting")
  const compared: string[] = []
  if (calibration.detected === true) for (const [kind,key] of [["speed","paperSpeedMmPerSecond"],["gain","gainMmPerMv"]] as const) {
    if (derived[kind].size === 1) {
      compared.push(kind)
      const value = [...derived[kind]][0], inferred = Number(calibration[key])
      if (!Number.isFinite(inferred) || Math.abs(value-inferred) > .01*Math.max(Math.abs(value),Math.abs(inferred))) reasons.push(`printed-${kind}-disagrees-with-pulse-grid`)
    }
  }
  const state = unsupported ? "unsupported" : reasons.length ? "conflict" : compared.length === 2 ? "corroborated_inference" : compared.length ? "partially_corroborated" : "unresolved"
  const assumptions = record(reconciliation.pulseAssumptions)
  if (reconciliation.version !== (resolvedSpeed ? 2 : 1) || reconciliation.state !== state || reconciliation.quantitativeBlocked !== Boolean(reasons.length) ||
      JSON.stringify(reconciliation.reasons) !== JSON.stringify(reasons) || JSON.stringify(reconciliation.compared) !== JSON.stringify(compared) ||
      assumptions.durationSeconds !== (resolvedSpeed ? null : .2) || assumptions.amplitudeMv !== 1 || reconciliation.acquisitionSampleRateHz !== null) fail()
}

type PhysicalCalibration = {
  detected?: boolean; confidence?: number; paperSpeedMmPerSecond?: number; gainMmPerMv?: number
  reconciliation?: CalibrationReconciliation; printedSettings?: PrintedSettingsReport
}

export function calibrationIsUsable(calibration: PhysicalCalibration | undefined) {
  return calibration?.detected === true && (calibration.confidence ?? 0) >= .35 &&
    [25,50].includes(calibration.paperSpeedMmPerSecond ?? 0) && [5,10,20].includes(calibration.gainMmPerMv ?? 0)
}

export function calibrationPublicationBlock(calibration: PhysicalCalibration | undefined) {
  if (calibration?.reconciliation?.quantitativeBlocked) {
    return calibration.reconciliation.state === "unsupported" ? "unsupported_calibration" as const : "calibration_conflict" as const
  }
  // A recognized nondefault setting cannot inherit the backend's default units
  // when pulse/grid evidence failed. Printed text alone does not fix pixel scale.
  if (!calibrationIsUsable(calibration) &&
      (calibration?.printedSettings?.values.gain.some(value => value !== 10) ||
       calibration?.printedSettings?.values.speed.some(value => value !== 25) ||
       (calibration?.detected &&
        (calibration.gainMmPerMv !== 10 || calibration.paperSpeedMmPerSecond !== 25)))) {
    return "unsupported_calibration" as const
  }
  return undefined
}

export function bindPrintedCalibrationSource(printed: PrintedSettingsReport | undefined, source: { sha256: string; width: number; height: number }) {
  if (printed?.sourceSha256 !== source.sha256) {
    throw new EvidenceContractError("calibration_source_identity_mismatch", "Printed-setting evidence does not identify the admitted source.")
  }
  if (printed.sourceSize.width !== source.width || printed.sourceSize.height !== source.height) {
    throw new EvidenceContractError("calibration_source_coordinate_mismatch", "Printed-setting coordinates do not match the admitted source raster.")
  }
}
