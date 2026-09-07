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
}
export type CalibrationReconciliation = {
  version: 1
  state: "unsupported" | "conflict" | "corroborated_inference" | "partially_corroborated" | "unresolved"
  reasons: string[]
  compared: Array<"speed" | "gain">
  quantitativeBlocked: boolean
  pulseAssumptions: { durationSeconds: 0.2; amplitudeMv: 1 }
  acquisitionSampleRateHz: null
}

const fail = (): never => { throw new EvidenceContractError("invalid_calibration_reconciliation", "Printed calibration evidence is malformed or disagrees with its physical-unit decision.") }
function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return fail()
  return value as Record<string, unknown>
}
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value)

/** Recompute cross-field decisions; a serialized flag cannot hide a conflict. */
export function validatePrintedCalibration(calibration: Record<string, unknown>) {
  if (calibration.printedSettings === undefined && calibration.reconciliation === undefined) return
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
  if (reconciliation.version !== 1 || reconciliation.state !== state || reconciliation.quantitativeBlocked !== Boolean(reasons.length) ||
      JSON.stringify(reconciliation.reasons) !== JSON.stringify(reasons) || JSON.stringify(reconciliation.compared) !== JSON.stringify(compared) ||
      assumptions.durationSeconds !== .2 || assumptions.amplitudeMv !== 1 || reconciliation.acquisitionSampleRateHz !== null) fail()
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
