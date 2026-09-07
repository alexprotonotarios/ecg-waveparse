import type { RunRecord } from "@/lib/runs"
import { EvidenceContractError, positivePhysicalValue, unitFraction } from "@/lib/digitizer/physical-contracts"

/** Additive validation of legacy records: missing evidence stays missing. */
export function validateDurableRun(value: unknown): RunRecord {
  const fail = (issue: string): never => { throw new EvidenceContractError(issue, `Invalid stored run: ${issue}.`) }
  if (!value || typeof value !== "object" || Array.isArray(value)) return fail("invalid_run_object")
  const run = value as RunRecord
  if (typeof run.id !== "string" || typeof run.localPath !== "string" ||
      !["reference", "upload"].includes(run.source) ||
      !["completed", "needs_review", "partial", "queued", "running", "pending_digitizer", "timed_out", "failed"].includes(run.status) ||
      !run.assets || typeof run.assets !== "object" || Array.isArray(run.assets)) return fail("invalid_run_header")
  for (const key of ["sampleRateHz", "effectiveSampleRateHz", "paperSpeedMmPerSecond", "gainMmPerMv"] as const) {
    if (run[key] !== undefined && !positivePhysicalValue(run[key])) return fail(`invalid_${key}`)
  }
  if (run.calibrationConfidence !== undefined && !unitFraction(run.calibrationConfidence)) return fail("invalid_calibration_confidence")
  for (const key of ["leadCount", "fileSizeBytes"] as const) {
    if (run[key] !== undefined && (!Number.isSafeInteger(run[key]) || run[key]! < 0)) return fail(`invalid_${key}`)
  }
  if (run.retention && ((run.retention.version !== 1 && run.retention.version !== 2) ||
      run.retention.policy !== `lean-final-evidence-v${run.retention.version}` || !Array.isArray(run.retention.removedAssets))) return fail("invalid_retention_contract")
  if (run.reliability) {
    const reliability = run.reliability
    for (const key of ["sourceWidth", "sourceHeight", "nominalSampleRateHz", "effectiveSampleRateHz"] as const) {
      if (!positivePhysicalValue(reliability[key])) return fail(`invalid_reliability_${key}`)
    }
    for (const key of ["annotationMaskedPixels", "uncertainSampleCount", "missingSampleCount"] as const) {
      if (!Number.isSafeInteger(reliability[key]) || reliability[key] < 0) return fail(`invalid_reliability_${key}`)
    }
    if (!unitFraction(reliability.annotationMaskedFraction) || reliability.morphologyReconstructed !== false || reliability.originalPreserved !== true) return fail("invalid_reliability_evidence")
  }
  return run
}
