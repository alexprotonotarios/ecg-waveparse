import type { RunAssetKey, RunRecord } from "@/lib/runs"

/** Quantitative evidence survives every human review outcome. */
export const PERMANENT_EVIDENCE_ASSETS = new Set<RunAssetKey>([
  "input", "diagnostic", "paperRender", "probability", "canonicalCsv",
  "segmentsCsv", "uncertaintyCsv", "segmentMapJson", "metadataCsv", "provenanceJson", "reviewJson",
])

export type EvidenceAvailability = "recorded" | "unavailable_historical" | "not_recorded" | "not_applicable"
export type QuantitativeEvidenceAvailability = {
  version: 1
  segments: EvidenceAvailability
  uncertainty: EvidenceAvailability
}

// Describes persisted references, not a substitute for asset hash verification.
// In particular, never recreate deleted uncertainty from the signal or review.
export function quantitativeEvidenceAvailability(run: Pick<RunRecord, "assets" | "retention">): QuantitativeEvidenceAvailability {
  const state = (key: "segmentsCsv" | "uncertaintyCsv"): EvidenceAvailability => {
    if (run.assets[key]) return "recorded"
    if (run.retention?.removedAssets.includes(key)) return "unavailable_historical"
    return run.assets.canonicalCsv ? "not_recorded" : "not_applicable"
  }
  return { version: 1, segments: state("segmentsCsv"), uncertainty: state("uncertaintyCsv") }
}
