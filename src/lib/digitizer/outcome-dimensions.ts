import type { RunRecord } from "@/lib/runs"

export function describeOutcome(run: Pick<RunRecord, "status" | "processing" | "publicationDecision" | "assets" | "review" | "reliability">) {
  const quantitative = Boolean(run.assets.canonicalCsv)
  const requiredEvidence = ["input", "canonicalCsv", "segmentsCsv", "uncertaintyCsv", "segmentMapJson", "provenanceJson"] as const
  const missingEvidence = quantitative ? requiredEvidence.filter(key => !run.assets[key]) : []
  return {
    version: 1 as const,
    processingState: run.processing?.state ?? (["queued", "running", "timed_out", "failed"].includes(run.status) ? run.status : "completed"),
    reconstructionOutcome: quantitative ? "quantitative_output" as const : run.assets.diagnostic ? "diagnostic_only" as const : "no_output" as const,
    quantitativeEligibility: quantitative && run.publicationDecision?.outcome === "needs_review"
      ? missingEvidence.length ? "incomplete_evidence" as const : "for_source_review" as const
      : "unavailable" as const,
    reviewState: run.review?.decision ?? (quantitative || run.assets.diagnostic ? "pending" as const : "not_applicable" as const),
    missingEvidence,
    advisoryInputQuality: run.reliability?.inputQualityOutcome ?? "not_recorded",
    advisoryReasons: run.reliability?.inputQualityReasons ?? [],
    clinicalValidationUse: false as const,
  }
}
