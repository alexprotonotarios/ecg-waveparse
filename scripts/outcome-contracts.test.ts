import assert from "node:assert/strict"
import test from "node:test"
import { describeOutcome } from "../src/lib/digitizer/outcome-dimensions"
import { validateDurableRun } from "../src/lib/digitizer/record-contracts"
import { canAcceptQuantitativeReview } from "../src/lib/digitizer/review-policy"
import { retainReviewableOutputWithAdvisoryQuality } from "../src/lib/digitizer/publication-policy"
import type { RunRecord } from "../src/lib/runs"

const asset = { label: "synthetic", path: "storage/runs/run_test/synthetic" }
const base: RunRecord = { id: "run_test", localPath: "storage/runs/run_test", source: "upload", status: "needs_review", createdAt: "2026-09-06", updatedAt: "2026-09-06", fileName: "synthetic.png", sampleRateHz: 500, assets: { input: asset }, publicationDecision: { policyId: "test", policyVersion: 1, outcome: "needs_review", reasonCode: "source_verified" } }

test("durable records reject nonphysical rates while reading legacy evidence absence", () => {
  assert.equal(validateDurableRun(base), base)
  for (const value of [-1, 0, NaN, Infinity, null, "500"]) assert.throws(() => validateDurableRun({ ...base, sampleRateHz: value }), /invalid_sampleRateHz/)
  assert.throws(() => validateDurableRun([]), /invalid_run_object/)
  assert.throws(() => validateDurableRun({ ...base, retention: { version: 99 } }), /invalid_retention_contract/)
})

test("advisory quality and human decisions cannot hide missing quantitative evidence", () => {
  for (const outcome of ["acceptable", "review", "insufficient", undefined] as const) {
    const decision = { outcome: "failed", reasonCode: "no_publishable_candidate", partialLeadSelection: false } as const
    assert.equal(retainReviewableOutputWithAdvisoryQuality(decision, outcome), decision)
    for (const review of [undefined, "accepted", "rejected"] as const) {
      const run = { ...base, assets: { ...base.assets, canonicalCsv: asset }, ...(review ? { review: { decision: review, reviewer: "synthetic", notes: "test", updatedAt: "2026-09-06", eventCount: 1 } } : {}) }
      const dimensions = describeOutcome(run)
      assert.equal(dimensions.quantitativeEligibility, "incomplete_evidence")
      assert.equal(dimensions.reviewState, review ?? "pending")
      assert.equal(dimensions.clinicalValidationUse, false)
    }
  }
})

test("v2 acceptance requires permanent evidence and remains separate from processing", () => {
  const assets = { input: asset, canonicalCsv: asset, segmentsCsv: asset, uncertaintyCsv: asset, segmentMapJson: asset, provenanceJson: asset }
  assert.equal(describeOutcome({ ...base, assets }).quantitativeEligibility, "for_source_review")
  assert.equal(canAcceptQuantitativeReview({ ...base, assets, retention: { version: 2 } }), true)
  assert.equal(canAcceptQuantitativeReview({ ...base, assets: { ...assets, uncertaintyCsv: undefined }, retention: { version: 2 } }), false)
})
