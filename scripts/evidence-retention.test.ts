import assert from "node:assert/strict"
import test from "node:test"
import { quantitativeEvidenceAvailability } from "../src/lib/digitizer/evidence-retention"

test("legacy compaction reports removed uncertainty without inventing evidence", () => {
  const evidence = quantitativeEvidenceAvailability({
    assets: { canonicalCsv: { label: "signal", path: "signal.csv" } },
    retention: { version: 1, policy: "lean-final-evidence-v1", compactedAt: "2026-09-05", retainedAssets: ["canonicalCsv"], removedAssets: ["segmentsCsv", "uncertaintyCsv"], reviewArtifactsRetained: false, reclaimedBytes: 0, cumulativeReclaimedBytes: 0 },
  })
  assert.equal(evidence.uncertainty, "unavailable_historical")
  assert.equal(evidence.segments, "unavailable_historical")
  assert.equal(quantitativeEvidenceAvailability({ assets: {} }).uncertainty, "not_applicable")
  assert.equal(quantitativeEvidenceAvailability({ assets: { canonicalCsv: { label: "signal", path: "signal.csv" } } }).uncertainty, "not_recorded")
})
