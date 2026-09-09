# API contract v1

| Capability | JavaScript / TypeScript | Python |
| --- | --- | --- |
| Configure | `new Digitizer({workspaceDir, runtimeDir?})` | `Digitizer(workspace_dir=..., runtime_dir=None)` |
| Digitize | `await digitize(inputPath, {device?, timeoutMs?, signal?, onProgress?})` | `digitize(input_path, device="cpu", timeout_ms=1800000)` |
| Async Python | — | `await digitize_async(...)` |
| Read | `await getRun(runId)` | `get_run(run_id)` / `await get_run_async(run_id)` |
| Interpreter evidence | `await getEvidence(runId)` | `get_evidence(run_id)` / `await get_evidence_async(run_id)` |
| Review | `await review(runId, {decision, reviewer, notes, confirmations?})` | `review(run_id, decision=..., reviewer=..., notes=..., confirmations=None)` / async equivalent |

Inputs: local PNG/JPEG/WebP/single-frame TIFF paths, at most 50 MB, 80 million
decoded pixels and 32,768 pixels on either edge. Source decoding and capture
preflight run before admission. Unsupported devices, protocol versions and timeout
bounds are rejected. Result JSON keeps the engine's fields, with schema version,
WaveParse provenance and absolute artifact paths added. TypeScript declarations
describe the existing engine metadata; Python TypedDict describes common result
fields and leaves detailed evolving QA/provenance maps extensible.

The private subprocess protocol uses one versioned JSON request on stdin and
newline-delimited events on stdout: `started`, followed by `result` or `error`.
Other diagnostics go to stderr. This is an internal transport, not a network API.
The public API owns serialization and process cleanup.

Stable client errors include `invalid_request`, `invalid_input`, `not_found`,
`runtime_unavailable`, `run_busy`, `cancelled`, `protocol_error`, and `runner_failed`.
Existing engine integrity/storage errors retain their code when available; unknown
exceptions use `execution_error`. An error may identify an already-admitted run.
Timeout is a persisted `timed_out` result; extraction abstention is a `failed` result.

`invalid_evidence_contract` identifies invalid physical or durable evidence.
Its internal `issue` distinguishes rates, crops, transforms, counts, timing and
artifact/segment identity. Intentional CSV missingness is an empty cell, never
a zero-valued reconstruction.

Review uses `decision: "accepted"` or `"rejected"` in JavaScript, and
`decision="accepted"` or `"rejected"` in Python. Both require nonempty reviewer
and notes. Acceptance also requires an eligible quantitative output
cryptographically bound to its source and all three `confirmations` fields:

| Field | What the reviewer confirms |
| --- | --- |
| `sourceCompared` | The output was compared with the original image. |
| `leadIdentityVerified` | Lead identities were checked against the source. |
| `scaleAndGapsReviewed` | Calibration, scale and missing signal were reviewed. |

These keys are camelCase in both the JavaScript object and the Python dictionary.
Each must be `true` (`True` in Python) for acceptance; set it only after that
review has taken place. Rejection does not require acceptance confirmations.
`getRun` checks available artifact integrity. Review compaction can remove
transient artifact paths from the next returned result.

Retention policy `lean-final-evidence-v2` permanently keeps the compact signal,
per-sample uncertainty and `segmentMapJson`, together with source, canonical
signal, overlay, provenance and audit. Acceptance or rejection changes review
state without changing samples or removing uncertainty. Review events record
the hashes of every derived artifact actually reviewed. Transient candidate
images, tensors and alternate runs remain disposable. Retained storage depends on exported sample and uncertainty counts.

`quantitativeEvidence` distinguishes recorded, unavailable historical,
not-recorded and not-applicable evidence. Older runs remain readable under
their original contract; deleted uncertainty is never reconstructed from a
review decision. New v2 acceptances require the complete permanent bundle.

`outcomeDimensions` separates processing state, reconstruction outcome,
quantitative eligibility, human review and advisory input-quality findings.
`for_source_review` means the evidence is available to inspect. It is not an
independent fidelity pass or clinical validation. A legacy quantitative result
with missing permanent evidence reports `incomplete_evidence`. Blur and
resolution warnings do not override missing identity, units or timing evidence.

`segmentMapJson` binds each exported lead segment to source/signal/run hashes,
printed panel position, local sample indices and half-open support. Internal
gaps and panel boundaries are retained. Export sampling at 500 Hz is not an
acquisition-rate claim. Acquisition time/simultaneity remains unresolved. Source
regions may be inferred from layout bands and inverse transforms for navigation;
they are not exact verified trace-support masks. Unsupported regions remain
unresolved. Repeated source strips not exported
by a canonical column are listed as omissions, and Cabrera polarity conversion
is explicit. A paper render is for inspection; quantitative measurements use
the CSV and source evidence.

The interpreter evidence bundle binds original source, run, canonical signal,
segments and their artifact identities. Quantitative reconstruction requires
the canonical and compact CSVs, uncertainty, segment map and provenance. An
abstention still returns its verified original image for local inspection.
`generatedSamplesPermitted` is false and `networkTransmission` is `not_performed`.
The original input may be bound by the run-level source hash without a duplicate
input-asset identity; mismatched present identities are rejected.

`validateMeasurementReference` is an internal contract helper for integrations:
run/source/canonical/immutable-segment IDs must match, and the half-open requested
sample interval must be returned, within bounds and gap-free. A valid reference
still requires calibration and uncertainty review. The helper does not measure,
diagnose, fill gaps or approve an ECG. No external interpreter service is called.

`executionProfile` provides named stage durations and runner RSS with scope
limitations. Candidate `runtimeMs` includes model loading and extraction;
process-tree RSS and working-disk samples are separate profiler evidence.
