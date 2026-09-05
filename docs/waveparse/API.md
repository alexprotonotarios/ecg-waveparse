# API contract v1

| Capability | JavaScript / TypeScript | Python |
| --- | --- | --- |
| Configure | `new Digitizer({workspaceDir, runtimeDir?})` | `Digitizer(workspace_dir=..., runtime_dir=None)` |
| Digitize | `await digitize(inputPath, {device?, timeoutMs?, signal?, onProgress?})` | `digitize(input_path, device="cpu", timeout_ms=1800000)` |
| Async Python | — | `await digitize_async(...)` |
| Read | `await getRun(runId)` | `get_run(run_id)` / `await get_run_async(run_id)` |
| Review | `await review(runId, {decision, reviewer, notes, confirmations?})` | `review(run_id, decision=..., reviewer=..., notes=..., confirmations=None)` / async equivalent |

Inputs: local PNG/JPEG/WebP/TIFF paths, at most 50 MB. Source decoding and capture
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

Review requires nonempty reviewer and notes. Acceptance requires all three original
confirmations and an eligible quantitative output cryptographically bound to its
source. `getRun` checks available artifact integrity. Review compaction can remove
transient artifact paths from the next returned result.
