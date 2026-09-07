# Engineering campaign results — 6 September 2026

**Historical snapshot:** this report describes the first campaign and its
archived artifacts. It is not a completion certificate for the Pro report or
the current working tree. The [acceptance audit and continuation](COMPLETION-WORK-2026-09-06.md)
record subsequent changes, actual decoder experiments and outstanding checks.
References below to “final” mean that earlier campaign's frozen artifacts.

The GPT-6 Pro feedback was checked against local baseline commit
`9a020c4f06d4dfdfbf59d0cb18660f50671b1e39`, then implemented as the bounded
work packages in [the ledger](ENGINEERING-IMPROVEMENTS.md). The report's public
review revision `bbf1dd3` is a different source identity.

The main result is a stronger measurement/evidence contract and reproducible
evaluation. **No improvement in extraction accuracy is claimed.** The final
regression preserves every baseline waveform, while the new public pilot
exposes substantial failures beyond that deterministic family. Experimental
additions were retained as experiments where they failed the adoption criteria
or lacked model-rights evidence.

## Implementation and validation

| Area | Result |
| --- | --- |
| Numerical/semantic measurement | Scoring v6 separates local shape from canonical placement and source-label identity; truth, annotations and uncertainty retain their coordinate frames and gaps. |
| Evidence lifecycle | Permanent compact/uncertainty/segment evidence, immutable segment and omission IDs, interval ancestry/lineage, all-reviewed-asset hashes and explicit unavailable legacy evidence. |
| Contracts and safety | Physical/evidence/durable-record validation; malformed/decompression/multiframe input limits; consistent client/input errors; durable-write, tamper, interruption, timeout and recovery checks. |
| Policy and integration | Separate processing, reconstruction, eligibility and review dimensions; local `getEvidence` / `get_evidence`; original inspection after abstention; no automatic acceptance or external image fallback. |
| Calibration and evaluation | Historical numerical profile reproduced exactly with a hashed fitting recipe; feature compatibility validation; grouped collection/exposure protocols; frozen public pilot and correction audit. |
| Maintainability and installation | Focused modules extracted without changing waveforms; unfiltered CI with fast/native tiers; pinned non-root Linux recipe; executable Python/JavaScript examples and source-package audits. |

Local validation passed **208 TypeScript tests and 259 Python tests**, type
checking, lint and the Next.js production build. Installed package checks cover
ESM/CommonJS, NodeNext types, Python sync/async and shared runtime identities.
The final-package lifecycle test preserved evidence through review, compaction,
reopen, tampering checks and dead-worker recovery; cancellation after source
admission completed in 110.4 ms. The final installed evidence reader verified
all 24 terminal regression/public-pilot records. Worked examples matched across
both interfaces for a clean result, a difficult result and an abstention.

The detached [artifact receipt](../../dist/2026-09-06-final-artifacts.json),
delivered alongside the archives and retained with the local campaign, records
their hashes and corresponding-source rebuild checks. It is not bundled inside
the archives that it hashes. Frozen inference
package D uses runtime payload
`a6c999ca9719ba4f3c2d2a3d4625052836016812a20192d3cb673dcdf76e1249`.
Source/documentation and scorer changes after D are separately identified;
the final executable payload and language entry points are verified against D.

## What the measurements establish

| Evidence stratum | All-attempt result | Conditional fidelity and limits |
| --- | --- | --- |
| [Deterministic final regression](verification/2026-09-06-iteration-d.json), one signal family | 8 attempts: 7 returned for review, 1 abstained, 0 overall runtime failures, 0 accepted | All 7 CSVs and all source/diagnostic/paper bytes match the inspected baseline successors; 194,007 samples, 28.62 µV pooled RMSE, 0.9599 mean correlation, 99.32% mean coverage. |
| [Frozen public PTB pilot](PUBLIC-PILOT-2026-09-06.md), 16 patient groups | 16 attempts: 12 returned, 4 abstained, 0 overall runtime failures; **6/16 reference-qualified**, 0 accepted | 412,036 compared samples; 121.15 µV pooled RMSE, 103.57 µV mean case RMSE, 0.7700 correlation, 98.17% coverage. Rendered acquired signals; unknown model-training overlap. |
| [CPU/platform/interface comparisons](PERFORMANCE-2026-09-06.md), one clean source each | Linux CPU and native CPU Python/JavaScript completed and met frozen aggregate margins | Linux approximates MPS closely; native CPU differs in pointwise morphology/missingness. One internal JS-run candidate timed out. These are not sample-level or broad platform-equivalence claims. |

The current library regression contract passes **66/66 checks**. Historical
gates still pass **30/32**, retaining the original low-resolution/perspective
status mismatches. The known low-resolution narrow-feature limit, 3×4 fidelity
limit and 12×1 synthetic abstention remain documented. Gates were not weakened
to relabel those outcomes.

Two new evaluation defects were corrected transparently: pilot page duration
was initially used as panel duration, and full-page truth/uncertainty needed
explicit canonical indices. The original reports and corrected rescoring chain
remain retained. No pilot image, truth or extracted CSV changed; waveform
fidelity metrics and qualification thresholds stayed fixed.

## Experimental and conditional decisions

The [decoder/source-verifier/selection experiments](EXPERIMENTS-2026-09-06.md)
did not justify replacing the default extraction or selection policy. The new
geometric tracer failed its prespecified crop improvement gate; its full-pipeline
promotion therefore stopped. Local warp and two-sided source verification remain
bounded advisory experiments. The alternative learned decoder's weight licence
was recorded as unknown; its weights were not downloaded or adopted.

Pulse/grid scale and coordinate transforms are tested; printed-setting OCR and
independent patient landmark adjudication are not claimed. Source bands marked
inferred are not exact recovered source crops. Acquisition simultaneity and
original sampling remain unknown without supporting source evidence. The
[source-import feasibility decision](SOURCE-IMPORT-FEASIBILITY.md) admits no
new PDF/native/vector adapter without a concrete input requirement.

The reserved final evaluation collection contained zero records at this
snapshot. The Pro report requires controlled final evaluation, grouped before
image variants, with controlled or adjudicated endpoints; it does not require
an external human custodian or a clinical validation study as a prerequisite
for engineering completion. Model-specific rights enquiries remain unresolved.
Hosted CI, registry publication, commercial model-rights approval and downstream
activation are separate states.

## Review, preservation and rollback

Changes are prepared in the local working tree; no commit, push or registry
publication was made by this campaign. The untouched feedback, source snapshots,
fixtures, failed attempts, scores, model/runtime references and review audits
are retained under `benchmark/results/gpt6pro-20260906/` with a keep marker.
The historical calibration run is also pinned. Storage audit was read-only;
no pruning occurred. Existing paused benchmark groups remain paused. Docker
Desktop was restored to its prior stopped state after the Linux test.

Review the [20 work-package decisions](ENGINEERING-IMPROVEMENTS.md) before
release. To compare or roll back code, use an isolated baseline checkout and
preserve all new evidence. Do not run the old v1 compactor over v2 review bundles,
since it does not preserve their uncertainty contract. Any future accuracy
tuning on the now-exposed pilot requires a new evaluation set.
