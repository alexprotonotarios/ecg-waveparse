# GPT-6 Pro engineering improvement campaign

Started 6 September 2026 from local source commit
`9a020c4f06d4dfdfbf59d0cb18660f50671b1e39`. The supplied review inspected the
separate public source-preview repository at `bbf1dd3`; those identities must
not be conflated. The review is an engineering proposal with limited execution
evidence. Its findings are checked against this checkout before implementation.

The objective is to complete each applicable recommendation and its tests.
Experimental packages require a recorded comparison and adoption decision;
they do not require shipping an inferior or unlicensed decoder. Independent
clinical evidence, maintainer licence replies and a release-owner decision
cannot be manufactured by code or replaced by regression tests.

The [first campaign results](ENGINEERING-RESULTS-2026-09-06.md) describe its archived
snapshot. [Continuation evidence](COMPLETION-WORK-2026-09-06.md) and the
[criteria audit](ACCEPTANCE-AUDIT-2026-09-06.md) record current implementation,
final results and remaining hosted/owner decisions.

**Completion correction, 6 September 2026:** The first campaign completed local
implementation and bounded experiments, but did not satisfy every acceptance
criterion in the Pro report. Its earlier goal-complete decision was premature.
Work has resumed against the original requirements. The historical receipts and
archives remain evidence of that earlier source snapshot, not subsequent edits.

## Execution and evidence

The untouched source snapshot, supplied report, generated fixture bytes, truth,
baseline package and per-case results live in the retained local directory
`benchmark/results/gpt6pro-20260906/`. It may contain private repository source
and is excluded from public package/source allowlists. The baseline uses a
separate installed consumer and runtime. Historical quality gates remain
unchanged. Existing paused campaigns remain paused.

Each result must retain all attempted inputs in its denominator, distinguish
quantitative yield from reference-qualified yield and conditional fidelity,
and report missingness, semantic outcomes, resource use and review separately.
No output is clinically accepted by this campaign.

## Work-package ledger

| ID | Acceptance evidence to produce | State |
| --- | --- | --- |
| WP-001 | [Frozen baseline and failure catalogue](BASELINE-2026-09-06.md); exact source, fixtures and outputs retained; 8/8 infrastructure completion; historical 2/32 failures retained, library 66/66 passes | Complete locally; unchanged baseline and failure catalogue retained |
| WP-002 | [Scoring v7](SCORING-CONTRACT.md), explicit transformations, wrong-panel/identity/gap/shift/resampling tests, installed harness integration | Implemented and tested; policy-v3 regression retains all seven original canonical CSVs and strict semantic passes; legacy PMcardio timing is reconciled only from independently declared panel samples |
| WP-003 | Permanent uncertainty/segment evidence, all-reviewed-asset hashes, legacy absence and tamper/lifecycle tests | Complete locally; final D installed review/compaction/reopen/tamper lifecycle passed |
| WP-004 | Unfiltered PR/main triggers, fast contracts, shared fixture bytes, native platform matrix and a required aggregate that fails on missing prerequisites | Trigger and drift/tamper/denominator tests pass locally; exact-candidate hosted checks and branch-protection evidence outstanding |
| WP-005 | Physical/evidence, durable-record and segment validators; explicit compatibility; adversarial tests | Complete locally; final malformed nested evidence guards passed the full contract tests and installed regression |
| WP-006 | Exact historical fitting run located; numerical profile reproduced identically; hashed fitting recipe and parameter mismatch rejection added; extraction dirty snapshot still unavailable | Complete locally; profile reproduced exactly, historical dirty-source limit recorded |
| WP-007 | [Evaluation specification](EVALUATION-PROTOCOL.md), split/protocol manifests and denominator/group-safe reports | [Final 25-patient evaluation](FINAL-EVALUATION-2026-09-06.md) complete with one access: 19 returned, six abstained, zero runtime failures, four reference-qualified. Patient intervals, strata, all failures and 19 visual inspections retained. Controlled measurements and eight grouped development captures remain separate |
| WP-008 | [Current geometry/calibration evidence](COMPLETION-WORK-2026-09-06.md), source transform chain and physical scale tests | Actual OCR/refusals and global/local geometry experiments complete. Full-page testing exposed and fixed neural gain and pulse-scale errors; corrected installed package returns 5/8 cases, all five passing fixed scale/fidelity gates; three refusals remain explicit. Local warp stays experimental |
| WP-009 | Immutable segment map added to exports/retention; canonical/compact round trips, explicit repeated-segment omissions and unknown acquisition timing | Segment/omission identities, source bands, timing and lineage tests pass; actual short/long nondefault pages refuse quantitative interpretation |
| WP-010 | [Decoder comparisons and no-adoption decisions](COMPLETION-WORK-2026-09-06.md) | Controlled crop and full-image candidate pipeline comparisons now run, including an actual pretrained per-lead model; alternatives failed the fixed safeguards, so current backends remain |
| WP-011 | [Two-sided verification against actual reconstruction errors](COMPLETION-WORK-2026-09-06.md) | Controlled fault fixtures plus 168 actual lead paths evaluated; 83/98 low-error paths flagged at the fixed threshold, so the prototype remains advisory with incomplete full-image attribution disclosed |
| WP-012 | [Ancestry, lineage and fixed/fallback/selector/oracle ablation](COMPLETION-WORK-2026-09-06.md) | Four-input, three-group production/exhaustive comparison and candidate profiling complete on EXIF-v4; both selectors return the same two outputs, with extra exhaustive cost and no selected-output improvement. Retain current production selection |
| WP-013 | Advisory operation renamed with compatibility alias; processing/output/eligibility/review dimensions and missing-evidence checks added | Complete locally; independent dimensions and review/policy tests pass |
| WP-014 | [Responsibility-focused module architecture](ARCHITECTURE.md) and installed numerical equivalence | Incremental decomposition complete; final D retained all baseline CSVs and outcomes |
| WP-015 | Stage/resource profiling and platform/interface comparison with explicit budgets | Twelve current installed calls complete: all return, 7/9 full comparisons pass. Low-resolution candidate-count and device/correlation drift plus one 9 GiB memory-budget overrun remain failures. Cold/warm and observed-neural cancellation evidence is recorded; no full equivalence, cache or worker promotion claimed |
| WP-016 | [Threat boundaries and fault verification](LOCAL-SAFETY.md), including malformed inputs and offline inference | Complete locally; fault/privacy/input/lifecycle checks and network-denied macOS/Linux inference pass |
| WP-017 | [Executable examples/evidence matrix](README.md), [non-root container](LINUX-CONTAINER.md), minimal source toolchain and installed payload parity | Fresh reduced source install uses nine packages; build, 108 contracts, NodeNext and both installed language consumers pass. Runtime/interfaces and source lock reproduce exactly. Three current evidence examples match across languages; final report is complete and final artifact identity is recorded separately |
| WP-018 | Exact component/model evidence, current enquiries, explicit bounded release decision | Inventory and bounded no-release decision recorded; model-rights clarification and exact-artifact owner release decision outstanding |
| WP-019 | [Native/vector/PDF source import feasibility decision](SOURCE-IMPORT-FEASIBILITY.md) | Feasibility decision complete; no demand-justified adapter admitted |
| WP-020 | [Local interpreter contract](API.md), immutable references and installed source-inspection demonstrations | Complete locally; JS/Python source/segment/abstention demonstrations match exactly |

## Change and promotion rules

For each numerical or policy change, record the failure/work-package IDs,
assumptions, before/after results on identical input bytes, semantic/morphology/
coverage effects, resources, provenance and rollback. Add a targeted regression.
Refactors are checked independently of algorithm changes. Do not tune on a
final evaluation set or alter thresholds to relabel existing failures.

Final completion requires the ledger to link actual implementation and test
evidence, with any external prerequisites or conditional non-adoption stated
explicitly. Local checks, hosted CI, publication and runtime activation are
separate states.
