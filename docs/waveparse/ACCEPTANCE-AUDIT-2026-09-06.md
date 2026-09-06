# Pro report acceptance audit

This audit follows the supplied 6 September report, reviewed against public
source commit `bbf1dd38705cf38588d2bc521493528b0833bbc4`. The private development
base is `9a020c4f06d4dfdfbf59d0cb18660f50671b1e39`. These are different histories.
The original report and raw evidence are retained locally; neither private
history nor ECG source files belong in the public export.

**All requirements are not yet closed.** The [final evaluation](FINAL-EVALUATION-2026-09-06.md)
completed all 25 patients once: 19 returns, six abstentions, zero runtime failures
and four reference-qualified cases. Its 6 GiB sampled-memory budget failed.
The exact candidate has not passed hosted release verification, and the owner
has not made its exact-artifact distribution/use decision. Failed experiment
gates remain failures even when the required comparison and no-adoption
decision are complete. No result is clinically accepted.

## Criteria and evidence

| Work package | Criteria checked and evidence | Disposition |
| --- | --- | --- |
| WP-001 Baseline | [Baseline instructions, identities and stable failure catalogue](BASELINE-2026-09-06.md). All eight attempted inputs retained; exact CSV reproduction distinguished from renderer and tolerance differences. Baseline was frozen before numerical changes. Later installed regression reproduces all nine returned CSVs across eleven development inputs. | Local deliverable complete. Historical two-of-32 gate failures remain; the separate library gate passes 66/66. |
| WP-002 Scoring | [Versioned coordinate/scoring contract](SCORING-CONTRACT.md); `test_scoring_coordinates.py` and `test_score_digitization.py`. Wrong panels and swapped identities fail separately from shape; canonical and internal gaps remain positioned; uncertainty follows the same transform; NaN, unequal length, duplicate, partial and single-point cases have defined outcomes; derivatives do not cross gaps. | Implemented and tested. Lead-local metrics remain separate; alignment is bounded, not proof of timing identity. |
| WP-003 Retention | Retention policy v2 and `evidence-retention.test.ts`, `runs-review.test.ts`, `run-storage-policy.test.ts`. Installed review/compaction/reopen/export/tamper lifecycle retains uncertainty locations/statuses, segment map and hashes. Historical deleted evidence remains explicitly unavailable. | Local deliverable complete; retained storage is measured. No reconstruction of historically deleted uncertainty. |
| WP-004 CI | [Trigger coverage and aggregate requirements](HOSTED-CI.md); `waveparse-ci.test.ts` and `test_platform_evidence.py`. PR/main triggers are unfiltered. Shared fixture bytes, payload identity, case count/uniqueness, semantics, waveform and uncertainty are compared. The aggregate fails on failed/skipped prerequisites. | Local workflow/tests complete. Exact-candidate hosted execution and required-check enforcement remain open. Private plan rejects branch protection; public preview is currently unprotected. |
| WP-005 Validators | [Physical/evidence contract](API.md), shared TypeScript/Python validators and adversarial `physical-contracts`, `segment-evidence`, `geometry-evidence` and durable-record tests. Negative/nonfinite/inconsistent/out-of-bounds/unsupported evidence fails; old artifacts have explicit compatibility; required absent evidence is not verified; missing samples round-trip. | Implemented and tested. Final evaluation semantic/fidelity failures cannot be relabelled by contract-test passes. |
| WP-006 Selector fitting | [Exact fitting provenance](CALIBRATION-PROVENANCE.md); reproducible command, hashed recipe and feature identity guards. Located historical fitting inputs reproduce the profile numerically; parameter mismatch is rejected. Final groups were excluded from fitting. | Complete within recorded provenance. Historical dirty extraction source is unavailable and explicitly not claimed reproducible. No newly fitted selector is promoted. |
| WP-007 Independent evaluation | [Dataset/split/annotation protocol](EVALUATION-PROTOCOL.md), patient grouping and one-use final access control. Controlled morphology/measurement fixtures, exposed development captures and 25 previously unused local PTB patient groups are separate. Sources, licences, truth, membership, tolerances and source lock were fixed before final outcomes; sample planning targets coarse overall yield precision. | [Final denominator, strata, patient intervals and visual failure report](FINAL-EVALUATION-2026-09-06.md) complete: 19/25 returned, 4/25 qualified, 11/19 strict semantic passes. Cross-dataset subject linkage and model-training overlap remain unknown. Actual capture evidence is exposed development data, not a fresh scan/photo final cohort. |
| WP-008 Geometry/calibration | [Measured physical and local-grid experiments](COMPLETION-WORK-2026-09-06.md). Independent x/y scale and round trips; OCR alternatives/defaults/conflicts; 25/50 mm/s and 5/10/20 mm/mV; ambiguous periods and unequal-axis images. Full pipeline gain normalization and conservative grid-period correction fix observed unit errors. Four predefined distorted sheets improve under local mapping; flat case remains within margin and unsupported maps refuse. | Five of eight full physical-setting cases return and pass fixed scale/fidelity gates; three refuse. Local warp remains experimental. Calibration error is reported separately from waveform error. |
| WP-009 Segments/time | Immutable source labels, normalized leads, crops, panel/segment identity, polarity and calibration references; unknown acquisition simultaneity stays unknown. `segment-evidence.test.ts`, `lineage.test.ts` and scoring/label tests cover repeated leads, order/swaps/crops/rhythm omissions. Canonical/compact round trips preserve gaps. Actual short/long pages refuse unsupported quantitative output. | Implemented and tested; unsupported leads and intervals remain missing. Recognizing a layout does not imply supported acquisition timing. |
| WP-010 Decoders | [Paired crop and untouched-image comparisons](COMPLETION-WORK-2026-09-06.md) include directional geometry and a pinned actual pretrained per-lead model, with model/dependency/rights inventory and fixed budgets. Sixteen crops retain all missing paths; eight full images retain all candidate outcomes. | Comparison and no-adoption decision complete. Learned and directional alternatives fail fixed adoption safeguards; keep the existing backends. No training or new production model. |
| WP-011 Two-sided verification | Controlled omitted-apex, fabricated-notch, wrong-row, text-following and thin-feature fixtures; local maps/reasons and original-coordinate/shared-mask dependence are recorded. Independently truth-scored 168 reconstructed lead paths: 68/70 high-error flags and 83/98 low-error flags. | Evaluation complete; false flags rule out automatic veto. Prototype stays advisory. Full-image annotation attribution is incomplete and is not claimed solved. |
| WP-012 Selection/fallback | Candidate model/preparation/geometry/calibration ancestry and lineage contracts. Four inputs from three groups have separate production/exhaustive runs, fixed backend, candidate/fallback, selected output and evaluation-only oracle scores. Both selectors return the same two outputs and abstain on the other two; exhaustive processing costs more. | Ablation and retain-current-policy decision complete. No selector change or unsupported calibrated-confidence claim. Small development experiment, not final-set selector tuning. |
| WP-013 Review policy | [Policy decision/API documentation](API.md), `outcome-contracts.test.ts`, `capture-quality.test.ts` and review tests. Processing, advisory capture quality, output evidence, eligibility and human review are separate; warning does not hide hard missing evidence. Acceptance/rejection binds exact assets and leaves extraction policy intact. Partial output limits are explicit. | Implemented and tested. `needs_review` is not acceptance. |
| WP-014 Decomposition | [Module architecture](ARCHITECTURE.md) and focused tests. Responsibility-focused extraction separates geometry, contracts, ancestry, retention, scoring and policy while preserving shared engine interfaces. Frozen refactor-only regression retains exact outputs/outcomes; later numerical changes have separate identities and experiments. | Local deliverable complete. Current development regression also reproduces all nine returned canonical CSVs exactly. |
| WP-015 Performance/platform | [Stage/resource/device matrix](PERFORMANCE-2026-09-06.md): twelve installed interface/device calls, 24 actual cold/repeat forwards, resource budgets and actual-neural cancellation through both languages. Concurrency and cancellation/recovery tests preserve coherent records and bound worker ownership. | Measurement/review deliverable complete. Seven of nine full comparisons pass; candidate-count/device/correlation drift and one CPU memory-budget overrun remain failures. No full equivalence, cache or persistent-worker claim. |
| WP-016 Lifecycle/privacy/input | [Threat boundaries and fault evidence](LOCAL-SAFETY.md); malformed/oversized inputs, path/integrity faults, interruption and durable terminal-state tests. Network-denied inference succeeds after separate runtime setup. Authorized workspace/cache boundaries and non-root Linux deployment assumptions are explicit. | Local deliverable complete within tested scope; historical Linux payload is labelled. Automated testing is not complete security proof. |
| WP-017 Developer experience | [Executable installed examples](WORKED-EXAMPLES.md), [API](API.md) and [deployment recipe](LINUX-CONTAINER.md). Minimal source lock preserves exact versions/integrities: 34 package records, nine packages installed on this Mac. Fresh build, 108 package contracts, NodeNext/ESM/CommonJS and Python consumers pass; 37 runtime files and executable interfaces reproduce exactly. | Current local build/install and final reporting complete; final documentation artifact assembly is recorded separately. Current native hosted Linux/macOS evidence remains with WP-004. |
| WP-018 Release/rights | [Component/model evidence](LICENSING.md), separate model-specific enquiries, notices, source allowlist and artifact hashes. Existing source-preview terms are distinct from model distribution/use rights; no blanket MIT/commercial clearance. | No registry release authorized or performed. Exact-artifact owner go/no-go and unresolved rights clarification remain open. Development and bounded source preparation continue. |
| WP-019 Native/vector import | [Feasibility and admission decision](SOURCE-IMPORT-FEASIBILITY.md). Raster, vector/PDF and native signals require explicit source identity/conversion loss and shared evidence; unsupported types fail closed. | Feasibility deliverable complete. No demand-justified adapter was admitted, so vector-adapter reconstruction tests are conditional and not represented as executed. |
| WP-020 Interpreter | [Local immutable evidence contract](API.md), `interpreter-evidence.test.ts` and installed JS/Python demonstrations. Source/run/segment and measurement references are hash-bound; original inspection survives abstention; no generated numerical sample enters reconstruction. | Local contract/demonstration complete. No diagnostic model, new training or invented morphology. |

## Interpretation of failed gates

The development labelled-measurement experiment returned 28 of 32 endpoints
meeting the predeclared source-visibility proxy (87.5%). All 28 returned endpoints
met 75 µV / 20 ms error limits, but coverage fails the unchanged 90% gate. Eight
grouped PMcardio development captures returned no quantitative output after the
EXIF coordinate fix; all eight abstained without runtime failure. These findings
are preserved alongside the successful controlled tests, not averaged away.

The report explicitly permits retaining the baseline after an unsuccessful
decoder, verifier or selection experiment. Their deliverable is evidence and a
bounded adoption decision. Completing that work does not mean those methods
passed adoption gates or solved general scan/photo fidelity.

The final set adds 4/25 reference-qualified cases (16.0%, 95% Wilson interval
6.4%–34.7%). Conditional on 19 returns, pooled RMSE is 136.472 µV, mean patient
correlation 0.71918 and mean coverage 97.5%. All 19 source/diagnostic/render sets
were inspected; visible peak loss, panel mistakes and a gross lead/timing mismatch
remain recorded even where metadata semantics passed. Peak sampled process-tree
RSS was 7.779 GiB against the frozen 6 GiB investigation budget. The unchanged
time and workspace budgets passed. The final set is now exposed and cannot be
reused silently for tuning or another independent final claim.

## Delivery and review boundary

Changes and evidence are local. The prepared public-source review checkout
contains only audited source over the existing public history; no private
history is imported. No commit, push, pull request, merge, deployment, registry
publication or runtime activation has been performed by this continuation.

The requested Autoreview pass covered the first campaign's selected Git scope
at P0 priority. It completed scoped-clean; that is not a certificate for later
continuation edits. The skill permits one bounded pass unless the user requests
another. The current contract, consumer and numerical checks are reported on
their own evidence.
