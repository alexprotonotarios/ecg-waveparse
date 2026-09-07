# Pro report acceptance audit

This audit follows the supplied 6 September report, reviewed against public
source commit `bbf1dd38705cf38588d2bc521493528b0833bbc4`. The private development
base is `9a020c4f06d4dfdfbf59d0cb18660f50671b1e39`. These are different histories.
The original report and raw evidence are retained locally; neither private
history nor ECG source files belong in the public export.

**The authorized implementation, testing and bounded-experiment deliverables
are complete.** All 20 work packages have evidence and a disposition below.
The owner approved bounded source-preview delivery and deferred registry
publication/model redistribution; see the [delivery record](DELIVERY-2026-09-06.md).
The [macOS CPU correction](MACOS-CPU-2026-09-06.md) is committed in draft PR #1
at `693f6bf`. Local and hosted Linux/macOS regressions each complete all eight
attempts: seven returns, one abstention, no runtime or candidate failures,
7/7 strict semantic passes and 66/66 library gates. All required hosted checks
pass on the exact candidate; independent comparison of the downloaded platform
evidence reproduces the same eight-case pass. The initial Mac timeout and poor
fallback results remain recorded as a failed earlier attempt.

This does not mean every accuracy or resource gate passed. The separate
[final evaluation](FINAL-EVALUATION-2026-09-06.md) completed 25 patients once:
19 returns, six abstentions, zero runtime failures and four reference-qualified
cases. Its 6 GiB sampled-memory budget failed. The current hosted Mac took
57.09 minutes for extraction and its profiler recorded a sampling timeout;
those limitations remain explicit. Failed experiment gates remain failures even
when the required comparison and no-adoption decision are complete. No result
is clinically accepted, and model-specific rights remain unresolved.

## Criteria and evidence

| Work package | Criteria checked and evidence | Disposition |
| --- | --- | --- |
| WP-001 Baseline | [Baseline instructions, identities and stable failure catalogue](BASELINE-2026-09-06.md). All eight attempted inputs retained; exact CSV reproduction distinguished from renderer and tolerance differences. Baseline was frozen before numerical changes. Later installed regression reproduces all nine returned CSVs across eleven development inputs. | Local deliverable complete. Historical two-of-32 gate failures remain; the separate library gate passes 66/66. |
| WP-002 Scoring | [Versioned coordinate/scoring contract](SCORING-CONTRACT.md); `test_scoring_coordinates.py` and `test_score_digitization.py`. Wrong panels and swapped identities fail separately from shape; canonical and internal gaps remain positioned; uncertainty follows the same transform; NaN, unequal length, duplicate, partial and single-point cases have defined outcomes; derivatives do not cross gaps. | Implemented and tested. Lead-local metrics remain separate; alignment is bounded, not proof of timing identity. |
| WP-003 Retention | Retention policy v2 and `evidence-retention.test.ts`, `runs-review.test.ts`, `run-storage-policy.test.ts`. Installed review/compaction/reopen/export/tamper lifecycle retains uncertainty locations/statuses, segment map and hashes. Historical deleted evidence remains explicitly unavailable. | Local deliverable complete; retained storage is measured. No reconstruction of historically deleted uncertainty. |
| WP-004 CI | [Trigger coverage and aggregate requirements](HOSTED-CI.md); `waveparse-ci.test.ts` and `test_platform_evidence.py`. PR/main triggers are unfiltered. Shared fixture bytes, payload identity, case count/uniqueness, semantics, waveform and uncertainty are compared. The aggregate fails on failed/skipped prerequisites; partial retained evidence cannot pass. | [Two required Actions checks are enabled](DELIVERY-2026-09-06.md), preserving administrator behavior. Run 34062261795 passes all four jobs at candidate `693f6bf`; the complete eight-case Linux/macOS comparison independently reproduces the hosted result. Downloaded npm/wheel/sdist bytes, all 255 source files and 38 runtime files per language verify. The initial Mac timeout remains failed historical evidence. |
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
| WP-015 Performance/platform | [Stage/resource/device matrix](PERFORMANCE-2026-09-06.md): twelve installed interface/device calls, 24 actual cold/repeat forwards, resource budgets and actual-neural cancellation through both languages. Concurrency and cancellation/recovery tests preserve coherent records and bound worker ownership. | Measurement/review deliverable complete. The earlier CPU/MPS/interface matrix retains its 7/9 passes and drift/memory failures. The [CPU correction](MACOS-CPU-2026-09-06.md) completes all 63 candidates locally and on both hosted platforms; full hosted parity passes 8/8. Local/Linux/hosted-Mac extraction takes 16.85/32.50/57.09 minutes. Sampled RSS is 5.40/8.56/4.98 GiB; hosted Mac sampling records `TimeoutExpired`, so its peak is incomplete evidence. Its slowest candidate is 170.41 s against 180 s. Older final results and mixed CPU/MPS failures are not revised, and no persistent worker is promoted. |
| WP-016 Lifecycle/privacy/input | [Threat boundaries and fault evidence](LOCAL-SAFETY.md); malformed/oversized inputs, path/integrity faults, interruption and durable terminal-state tests. Network-denied inference succeeds after separate runtime setup. Authorized workspace/cache boundaries and non-root Linux deployment assumptions are explicit. | Local deliverable complete within tested scope; historical Linux payload is labelled. Automated testing is not complete security proof. |
| WP-017 Developer experience | [Executable installed examples](WORKED-EXAMPLES.md), [API](API.md) and [deployment recipe](LINUX-CONTAINER.md). Minimal source lock preserves exact versions/integrities: 34 package records, nine packages installed on this Mac. The CPU-corrected artifacts pass 109 package contracts, 93 fast Python and 14 native wrapper checks plus NodeNext/ESM/CommonJS and Python consumers; all 38 runtime files per language verify. The installed Python wheel reads all eight real JavaScript regression results. | Current local build/install and final reporting complete; final documentation artifact assembly is recorded separately. Current native hosted Linux/macOS evidence remains with WP-004. |
| WP-018 Release/rights | [Component/model evidence](LICENSING.md), separate model-specific enquiries, notices, source allowlist and artifact hashes. Existing source-preview terms are distinct from model distribution/use rights; no blanket MIT/commercial clearance. | [Owner decision recorded against the prepared artifacts](DELIVERY-2026-09-06.md): approve bounded source preview and hosted verification; defer registry publication and model redistribution. Rights clarification remains unresolved. The distribution/use decision deliverable is complete within that limited scope. |
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

The bundled package reports describe their preparation snapshot. The subsequent
[delivery record](DELIVERY-2026-09-06.md) records the approved commit, push and draft
PR over the existing public history; no private history is imported. There has
been no merge, deployment, registry publication or runtime activation.

The requested Autoreview pass covered the first campaign's selected Git scope
at P0 priority. It completed scoped-clean; that is not a certificate for later
continuation edits. The skill permits one bounded pass unless the user requests
another. The current contract, consumer and numerical checks are reported on
their own evidence.
