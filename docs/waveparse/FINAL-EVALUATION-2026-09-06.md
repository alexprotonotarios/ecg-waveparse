# Frozen 25-patient engineering evaluation

All **25 patients were attempted once**. The installed package returned **19 quantitative outputs**, abstained on **6**, and had **0 overall runtime failures**. **4/25 (16.0%) met every prespecified semantic and numerical gate.** No output was accepted for clinical or quantitative use.

Returned yield was 76.0% (95% Wilson interval 56.6%–88.5%); reference-qualified yield had a 95% Wilson interval of 6.4%–34.7%. All 25 patients remain in these denominators. Qualification is an engineering label, not clinical validation.

Conditional on the 19 returned outputs and 539,564 compared samples, pooled RMSE was **136.472 µV**, mean patient correlation **0.71918**, and mean patient coverage **97.5%**. Strict semantics passed for 11 returns and failed for 8. Abstentions have no reconstructed-signal error; they are not zero-error cases.

## Design and independence

This is a final engineering evaluation of controlled renders of acquired public PTB waveforms. It is not a cohort of actual photographs or scans. Twenty-five patient groups were fixed before waveform download, excluding all 17 documented locally exposed PTB patient groups. The planning target was a coarse approximately ±20 percentage-point overall yield estimate; small strata do not support narrow precision claims. Cross-dataset subject linkage and pretrained-model training overlap remain unknown.

The original final source lock was superseded for development gain/grid corrections before any final outcome was exposed. The replacement preserved patient membership, inclusion rules, truth and all 75 prepared image/truth/annotation byte sequences. Source, policy, scorer and gates were frozen before this one evaluation. The access log contains one start and its matching completion. Source/scorer, manifest, payload and all fixture hashes were rechecked by the harness after execution. No case was rerun, dropped or used to retune the implementation.

The unchanged gates are strict semantic pass, global aligned RMSE ≤100 µV, mean lead correlation ≥0.9 and mean lead coverage ≥0.9. Alignment is limited to 40 ms. The final set has no adjudicated clinical interval or fine-morphology endpoint labels; zero labelled events is not perfect morphology preservation. Separate controlled visibility/measurement experiments are reported in [the continuation evidence](COMPLETION-WORK-2026-09-06.md).

## Conditional patient-level uncertainty

| Endpoint on returned patients | Patient mean | 95% patient-bootstrap interval |
| --- | ---: | ---: |
| RMSE (µV) | 121.109 | 81.808–163.973 |
| Correlation | 0.71918 | 0.54604–0.86581 |
| Coverage fraction | 0.97479 | 0.95315–0.99267 |

These intervals resample independent returned patients with the fixed seed and 2,000 bootstrap repetitions. The patient-mean RMSE differs from the sample-pooled RMSE above. Neither interval family adds outcomes for abstaining patients.

## Predeclared layout and degradation strata

| Layout / degradation | Attempted | Returned | Reference-qualified | Conditional pooled RMSE (µV) |
| --- | ---: | ---: | ---: | ---: |
| standard_12x1 / jpeg_compression | 3 | 1 | 0 | 102.433 |
| standard_12x1 / rotation | 3 | 1 | 0 | 39.785 |
| standard_3x4 / clean | 3 | 3 | 0 | 174.956 |
| standard_3x4 / perspective | 3 | 2 | 1 | 217.629 |
| standard_6x2 / blur | 3 | 3 | 1 | 102.890 |
| standard_6x2 / clean | 7 | 7 | 2 | 132.000 |
| standard_6x2 / low_resolution | 3 | 2 | 0 | 198.785 |

Marginal strata, every failed gate and complete conditional metrics are retained in the [compact continuation receipt](verification/2026-09-06-continuation.json). No superiority comparison is made against the earlier unpaired 16-patient pilot.

## Every attempted patient

| Case | Outcome | Semantics | RMSE (µV) | Correlation | Coverage | Reference-qualified |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 01 | quantitative_needs_review | passed | 170.483 | 0.8514 | 99.8% | no |
| 02 | quantitative_needs_review | failed | 118.286 | 0.9268 | 99.2% | no |
| 03 | quantitative_needs_review | passed | 29.026 | 0.9884 | 99.9% | yes |
| 04 | quantitative_needs_review | passed | 102.433 | 0.9319 | 94.3% | no |
| 05 | quantitative_needs_review | passed | 65.365 | 0.9082 | 99.8% | yes |
| 06 | abstention | unavailable | unavailable | unavailable | — | no |
| 07 | quantitative_needs_review | failed | 139.525 | 0.3237 | 87.0% | no |
| 08 | abstention | unavailable | unavailable | unavailable | — | no |
| 09 | quantitative_needs_review | failed | 54.870 | 0.9377 | 98.4% | no |
| 10 | quantitative_needs_review | failed | 39.489 | 0.9625 | 99.1% | no |
| 11 | quantitative_needs_review | failed | 67.637 | 0.8923 | 99.8% | no |
| 12 | abstention | unavailable | unavailable | unavailable | — | no |
| 13 | quantitative_needs_review | failed | 58.041 | 0.9261 | 99.9% | no |
| 14 | quantitative_needs_review | passed | 308.775 | 0.0589 | 97.4% | no |
| 15 | abstention | unavailable | unavailable | unavailable | — | no |
| 16 | quantitative_needs_review | failed | 39.785 | 0.9877 | 99.5% | no |
| 17 | quantitative_needs_review | passed | 45.157 | 0.9722 | 99.6% | yes |
| 18 | quantitative_needs_review | passed | 276.437 | -0.0138 | 98.9% | no |
| 19 | quantitative_needs_review | passed | 162.421 | 0.7552 | 99.6% | no |
| 20 | abstention | unavailable | unavailable | unavailable | — | no |
| 21 | quantitative_needs_review | failed | 82.557 | 0.9362 | 97.8% | no |
| 22 | quantitative_needs_review | passed | 24.085 | 0.9768 | 99.9% | yes |
| 23 | quantitative_needs_review | passed | 246.135 | 0.2432 | 82.8% | no |
| 24 | abstention | unavailable | unavailable | unavailable | — | no |
| 25 | quantitative_needs_review | passed | 270.558 | 0.0991 | 99.4% | no |

## Source and output inspection

All 19 returned cases had their original, diagnostic view and paper render inspected. This is an agent visual audit, not blinded adjudication or a human acceptance event. Visual findings do not alter the prespecified numerical classifications.

| Case | Recorded concern |
| --- | --- |
| 01 | Several tall narrow precordial R peaks, especially V3-V5, are strongly attenuated or truncated in the selected trace/render. V5 is particularly discrepant. Twelve panels and gross polarity are present; this is not sufficient fidelity or clinical acceptance. |
| 02 | V2 contains sharp downward excursions at the falling T-wave locations where the selected path meets tall V3 strokes in the source; this is visible row interference. V4 has missing narrow peaks and gaps. Some peak apices are already clipped by the rendered source boundary, limiting what can be reconstructed. The numerical semantic failure remains a failure; no visual or clinical acceptance. |
| 03 | All twelve panels, predominantly negative II/III/aVF and V2-V6 complexes, beat pattern and gross polarity follow the blurred source. Fine deflections and nadir amplitudes are limited by source resolution; visual inspection does not establish quantitative or clinical acceptance. |
| 04 | The 12x1 row order and baseline wander are recognizable, but overlapping tall precordial complexes are poorly recovered. V4/V5 contain attenuated or missing complexes and altered polarity; II/aVL/aVF and V4 have explicit gaps. V3 shows split notched troughs near crossing strokes. No quantitative or clinical acceptance inferred from the plausible overall render. |
| 05 | All twelve panels and beat timing are recognizable, but narrow negative III/aVR deflections and tall V3/V4 positive peaks are visibly shortened. T-wave shape and polarity are broadly retained. This does not offset the peak-amplitude loss or establish acceptance. |
| 07 | Low-resolution source shows negative V4 components and narrow III/aVF deflections that are substantially attenuated in the render; several panel ends develop abrupt baseline steps. The diagnostic overlay also draws through calibration pulses. It is not evidence that those pixels entered the canonical CSV; export scoring and segment boundaries remain authoritative. No clinical acceptance. |
| 09 | Several clearly visible V4/V5 QRS complexes are absent or replaced by gaps; only a subset of tall peaks survives. aVR negative deflections are markedly attenuated, and the right edge of V3 contains a large downward excursion not supported by that lead's source path. All twelve panels are represented, but selected morphology is not faithfully preserved. |
| 10 | All twelve 3x4 panels, beat counts, gross polarity and V2/V3 T-wave shapes are recognizable; no obvious wrong-row switch observed. Some narrow extrema are rounded or attenuated at source resolution. This qualitative check does not replace fixed semantic/fidelity scoring or confer acceptance. |
| 11 | The selected path visibly shortens narrow positive I/aVL/V4/V5 peaks and negative III/aVR/V1 troughs in the blurred source. Beat pattern, row identity and slower waveform contour are recognizable; narrow-feature fidelity is limited and no acceptance follows. |
| 13 | All twelve panels and broad beat pattern are represented. Narrow negative components in V1/V2/aVR and III are substantially attenuated; the deep V2 S components are particularly poorly retained. V3/V4 positive/negative complexes retain broad shape but reduced extrema. No acceptance follows from panel completeness. |
| 14 | The perspective source has large precordial complexes crossing adjacent rows; their source-labelled panel morphology is not preserved in the selected render. The V4-V6 panels lose the two tall source peaks and show displaced or inverted-looking complexes; several panel transitions have baseline steps. The numerical and semantic gates remain authoritative; the recognizable 3x4 layout does not establish correct panel reconstruction. |
| 16 | The corrected 12x1 row order, gross polarity and beat pattern generally follow the rotated source. aVL contains several explicit gaps and cut-off peaks in the middle/right portion near close aVR/aVL strokes. Other rows are more consistent at this viewing scale. This remains an unaccepted output; source review and numerical evidence are both required. |
| 17 | All twelve panels, broad negative V2-V4 complexes, gross polarity and beat pattern follow the source. Fine notches and extrema remain raster-limited; no obvious row jump or invented beat was seen at this viewing scale. Numerical qualification and clinical acceptance are separate. |
| 18 | Panel boundaries visibly disagree with source content: the final positive I complex appears under aVR, and neighbouring panel content is displaced across subsequent labels. The very tall V4/V5 peaks are largely absent or replaced by small complexes. V4 apices are clipped in the source, but visible lower strokes and V5 peaks also fail to reconstruct faithfully. A nominal 3x4 layout and plausible waveform are not sufficient identity/timing evidence; no acceptance. |
| 19 | Normal narrow V3-V5 peaks and several limb-lead extrema are strongly attenuated; V3 also loses the large positive component of the broad middle complex. The broad V1/V2 middle complex and surrounding T-wave contour are recognizable, but their visibility does not compensate for the missing narrower components. Twelve panels are present; no quantitative or clinical acceptance. |
| 21 | V3-V5 have missing or truncated complexes and explicit gaps; much of the right-hand V4/V5 source morphology is absent. Narrow negative aVR/V1/V2 components are also attenuated. Broad rhythm and baseline changes are recognizable in less affected regions. No visual or clinical acceptance; the losses remain recorded alongside the fixed numerical outcome. |
| 22 | All twelve panels, beat pattern and gross polarity broadly follow the perspective source, including low-amplitude limb leads and tall V1-V3 complexes relative to V4-V6. No obvious gross panel switch was observed at this viewing scale. Fine notches and negative components are smoothed; quantitative scoring and source coordinates remain necessary. No visual or clinical acceptance. |
| 23 | The tall narrow V3-V6 source peaks are strongly attenuated, and several narrow limb-lead negative components are lost. Additional rapid excursions appear at the right ends of the precordial panels; V5 also contains a deep extra excursion near its middle. All twelve panels and broad rhythm are recognizable, but low-resolution tracking is not faithful enough to infer acceptance. |
| 25 | The exported precordial panels visibly resemble continuations of the corresponding limb-lead rows instead of the source-labelled V1-V6 morphology. For example, V1 is rendered as positive I-like complexes although the source V1 complexes are predominantly negative; V4 is rendered as negative aVR-like complexes while the source has tall positive V4 complexes. Each rendered half contains two main beats versus four in the corresponding source panel. This is a gross identity/timing mismatch, not a successful twelve-lead reconstruction. No visual or clinical acceptance. |

Narrow peaks can be lost despite a plausible render or a passing aggregate score. Clipped source boundaries and overlapping traces further limit recoverability. The diagnostic paper render is baseline-centred and must not replace source/CSV evidence for amplitude or ST measurements.

## Resources and identities

MPS calls had median 82.24 s, nearest-rank p95 174.91 s and maximum 190.63 s. Peak sampled process-tree RSS was 7.779 GiB and workspace usage 0.349 GiB. The complete evaluation took 39.07 minutes.

Predeclared investigation limits were 660 s observed call time, 6 GiB sampled process-tree RSS and 1 GiB evaluation workspace, with one heavy evaluation at a time and a 600 s extraction deadline. Resource checks: observedCallBudget=pass, sampledProcessTreeRssBudget=FAIL, sampledWorkspaceBudget=pass, profilerCompleted=pass, resourceMeasurementsComplete=pass.

Sampling may count shared pages repeatedly and can miss brief peaks. Workspace size excludes the separately installed package/runtime and other retained experiments. These are observations, not hard OS-cap guarantees.

- Payload manifest: `e84bef376703e8029ac673cfbc1f85d6cbc5fbb1db608367ca221e7e4c22150b`.
- Runtime/model/dependency manifest: `d9465db541f0d2ad0541eb04f36e3fee930a8bbbc1a50135fe2950b63203fc3a`.
- Final protocol: `85985d1858b1731df026c05bc16681974caa5931c2a77ac6cef2b11a6665f303`.
- Fixture manifest: `a98aff5820f9694926d0b68c629ce8a8b9a6aa54aebd08b1bc37dcb59f90529d`.
- Completed report: `2a9cbfce9c9c06e2e11f9fb65afb23aecab694f60dbd65d9c59996293b268413`.

The complete local record is `benchmark/results/pro-completion-20260906/final-set-v2/`. The public receipt contains compact metrics and identities, not patient membership, waveform source files or model weights. These results do not confer release approval, model-rights clearance or passing hosted CI. See [the acceptance audit](ACCEPTANCE-AUDIT-2026-09-06.md) for remaining requirements.
