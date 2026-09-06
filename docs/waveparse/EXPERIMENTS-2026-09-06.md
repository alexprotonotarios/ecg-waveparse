# Decoder, geometry, verification and selection experiments

These are engineering experiments with explicit non-adoption decisions. The
production decoder, candidate policy and historical selector profile remain
unchanged. None of the results establish diagnostic accuracy.

## Controlled decoder comparison

`benchmark/protocols/decoder-experiment.v1.json` was frozen before execution.
`scripts/run_engineering_experiments.py` evaluates four backends on identical
calibrated crops and source-darkness masks. Sixteen crops cover eight constructed
morphologies at two raster scales. They derive from **one piecewise signal
family**, not sixteen independent recordings. All 64 backend attempts returned.

| Backend | Mean case RMSE, µV |
| --- | ---: |
| Upstream probability centroid | 18.886 |
| Current WaveParse probability ridge | 30.556 |
| Native connected ink | 30.844 |
| Experimental direction/connection tracer | 31.860 |

The experimental tracer's paired improvement was **−1.304 µV** against the
current ridge, failing the required 5 µV mean improvement. The protocol also
limits individual regression to 10 µV, coverage loss to one percentage point,
and runtime to five seconds per crop. The maximum observed alternative runtime
was below 0.1 seconds. Gaps remain missing and ambiguous vertical branches are
flagged. Scale-dependent results remain separate in the retained report.

**Decision:** retain the current backend. The alternative failed the crop gate
for admission to a full-pipeline promotion test. The centroid's advantage on
these constructed masks does not establish a full-image segmentation advantage.
The ordinary full pipeline is separately checked on untouched regression inputs
and the frozen public waveform pilot.

The candidate [heatmap implementation](https://github.com/dangnh0611/kaggle_ecg_digitization)
was inspected at commit `8358d5b09a5b9519275e9b54c11716da5752a9d6`.
Its repository declares MIT, but its linked
[checkpoint dataset](https://www.kaggle.com/datasets/dangnh0611/ecg-checkpoints)
reported licence `Unknown` at inspection (version 11). Repository licensing does
not settle checkpoint rights. Weights were not downloaded, mirrored or run.
The README's training hardware requirements are not an inference measurement.
**Decision:** no adoption or performance claim without identifiable weights,
applicable rights and a supported, bounded inference recipe.

## Geometry and physical calibration

`sourceTransformChain` records original-to-working scaling, homography and crop,
plus the inverse, using pixel-edge coordinates. Horizontal and vertical scales
remain distinct. Geometry metadata does not perform waveform resampling.
Published provenance records the separate decoder/resampling operations.

Known-point transform tests round-trip within 1e-8 pixels. The calibration
detector passes six combinations of 25/50 mm/s and 5/10/20 mm/mV on a 4 px/mm
grid: exact selected speed/gain, with coordinate budgets of 0.1 px/mm, 1 ms and
0.01 mV. Existing tests cover ambiguous major/minor periods and refusal without
sufficient pulse/grid evidence. The pulse assumes 200 ms and 1 mV; printed
settings are explicitly **unresolved**, not silently confirmed by defaults.

The experimental local grid spline uses known landmarks on four rows. Flat,
10-pixel-curved and 20-pixel-curved cases gave global affine residuals of 0,
3.545 and 7.090 pixels versus local residuals of 0, 0.0056 and 0.0112 pixels.
Maximum round-trip error was 2.84e-14 pixels. Folds, unsupported extrapolation,
row disagreement and invalid slope budgets are refused.

**Decision:** keep `VerticalGridWarp` as a callable experiment with no production
route. This is a coordinate feasibility result; it needs independent grid-node
detection and full-image comparisons before automatic correction is justified.

## Two-sided source verification

`verify_source_trace` compares mapped paths with the original source within a
lead region. It reports both unsupported path points and omitted source ink,
local distance arrays and component evidence. Grid, text, pulse, annotation and
ambiguous masks have separate roles; missing exclusions make attribution
incomplete. Shared decoder masks must be declared. Nothing alters samples or
normalises unusual morphology.

| Known construction | Unsupported path fraction | Omitted ink fraction |
| --- | ---: | ---: |
| True irregular path | 0 | 0 |
| Omitted apex | 0.1167 | 0.2429 |
| Fabricated notch | 0.0500 | 0.0314 |
| Wrong row | 1 | 1 |
| Text-following path | 0.1333 | 0.1029 |

All four faults exceed the frozen 0.02 fault fraction, and the true path meets
both 0.02 maxima. Thin ambiguous ink is retained as ambiguity, and NaN gaps are
never bridged. This controlled evaluation uses known source constructions
independent of the selected model's segmentation. It is not patient validation
or a calibrated acceptance threshold. **Decision:** advisory experimental
evidence only; no production eligibility change.

## Selection and fallback ablation

`scripts/selection_ablation.py` reads the exact historical 100-case extraction
run used for calibration: 72 development and 28 already-exposed validation
cases. It hashes 704 score files. The original extraction identity ends in
`-dirty`, and the exact dirty source is unavailable. Scoring predates strict
semantic v6; every reference-qualified yield is therefore unavailable.

| Policy | Returned / 100 | Mean returned-case RMSE, µV | Coverage | Correlation |
| --- | ---: | ---: | ---: | ---: |
| Fixed default | 43 | 163.00 | 0.9300 | 0.6275 |
| Fixed native, selected using development only | 56 | 186.01 | 0.9638 | 0.5471 |
| Default, native only when output absent | 66 | 157.79 | 0.9430 | 0.6433 |
| Historical actual selector | 6 | 51.39 | 0.9525 | 0.9529 |
| Offline truth oracle over generated candidates | 84 | 105.98 | 0.9384 | 0.8670 |

The actual selector returned 6/72 development cases and 0/28 validation cases.
Fixed/raw candidates have not passed that selector's eligibility policy;
their higher return rate is not usable or clinically qualified yield.
On the five common returned cases, actual selection, the development-chosen
fixed candidate and the oracle had mean RMSE 44.93, 56.94 and 39.48 µV.
Every additional missing-only fallback has its own row in the retained receipt.
Total generated-candidate runtime was 32,923,099 ms; this summed work is not
wall time, and selected-candidate times exclude preprocessing/shared caches.

**Decision:** no selector simplification or threshold change is supported by
this exposed, incomplete historical evidence. Candidate ancestry now records
shared engine models, preprocessing, geometry, calibration and parents without
calling different IDs independent. Fusion and peer repair record interval
contributors and time/amplitude transforms. The oracle remains offline only.

## Retained evidence

Local campaign directory: `benchmark/results/gpt6pro-20260906/`.
`controlled-experiments-v2/report.json` supersedes v1's mistaken independent
family grouping; numerical results are unchanged. V1 is retained for audit.
`selection-ablation.json` retains all policies, denominators, hashes and caveats.
These working evidence files are excluded from public package/source archives.
