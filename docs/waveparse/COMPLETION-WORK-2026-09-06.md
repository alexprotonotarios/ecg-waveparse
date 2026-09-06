# Completion work following the acceptance audit

The original Pro report remains the requirement source. The earlier campaign
receipt records an earlier source snapshot; it does not establish completion
of the work below. This continuation has not been committed, published or
clinically validated.

## Printed settings and quantitative refusal

The pipeline now reads explicit printed speed/gain tokens from the untouched
source with provisioned local OCR. It retains the numeric alternatives,
confidence, original-image boxes and source hash. Unrelated OCR text is not
retained. Supported settings are 25/50 mm/s and 5/10/20 mm/mV. Defaults are not
reported as observations. Pulse assumptions (200 ms and 1 mV) remain explicit;
no acquisition sample rate is inferred from printing.

The shared Python/TypeScript contract recomputes conflicts and unsupported
settings. A conflict stops candidate execution before quantitative export,
while keeping the original available for inspection. Printed values alone do
not resolve an ambiguous physical grid. Old reports without printed evidence
retain their documented historical meaning.

Publication policy v3 identifies the new hard calibration gate. Its existing
numerical thresholds and selector feature contract remain unchanged; historical
policy v2 and its fitting inputs are retained for reproduction. New outcomes
therefore have a distinct policy identity even when their waveform is unchanged.

The benchmark renderers were also corrected: a 200 ms pulse spans 10 mm at
50 mm/s, and 20 mm/mV pulses need sufficient top margin. Both renderers now pass
image-based pulse detection across all six supported speed/gain combinations.
Their default 25 mm/s, 10 mm/mV rendering is unchanged.

The frozen ten-image development protocol covers all six supported combinations,
speed/gain conflicts, unsupported speed and absent settings. Actual Apple Vision
accurate OCR and RapidOCR each passed 10/10. The first sandboxed Apple execution
failed to start OCR; its subsequent fast-mode execution recognized text at
confidence 0.5 and failed the unchanged 0.85 threshold. Accurate mode is used
only for printed settings. Existing fast lead-label recognition is unchanged.
All attempts remain in `printed-calibration-receipt.json` under the local
`benchmark/results/pro-completion-20260906/` evidence directory.

The separately installed npm artifact also passed all three actual refusal
cases: source preparation, OCR, shared policy, no candidates/CSV, durable reload
and original-image inspection. These are synthetic refusal tests, not evidence
that every valid photographed ECG can be calibrated. Source SHA checks bind
the OCR evidence to the admitted original. Focused Python tests passed 96/96
and TypeScript tests passed 11/11 before the subsequent grid test additions.

## Detected grid correction

The experimental detector now estimates grid rows from coloured source pixels.
It requires consistent row spacing, continuous support and a shared smooth
vertical displacement. It refuses grey grids, missing support and row-dependent
distortions. Its versioned forward/inverse mapping records the spline nodes,
support bounds, one resampling and unknown absolute translation. Production
extraction continues to use its global geometry route.

The predefined image experiment uses four 12-panel sheets (one controlled
signal family), with curvature amplitudes 0/4/10/20 pixels. The current global
page correction is compared with detected local correction using the same crop
decoder. Global outputs are mapped back through their actual homography for
physical error scoring; crop identity and calibration are controlled in this
experiment. They are not obtained from the waveform truth by the grid detector.

| Curvature | Global waveform RMSE | Local waveform RMSE | Local grid-coordinate RMSE |
| --- | ---: | ---: | ---: |
| 0 px | 4.219 µV | 4.166 µV | 0 px |
| 4 px | 59.874 µV | 7.499 µV | 0.143 px |
| 10 px | 148.357 µV | 6.411 µV | 0.107 px |
| 20 px | 296.898 µV | 5.132 µV | 0.083 px |

All four sheets passed the frozen waveform, geometry, coverage and runtime
gates; all three unsupported constructions were refused. Round-trip error was
at most 5.7e-14 pixels. Each image completed within 1.2 seconds in this run.
The first harness run could not score globally transformed frames and counted
both affected cases as failures. The corrected harness scores those frames;
it uses identical image bytes and unchanged gates. Both reports are retained
as `detected-grid-v1` and `detected-grid-v2`. No patient or general photographic
performance claim follows from this family, and no production promotion is made.

## Completed continuation experiments

The continuation evidence is retained in
`benchmark/results/pro-completion-20260906/`. It now also includes:

- An actual pretrained per-lead heatmap comparison using the publicly available
  Felix ECG-Digitiser M1 checkpoint, pinned to its repository revision and LFS
  SHA-256. The CPU adapter uses 13 semantic channels, two threads, no ensemble
  or mirroring, and safe `weights_only` loading. It is a research adapter, not
  a claim to reproduce the upstream M3 ensemble. Nothing was trained, mirrored
  publicly or added to production.
- Sixteen identical crop inputs: only 10 learned paths had comparable samples;
  the complete-return, coverage and per-case regression gates failed. On the
  eight untouched full images, native candidates returned 8/8, directional
  candidates 6/8 and learned candidates 0/8 under the fixed geometry/identity
  and resource constraints. Keep the existing backends. These are complete
  candidate pipelines; they do not replace the application selection ablation.
- Two-sided verification of 168 actual reconstructed lead paths: it flagged
  68/70 paths above 75 µV RMSE, but also 83/98 paths at or below that threshold.
  This false-flag rate rules out using the prototype as an automatic veto.
  Text/annotation attribution is incomplete on these full images. Controlled
  attribution fixtures and the shared-mask limitation remain explicit.
- A scan/photo grouping audit: 6,600 metadata rows contain 700 signal-variant
  identifiers but only 100 underlying recordings, all already exposed in the
  development corpus. All 6,600 waveform-key lookups passed. The audit also
  identified 110 digital downsampling images historically labelled photographs.
  These records cannot become fresh final evaluation groups.
- Scorer v7 adds fixed-window signed amplitude and level-duration measurements,
  preserving raw/aligned frames, unavailable endpoints and errors introduced by
  the measurement detector itself. Three controlled source resolutions contain
  72 labels; 32 meet the predeclared raster visibility proxy. On unchanged truth,
  mean detector error is below 0.002 ms and 0.02 µV. Actual installed extraction
  returned the 6/3 pixels-per-mm images and abstained on the 1 pixel-per-mm image.
  All 28 returned visible endpoints met the fixed 75 µV amplitude / 20 ms duration
  limits in both raw and aligned frames. The remaining four visible endpoints
  are unavailable: 28/32 (87.5%) coverage fails the preset 90% coverage gate.
  All 72 labels remain in the report. Source and diagnostic panels were visually
  inspected; these are controlled engineering measurements, not clinical QRS/QT.
- Policy-v3 regression returned nine of eleven inputs (the original eight plus
  the three labelled sheets), with two abstentions and no runtime failures.
  The seven original returned canonical CSVs remained byte-identical to baseline.
  Its 253,966 compared samples and pooled 27.262 µV RMSE are conditional on the
  nine returns across two controlled families; they are not a patient estimate.
- Eight separately grouped PMcardio development recordings were then evaluated
  from their exact source/truth bytes: four photographs, two scans and two
  digital exports. The first run returned no quantitative outputs, with six
  abstentions and two runtime failures. Both failures arose because OpenCV
  applied EXIF orientation to printed-setting coordinates while admission and
  preparation retained stored raster coordinates. The fix preserves that raster
  frame; all eight EXIF rotations/reflections match the admitted pixels in tests.
  The repeated eight-input run had eight abstentions and no runtime failures.
  Yield and reference-qualified yield remain 0/8; RMSE/correlation/coverage are
  unavailable. The 95% Wilson yield interval is approximately 0–32.4%, and the
  recordings were already exposed development data. This is a reliability fix,
  not evidence of useful scan/photo reconstruction. Original and repeated runs
  are retained as `capture-evaluation-v1` and `capture-evaluation-v2`.
- Actual first/warm inference now covers four inputs on each device, with one
  new model and two repeat forwards per input: 24 forwards total. Each device
  produced nine finite candidates and three empty 12x1 arrays. Within each
  device, repeated arrays, missingness, layouts and scales were exact. CPU/MPS
  first-forward RMSE differences were 0.173/2.972/6.290 µV for the three nonempty
  inputs, with maximum point differences up to 254.67 µV. This is backend-stage
  evidence, not application policy or uncertainty parity. Peak sampled RSS was
  8.800 GiB CPU / 2.726 GiB MPS. Warm runs were not consistently faster, so no
  persistent worker or cache is adopted. `warm-inference-receipt-v2.json` corrects
  a derived-summary field error; it does not replace or rerun the measured forwards.
- A fresh lightweight environment now passes all 88 fast Python checks, with
  OpenCV explicitly declared and module imports fixed for the CI invocation.
  The 209-test TypeScript run passed 207 in the sandbox; the two localhost-server
  checks were blocked by sandbox permissions and passed with localhost enabled.
  Lint and type checking passed. Policy-v3 installed calibration refusals passed
  3/3; an earlier invocation's three missing-fixture-path failures are retained.

The selection/resource ablation completed production and exhaustive profiles
separately on four inputs from three groups (eight application runs). Both
profiles returned the same two controlled cases and abstained on the two
capture recordings. On common returned cases, mean RMSE was 82.840 µV for the
development-chosen fixed native backend, 21.296 µV for either application
selector and 20.407 µV for the evaluation-only exhaustive oracle. Production
took 79–139 seconds per input; exhaustive selection took 104–201 seconds.
Extra search produced no selected-output improvement in this comparison, so
the current production policy remains. Candidate ancestry, subprocess memory,
timings and all missing/failed candidates are retained. Standalone candidate
identity is not copied from the selected output. This experiment used the
EXIF-v4 payload, before the gain/grid corrections below; it is not independent
validation of a newly fitted selector.

Actual neural cancellation also completed through both installed interfaces.
A local test hook recorded the first convolution returning inside the real
upstream U-Net before cancellation. JavaScript cleanup took 153.565 ms and
Python cleanup 145.205 ms. Both left a coherent failed record, inspectable
original, no quantitative CSV and no surviving inference child. This is measured
cancellation on EXIF-v4, not a throughput estimate or only an admission-stage
abort. The hook did not modify installed code or model weights.

## Physical units in the complete installed pipeline

The full-page setting experiment found a real error beyond successful OCR:
20 mm/mV was detected and recorded, but the neural normalizer still used its
default 10 mm/mV conversion. The wrapper now passes the trusted gain into the
upstream physical-unit conversion and carries it through repeats and CPU
confirmation. Default 10 mm/mV inference configuration is unchanged. Historical
selector priors fitted at 10 mm/mV are incompatible with other gains; they are
not extrapolated to the corrected values. Unresolved or low-confidence physical
calibration for a known nondefault setting cannot silently fall back to defaults.

The detector also had a 3.33% horizontal-scale bias on one 25 mm/s page because
its short calibration pulse span dominated the estimate. A conservative median
of consistent, already detected row-grid periods now refines that scale while
retaining the prior estimate and its evidence. Unresolved grid periods,
inconsistent rows or implausible changes are rejected. Vertical and horizontal
scales remain separate. Eight constructed pages plus two unequal-axis resizes
pass the predefined 2% physical-coordinate tolerance; no local image warp was
promoted into production.

The combined corrections were built into the installed calibration-v6c payload
(`e84bef376703e8029ac673cfbc1f85d6cbc5fbb1db608367ca221e7e4c22150b`).
Its eight full application attempts cover 25/50 mm/s, 5/10/20 mm/mV and short/
long page durations. Five returned reviewable quantitative output, three
abstained and none failed at runtime. All five returns passed the fixed scale,
semantic, RMSE, correlation and coverage gates. Their 149,720 compared samples
had pooled RMSE 31.763 µV, mean correlation 0.96445 and mean coverage 0.99813.
Maximum returned absolute horizontal scale error was 0.312%; vertical error
was 0%. These are conditional results from one controlled family.

On the identical 20 mm/mV page, RMSE fell from 140.258 to 30.281 µV after gain
correction. Its source, diagnostic view and paper render were visually compared:
the twelve panels, QRS polarity and V1 morphology follow the source. Numerical
gain accuracy comes from the canonical CSV/truth comparison, not the render.
No human acceptance event was created. The 25 mm/s, 5 mm/mV input lacked trusted
identity; both nondefault durations lacked a publishable candidate. All three
retain their original and export no quantitative signal. They are refusals,
not fidelity passes. Prior failed harness attempts and the pre-fix outputs
remain in the evidence directory.

The combined implementation passed 213 TypeScript and 293 Python tests, plus
focused checks added for CI evidence comparison. Both installed packages have
37 identical runtime files, with every hash checked against the shared payload
manifest. The first declaration build failed because an ES2023 array method was
used under the ES2022 contract; replacing it with compatible sorting fixed the
build without raising the package's language target.

## Installed interface, device and source-install checks

The current installed matrix completed twelve calls: three input shapes/
resolutions from one deterministic family, both languages, and CPU/MPS. All
twelve returned reviewable outputs with strict semantic passes and no overall
runtime failures. Corresponding Python/JavaScript waveforms, missingness,
uncertainty statuses/spreads and terminal review outcomes were exact on each
device. Seven of nine full comparison groups passed their predeclared limits.

One low-resolution CPU comparison failed exact candidate-count equality. Its
JavaScript geometry candidate failed at 180.94 seconds, consistent with the
configured 180-second candidate deadline; the Python candidate completed at
166.17 seconds. Both selected the same waveform. CPU versus MPS also failed
full selected-policy identity and the correlation margin for that input: both
remained `needs_review` / `reviewable_constrained`, but selected different
candidates. CPU RMSE was 30.199 µV versus 26.383 µV on MPS; mean correlation
was 0.966257 versus 0.975562. The 0.009305 loss exceeds the fixed 0.005 margin.
No comparison or tolerance was removed to make this a pass.

Median/max call time was 111.03/120.65 seconds on MPS and 300.33/484.77 seconds
on CPU (six calls each). Peak sampled process-tree RSS was 3.523 GiB MPS and
9.193 GiB CPU. The latter exceeds the 9 GiB investigation budget in one run;
all call-time and workspace-size budgets passed. Shared pages can be counted
in multiple processes; this is not a hard OS memory cap or an OOM observation.
Full numerical/platform equivalence is not established. No persistent worker,
cache, timeout extension or selector retuning is adopted.

The first matrix launcher followed the Python virtualenv's executable symlink
to the system interpreter, bypassing its installed wheel. That attempt was
stopped after two JavaScript completions and one Python import failure. The
corrected launcher preserves the virtualenv path and the complete twelve-call
repeat above. Planned and actual start counts are now separate. Both attempts
and their source/payload identities remain retained.

The source distribution now installs nine packages on this Mac, compared with
589 for the previous application-derived toolchain. It retains only TypeScript,
esbuild, tsx, Node declarations and a YAML parser. The source lock contains the
34 reachable package records across platforms rather than all 691 application
records. Every retained version/integrity record and both supported platform
binaries match the original lock. UI dependencies remain in the application
checkout; UI-only helpers are excluded from the public library export.

A fresh source install rebuilt both distributions, passed 108 package tests
and passed installed ESM/CommonJS, NodeNext, Python sync/async and source-allowlist
checks. Its runtime payload and all executable language entry points are byte-
identical to calibration-v6c; the generated source lock also reproduces exactly.
The preceding clean-source run passed 92 Python contracts. Lint and the local
Next.js production build passed. Source audit found no images, models, original
Git history, symlinks or recognised credential patterns; its documented limits
remain. A NodeNext declaration import-query defect was found and corrected
during these consumer checks; its failed attempt is retained. These are local
results, not hosted CI or publication.

## Completed final evaluation and development regression

The final engineering set has a frozen 25-patient PTB membership, excluding
all 17 documented locally exposed PTB patient groups. Its planning target is a
coarse approximately ±20 percentage-point overall yield estimate, not per-stratum
or clinical precision. Membership was pinned before waveform download; source,
policy, scorer and gates were frozen before final execution. The owner-only
collection uses a one-use access log; cross-dataset subject linkage and
model-training overlap remain unknown. Its first source lock was superseded for
the development gain/grid fixes before any final outcome was exposed. The v2
protocol preserves membership and gates; automated checks prove
all 75 prepared source/truth/annotation files are byte-identical to v1. The
renewed evaluation source lock remains valid after the packaging-only changes.

The [final evaluation](FINAL-EVALUATION-2026-09-06.md) completed all 25 attempts
once, with 19 quantitative returns, six abstentions and no overall runtime
failures. Four of 25 qualified under every frozen gate (16.0%; 95% Wilson interval
6.4%–34.7%). Among the 19 returns and 539,564 compared samples, pooled RMSE was
136.472 µV, mean patient correlation 0.71918 and mean coverage 97.5%. Strict
semantics passed for 11 and failed for eight. All 19 source/diagnostic/render
sets were inspected, with visible morphology and identity failures recorded.
No output was accepted and no final outcome was used for retuning.

Median/p95/max call times were 82.244/174.909/190.628 seconds. Peak sampled
process-tree RSS was 7.779 GiB, exceeding the predeclared 6 GiB budget; the
660-second call and 1 GiB workspace budgets passed. These controlled renders of
acquired waveforms are not actual scan/photo validation. All patient intervals,
strata and failed gates remain in the
[current compact receipt](verification/2026-09-06-continuation.json).

The current eleven-input development regression also completed: nine returned,
two abstained and none failed at runtime. All nine canonical CSVs are byte-
identical to the preceding installed regression; all eleven source, truth,
terminal outcome and publication-decision identities reproduce. The 253,966
compared samples have pooled RMSE 27.262 µV, mean correlation 0.94580 and mean
coverage 99.5%. The library gate passes 66/66; the historical two-of-32 failures
remain unchanged. This is regression reproduction, not independent accuracy.

## Remaining delivery requirements

The [criteria audit](ACCEPTANCE-AUDIT-2026-09-06.md) remains open for exact-candidate
hosted release verification and the release owner's exact-artifact decision.
Final report/archive identities are recorded separately from the numerical
payload so documentation changes do not masquerade as new inference evidence.
CI now prepares
one shared set of fixture bytes for both native platforms and compares waveform,
missingness, uncertainty, semantics and policy under predefined limits. The
aggregate check runs even when its prerequisite jobs fail; identical refusals
are labelled refusal parity. Quantitative drift, altered inputs, duplicates,
missing cases and evidence tampering fail the local comparison tests. These are
prepared workflow changes, not observed hosted results.

GitHub currently refuses branch protection on
the private development repository's plan; the public source-preview repository
supports the feature but currently has no protection. Neither fact is a passing
required-check enforcement result.
