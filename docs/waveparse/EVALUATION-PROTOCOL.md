# Evaluation collections and annotation protocol, v1

Frozen protocol date: 6 September 2026. Engineering thresholds are not clinical
validation. No new final patient cohort is claimed or available in this checkout.

## Collections and admission

| Collection | Membership and use | Evidence boundary |
| --- | --- | --- |
| Fast regression | Frozen eight-image campaign; one deterministic signal family | Repeatedly inspected; regression only |
| Development/tuning | Existing paired corpora and new controlled geometry/decoder fixtures | All previously accessed splits are development evidence, including historically named held-out splits |
| Frozen public engineering pilot | 16 distinct PTB patient groups; waveform membership, image degradations and qualification gates frozen before waveform download | Acquired signals rendered to images; no photographed/scanned source or independent model-training-overlap claim |
| Final evaluation | Reserved, access-controlled patient/recording cohort; currently zero admitted records | Requires lawful truth, unexposed groups and a membership/code/gate freeze before any run; external custodianship is useful but is not a Pro-report prerequisite |

The public register is `benchmark/protocols/evaluation-collections.v1.json`.
Private membership remains in controlled manifests. Each admitted final case
requires `evaluationProvenance`: collection, licence-evidence reference, truth
method, pretrained-training overlap (`known`, `none_documented`, or `unknown`),
`previouslyExposed: false`, and an access-control reference. Absence is not
clearance. Known training overlap is reported as its own stratum and excluded
from a claim of model-independent validation.

Group by patient, recording and original signal/source **before** rendering,
photographing or cropping. Aliases and identical image/truth hashes cannot cross
splits. Synthetic families cannot masquerade as independent patient records.
The validator emits a membership digest and counts without patient identifiers.
Retain that digest, exact image/truth/annotation hashes, source/payload/model
hashes, scorer, gates and review policy for every evaluation.

Before final admission, the evaluation owner records inclusion/exclusion criteria,
intended use, endpoint precision and strata. Plan sample size at the independent
recording/patient level. A yield proportion with a desired 95% half-width `h`
can start from `1.96² p(1-p)/h²`, with conservative `p=.5`; then account for
stratification and clustering. This approximation does not replace a formal
sample-size plan for correlated or rare endpoints. No arbitrary image count
establishes validation.

Every final access is logged with membership digest, protocol digest, source
identity, purpose, requester, date and whether outcomes were exposed. A set
used to change code, thresholds or fitting becomes development data. A fresh
evaluation set is then required for the next final claim.

## Required strata and labels

Record layout and repeated/rhythm segments; vendor/template; scan, photograph
or screenshot; source resolution; grid colour and minor/major period; JPEG
compression; blur; rotation, perspective and curvature; annotations; overlapping
traces; clipping; and missing/conflicting calibration. Include unfamiliar
layouts, cropped/swapped labels, unsupported durations/settings and non-ECG
inputs as refusal cases. A missing label is not a normal-layout label.

Annotate narrow notches, steep/negative/broad QRS components, low-amplitude
segments, pacing-like marks and clipped regions. Store original truth sample
indices and original-image regions, with separate visible, ambiguous and
unrecoverable states at the **actual source resolution**. Subpixel events that
disappear in the raster cannot be scored as visible omissions. Distinguish a
waveform from text, calibration, grid and other marks; preserve uncertain masks.

Controlled fixtures use exact construction landmarks. Patient landmarks require
two readers and adjudication, recording disagreement. Report interval/amplitude
errors against these labels separately from errors introduced by any automated
measurement detector. Never infer simultaneity from panel position or algebra.

## Required report

Use `benchmark/protocols/report-template.v1.json`. Report every attempted input,
returned/partial signals, diagnostic-only outcomes, abstentions, timeouts,
infrastructure/scoring errors and missing results. Report semantic identity,
placement, calibration and missingness independently of local aligned fidelity.
Reference-qualified yield requires prespecified semantic **and** endpoint gates;
without those gates it is `null`, never equivalent to returned-signal yield.

Include raw/aligned RMSE, correlation, coverage, event/interval/amplitude errors,
false deflections, annotation recovery, uncertainty discrimination, CPU/device,
runtime/RSS/disk and human review state. Report stratum and common-returned-case
comparisons plus all-attempted yield. Bootstrap independent group means, not
image variants; one synthetic family has no defensible group-level interval.

The executable accounting lives in `ecg_benchmark/evaluation.py` and is used by
the benchmark report. Tests reject alias/hash leakage, previously exposed final
data, duplicate results, missing gates and artificially inflated group counts.
