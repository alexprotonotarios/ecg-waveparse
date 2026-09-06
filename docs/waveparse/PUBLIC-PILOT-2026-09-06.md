# Frozen public waveform engineering pilot — 6 September 2026

**16 attempted inputs from 16 distinct PTB patient groups: 12 quantitative
outputs for review, four abstentions, zero infrastructure failures, six
reference-qualified outputs, zero accepted runs.** Returned-output yield was
75%; reference-qualified yield was **37.5% of all attempts**. This supports no
expanded accuracy or clinical-use claim.

The [versioned receipt](verification/2026-09-06-public-pilot.json) contains
per-case/per-lead results, raw and aligned error, error tails, semantics,
uncertainty bins and risk versus retained coverage, input/prediction hashes,
package/scorer identities, resource measurements and visual observations.

## Corpus and frozen choices

The source is the [PTB Diagnostic ECG Database v1.0.0](https://physionet.org/content/ptbdb/1.0.0/),
whose files are listed under Open Data Commons Attribution License v1.0.
Attribution: Bousseljot, Kreiseler and Schnabel, *Biomedizinische Technik* 40,
supplement 1 (1995), p317; and [Pollard et al., PhysioNet as a global platform
for biomedical research](https://doi.org/10.1038/s44360-026-00096-z) (2026).

Patient directories were ordered by a fixed hash salt; one recording per
patient was selected before waveform download. Previously used PTB patient262
was excluded. The protocol hash is
`5f2862f72be4345c9029d54f1b18afeacb45a0b41b4f350d8883f73378aeba90`.
The 16-group plan used a coarse overall yield precision calculation, not a
clinical sample-size justification. Membership, layout, image conditions and
gates were frozen together.

Verified 1000 Hz source waveforms were resampled to 500 Hz with padding, cropped
to seconds 10–20, median-centred and rendered as ten-second ECG pages. Source
file hashes and per-channel WFDB checksums were checked. The independent
reference represents the visible panels. Images are **digitally rendered
acquired waveforms**, not photographs or scans. Patient grouping is known
within PTB; cross-dataset subject linkage and pretrained-model overlap remain
unknown. No patient morphology/interval annotations were invented.

Qualification required strict segment semantics, per-case aligned RMSE ≤100 µV,
mean lead correlation ≥0.90 and mean lead coverage ≥0.90. These are engineering
gates, not universal clinical thresholds. Alignment allows at most 40 ms and a
median baseline correction; raw error remains separately reported.

## Results and limitations

Among the 12 returned outputs, **412,036 samples** were compared. Pooled aligned
RMSE was **121.15 µV**; mean case RMSE was **103.57 µV**, mean correlation
**0.7700**, and mean coverage **98.17%**. Nine outputs passed strict semantics;
three retained unverified source-label identity. Only six passed semantics and
the numerical gates together.

| Layout | Attempts | Returned | Reference-qualified | Mean case RMSE among returned | Mean correlation | Mean coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6×2 | 8 | 7 | 2 | 125.28 µV | 0.7188 | 97.74% |
| 3×4 | 4 | 2 | 1 | 124.95 µV | 0.6427 | 99.46% |
| 12×1 | 4 | 3 | 3 | 38.65 µV | 0.9742 | 98.30% |

These small, differently constituted strata do not establish a layout ranking
or a causal effect of image degradation. Both perspective cases abstained.
The worst returned case had RMSE 321.86 µV and correlation 0.0514; visual review
showed substantial precordial morphology and beat-presentation disagreement.
High coverage and a plausible paper render did not make that result accurate.

Exploratory patient-group bootstrap 95% intervals were 50–93.75% for returned
yield, 12.5–62.5% for reference-qualified yield, 57.48–158.27 µV for mean case
RMSE, and 0.576–0.942 for mean correlation. They do not support narrow
per-stratum or clinical precision claims.

All 16 source/diagnostic/paper overviews were visually inspected. The audit
records altered peaks/components, a clear wrong-morphology result, explicit
gaps and the four preserved-source abstentions. This was a resized overview,
not full-resolution two-reader clinical adjudication. No acceptance decision
was submitted. Calibrated source settings remain inferred from pulse/grid
evidence; printed-setting recognition and acquisition sampling are not claimed.

The network-denied macOS/MPS run took 1,527.51 seconds. Median call time was
82.84 seconds; nearest-rank p95/max was 168.52 seconds (n=16). Sampled process-tree
RSS peaked at 10.35 GB and workspace size at 297.52 MB; no sampling errors
occurred. RSS may count shared pages more than once, and runtime/package storage
is separate. These are observations, not broad capacity guarantees.

## Evaluation corrections and exposure

The first report wrongly treated the ten-second page duration as each panel's
duration, and the original scorer assumed lead-local truth support. That
incorrectly failed later panels. A corrected manifest records 5, 2.5 or 10
seconds per panel and explicitly marks canonical-display truth. A second audit
made uncertainty follow the CSV's explicit `canonical_sample` indices when
comparing full-page truth. Tests cover all three layouts, internal gaps, later
panels and compact/canonical equivalence.

The original report, corrected metadata and every rescore remain retained.
Final scoring is `public-pilot-rescored-v3`; its receipt binds the correction
chain. All image, truth and extracted CSV bytes remained fixed. All waveform
fidelity values remained identical; semantic and uncertainty accounting were
corrected. Gates and extraction policy were not changed using pilot outcomes.

Extraction used installed package C, payload
`e76950ceef9fc9bb19fe5a8a449b88ed47eddeebbed2864bd3128358ee825019`.
Final evaluation source is
`174fd48d5c1e8aca8b0ca65e2c85c4c9aa5f8ce2709e4d66969397c9530f8f98`.
The pilot is now exposed. Any future tuning using it is development work and
requires a fresh evaluation set. The reserved, access-controlled final clinical
collection still contains zero admitted records.
