# Scoring contract, version 7

Version 7 adds labelled measurement endpoints to version 6's coordinate,
waveform, semantic and uncertainty metrics. Existing metric definitions and
historical reports remain unchanged.

`rawRmseUv` is explicitly **lead-local, unaligned**. Aligned metrics remove a
bounded time shift and median baseline offset on that local frame. Neither
metric validates the printed panel or acquisition simultaneity. Official-style
PhysioNet compatibility remains a separately labelled metric family.

Every lead reports `frameTransform`: input counts/rates, selected panels,
panel/crop offsets, gap-preserving resampling, alignment shift, baseline
correction and compared overlap. Frame selection precedes resampling so that
endpoint rounding cannot accidentally defeat canonical-panel detection.

`--coordinate-contract` supplies independent expected segment IDs, lead IDs,
panel indices, polarity, half-open display-time support and page duration.
Alternatively `--case-metadata` derives these expectations from a benchmark's
declared layout and duration. Candidate values never select the expected panel.
`segmentDurationSeconds` means one panel, not the whole page. Truth explicitly
declares `lead_local` or `canonical_display`; canonical truth retains its
independently specified panel offset and internal gaps. Its array length must
match the declared frame.
`--candidate-segments` supplies the exported identity record for comparison;
CSV headers alone leave source-segment identity unverified. Missing metadata
returns `not_evaluated`, never an implicit pass.

Strict semantics reports original-canvas length, expected/returned/missing
support, samples outside that support, unexpected leads and identity mismatches.
Missingness is measured separately from placement; a correctly located partial
signal is not automatically reference-qualified. Repeated lead segments require
an explicit multi-segment representation and are refused by this one-segment-
per-column scorer rather than silently overwritten.

All-NaN or single-point leads have no comparable waveform result (the existing
alignment requires 20 comparable samples). Unequal canvases and duplicated
panels fail strict placement. Internal and boundary gaps retain their positions;
derivative comparisons never connect across missing samples. Alignment does not
invert polarity, relabel leads or change the time/voltage scale.

Annotations use original truth sample indices. Uncertainty uses exported
`lead_sample` and, when present, explicit `canonical_sample` indices. The
canonical map is used for an input spanning that canvas; a compact input uses
lead-local indices. Both map through the declared crops and sample rates;
uncertainty additionally follows the alignment shift. Interpolated locations
inherit both neighbours' limitations, and absent uncertainty records remain
unavailable. Candidate spread is a disagreement score, not a calibrated
confidence interval. Tests cover nonzero shifts, resampling and gaps.

Semantic status and quantitative gate status must accompany conditional error
metrics. A local shape score cannot compensate for a wrong panel or identity.

## Controlled or adjudicated measurements

Optional per-lead `measurements` annotations name signed amplitude or duration
above a fixed level. Their label method, source visibility, physical reference
value, polarity and search/baseline windows are frozen before extraction. Windows
follow the same panel/crop/rate mapping as event annotations, retaining fractional
sample coordinates. Missing samples or multiple crossing candidates yield an
unavailable/ambiguous measurement; the scorer does not bridge them or choose a
truth-favoured crossing.

Raw and aligned measurement reports each retain the complete label denominator.
They separate measurement-detector error on truth, the increment from replacing
truth by the reconstruction, and total error against the independent label.
Non-visible source features remain counted but excluded from the source-visible
accuracy denominator. Signed amplitudes preserve negative complexes. These
fixed engineering endpoints are not validated clinical QRS/QT delineation.

Controlled raster labels use a recorded counterfactual render of the same
synthetic construction with only the labelled feature removed. Contrast and
source-pixel localisation support define a conservative recoverability proxy;
pixel difference alone does not establish visibility. Counterfactuals are never
used to alter a clinical source or fill a reconstructed waveform. The source
image, counterfactual hashes, source region and ambiguity decision are retained.
