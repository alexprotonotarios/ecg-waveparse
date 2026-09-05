# Review of the two historical smoke mismatches

Reviewed 5 September 2026 against the installed 0.1.0 payload
`ade3c347c983238365078466537f950166684da5956e079fe338039cec19265f`.
The original six-case gate configuration and its failed 30/32 comparison are
preserved. A separate installed-library configuration now checks the current
review policy and quantitative recovery: **66/66 checks pass** on the same saved
regression run. This is an engineering review, not clinical acceptance.

## Why the old expectations differ

The original smoke expectations required both cases to produce 12 leads, global
RMSE no greater than 40 µV, uncertainty AUROC at least 0.8 and safe usable fraction
at least 0.65. In the data-bearing workspace history, commit `d4a52cc` replaced
these recovery checks with expected refusals. Commit `f608a2f` changed the
perspective refusal reason. Neither threshold change is hidden by this review.

Subsequently, commit `177def7` (21 August) deliberately changed the publication
policy: constrained recoverable traces remain `needs_review`, and input-quality
classifiers attach fidelity warnings instead of forcing refusal. Current policy
tests assert this behaviour. The old smoke refusal expectations were not updated.
Those historical commits remain in the private workspace; their git history is
not part of the clean release repository.

The new `benchmark/suites/waveparse-smoke-gates.json` restores the original
40 µV / 0.8 / 0.65 limits for these two cases. It also requires the actual app
status and publication outcome to remain `needs_review`, with `reviewRequired`
explicitly true. A benchmark row labelled `completed` means scoring completed;
it must not be confused with a user accepting a result in the app.

## Evidence and limitations

| Measure | 900-pixel low-resolution page | Perspective page |
| --- | ---: | ---: |
| Global aligned RMSE | 26.38 µV | 18.51 µV |
| Mean lead correlation | 0.9756 | 0.9882 |
| Mean lead coverage | 99.99% | 99.49% |
| Uncertainty AUROC | 0.9859 | 0.9510 |
| Safe usable fraction | 95.83% | 75.93% |
| Annotated events preserved | 2/3 | 3/3 |
| Reported effective sampling | 90 Hz | 220 Hz |
| Confidence | lower | lower |
| Review state | needs_review | needs_review |

Both cases recover twelve correctly positioned traces in the inspected source
and diagnostic views, with no gross row swap or calibration pulse selected as a
lead. This visual inspection does not certify subtle waveform detail. All sixteen
source/export identities and all thirty installed runtime payload files were
rechecked; the CSVs, source images and truth are unchanged from the scored run.
No new inference was needed to adjudicate an expectation change.

The low-resolution case loses the annotated narrow V3 R-prime notch: it is no
longer a local turning point. Its main V3 R peak is also attenuated by about
248 µV. The high whole-trace correlation therefore does not establish fine-detail
fidelity. Its source is only 900 pixels wide across a ten-second page; 90 Hz is
a conservative upper bound from the full page width, and the 500 Hz CSV is an
interpolation grid. The new checks reject reporting an effective rate above
90 Hz, dropping review, or losing either of the two larger annotated events.
They do not require the lost notch to stay lost: a real improvement may recover it.

The perspective case preserves all three annotated events, including a V3 notch
with 126.28 µV local prominence. The new checks require those events and retain
the existing 80 µV notch-prominence threshold used by the dedicated notch case.
Its lower-confidence and uncertainty evidence still requires source review.

The full regression denominator remains eight images from one synthetic family:
seven quantitative results requiring review, one 12x1 abstention and zero runtime
failures. The 3x4 result remains limited. Passing these smoke checks does not
establish general ECG accuracy, measurement suitability or diagnostic validity.

## Reproduction and audit

Use `scripts/check_waveparse_regression.py` with the new configuration as shown
in [REGRESSION.md](REGRESSION.md). Focused negative tests verify that automatic
acceptance, absent/false review evidence, inflated sampling rates and excessive
waveform error still fail.

- [Adjudication identities, metrics and events](verification/2026-09-05-smoke-adjudication.json)
- [Current library gate result](verification/2026-09-05-library-smoke-gates.json)
- [Unchanged historical gate result](verification/2026-09-05-historical-smoke-gates.json)
