# Version 0.1.0 release verification — 5 September 2026

This records the local preparation baseline. The clean source repository has
since been created privately at `alexprotonotarios/ecg-waveparse`; the native
workflow follow-up is recorded in HOSTED-CI.md. Nothing has been published to
npm/PyPI or made public.

## Current package identity and reproducibility

The tested runtime payload manifest SHA-256 is:

```text
ade3c347c983238365078466537f950166684da5956e079fe338039cec19265f
```

The separately installed engine/model/dependency manifest SHA-256 is:

```text
d9465db541f0d2ad0541eb04f36e3fee930a8bbbc1a50135fe2950b63203fc3a
```

These identify different things: changes to WaveParse's own selection code can
change the payload while leaving the upstream runtime manifest unchanged. All
preparation artifacts still use unpublished version 0.1.0; distinguish them by
their hashes. Final distribution hashes are recorded in `dist/SHA256SUMS`.

- Built npm, Python wheel/sdist and an allowlisted corresponding-source archive.
- Fresh external npm/wheel installations passed ESM, CommonJS, TypeScript
  NodeNext and Python synchronous/asynchronous consumer checks.
- All 30 runtime files match between the two installed language packages.
- The clean source rebuild produces the same runtime payload as the checkout
  and the installed package used for regression.
- The source audit excludes images, CSVs, models, storage, old Git history and
  recognised credential patterns. The audit is a bounded automated check,
  not a proof that arbitrary secrets can never occur in source text.
- Full original licence texts and attribution are included in both packages
  and the source archive. Remaining model-rights questions are in LICENSING.md.

## Shared-code tests

The updated checkout passed 184 JavaScript tests, 222 Python tests, TypeScript
checking, ESLint and the Next.js production build. Its clean source export
independently passed 79 JavaScript and 49 Python checks.

Two HTTP tests initially could not bind loopback sockets inside the restricted
sandbox. The complete suite passed when rerun with local networking enabled.
An initial MPS benchmark invocation was refused because that sandbox hid GPU
availability; the recorded regression uses an environment where MPS is available.

## Installed-package engineering regression

The suite contains eight images from one deterministic waveform family: six
6x2 variants (clean, arrow, notch, low resolution, blur and perspective), one
clean 3x4 image and one clean 12x1 image. This is not a patient cohort, an
independent locked validation set or clinical validation. The scorer preserves
missingness and timing and permits at most 40 ms alignment.

The first run returned eight reviewable CSVs, but the 3x4 and 12x1 native
candidates had unverified waveform time origins and included non-waveform ink.
Their RMSE/correlation were 388.99 µV/0.0134 and 223.06 µV/0.0396 respectively.
Both were subsequently rejected through the review API. The original results
remain recorded in `verification/2026-09-05-regression-before.json`.

The shared engine now excludes native 3x4/12x1 candidates whose panel timing is
unverified, including through the peer-corroboration fallback. The Python
3x4 fidelity report also requires source timing. The library wrapper now keeps
policy abstention distinct from a worker failure.

The corrected 3x4 run selected a different candidate: RMSE 74.45 µV,
correlation 0.8094 and coverage 97.09%. It remains a limited, unaccepted result;
two of three annotated morphology events were preserved. The corrected 12x1
run abstained with `no_publishable_candidate`, retaining source and provenance
without exposing a CSV. One fallback candidate reached its existing five-minute
limit. No timeout or quality threshold was relaxed.

The corrected clean 6x2 canonical CSV matches the earlier MPS baseline exactly:
`916e38c071c8b7d5cc290589f6ae9bea96ffbfd6aff121468a8231ecc1fbea23`.
All six 6x2 canonical CSVs are byte-for-byte unchanged by the timing fix.
The complete rerun returned seven quantitative results for review out of eight
images (87.5% yield), one abstention and no overall runtime failures. None was
accepted as clinically reviewed evidence. Conditional on the seven returned
outputs, pooled RMSE was 28.62 µV, mean case correlation 0.9599 and mean case
coverage 99.32%. Those metrics do not include a reconstructed 12x1 result.

| Synthetic case | Result | RMSE (µV) | Correlation | Coverage |
| --- | --- | ---: | ---: | ---: |
| 6x2 clean | needs_review | 16.21 | 0.9912 | 99.95% |
| 6x2 arrow | needs_review | 16.10 | 0.9912 | 98.85% |
| 6x2 notch | needs_review | 15.83 | 0.9920 | 99.91% |
| 6x2 low resolution | needs_review | 26.38 | 0.9756 | 99.99% |
| 6x2 blur | needs_review | 28.27 | 0.9720 | 99.96% |
| 6x2 perspective | needs_review | 18.51 | 0.9882 | 99.49% |
| 3x4 clean | needs_review | 74.45 | 0.8094 | 97.09% |
| 12x1 clean | abstention | — | — | — |

The exact payload, source/truth hashes, scorer, all candidate outcomes and
metrics are in `verification/2026-09-05-regression-after.json`. Wall time for
the eight sequential calls was about 24.4 minutes on this MPS-enabled Mac.

The unchanged historical smoke gates reported 30 of 32 checks satisfied and
**two status mismatches**: low resolution and perspective were expected to
abstain but returned reviewable outputs. That historical comparison remains
failed. The subsequent [evidence review](SMOKE-ADJUDICATION.md) traced the
mismatch to an August policy change. A separate library configuration restores
the original quantitative limits and adds review-state and morphology checks;
all 66 checks pass on the saved run. Low resolution still loses the narrow V3
notch and remains a limited, lower-confidence result. No inference policy was
relaxed and no clinical result was accepted by this adjudication.


## Platform and lifecycle evidence

Explicit setup and doctor previously passed on Apple Silicon macOS and Linux
x86-64 (Debian Bookworm under Docker emulation), with Node 22.22.2 and CPython
3.12.9. Linux used CPU-only Torch 2.12.0+cpu and torchvision 0.27.0+cpu. The
earlier installed Python/Linux CPU smoke completed all seven candidates and
produced a reviewable 6x2 CSV, SHA-256
`1256bd8ea544a1f348daec51a2860541b00bfb1470d9c11faf62b44d4a28ad50`.
That smoke predates the panel-timing safeguard and is an emulated result.
The subsequent native macOS/Linux checks are recorded separately in HOSTED-CI.md.

Earlier package lifecycle checks exercised review rejection, refusal of
acceptance without confirmations, source integrity, deliberate synthetic-copy
tamper detection, cancellation after admission, terminal compaction and recovery
of a dead runner. No result was accepted as clinically reviewed evidence.

## Neo dev integration

The real desktop flow passed authenticated upload, queued/running refresh,
worker extraction, backend artifact storage, source/artifact hash checks,
completed-result refresh and persisted engineering rejection. The resumed
authenticated gate passed 18 tests with three expected skips; only one is the
real ECG test. No result was accepted clinically and production was untouched.
Authentication used Neo's genuine WorkOS test-session helper. The test did not
exercise a hosted preview domain or the interactive sign-in/callback flow.

Testing found and fixed two UI bugs: the saved upload still appeared pending,
and draft initialisation could overwrite the first title edit. A shared dev
deployment collision removed the worker endpoint; that attempt was cancelled
without a committed result. The successful retry used an isolated dev backend.
A browser hash-selector error was then fixed, and inspection/rejection resumed
against the same completed run instead of redoing extraction.

`verification/2026-09-05-neo-dev.json` binds this evidence to the original npm
archive SHA-256
`5fff85cb6337231cbb946ea63a5970ba4663fec400513550eb5c85a83c3c9863`.
Its package identity is separate from the corrected payload above; a subsequent
vendor update must retain that boundary in its verification record.

## Remaining public-release work

1. Retain the completed smoke adjudication above and confirm the limited
   3x4/12x1 scope of the initial research/developer release.
2. Complete the model-specific licence clarifications in LICENSING.md and
   approve the stated mixed-licence allocation.
3. Bind the intended publication artifacts to the supported-platform checks in
   HOSTED-CI.md; changes to the engine require corresponding regression evidence.
4. Complete registry publishing configuration using PUBLISHING.md. Both
   `alexprotonotarios` accounts and 2FA were verified on 6 September 2026;
   package ownership and publisher connections are still separate steps.
5. Approve the exact release artifacts and clean-source public visibility.

The documented 3x4 accuracy limitation and 12x1 abstention must accompany any
research/developer preview. The saved historical clinical-validation metrics
have not been relabelled as current, and no diagnostic-use claim is established.
