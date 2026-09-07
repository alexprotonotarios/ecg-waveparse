# Native package CI — 5 September 2026

The results below are historical. The 6 September Pro continuation is preparing
a new native package matrix and does not claim that this older successful run
verifies its changed artifacts. The new workflow removes PR path filters, adds
fast contracts on every PR and generates one shared set of source/truth bytes
for Ubuntu and macOS. A separate aggregate compares waveform, missingness,
uncertainty, semantic and policy results under the fixed
`benchmark/protocols/platform-parity.v1.json` limits. It runs after failed or
skipped prerequisites and fails unless both execution tiers succeeded.
Identical abstentions establish refusal parity only. The recommended required
checks are `Fast package contracts` and `Supported platform equivalence`.
Current hosted execution and required-check enforcement remain unverified;
changing the workflow file alone does not enable branch protection.

The clean source repository was private when these checks ran on 5 September 2026:
[alexprotonotarios/ecg-waveparse](https://github.com/alexprotonotarios/ecg-waveparse).
Its initial 110-file source snapshot contains no original ECGs, model files,
storage or history from the original data-bearing repository. No package has
been published and no production service has been deployed.

## Run identity and scope

[Native package workflow 33990248411](https://github.com/alexprotonotarios/ecg-waveparse/actions/runs/33990248411)
tests source commit `701b35619cd69a9b7cf505db5c6ee6de19e946e6` on native
Ubuntu 24.04 x86-64 and macOS 15 ARM64 runners, using Node 22.22.2 and CPython
3.12.9. CPU inference is explicitly selected on both platforms.

The workflow builds npm, wheel, sdist and corresponding-source archives; audits
the exported source; runs package/reliability and installer tests; checks fresh
ESM/CommonJS/NodeNext and Python sync/async consumers; explicitly sets up the
locked inference runtime; inventories dependencies and all five model hashes;
runs the native extractor unit tests; and performs real synthetic extraction.
The lifecycle smoke verifies persisted CSV identity, refusal of acceptance
without confirmations, engineering rejection/compaction, tamper detection,
cancellation and abandoned-run recovery. Nothing is clinically accepted.

Both jobs completed successfully and their uploaded artifacts were downloaded
and independently checked:

| Native runner | Job duration | JavaScript tests | Python checks | Real extraction/lifecycle |
| --- | ---: | ---: | ---: | --- |
| Ubuntu 24.04 x86-64 | 6m37s | 79 passed | 2 installer + 47 native passed | Passed |
| macOS 15 ARM64 | 17m55s | 79 passed | 2 installer + 47 native passed | Passed |

The npm tarball and Python wheel from both runners are byte-for-byte identical
to the prepared packages tested locally and, for npm, installed in Neo. Both
language artifacts on both hosts contain the same 30 verified runtime files.
Each installed inference runtime inventories 66 distributions and verifies all
five model hashes. The shared payload manifest is
`ade3c347c983238365078466537f950166684da5956e079fe338039cec19265f`, with runtime
manifest `d9465db541f0d2ad0541eb04f36e3fee930a8bbbc1a50135fe2950b63203fc3a`.

The exact package hashes, source-archive comparisons, native inference CSV
hashes, runner results and GitHub artifact identifiers are recorded in
`verification/2026-09-05-native-ci.json`. Source archives differ at archive
level but contain identical regular files across the two runners. Compared
with local preparation, the Python sdist differs only in `.gitignore`, while
the repository source archive contains the workflow correction below.

GitHub emitted an action-runtime Node 20 deprecation warning and ran those
actions on Node 24. The explicitly installed library Node version remained
22.22.2. This warning did not fail either job.

This report was added after the successful run; the tested code revision above
remains explicit. Reporting-only changes do not change its engine payload.

## Initial workflow correction

The first source push produced a workflow parsing failure
([run 33990125880](https://github.com/alexprotonotarios/ecg-waveparse/actions/runs/33990125880));
no native test job started. `runner.temp` had been used in a job-level environment
block, where that context is unavailable. Commit `701b356` moved `MPLCONFIGDIR`
to the inference step environment. This changed only workflow configuration,
not the shared engine payload. The failed attempt remains in the run history.
See GitHub's [context availability reference](https://docs.github.com/en/actions/reference/workflows-and-actions/contexts#context-availability).

This checks package/runtime operation on native hosts. The fixture is generated
from the deterministic engineering suite on each host; it is not patient data.
Generated font/rendering differences and CPU numerical differences are not
controlled as a cross-platform waveform-equivalence study. The reconstruction
limitations and two historical smoke decision mismatches in VERIFICATION.md
remain separate release considerations, along with model-licence clarification
and registry ownership/configuration.
