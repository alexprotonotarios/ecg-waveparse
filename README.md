# ECG WaveParse

**ECG image-to-signal digitization, running locally in your own application.**

WaveParse provides Python and JavaScript/TypeScript interfaces to the same
versioned TypeScript/Python engine. It extracts lead signals and produces CSVs,
a diagnostic overlay, quality/provenance evidence and a clean ECG-paper render.
It does not diagnose ECGs. Outputs may require review, be incomplete, or abstain.

**Source preview:** the library source is available in
[alexprotonotarios/ecg-waveparse](https://github.com/alexprotonotarios/ecg-waveparse).
Version **0.1.0 has not been published to npm or PyPI**. Build the artifacts below;
do not assume a registry package with the same name is this project.

The 6 September engineering changes are prepared locally. The public preview
and the first campaign's archived artifacts precede the
[continuation evidence](COMPLETION-WORK-2026-09-06.md). The
[25-patient final evaluation](FINAL-EVALUATION-2026-09-06.md) returned 19 signals,
abstained on six and qualified four under the fixed engineering gates. Hosted
verification and the owner's exact-artifact decision remain open in the
[criteria audit](ACCEPTANCE-AUDIT-2026-09-06.md). Reproduce each recorded
experiment using its matching source archive and payload identity; the earlier
receipt does not certify the current working tree.

Model weights and patient ECGs are not included. Explicit runtime setup downloads
upstream components separately. Model-specific licence clarification remains open;
see the [licensing evidence](https://github.com/alexprotonotarios/ecg-waveparse/blob/main/docs/waveparse/LICENSING.md)
and [third-party notices](https://github.com/alexprotonotarios/ecg-waveparse/blob/main/THIRD_PARTY_NOTICES.md)
for the code licences and runtime boundary.

## Requirements

| Component | Supported v1 configuration |
| --- | --- |
| OS | Apple Silicon macOS; Linux x86-64 with glibc 2.28+ (Ubuntu 24.04 is the CI target) |
| Node | 22.22.2, required for both language interfaces |
| Python | CPython 3.12.9, required for both language interfaces |
| Compute | CPU by default; MPS can be requested on a supported Mac |
| Storage | At least 6 GiB free before setup/first processing; active runs reserve additional space |

On minimal Debian/Ubuntu images, install `libgl1` and `libglib2.0-0` for OpenCV
before setup. The package workflow targets Ubuntu 24.04 and Apple Silicon macOS.

The Python package includes the compiled Node runner; you do not separately
install the npm package. The npm package includes the Python processing scripts;
you do not separately install the Python package. Both still need Node, Python
and an explicitly installed inference runtime. This is a server/local library,
not browser JavaScript or a hosted API client. Windows, Linux ARM and CUDA are
outside the initial support matrix.

## Build from source

With the supported Node/Python versions above and pnpm 10.34.5 installed:

```sh
git clone https://github.com/alexprotonotarios/ecg-waveparse.git
cd ecg-waveparse
python3.12 -m venv .venv
. .venv/bin/activate
pnpm install --frozen-lockfile
python -m pip install -r requirements-waveparse-build.txt
pnpm waveparse:build
python -m build --no-isolation packages/python --outdir dist
```

This creates the npm tarball, Python wheel/sdist and corresponding-source archive
in `dist/`. Building the packages does not download inference models or process
ECGs. Install the resulting package into your application as shown below.

## Install a built package

JavaScript/TypeScript, inside your application's backend or worker project:

```sh
pnpm add /absolute/path/to/dist/ecg-waveparse-0.1.0.tgz
pnpm exec ecg-waveparse setup --python /absolute/path/to/python3.12
pnpm exec ecg-waveparse doctor --workspace-dir ./ecg-results
```

Python, inside your application's virtual environment:

```sh
python -m pip install /absolute/path/to/dist/ecg_waveparse-0.1.0-py3-none-any.whl
python -m ecg_waveparse setup
python -m ecg_waveparse doctor --workspace-dir ./ecg-results
```

`setup` is the only operation that downloads anything. It retrieves the locked
upstream source and weights, creates a private Python environment and verifies
source/model identities and dependency versions. Git and Git LFS are not required.
Run `setup` again to verify/reuse an installed runtime. Interrupted installations
retain a `.partial` staging directory for resumption; a corrupt activated runtime
is refused, rather than overwritten. Use a new `--runtime-dir` to replace one.

The default cache lives under `~/Library/Caches/ecg-waveparse` on macOS and
`${XDG_CACHE_HOME:-~/.cache}/ecg-waveparse` on Linux, separated by manifest/release
and platform. Set `--runtime-dir /absolute/path` during setup and pass the same
directory to the API when managing your own location. Set `WAVEPARSE_NODE` to
a Node executable when it is not on PATH. JavaScript setup also accepts
`WAVEPARSE_PYTHON`; the Python CLI defaults to its current interpreter.

## JavaScript and TypeScript

```ts
import { Digitizer, WaveParseError } from "ecg-waveparse";

const digitizer = new Digitizer({ workspaceDir: "./ecg-results" });
try {
  const result = await digitizer.digitize("./ecg.png", {
    device: "cpu",
    timeoutMs: 1_800_000,
    onProgress: ({ runId }) => console.log("Started", runId),
  });
  console.log(result.id, result.status, result.publicationDecision);
  console.log(result.assets.diagnostic?.absolutePath);
  console.log(result.assets.canonicalCsv?.absolutePath);
} catch (error) {
  if (error instanceof WaveParseError) console.error(error.code, error.message, error.runId);
  else throw error;
}
```

CommonJS uses `const { Digitizer } = require("ecg-waveparse")`. Type declarations
are included; no `@types` package is needed. An optional `AbortSignal` can cancel
`digitize`; the runner stops child processing and preserves the admitted source.

## Python

```python
from ecg_waveparse import Digitizer, WaveParseError

digitizer = Digitizer(workspace_dir="./ecg-results")
try:
    result = digitizer.digitize("./ecg.png", device="cpu")
    print(result["id"], result["status"])
    if "canonicalCsv" in result["assets"]:
        print(result["assets"]["canonicalCsv"]["absolutePath"])
except WaveParseError as error:
    print(error.code, str(error), error.run_id)
```

For async applications, use `await digitizer.digitize_async("./ecg.png")`.
`get_run_async`, `get_evidence_async` and `review_async` are also available. Async cancellation stops
the runner and waits for its cleanup. The Python result is a typed dictionary;
its JSON keys match JavaScript, including `canonicalCsv` and `publicationDecision`.

## Results, artifacts and review

`getRun(id)` / `get_run(id)` returns the persisted result, or `null` / `None`.
Asset integrity is checked when results are reopened. A run whose local owner
has exited is recovered as failed with its admitted source preserved. The result includes:

- `schemaVersion`, run `id`, `status`, timestamps and source identity;
- `layout`, sample rate, calibration fields and selected candidate when available;
- `publicationDecision`, `reliability`, candidate QA, review and processing state;
- `waveparse`: package version, runtime-manifest SHA-256 and selected device;
- `assets`: available artifact paths, MIME information and recorded identities.

`needs_review` is not acceptance. `partial` is diagnostic-only incomplete evidence.
`failed` can be a valid extraction abstention; `timed_out` preserves the source.
These are normal results. Setup/request/integrity failures throw `WaveParseError`.
There is no automatic conversion of unavailable measurements into normal values.

Canonical CSVs preserve the nominal 500 Hz page timebase and missing samples.
Leads printed in different panels must not be assumed simultaneous. Compact
CSVs use lead-local panel time while retaining internal gaps. Use the CSV and
source for quantitative work. The baseline-centred paper render is a visual aid,
not an absolute ST-level reference. Refer to runtime provenance for calibration
and evidence restrictions. Current evidence is engineering reconstruction
evidence, not clinical validation or diagnostic accuracy.

After visually comparing source and overlay, an eligible run can be reviewed:

```ts
const runId = "run_REPLACE_WITH_YOUR_RUN_ID";
const reviewed = await digitizer.review(runId, {
  decision: "accepted", reviewer: "reviewer-identifier",
  notes: "Recorded findings from source-versus-overlay review.",
  confirmations: {
    sourceCompared: true, leadIdentityVerified: true, scaleAndGapsReviewed: true,
  },
});
```

Python uses `digitizer.review(run_id, decision="accepted", reviewer="...",
notes="...", confirmations={...})` with the same confirmation keys. Confirmations
must describe a review actually performed. Rejection uses `decision="rejected"`.
The original policy enforces eligibility; incomplete output cannot be accepted
as a complete quantitative run.

Workspaces own their `storage/runs/<id>` directories. Inputs are copied and hashed;
the original file is never modified. Candidate intermediates are compacted at
terminal completion. Retention v2 permanently preserves compact CSV, uncertainty
CSV and segment map alongside the source, canonical CSV, overlay, provenance and
review audit. Reopen the result after review rather than caching old paths.
Historical runs with already-deleted uncertainty explicitly report it unavailable.

`getEvidence(id)` / `get_evidence(id)` returns a verified local source/segment
bundle. It preserves original-image inspection after abstention and never
generates missing waveform samples. Segment records distinguish printed panel
position from unknown acquisition time. See [API.md](API.md) for identity-bound
measurement references and [examples](#worked-examples-and-evidence).

## Worked examples and evidence

`examples/waveparse.mjs` and `examples/waveparse.py` exercise the installed
package, then inspect source availability, reconstruction state and segment
count. Copy the JavaScript example into your installed consumer project; execute
Python with the environment containing the wheel. Both accept an optional
existing run ID to inspect a result without reprocessing:

```sh
node waveparse.mjs INPUT WORKSPACE RUNTIME [EXISTING_RUN_ID]
python waveparse.py INPUT WORKSPACE RUNTIME [EXISTING_RUN_ID]
```

The [worked examples](WORKED-EXAMPLES.md) cover a clean input, a difficult
low-resolution input and an abstention, with source/overlay inspection and
explicit truth comparisons.

The [baseline](BASELINE-2026-09-06.md) contains **eight attempted images from one
synthetic signal family**: seven quantitative results requiring review, one
12x1 abstention, zero overall runtime failures and zero accepted results.
Conditional pooled RMSE was 28.62 µV, mean case correlation 0.9599 and coverage
99.32%. That is engineering regression evidence, not independent clinical
accuracy. Strict scoring and subsequent source/package identities are tracked
in [the improvement ledger](ENGINEERING-IMPROVEMENTS.md).

| Capability | Evidence boundary |
| --- | --- |
| 6x2 synthetic regression | Clean and five degraded variants return signals; low-resolution narrow-feature loss remains documented. |
| 3x4 synthetic regression | A limited result returned at 74.45 µV RMSE; it still requires source review. |
| 12x1 synthetic regression | Abstention is preserved; support is input-dependent, not promised by recognizing a layout name. |
| Physical units | Printed speed/gain OCR, conflicts and pulse/grid evidence are checked. Nondefault settings require physical evidence. Five of eight controlled full-page setting cases return and pass scale/fidelity gates; three refuse. Source acquisition rate is not inferred from printed speed. |
| Uncertainty and timing | Per-sample missingness, interval lineage and immutable segment maps survive review/compaction. Candidate spread is not calibrated confidence. |
| PDF, native waveform and vector import | Not exposed by v1; [feasibility decision](SOURCE-IMPORT-FEASIBILITY.md) records admission conditions. |
| Alternative decoders/local warp | [Current measured experiments and no-adoption decisions](COMPLETION-WORK-2026-09-06.md); local warp remains experimental. |
| Final public waveform evaluation | [25 patient groups](FINAL-EVALUATION-2026-09-06.md): 19 returned, six abstained, four met all fixed engineering gates. Conditional pooled RMSE 136.47 µV, mean correlation 0.7192, mean coverage 97.5%. Rendered acquired signals; no clinical validation. |
| Historical public waveform pilot | [Earlier 16 patient groups](PUBLIC-PILOT-2026-09-06.md): 12 returned, four abstained, six qualified. Different patients and payload; not a paired comparison. |
| Deployment | [Non-root Linux recipe](LINUX-CONTAINER.md), [local safety boundaries](LOCAL-SAFETY.md), explicit model setup. |

The public source checkout installs only the TypeScript/esbuild/tsx toolchain,
Node type declarations and the YAML parser used to preserve its dependency lock.
Its lockfile is derived from the application lock
without changing the toolchain versions. The separate application checkout
retains its UI dependencies; UI-only helpers are absent from the library source
export. The installed npm package has zero external JavaScript dependencies,
and the wheel has no pip runtime dependencies in its package metadata. Both
use the separately provisioned, locked inference environment.

## Integration, offline operation and updates

In web applications run the library inside a backend worker that has the supported
runtime, writable storage, adequate memory and a long enough job lifetime. A package
install does not give a browser or restricted serverless runtime these capabilities.
Each call launches a local runner. Bound application-level parallelism to the
available CPU/memory; the existing storage admission policy also reserves space.

After successful setup and doctor verification, inputs and outputs stay local and
processing needs no network. Setup can be performed in the same deployment image
ahead of time. A runtime's virtual environment depends on the installed base Python;
do not copy it between operating systems or incompatible base installations.

Pin the package version and dependency lockfile. Upgrading either language package
requires its matching runtime manifest (`setup` installs/reuses the matching cache).
Existing results retain their version and hashes; reprocessing creates a new run.
No registry release silently rewrites stored ECG results.

## Troubleshooting and contributing

- **Missing Node/Python:** install the exact supported versions and check PATH or
  `WAVEPARSE_NODE` / `--python`.
- **Runtime unavailable or integrity mismatch:** run `doctor`, then use a fresh
  runtime directory if installed contents are corrupt. Downloads never happen implicitly.
- **No quantitative CSV:** inspect status, publication reason and source/diagnostic
  evidence. A clean-looking render is not proof of successful extraction.
- **Storage refused:** free space outside active runs; do not lower the admission floor.
- **Timeout:** v1 allows 60,000–7,200,000 ms; default 1,800,000 ms. Preserve the source
  and investigate the input/runtime before retrying. This bounds extraction after
  admission; runtime verification and capture preflight run first with their own
  limits. Cancellation cleanup can take up to 30 seconds.

See `CONTRIBUTING.md`, `API.md` and `RELEASING.md` in the WaveParse documentation
directory of the clean source archive. See `THIRD_PARTY_NOTICES.md` for licensing
boundaries and the remaining registry-release work. No patient/reference ECGs are distributed.
`VERIFICATION.md` records the tested platforms, synthetic parity results and
remaining release checks. These documents are also bundled in each package's `docs` directory.
`REGRESSION.md` describes the installed-package engineering suite;
`LICENSING.md` and `PUBLISHING.md` record the remaining publication decisions.
`CHANGELOG.md` records changes to both the interfaces and extraction behaviour.
