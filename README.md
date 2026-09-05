# ECG WaveParse

**ECG image-to-signal digitization, running locally in your own application.**

WaveParse provides Python and JavaScript/TypeScript interfaces to the same
versioned TypeScript/Python engine. It extracts lead signals and produces CSVs,
a diagnostic overlay, quality/provenance evidence and a clean ECG-paper render.
It does not diagnose ECGs. Outputs may require review, be incomplete, or abstain.

Version **0.1.0 is release preparation**. Packages have not been published to
npm or PyPI. Use the built artifacts below; do not assume a registry package
with the same name is this project.

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

## Install a prepared package

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
`get_run_async` and `review_async` are also available. Async cancellation stops
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
terminal completion. Review-only compact/uncertainty artifacts are removed after
a review decision. Reopen the result after review rather than caching old paths.
Store permanent references to the canonical CSV and provenance, not working files.

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
boundaries and publication blockers. No patient/reference ECGs are distributed.
`VERIFICATION.md` records the tested platforms, synthetic parity results and
remaining release checks. These documents are also bundled in each package's `docs` directory.
