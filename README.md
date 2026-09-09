# ECG WaveParse

Convert an ECG image into lead-signal CSVs, an overlay on the original image,
and a clean ECG-paper render. WaveParse runs locally through a command-line
tool or a Python / JavaScript / TypeScript API.

## Requirements

- CPython **3.12.9** and Node.js **22.22.2** (both are required).
- Apple Silicon macOS or Linux x86-64 with glibc 2.28 or newer.
- At least **6 GiB of free disk space** before setup and the first run.
- Internet access for initial runtime setup. Digitization runs locally.

On Debian/Ubuntu, install `libgl1` and `libglib2.0-0`. CPU is the default;
Apple Silicon users can request `--device mps`. Windows, Linux ARM and CUDA
are not currently supported.

## Install and run

Download the wheel or npm archive from the
[GitHub releases page](https://github.com/alexprotonotarios/ecg-waveparse/releases).
Choose either installation method. Each package includes the shared runner;
you do not need to install both packages.

Release downloads contain the code from their release tag. To use the code in
your current Git checkout, follow [Build from source](#build-from-source).

Python, in a virtual environment:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install ./ecg_waveparse-0.1.0-py3-none-any.whl
python -m ecg_waveparse setup
python -m ecg_waveparse doctor --workspace-dir ./ecg-results
python -m ecg_waveparse digitize --input ./ECG.png --workspace-dir ./ecg-results
```

JavaScript / TypeScript, from your application directory:

```sh
npm install ./ecg-waveparse-0.1.0.tgz
npx --no-install ecg-waveparse setup --python /path/to/python3.12
npx --no-install ecg-waveparse doctor --python /path/to/python3.12 --workspace-dir ./ecg-results
npx --no-install ecg-waveparse digitize --input ./ECG.png --workspace-dir ./ecg-results
```

Replace `/path/to/python3.12` with your CPython 3.12.9 executable. The installed
runtime uses its own environment. `setup` explicitly downloads the pinned
engine, model weights and dependencies and verifies their hashes. It can take
several minutes. Subsequent `doctor` and digitization calls do not download them.

The runtime defaults to a versioned directory under `~/Library/Caches/ecg-waveparse`
on macOS or `${XDG_CACHE_HOME:-~/.cache}/ecg-waveparse` on Linux. To store it
elsewhere, pass the same `--runtime-dir /absolute/path` to setup, doctor and
digitize. A workspace contains private source images and extracted signals;
choose a directory you control.

## Use the API

Python:

```python
from ecg_waveparse import Digitizer

digitizer = Digitizer(workspace_dir="./ecg-results")
result = digitizer.digitize("./ECG.png")
print(result["status"], result["assets"])
```

JavaScript / TypeScript:

```js
import { Digitizer } from "ecg-waveparse";

const digitizer = new Digitizer({ workspaceDir: "./ecg-results" });
const result = await digitizer.digitize("./ECG.png");
console.log(result.status, result.assets);
```

See [the API reference](docs/API.md) for asynchronous Python calls, progress,
cancellation, retained evidence and recording a human review.

## Inputs and outputs

Inputs are PNG, JPEG, WebP or single-frame TIFF files up to 50 MB, 80 million
decoded pixels and 32,768 pixels on either edge. Use the original scan or export
where possible, with the full sheet, readable labels and visible calibration.

A returned quantitative result includes the untouched source, canonical
12-lead CSV, compact lead-segment CSV, uncertainty CSV, segment map, source
overlay, ECG-paper render, and metadata with file identities and calibration.
Unavailable samples remain missing; a 500 Hz export does not imply that the
image contained 500 Hz of information. The paper render uses 25 mm/s and
10 mm/mV and is baseline-centred for inspection.

Check `status` before using an output:

| Status | Meaning |
| --- | --- |
| `needs_review` | A result is available for comparison with the original image. |
| `partial` | Incomplete diagnostic output; not an eligible full quantitative result. |
| `failed` | Extraction was refused or failed; inspect the recorded reason. |
| `timed_out` | Processing exceeded its time limit; the admitted source is preserved. |

WaveParse does not diagnose ECGs. Compare the source and overlay, lead identity,
scale, gaps and uncertainty before quantitative use. Use the CSV and source
evidence for measurements; a plausible paper render alone does not establish
fidelity. Human acceptance or rejection preserves the extracted samples and
their uncertainty.

## Troubleshooting

- **Runtime unavailable:** run `setup`, then `doctor` with the same runtime path.
  Verify the exact Node and Python versions above.
- **Runtime integrity error:** preserve the existing directory and set up a new
  runtime directory. Do not replace a model or dependency manually.
- **No quantitative CSV:** inspect the result's status and reason. Check source
  quality, lead labels and calibration; a refused extraction is a valid outcome.
- **Timeout or high memory use:** run one image at a time. The default timeout is
  30 minutes; `--timeout-ms` accepts 60,000–7,200,000. High-resolution images can
  require substantial memory and CPU time.

## Build from source

Use the source archive from the same release, or clone this repository. In the
checkout, with the required Node and Python versions:

```sh
corepack pnpm install --frozen-lockfile
python3.12 -m venv .build-venv
. .build-venv/bin/activate
python -m pip install -r requirements-waveparse-build.txt
corepack pnpm waveparse:build
python -m build --no-isolation packages/python --outdir dist
```

The `dist` directory contains the npm archive, Python wheel and source archives.
For example, install the newly built wheel from the checkout root into the active
virtual environment:

```sh
python -m pip install ./dist/ecg_waveparse-0.1.0-py3-none-any.whl
```

For a separate JavaScript application, install the archive at
`/absolute/path/to/checkout/dist/ecg-waveparse-0.1.0.tgz`. Then run the setup,
doctor and digitize commands in [Install and run](#install-and-run).
Building packages does not download the inference models. To verify the built packages, run
`WAVEPARSE_PYTHON=python corepack pnpm waveparse:verify`.

Build and package checks run locally. This repository does not include GitHub
Actions workflows; run these checks before submitting source changes.

## Licence

Independently authored WaveParse code is MIT-licensed. Adapted Open-ECG-Digitizer
components retain CC BY-SA 4.0. Runtime setup retrieves upstream components
under their own terms. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model-specific terms have not
been independently confirmed; model weights are downloaded separately.
