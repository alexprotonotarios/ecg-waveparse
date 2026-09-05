# Installed-package engineering regression

This suite exercises the release through the public JS API, using eight images
from one deterministic waveform family. It covers clean 6x2, an arrow over V1,
narrow-notch resolution, 900-pixel input, blur, perspective, clean 3x4 and clean
12x1. It is not a clinical cohort or an independent locked validation set.

Generate the fixtures with the installed inference runtime's Python:

```bash
RUNTIME_PYTHON=/absolute/path/to/runtime/engine/.venv/bin/python
"$RUNTIME_PYTHON" scripts/generate_benchmark.py --suite benchmark/suites/smoke.json --output /tmp/waveparse-fixtures/smoke
"$RUNTIME_PYTHON" scripts/generate_benchmark.py --suite benchmark/suites/waveparse-3x4.json --output /tmp/waveparse-fixtures/3x4
"$RUNTIME_PYTHON" scripts/generate_benchmark.py --suite benchmark/suites/waveparse-12x1.json --output /tmp/waveparse-fixtures/12x1
```

Install the prepared npm tarball in a separate consumer project. The harness
resolves `ecg-waveparse` from that project's package.json and refuses the source
package directory. Use a new output directory for each run:

```bash
node scripts/waveparse-regression.mjs \
  --consumer /absolute/path/to/consumer \
  --runtime /absolute/path/to/runtime \
  --python "$RUNTIME_PYTHON" \
  --manifest /tmp/waveparse-fixtures/smoke/manifest.json \
  --manifest /tmp/waveparse-fixtures/3x4/manifest.json \
  --manifest /tmp/waveparse-fixtures/12x1/manifest.json \
  --output /tmp/waveparse-regression-new-run \
  --device cpu
```

`--device mps` is optional on supported Macs with GPU access. A restricted
sandbox can hide MPS; a runtime check failure is an environment failure, not an
ECG abstention. Default extraction timeout is 30 minutes per case. Cases run
sequentially and retain the normal compact terminal evidence. No policy or
candidate timeout is relaxed for the benchmark.

The harness verifies package payload and fixture hashes before and after the
run. It records the runtime manifest, scorer identity, platform/device, case
denominator, extraction decisions, candidate outcomes, elapsed time and canonical
hashes. Every terminal result is reopened through the public integrity-checking
API. CSVs are scored against deterministic truth with a 40 ms alignment bound,
preserving lead timing and missingness.

`report.json` is saved after each case; per-case score files include coverage,
RMSE, correlation, morphology and uncertainty. Infrastructure failures remain
visible in the denominator. Policy refusals count as abstentions. Quantitative
results still require source review and are never accepted automatically by this
harness. The returned process status checks runtime/scoring completion; it is not
a clinical-quality certification or a pass of the historical accuracy target.

Compare the six smoke cases with the existing quality expectations separately:

```bash
"$RUNTIME_PYTHON" scripts/check_waveparse_regression.py \
  --report /tmp/waveparse-regression-new-run/report.json \
  --gates benchmark/suites/smoke-gates.json \
  --output /tmp/waveparse-regression-new-run/historical-smoke-gates.json
```

The comparison preserves the gate configuration. A changed refusal/recovery
decision is reported as a mismatch, even when the new waveform score is good.

For the current installed-library recovery and mandatory-review contract, use:

```bash
"$RUNTIME_PYTHON" scripts/check_waveparse_regression.py \
  --report /tmp/waveparse-regression-new-run/report.json \
  --gates benchmark/suites/waveparse-smoke-gates.json \
  --output /tmp/waveparse-regression-new-run/library-smoke-gates.json
```

This also checks the app status, publication outcome and review flag, which the
benchmark's scoring-completion status cannot establish. The low-resolution case
must report no more than 90 Hz effective sampling. See
[SMOKE-ADJUDICATION.md](SMOKE-ADJUDICATION.md) for the unchanged historical
comparison, original quantitative thresholds and known lost morphology.

Rendering uses fonts available on the host. Generated image hashes can differ
between platforms, so compare each run against its saved manifest and truth;
do not infer numerical parity from a shared case name.
