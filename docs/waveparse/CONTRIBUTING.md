# Contributing to ECG WaveParse

Use Node 22.22.2, CPython 3.12.9 and pnpm 10.34.5. Run from the clean source root:

```bash
pnpm install --frozen-lockfile
python -m pip install -r requirements-waveparse-build.txt
pnpm waveparse:build
python -m build --no-isolation packages/python --outdir dist
pnpm waveparse:test
python -m unittest scripts.test_waveparse_runtime
WAVEPARSE_PYTHON="$(command -v python)" pnpm waveparse:verify
```

Both distributions must contain the same runtime payload. Change the shared
engine code once, then rebuild both wrappers. Keep API/schema changes separate
from numerical-behaviour changes in release notes. Run the native platform
workflow and installed-package regression when changing inference behaviour.

Use synthetic or explicitly permitted fixtures. Do not commit patient ECGs,
private run folders, model caches, credentials or local environment files. Never
overwrite original inputs or use generated images to alter ECG morphology.
Quantitative outputs require source comparison and explicit review; plots alone
do not establish measurement accuracy. Keep benchmark denominators, abstentions,
failures, coverage and waveform error together in reports.

The original development repository contains private historical data. Public
contributions belong in the clean source repository and must not import the
original git history. Preserve third-party notices and read LICENSING.md.
