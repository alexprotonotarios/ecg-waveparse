#!/usr/bin/env bash
# Disposable Linux x86-64 container smoke test. Mount this checkout at /source:ro.
set -euo pipefail
apt-get update -qq
apt-get install -y -qq libgl1 libglib2.0-0 ca-certificates curl xz-utils > /tmp/apt.log
cd /tmp
curl --fail --silent --show-error -O https://nodejs.org/dist/v22.22.2/node-v22.22.2-linux-x64.tar.xz
curl --fail --silent --show-error -O https://nodejs.org/dist/v22.22.2/SHASUMS256.txt
sed -n '/ node-v22.22.2-linux-x64.tar.xz$/p' SHASUMS256.txt | sha256sum -c -
tar -xf node-v22.22.2-linux-x64.tar.xz -C /usr/local --strip-components=1
mkdir /tmp/consumer
cd /tmp/consumer
npm install --ignore-scripts --no-audit --no-fund /source/dist/ecg-waveparse-0.1.0.tgz
python -m pip install --no-index --no-deps /source/dist/ecg_waveparse-0.1.0-py3-none-any.whl
python -m ecg_waveparse setup --runtime-dir /tmp/runtime
python -m ecg_waveparse doctor --runtime-dir /tmp/runtime --workspace-dir /tmp/results
python -m ecg_waveparse digitize --input /source/benchmark/generated/smoke/cases/clean/image.png --runtime-dir /tmp/runtime --workspace-dir /tmp/results > /evidence/linux-result.json
python - <<'PY'
import json
from ecg_waveparse import Digitizer
r = json.load(open('/evidence/linux-result.json'))
assert r['status'] == 'needs_review', r.get('message')
assert r['layout'] == 'standard_6x2'
assert Digitizer(workspace_dir='/tmp/results').get_run(r['id'])['assets']['canonicalCsv']
print('Linux Python package inference and integrity reopen passed', r['id'])
PY
