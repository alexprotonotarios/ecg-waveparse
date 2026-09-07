# Local Linux container

The recipe targets **Linux x86-64** with the pinned Python 3.12.9 base-image
digest, verified Node 22.22.2 archive, and the same installed npm/wheel payload.
It runs as UID/GID 10001. Apt OS dependencies are resolved at build time; the
finished image digest and package inventory must accompany any deployment.

From the source checkout, build the packages and image:

```sh
pnpm waveparse:build
python3.12 -m build --no-isolation packages/python --outdir dist
docker build --platform linux/amd64 -f docker/waveparse.Dockerfile -t waveparse-local:engineering .
docker volume create waveparse-runtime
docker volume create waveparse-work
docker run --rm --platform linux/amd64 --cap-drop ALL --security-opt no-new-privileges \
  -v waveparse-runtime:/runtime waveparse-local:engineering setup --runtime-dir /runtime/engine-runtime
```

Setup explicitly downloads the hash-locked upstream code, weights and Python
dependencies. It is separate from inference and does not establish model-rights
clearance. Read `LICENSING.md` before distribution or deployment.

Process a synthetic/licensed source mounted read-only, with egress disabled:

```sh
docker run --rm --platform linux/amd64 --network none --read-only \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 256 \
  --memory 6g --cpus 4 --tmpfs /tmp:rw,nosuid,size=1g \
  -v waveparse-runtime:/runtime:ro -v waveparse-work:/work \
  -v "$PWD/benchmark/generated/smoke/cases/clean:/input:ro" \
  waveparse-local:engineering digitize --input /input/image.png \
  --runtime-dir /runtime/engine-runtime --workspace-dir /work
```

The 6 GiB / four CPU settings are initial engineering limits, not measured
capacity guarantees for every source. Keep the runtime and work volumes private
and preserve the process until it returns a terminal record. The library enforces
input, timeout, job-claim and storage limits; the workspace needs its configured
free-space floor plus working reservations. Source and reconstructed signals
are both sensitive. A paper render is a visualization; quantitative work uses
the CSV with source review, segment timing, gaps and uncertainty.

Run `doctor` using the same runtime volume before inference. No ports, host
networking, privileged container, Docker socket or host home-directory mounts
are required. Set deployment ownership and retention explicitly. Linux/CPU
results and macOS/MPS results require measured comparison, not assumed parity.

The [6 September comparison](PERFORMANCE-2026-09-06.md) executed this isolation
configuration on one frozen synthetic clean input using the installed Python
CLI. The x86-64 image ran under Apple Silicon Docker emulation, exited zero,
returned a reviewable signal and passed the fixed aggregate comparison gates.
Sampled container memory reached 5.915 GiB, close to the 6 GiB limit. Native
Ubuntu matrix execution remains a separate hosted CI check. The receipt records
the exact image, resolved OS dependencies and identical npm/wheel payloads.
