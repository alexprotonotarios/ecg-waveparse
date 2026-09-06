# Measured execution budgets and platform differences

## Frozen final evaluation

The [25-patient final evaluation](FINAL-EVALUATION-2026-09-06.md) used the same
payload as the matrix below. Its 25 MPS calls returned 19 signals and six
abstentions, with no overall runtime failure. Median/p95/max call times were
82.244/174.909/190.628 seconds. Peak sampled process-tree RSS was 7.779 GiB,
exceeding its predeclared 6 GiB investigation budget. Peak evaluation workspace
was 0.349 GiB; the 1 GiB workspace and 660-second observed-call budgets passed.
One-second sampling covered 2,273 samples over 39.07 minutes. These are sampled
process-tree totals, not proof of a hard operating-system memory cap.

## Current installed macOS matrix

The gain/grid-corrected payload
`e84bef376703e8029ac673cfbc1f85d6cbc5fbb1db608367ca221e7e4c22150b`
completed twelve installed calls: three inputs from one deterministic signal
family, Python/JavaScript, and CPU/MPS. All twelve returned reviewable output,
with no overall runtime failures and twelve strict semantic passes. Seven of
nine full comparison groups passed the frozen protocol. This is a measured
matrix with failures, not a claim of full platform equivalence.

| Device, six calls each | Median call | Nearest-rank p95 / max | Peak sampled process-tree RSS | Peak workspace |
| --- | ---: | ---: | ---: | ---: |
| Apple Silicon MPS | 111.03 s | 120.65 s | 3.523 GiB | 154.13 MiB |
| Apple Silicon CPU | 300.33 s | 484.77 s | 9.193 GiB | 145.65 MiB |

Corresponding language outputs have identical waveforms, missingness,
uncertainty statuses/spreads and terminal review outcomes. One low-resolution
CPU pair fails exact uncertainty candidate-count equality: its JavaScript
geometry candidate failed after 180.94 seconds, consistent with the existing
180-second deadline, while the Python candidate completed in 166.17 seconds.
The generic retained failure message does not itself identify a timeout.

The same low-resolution input differs between devices. CPU/MPS RMSE is
30.199/26.383 µV and mean correlation is 0.966257/0.975562. The correlation loss
of 0.009305 exceeds the fixed 0.005 margin. Different selected candidate IDs
also fail full policy identity, although both terminal outcomes remain
`needs_review` / `reviewable_constrained`. One CPU call exceeds the 9 GiB
investigation budget. All call-time and workspace budgets pass. Sampled RSS
may count shared pages repeatedly; no OOM or hard memory-cap compliance is
claimed. No tolerance, deadline or selection policy was changed to hide drift.

The initial matrix attempt bypassed the installed wheel by resolving the Python
virtualenv executable symlink. Its two JavaScript completions and one Python
import failure remain recorded separately. The corrected launcher preserves
the virtualenv path and distinguishes planned calls from actual starts.

Full receipts, all attempts and resource samples are retained locally in
`benchmark/results/pro-completion-20260906/platform-matrix-v2/` and
`platform-matrix-receipt-v1.json`. The current hosted native Linux/macOS
comparison is prepared but has not run. The older Linux measurement below
has a different payload and does not certify this candidate.

## Archived first-campaign platform comparison

The tables below describe the archived first-campaign D payload. The subsequent
gain/grid payload and its broader installed interface/device matrix are tracked
in [the continuation report](COMPLETION-WORK-2026-09-06.md). Do not apply the older
platform comparisons to the new payload without its own measured results.

The [platform receipt](verification/2026-09-06-platforms.json) records the
predeclared protocol, exact package/runtime identities, all comparison gates,
per-lead differences, internal candidate outcomes, resource measurements and
Linux OS package inventory. All three CPU comparisons met the preset aggregate
limits. This is **one clean synthetic input per CPU/platform/interface**, not
broad platform equivalence.

| Execution | Aligned RMSE | Mean correlation | Coverage | Observed time |
| --- | ---: | ---: | ---: | ---: |
| macOS MPS, final D clean regression | 16.20953 µV | 0.991215 | 99.9467% | 83.30 s |
| macOS CPU, installed JavaScript | 20.17907 µV | 0.986271 | 99.9233% | 339.57 s |
| macOS CPU, installed Python | 20.17907 µV | 0.986271 | 99.9233% | 320.97 s |
| Linux x86-64 CPU, Python-installed CLI under Docker emulation | 16.20954 µV | 0.991215 | 99.9467% | 287.56 s |

Times cover the measured call/container lifetime; the MPS harness also scores
the output. This is not a controlled throughput ranking. All returned runs
require review; no clinical acceptance was made. Source bytes, runtime/model
manifest, policy outcome and strict semantic identity matched. The fixed
margins were at most +5 µV RMSE, −0.005 correlation and −0.005 coverage.

Linux differed from MPS by at most 0.00447 µV at a compared sample, with identical
missingness and uncertainty statuses. macOS CPU differed by as much as
515.75 µV at a sample, with 28 finite/missing locations and 485 uncertainty
labels differing. All three annotated synthetic morphology events remained
detected. Passing aggregate margins does not establish pointwise equivalence
or unchanged fine morphology.

Python and JavaScript CPU runs produced identical canonical/compact CSVs,
source, diagnostic and paper images. Their uncertainty locations, statuses and
spread values also matched. Candidate counts differed by one at 29,988 of
30,000 uncertainty rows: the JavaScript run's geometry path candidate reached
its existing approximately 180-second budget; the Python repeat completed it
in 158.09 seconds. Both overall jobs completed. The timeout is retained in the
receipt; no deadline or extraction policy was changed to remove it.

## Resource observations and limits

The eight-case final D MPS regression took 1,007.14 seconds including profiling.
Median per-case call time was 114.93 seconds; nearest-rank p95/max was 222.30
seconds (n=8). Sampled process-tree RSS peaked at 3.79 GB and workspace size at
233.11 MB. The original baseline took 810.25 seconds under different desktop
conditions; these single runs do not prove a speed improvement or isolate the
effect of the refactor. No speed improvement is claimed.

Median measured stage times across the final regression were runtime verification
3.68 s, capture preflight 5.61 s, admission 0.077 s, preprocessing 5.93 s,
OCR/layout 4.35 s and export/provenance 1.50 s. Median engine total was 102.08 s.
Nested stage times are not additive independent measurements. Candidate times
include their construction, inference and vectorization work.

Native CPU JavaScript/Python sampled process-tree RSS peaked at 8.42/8.53 GB;
their workspace sizes peaked at 80.93/84.80 MB. RSS can count shared pages in
multiple processes and is not an OS hard maximum. The separate public pilot
peaked at 10.35 GB sampled process-tree RSS across 16 inputs.

Linux ran as UID/GID 10001, with no network, a read-only root filesystem,
read-only source/runtime mounts, dropped capabilities, no-new-privileges,
256 PID limit, four CPU limit and 6 GiB memory limit. It exited zero without
OOM. Sampled Docker container memory reached **5.915 GiB**, close to that
limit; this clean-case success is not a capacity guarantee for other images.
Docker memory accounting differs from the macOS process-tree RSS measure.
All sampled network counters were zero; the Docker network mode was `none`.

Use one heavy evaluation at a time on this 18 GiB laptop. Keep the default
30-minute whole-job deadline unless a separately measured deployment requires
another budget. Cancellation immediately after source admission completed in
110.4 ms against the 30-second cleanup budget; this measurement does not claim
the same latency during neural inference. The existing process-group timeout,
interruption and recovery tests remain part of verification.

## Construction and optimization decision

Direct profiling of unchanged upstream construction measured 1.931 seconds
on first construction in a process and 0.099 seconds on a repeated construction.
The first signal-extractor initialization was 1.048 s; segmentation-model
initialization 0.087 s and layout initialization 0.034 s. The filesystem cache
could already be warm. This was object/model construction, not forward inference
or a disk-cold startup benchmark.

This evidence does not justify introducing a persistent worker or model cache
with a new lifetime/isolation contract. Keep the current architecture and its
timeouts; retain the profiling tools for a separately controlled performance
change. Setup and inference stay separate. Docker Desktop was started only for
these tests and restored to its previous stopped state afterward.

The continuation adds 24 actual forward passes (four inputs, cold plus two
within-process repeats, on each of CPU/MPS). Each device returned nine finite
candidates and three empty 12x1 arrays. Within-device waveforms, missingness,
layouts and scales were exact. CPU/MPS first-forward RMSE differences were
0.173, 2.972 and 6.290 µV across nonempty inputs; this does not establish
application policy or uncertainty parity. Peak sampled RSS was 8.800 GiB on
CPU and 2.726 GiB on MPS. Warm processing was not consistently faster, so the
no-cache decision remains.

Separately, observed neural-inference cancellation through the installed
EXIF-v4 interfaces completed in 153.565 ms (JavaScript) and 145.205 ms (Python).
A test hook confirmed a real U-Net convolution had returned before cancellation.
Both runs retained coherent failed records and inspectable originals, exported
no numerical CSV and left no surviving inference child. These observations
extend the earlier admission-only check and do not alter the 30-second cleanup
budget. The later calibration-v6c matrix is a separate experiment.
