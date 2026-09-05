# Version 0.1.0 verification — 5 September 2026

This records local release preparation. Nothing has been published to npm/PyPI,
pushed, or made public. The GitHub workflow is prepared but has not run remotely.

## Packaging and compatibility

- Built the npm tarball, Python wheel, Python sdist and allowlisted source archive.
- Installed npm and wheel artifacts into fresh projects outside this checkout.
- Verified all 30 shared runtime files against their SHA-256 manifest in both
  installations, plus bundled documentation and third-party notices.
- Passed JavaScript ESM/CommonJS execution and TypeScript NodeNext consumer checks.
- Passed Python synchronous/asynchronous reads with matching persisted results.
- Rebuilt the clean source archive independently; its runtime payload manifest
  matched the checkout build exactly. No original ECG images, reference outputs,
  neural weights, storage directories or Git history are included in distributions.

## Runtime and extraction

Fresh explicit setup and doctor passed on Apple Silicon macOS and Linux x86-64
(Debian Bookworm, Docker emulation). The Linux environment used CPU-only Torch
2.12.0+cpu and torchvision 0.27.0+cpu. Both used Node 22.22.2 and CPython 3.12.9.
The runtime manifest SHA-256 is:

```text
d9465db541f0d2ad0541eb04f36e3fee930a8bbbc1a50135fe2950b63203fc3a
```

Real extraction used the existing deterministic `smoke/clean` engineering image,
source SHA-256 `c717830e02f8cda47646296abce342265ace7a24a438d52671ba3399de7d1f6c`.

| Execution | Initial result | Selected candidate | Canonical CSV SHA-256 |
| --- | --- | --- | --- |
| Existing core, Mac MPS baseline | needs_review / standard_6x2 | ink-path-2200 | `916e38c071c8b7d5cc290589f6ae9bea96ffbfd6aff121468a8231ecc1fbea23` |
| Installed npm library, Mac MPS | needs_review / standard_6x2 | ink-path-2200 | `916e38c071c8b7d5cc290589f6ae9bea96ffbfd6aff121468a8231ecc1fbea23` |
| Installed Python library, Linux CPU | needs_review / standard_6x2 | ink-path-2200 | `1256bd8ea544a1f348daec51a2860541b00bfb1470d9c11faf62b44d4a28ad50` |

The matched Mac MPS comparison is byte-for-byte identical. All seven Linux CPU
candidates completed. Linux/MPS hashes differ; no cross-platform numerical identity
is claimed. A separate Mac CPU run produced reviewable native-grid output after
several neural candidates reached existing execution limits under system load.
Those limits and selection policies were not relaxed. Hardware, device and resource
availability can therefore affect candidate availability and the selected result.

Review rejection, refusal of acceptance without confirmations, retained-source
integrity, deliberate synthetic-copy tamper detection, cancellation after admission,
terminal compaction, and recovery of a dead local runner were exercised successfully.
Test outputs were not accepted as clinically reviewed ECGs.

## Existing app regression checks

- 183 JavaScript tests passed, including the package contract test.
- 222 Python tests passed, including installer archive-integrity tests.
- TypeScript checking and ESLint passed.
- Next.js production build passed without the initial broad-file-tracing warnings.
- The existing local server restart/recovery test passed.

This is packaging and engineering parity evidence, not a new accuracy benchmark
or clinical validation. The saved historical validation metrics have not been
relabelled as current. Complete the publication actions in
`THIRD_PARTY_NOTICES.md` and run the prepared native Ubuntu/macOS CI matrix before
public release. Final distribution checksums are in `dist/SHA256SUMS`.
