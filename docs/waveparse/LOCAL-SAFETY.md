# Local execution boundaries and fault verification

The supported deployment is a private local/backend worker with explicit runtime
setup, controlled storage ownership and bounded application concurrency. This
audit identifies exercised boundaries; it is not proof of complete safety.

| Boundary | Verification and behavior |
| --- | --- |
| Input decoding | PNG/JPEG/WebP/single-frame TIFF only; 50 MB API limit, 80 million decoded pixels, maximum edge 32,768. Multipage and malformed rasters fail as `invalid_input`; source bytes remain untouched. |
| Admission and storage | Free-space floor and active reservations; exclusive owner-bound claims; private file/directory modes; intake journal and crash recovery at journal/source/metadata/commit boundaries. |
| Untrusted paths | Run IDs, source/derived identities, ancestry symlinks, redirected paths and archive traversal are checked. Changed bytes fail closed. |
| Durable writes | Injected ENOSPC at open/write/fsync/rename/directory-fsync leaves an old or complete new record. Atomicity does not promise hardware durability on every filesystem. |
| Job lifecycle | Duplicate invocation, cancellation, whole-job timeout, dead worker, stale owner, process-tree cleanup and local-server restart are exercised. A live owner cannot be stolen by elapsed time alone. |
| Review | Concurrent events serialize; acceptance requires complete evidence, source identity and actual review confirmations. Source/artifact/audit tampering is refused. Review preserves v2 uncertainty and segments. |
| Interruption during review | The audit is written before metadata. An interrupted multi-file update can leave an unattested/mismatched audit; integrity checks refuse it. Preserve evidence for recovery rather than silently re-attesting a partial review. |
| Downloads and models | Setup is explicit; source/weights/dependency hashes are locked, archive members checked, and runtime doctor verifies identities. Experimental checkpoint rights are checked before use. |
| Deserialization | Feature-cache tensor loading uses `weights_only=True`, shape/type/finite checks and invalid-cache rejection. This is not permission to load arbitrary models or pickle files. |
| Network | The corrected macOS installed regression runs under an OS `deny network*` policy; an egress probe verifies enforcement. Linux uses `--network none` after setup. |
| Privacy | OCR telemetry is disabled and portable OCR tests check it. Inputs, OCR text, traces and provenance remain sensitive local data. No clinical images were uploaded for debugging. |

The subprocess protocol carries JSON over local stdin/stdout. The library does
not expose an authenticated network service. The optional UI binds loopback and
requires its local-use guard. Containers use an unprivileged UID, dropped
capabilities, a read-only input/runtime/root filesystem, private writable work
and temporary storage, and no host home or Docker socket mounts.

Default application error messages for malformed rasters omit source filenames.
Upstream subprocess diagnostics and retained local provenance can still include
paths or OCR text; do not forward them wholesale to external crash reporting.
The profiler reads only PID/PPID/RSS, not process arguments. A deployment must
choose log retention/access and storage cleanup deliberately. Run the storage
audit before pruning, and keep signed validation and explicitly pinned evidence.

Long-running extraction tests use untouched synthetic or publicly licensed
waveforms. Lifecycle acceptance checks in unit fixtures do not constitute
clinical review. Installed campaign runs remain unaccepted.
