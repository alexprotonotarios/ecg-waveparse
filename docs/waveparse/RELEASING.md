# Preparing a release

1. Confirm versions agree and run the supported-platform package workflow.
2. Build the npm tarball, Python wheel/sdist and allowlisted source archive.
3. Inspect package manifests and verify identical runtime payload hashes.
4. Install artifacts into clean external projects; exercise the documented APIs,
   CLI, real model inference, review and failure behaviour.
5. Save compact test evidence and artifact SHA-256 values in the release report.

Run `node scripts/audit-waveparse-source.mjs build/waveparse/source dist/source-audit.json`
before transferring clean source. `PUBLISHING.md` records the intended registry
configuration; `LICENSING.md` records what has and has not been established.

The clean library source is shared as a source preview. npm/PyPI publication is
disabled for this preparation. Before enabling registry uploads, complete the
licensing actions in `THIRD_PARTY_NOTICES.md`, verify ownership of
the npm/PyPI names and approve the exact artifacts. A 404 during a registry lookup
does not reserve a package name. No tokens are required for preparation.

The original `ecg_digitizer` development repository contains historical ECG data.
The separate `ecg-waveparse` repository uses an audited clean source history;
public contributions belong there. Do not import the original development
history or private data into it. Preserve the local originals and evidence.

After publication, consumers upgrade through ordinary dependency update PRs and
redeploy their worker. A tagged release binds package, engine, model and policy
identities. API compatibility and numerical behaviour are reviewed independently;
compatible syntax does not imply unchanged waveforms. Keep old results intact.
