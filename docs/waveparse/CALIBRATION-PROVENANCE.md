# Selector calibration provenance

The historical profile is preserved byte-for-byte. The exact named benchmark
run was located and the existing fitting algorithm regenerated the same JSON
profile, including its numerical priors and context decisions. The default
command previously pointed to another run and did not reproduce it. Both
outcomes are recorded in [the verification receipt](verification/2026-09-06-calibration.json).

The correct historical command uses:

```sh
python scripts/calibrate_candidate_selector.py \
  --manifest benchmark/generated/multisource-truth-v1/exhaustive-100-manifest.json \
  --run benchmark/results/current-contract3-semantic-exhaustive-100-c4-20260820/run.json \
  --output benchmark/results/selector-calibration-draft/profile.json
```

There are 72 development groups and 28 validation groups. The 698 recorded score
files used by fitting are hashed. No held-out score is opened while fitting or
choosing context activation. Historical extraction was recorded as
`6e3910827c617fc0034195eb5c15fa8061c5d6f3-dirty`; its complete modified source
snapshot was not recovered. Reproducing the fitted profile from saved scores
does not establish reproducibility of those original extraction runs.

New drafts include a recipe containing exact manifest/run file hashes,
individual score-set identity, source-file hashes, feature definitions,
hyperparameters, Python version, split-membership hashes and group/case counts.
The procedure is deterministic and has no random seed. Membership remains in
controlled local manifests; no private corpus is copied into source packages.
Extraction source-snapshot identity is recorded separately from fitting-source
identity and remains explicitly absent for the historical run.

The shared parameter contract binds each new profile's candidate IDs to its
actual numerical options. Reusing a name after changing vectorizer, sampling,
threshold, preprocessing or capabilities invalidates that profile match. The
existing profile is labelled `historical_unverified` for this newer parameter
contract; its historical policy is not silently relabelled as newly calibrated.
Replacing the production profile requires a separate fixed-data evaluation.

The CLI writes a draft by default and refuses to overwrite the historical
production configuration. The `.recipe.json` file is independently inspectable.
Changing feature semantics requires a new parameter/pipeline contract and a
new evaluation; changing a version number alone cannot recreate old evidence.
