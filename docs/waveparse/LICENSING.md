# Licence evidence and distribution boundary

Reviewed 5 September 2026. This is an evidence inventory, not an assurance that
the assembled inference runtime has an unrestricted commercial licence.

## Code allocation

The Python/JS interfaces, local subprocess protocol, setup tooling, TypeScript
orchestration, policy checks and independently authored deterministic processing
are MIT. The six layout YAML files listed in `THIRD_PARTY_NOTICES.md` adapt
Open-ECG-Digitizer's layout configuration and retain CC BY-SA 4.0. The two Python
subclass extensions are also offered under CC BY-SA 4.0 to avoid representing
that integration as wholly MIT. The corresponding source, notices and full
licence text accompany the clean release source and both language distributions.

The review compared the distributed Python/TypeScript/YAML source with pinned
upstream files and inspected the subclass boundaries and layout definitions.
No substantial copied Python or TypeScript block was found in that comparison.
The YAML layout definitions share upstream structure and lead arrangements.
This code inspection cannot establish authorship or rights to every historical
contribution; the release owner still reviews the stated allocation.

The clean source also includes benchmark scoring functions adapted from the
BSD-2-Clause PhysioNet 2024 evaluator at commit
`1a5135470e7fd9817633f055f3dadebb58fc89ef`. Its original copyright and licence
are retained in `LICENSES/PhysioNet-evaluation-2024-BSD-2-Clause.txt`.
Those scoring functions are absent from the distributed inference payload.

Open-ECG-Digitizer is fetched unmodified from commit
`97a15087d4abcda843da8c58ee74b1d8f47e6f9a`. Its original licence is hash-verified
at setup. Its source and weights are absent from npm, wheel, sdist and clean
source archives. The package invokes this separately installed runtime locally.

## Dependencies and model evidence

`licensing/runtime-darwin-arm64.json` inventories the installed locked macOS
runtime. Native CI produces the equivalent inventory on both supported platforms.
The inventories record exact distribution versions, declared licences, metadata
hashes, original notice-file paths/hashes, engine commit and all five model hashes.
Dependencies retain their original notices in the installed Python environment.

The inventory includes bundled-wheel components through the upstream wheels'
original third-party notices; a top-level BSD or MIT declaration does not
relicense every native library inside a wheel. For example, NumPy declares a
compound licence, and certifi/tqdm include MPL-2.0 material. `yacs` reports
`UNKNOWN` in metadata but its installed original LICENSE is Apache-2.0.

| Material | Evidence established | Remaining scope |
| --- | --- | --- |
| Open-ECG source | Pinned repository LICENSE is CC BY-SA 4.0 | Retain attribution and modification notices |
| Two ECG neural weights | Pinned upstream Git LFS objects; hashes and URLs recorded; repository licence retained | No separate model-specific grant or model card found in the pinned tree |
| RapidOCR 3.9.2 | Distribution and tagged source declare Apache-2.0 | Package metadata alone does not establish all converted model rights |
| Three OCR ONNX models | Shipped in the locked RapidOCR wheel; exact hashes and original model URLs recorded upstream | Confirm model-specific conversion/redistribution terms before mirroring |
| Apple Vision OCR | Uses the operating system framework on macOS; framework is not distributed | Operating system terms apply |
| ECG datasets and original scans | No patient images or historical run assets in the source/package allowlists | Any future validation corpus needs its own documented rights |

Evidence links:

- [Pinned Open-ECG licence](https://github.com/Ahus-AIM/Open-ECG-Digitizer/blob/97a15087d4abcda843da8c58ee74b1d8f47e6f9a/LICENSE)
- [Pinned Open-ECG README and citation request](https://github.com/Ahus-AIM/Open-ECG-Digitizer/blob/97a15087d4abcda843da8c58ee74b1d8f47e6f9a/README.md)
- [RapidOCR 3.9.2 licence](https://github.com/RapidAI/RapidOCR/blob/v3.9.2/LICENSE)
- [PaddleOCR licence](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE)

## Clarification prepared for the release owner

Before public publication, obtain an explicit answer from the ECG engine/model
maintainers confirming whether CC BY-SA 4.0 covers both named neural weight
files and their use/distribution in a separately installed commercial runtime.
Also confirm the three RapidOCR model files' redistribution terms. Ask whether
additional notices, training-data restrictions or model-specific licences apply.
No request has been sent to maintainers on the user's behalf.

These questions are deliberately concrete. The current packaging does not mirror
the models, invent an MIT grant for them, or assert that every application using
the library must adopt a particular licence.
