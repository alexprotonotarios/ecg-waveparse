# Changelog

## 0.1.0 — unpublished release preparation

- Python and JavaScript/TypeScript packages expose the same local engine and
  persisted result contract, with synchronous/asynchronous Python and ESM/CJS
  JavaScript interfaces.
- Explicit runtime setup pins and verifies upstream source, dependencies and
  model hashes. Processing runs locally after setup.
- Packages include installation/API documentation, corresponding source,
  licence notices, artifact integrity checks and review/cancellation support.
- Release verification found native 3x4/12x1 candidates with unverified panel
  time origins. Those candidates are excluded from quantitative selection,
  including through peer corroboration. The tested 12x1 case now abstains;
  the 3x4 fallback remains less accurate than the tested 6x2 cases.
- Policy abstention no longer receives a misleading worker-failure code.

See VERIFICATION.md for exact payload identities, measurements and limitations.
Intermediate preparation archives all use 0.1.0 and are distinguished by hashes;
none has been published. Once published, never replace an existing version's
artifacts. Each release must record API changes and numerical/selection changes
separately, even when the public function signatures remain compatible.
