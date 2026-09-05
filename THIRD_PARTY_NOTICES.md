# ECG WaveParse: licensing and release inventory

This is release preparation. Publication is disabled. MIT applies to independently
authored WaveParse software, not to every component of the assembled runtime.

| Material | Origin / treatment |
| --- | --- |
| Language interfaces, packaging, setup, runtime-path adapter | Independently authored WaveParse code; MIT. |
| TypeScript orchestration, deterministic processing, policy and rendering | Independently authored project implementation; MIT except the explicit file list below. |
| Layout configuration and two Open-ECG extension modules | CC BY-SA 4.0; attribution and modification details below. |
| Open-ECG-Digitizer | Ahus-AIM, commit `97a15087d4abcda843da8c58ee74b1d8f47e6f9a`; CC BY-SA 4.0. Downloaded by explicit setup with its original LICENSE; not included in npm/Python payloads or relicensed as MIT. |
| Neural model weights | Downloaded separately from that upstream revision; recorded hashes identify the exact files. Confirm their redistribution terms before mirroring or bundling them. |
| RapidOCR / ONNX OCR models and Python dependencies | Obtained in the locked runtime environment. RapidOCR 3.9.2 declares Apache-2.0. Package declarations, original notice-file hashes and model identities are in `docs/licensing/`; model-specific confirmation remains outstanding. |
| ECG images, reference outputs, benchmark datasets | Excluded from distributions and the clean source archive. The repository history is not a publishable source artifact. |
| Benchmark PhysioNet alignment/SNR compatibility functions | Adapted from `physionetchallenges/evaluation-2024`, commit `1a5135470e7fd9817633f055f3dadebb58fc89ef`, BSD-2-Clause. Present in the clean source benchmark tooling; absent from the inference runtime. Original notice retained in `LICENSES/PhysioNet-evaluation-2024-BSD-2-Clause.txt`. |

Upstream licence: https://github.com/Ahus-AIM/Open-ECG-Digitizer/blob/97a15087d4abcda843da8c58ee74b1d8f47e6f9a/LICENSE

For research using upstream code/data, retain the requested citation:
Stenhede E, Bjørnstad AM, Ranjbar A. Digitizing Paper ECGs at Scale: An
Open-Source Algorithm for Clinical Research. npj Digital Medicine (2026).
https://doi.org/10.1038/s41746-025-02327-1

## Attribution and modifications

The following distributed files use CC BY-SA 4.0, whose full text is retained at
`LICENSES/Open-ECG-Digitizer-CC-BY-SA-4.0.txt`:

- `ecg_pipeline/lead_layouts_reliable.yml`
- `ecg_pipeline/lead_layout_standard_6x2.yml`
- `ecg_pipeline/lead_layout_standard_6x2_with_r1_ignored.yml`
- `ecg_pipeline/lead_layout_standard_3x4.yml`
- `ecg_pipeline/lead_layout_standard_3x4_with_r1.yml`
- `ecg_pipeline/lead_layout_standard_12x1.yml`
- `ecg_pipeline/fidelity_inference_wrapper.py`
- `ecg_pipeline/reliable_signal_extractor.py`

The layouts adapt `src/config/lead_layouts_all.yml` from Open-ECG-Digitizer by
selecting supported layouts, constraining geometry, and handling rhythm rows.
Original attribution: Ahus-AIM / Elias Stenhede, Agnar Martin Bjørnstad and Arian
Ranjbar. WaveParse changes: Alexandros Protonotarios, 2026. The inference and
signal-extractor extensions call the upstream interfaces and add source-ink
support, deterministic seeds, feature caching and ordered ridge extraction.
These two extension modules are conservatively offered under the same CC BY-SA
terms. The language wrappers and setup tools remain under MIT.

The source paths above are under `runtime/` in the npm package and under
`ecg_waveparse/runtime/` in the Python package. Upstream code is downloaded
separately, unmodified, from the pinned commit; WaveParse does not claim upstream
endorsement. Retain this attribution, the licence text and modification notice
when redistributing those files. See `docs/LICENSING.md` for the review boundary.

`ecg_benchmark/scoring.py` implements bounded alignment and SNR compatibility
with the above PhysioNet evaluator. The `physionet_align_signals` and
`physionet_snr` functions retain BSD-2-Clause attribution. WaveParse adds
lead-local timing, missingness, coverage, uncertainty and morphology scoring.

## Before public distribution

1. Review the explicit mixed-licence allocation above and the inventory in
   `docs/LICENSING.md` against the exact release source.
2. Obtain model-specific clarification before calling the assembled runtime
   legally cleared or mirroring/bundling the model files. The absence of a
   separate model licence is not evidence of an unrestricted grant.
3. Verify registry ownership and inspect the allowlisted source archive. Do not
   make the existing data-bearing repository/history public as a shortcut.

These are publication blockers, not an assertion that all downstream applications
must use a particular licence. Local build and installation do not claim that
the assembled runtime is wholly MIT-licensed.
