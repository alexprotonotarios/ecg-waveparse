# Third-party notices

MIT applies to independently authored WaveParse software. Components listed
below retain their own licences.

| Material | Origin / treatment |
| --- | --- |
| Language interfaces, packaging, setup, runtime-path adapter | Independently authored WaveParse code; MIT. |
| TypeScript orchestration, deterministic processing, policy and rendering | Independently authored project implementation; MIT except the explicit file list below. |
| Layout configuration and two Open-ECG extension modules | CC BY-SA 4.0; attribution and modification details below. |
| Open-ECG-Digitizer | Ahus-AIM, commit `97a15087d4abcda843da8c58ee74b1d8f47e6f9a`; CC BY-SA 4.0. Downloaded by explicit setup with its original LICENSE; not included in npm/Python payloads or relicensed as MIT. |
| Neural model weights | Downloaded separately from that upstream revision; recorded hashes identify the exact files. Confirm their redistribution terms before mirroring or bundling them. |
| RapidOCR / ONNX OCR models and Python dependencies | Obtained in the locked runtime environment. RapidOCR 3.9.2 declares Apache-2.0. Original dependency notices are retained in the installed environment; model-specific confirmation remains outstanding. |

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
- `ecg_pipeline/probability_path.py` (extracted unchanged from the signal-extractor extension)

The layouts adapt `src/config/lead_layouts_all.yml` from Open-ECG-Digitizer by
selecting supported layouts, constraining geometry, and handling rhythm rows.
Original attribution: Ahus-AIM / Elias Stenhede, Agnar Martin Bjørnstad and Arian
Ranjbar. WaveParse changes: Alexandros Protonotarios, 2026. The inference and
signal-extractor extensions call the upstream interfaces and add source-ink
support, deterministic seeds, feature caching and ordered ridge extraction.
These extension modules and the extracted probability-path module are conservatively offered under the same CC BY-SA
terms. The language wrappers and setup tools remain under MIT.

The source paths above are under `runtime/` in the npm package and under
`ecg_waveparse/runtime/` in the Python package. Upstream code is downloaded
separately, unmodified, from the pinned commit; WaveParse does not claim upstream
endorsement. Retain this attribution, the licence text and modification notice
when redistributing those files.


Model files are downloaded by explicit setup from their recorded upstream
locations. They are not included in these archives. Model-specific terms have
not been independently confirmed; check the upstream terms for your intended
use before mirroring or redistributing the assembled runtime.
