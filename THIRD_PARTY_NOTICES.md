# Third-party notices

MIT applies to independently authored WaveParse software. Components listed
below retain their own licences.

| Material | Origin / treatment |
| --- | --- |
| Language interfaces, packaging, setup, runtime-path adapter | Independently authored WaveParse code; MIT. |
| TypeScript orchestration, deterministic processing, policy and rendering | Independently authored project implementation; MIT except the explicit file list below. |
| Layout configuration and Open-ECG extension modules | CC BY-SA 4.0; attribution and modification details below. |
| Open-ECG-Digitizer | Ahus-AIM, commit `97a15087d4abcda843da8c58ee74b1d8f47e6f9a`; CC BY-SA 4.0. Downloaded by explicit setup with its original LICENSE; not included in npm/Python payloads or relicensed as MIT. |
| Neural model weights | Downloaded separately from that upstream revision; recorded hashes identify the exact files. An upstream member's 26 September reply confirms the same CC BY-SA 4.0 licence covers those weights, with clear attribution and citation. Not bundled or mirrored. |
| RapidOCR / ONNX OCR models and Python dependencies | Obtained in the locked runtime environment. RapidOCR 3.9.2 declares Apache-2.0. Original dependency notices remain in the environment. The maintainer's model notice at commit `4a3070f304467e6d426e78a82afea5cb1181f305` names all three pinned OCR hashes and Apache-2.0 terms, with Baidu/PaddleOCR attribution. |

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
- `ecg_pipeline/compact_signal_extractor.py` (cropped storage for rejected centroid regions)
- `ecg_pipeline/cpu_unet_lifetime.py` (release consumed U-Net tensor references on Linux CPU)
- `ecg_pipeline/cpu_unet_decoder_lifetime.py` (release consumed encoder/decoder inputs on Mac CPU)
- `ecg_pipeline/probability_path.py` (extracted unchanged from the signal-extractor extension)

The layouts adapt `src/config/lead_layouts_all.yml` from Open-ECG-Digitizer by
selecting supported layouts, constraining geometry, and handling rhythm rows.
Original attribution: Ahus-AIM / Elias Stenhede, Agnar Martin Bjørnstad and Arian
Ranjbar. WaveParse changes: Alexandros Protonotarios, 2026. The inference and
signal-extractor extensions call the upstream interfaces and add source-ink
support, deterministic seeds, feature caching and ordered ridge extraction.
The compact signal extractor adapts the pinned upstream centroid region loop
to store independent occupied crops, preserving its calculations and refinement order.
The CPU U-Net lifetime module adapts `src/model/unet.py` to release consumed
encoder skips and upsampled tensors after concatenation, preserving the original
operator sequence and complete spatial normalization.
The Mac CPU input-lifetime module also adapts `src/model/unet.py`.
It traverses bare, unhooked encoder and decoder containers directly so consumed
feature and concatenated inputs are released after their convolution. The original
encoder input remains live for its actual skip operation. Leaf operations and
complete spatial normalization remain in order; hooked/custom and training paths
retain upstream dispatch. Linux continues to use its existing lifetime implementation.
These extension modules and the extracted probability-path module are conservatively offered under the same CC BY-SA
terms. The language wrappers and setup tools remain under MIT.

The source paths above are under `runtime/` in the npm package and under
`ecg_waveparse/runtime/` in the Python package. Upstream code is downloaded
separately, unmodified, from the pinned commit; WaveParse does not claim upstream
endorsement. Retain this attribution, the licence text and modification notice
when redistributing those files.


Model files are downloaded by explicit setup from their recorded upstream
locations. They are not included in these archives. The exact OCR model hashes
are covered by RapidOCR's
[model notice](https://github.com/RapidAI/RapidOCR/blob/4a3070f304467e6d426e78a82afea5cb1181f305/python/MODEL_LICENSES.md),
verified on 10 September 2026. Retain its attribution and licence material if
redistributing those models. The
[26 September Open-ECG reply](https://github.com/Ahus-AIM/Open-ECG-Digitizer/issues/47#issuecomment-5842963810)
confirms that the same CC BY-SA 4.0 licence covers the two named neural weights,
with clear attribution and citation. Fresh REST responses and both active weight
hashes were verified on 29 September. Retain the upstream licence and requested
research citation. The assembled runtime is not represented as wholly MIT-licensed.
No dependency or model version changed for this notice update.
