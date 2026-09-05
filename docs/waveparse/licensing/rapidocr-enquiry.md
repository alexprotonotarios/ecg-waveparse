# RapidOCR ONNX model licence enquiry

Status: approved by the release owner and posted as
[discussion 736](https://github.com/RapidAI/RapidOCR/discussions/736) on
5 September 2026. Reloaded and verified against the approved text and Q&A
category; awaiting a maintainer answer.

Destination: a public question in
[RapidOCR Q&A](https://github.com/RapidAI/RapidOCR/discussions/categories/q-a),
the channel requested by its issue template for questions.

Subject: Licence and notices for the three ONNX models bundled with rapidocr 3.9.2

## Proposed message

Hello,

I am preparing a separately installed local OCR dependency for WaveParse, an
ECG digitisation library. We pin the `rapidocr==3.9.2` wheel and verify the three
bundled ONNX files against these SHA-256 identities:

| File | SHA-256 |
| --- | --- |
| `PP-OCRv6_det_small.onnx` | `090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f` |
| `PP-OCRv6_rec_small.onnx` | `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884` |
| `ch_ppocr_mobile_v2.0_cls_mobile.onnx` | `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c` |

These identities match the ONNX entries in `default_models.yaml` for v3.9.2.
Your README identifies Baidu as the model copyright holder and declares the
project Apache-2.0.

Could you point us to the licence and required notices that cover these exact
converted models? In particular, does Apache-2.0 cover their commercial inference
use and redistribution, including any conversion-related rights, or are there
additional model-specific terms?

Users currently install the unmodified upstream wheel explicitly during runtime
setup. Our npm/Python interface packages do not contain these models, and we do
not mirror them. We retain the upstream dependency notices. We would like to
document the applicable terms accurately for downstream users before release.

Thank you,

Alexandros Protonotarios

## Preparation evidence

- [Tagged README](https://github.com/RapidAI/RapidOCR/blob/v3.9.2/README.md)
- [Tagged licence](https://github.com/RapidAI/RapidOCR/blob/v3.9.2/LICENSE)
- [Tagged model catalogue](https://github.com/RapidAI/RapidOCR/blob/v3.9.2/python/rapidocr/default_models.yaml)
- [Question-channel configuration](https://github.com/RapidAI/RapidOCR/blob/main/.github/ISSUE_TEMPLATE/config.yml)

The locked wheel's installed model catalogue and all three hashes were checked
on 5 September 2026. Package-level licensing is established; this question asks
for an explicit statement covering the converted models.
