"""Write review-only source-photo evidence without filling missing samples."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ._waveforms import clean, pack

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
ROWS = (("I", "aVR", "V1", "V4"), ("II", "aVL", "V2", "V5"), ("III", "aVF", "V3", "V6"))
SAMPLES = 5000
RATE = 500
ARM = "guarded-rays"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(clean(pack(value)), handle, indent=2, allow_nan=False)
        handle.write("\n")


def _csv(path: Path, header, rows) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _uv(value: float | None) -> str:
    return "" if value is None else format(value * 1000, ".17g")


def _window(path: dict) -> tuple[int, int]:
    return (0, SAMPLES) if path["lead"] == "rhythm_II" else (path["column"] * 1250, (path["column"] + 1) * 1250)


def validate_export(image: np.ndarray, result: dict) -> tuple[dict, dict]:
    """Validate the versioned evidence boundary before writing any artifact."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("A BGR uint8 source raster is required.")
    source = result.get("source", {})
    if not isinstance(source.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is None:
        raise ValueError("Invalid encoded or decoded source identity.")
    if source.get("decodedRasterSha256") != _sha(image.tobytes()) or source.get("imageSize") != [image.shape[1], image.shape[0]] or source.get("unchanged") is not True:
        raise ValueError("Diagnostic evidence does not match the source raster.")
    if result.get("schemaVersion") != 2 or result.get("method") != "source-photo-waveform-v1" or any(result.get(key) is not False for key in ("productionEligible", "semanticIdentityConfirmed", "quantitativeUseApproved")) or result.get("requiresReview") is not True:
        raise ValueError("Expected the diagnostic, review-required result contract.")
    if result.get("quantitativeUncertainty") != "unavailable" or result.get("zeroReviewFlagDoesNotMeanSafe") is not True or result.get("sourceGapsMayNotBeFilled") is not True:
        raise ValueError("Missing diagnostic uncertainty or gap limitations.")
    observation = result.get("observation", {})
    if result.get("state") != "diagnostic_available":
        if result.get("state") not in {"unresolved", "preferred_timing_available"} or observation.get("candidatePassed") is not False or result.get("selectedDiagnosticArm") is not None:
            raise ValueError("Inconsistent unsupported diagnostic result.")
        return {}, {}
    if observation.get("candidatePassed") is not True or result.get("selectedDiagnosticArm") != ARM:
        raise ValueError("Missing fixed diagnostic arm.")
    waveforms = observation.get("waveforms", {})
    arm = waveforms.get("arms", {}).get(ARM, {})
    if waveforms.get("sampleRateHz") != RATE:
        raise ValueError("Unsupported diagnostic sampling rate.")
    converted = arm.get("convertedPaths", [])
    reviews = arm.get("canonicalReview", [])
    paths = {p["lead"]: p for p in converted}
    flags = {p["lead"]: p["canonicalReviewRequired"] for p in reviews}
    expected = set(LEADS) | {"rhythm_II"}
    if len(converted) != 13 or len(reviews) != 13 or set(paths) != expected or set(flags) != expected:
        raise ValueError("All 12 primary leads and the separate rhythm II are required.")
    calibration = observation["sourceGrid"]["sourcePaths"]["calibration"]["reconciledCalibration"]
    if calibration.get("paperSpeedMmPerSecond") != 25 or calibration.get("gainMmPerMv") != 10:
        raise ValueError("This diagnostic exporter supports corroborated 25 mm/s and 10 mm/mV only.")
    sampled = observation.get("raySourcePaths", {}).get("arms", {}).get(ARM, [])
    sampling = {p["lead"]: p for p in sampled}
    if len(sampled) != 13 or set(sampling) != expected:
        raise ValueError("Missing source-sampling review evidence.")
    for lead, path in paths.items():
        expected_slot = (3, None) if lead == "rhythm_II" else next((r, c) for r, row in enumerate(ROWS) for c, name in enumerate(row) if name == lead)
        if (path["row"], path["column"]) != expected_slot or path.get("gainMmPerMv") != 10 or path.get("canonicalSampleRate") != RATE:
            raise ValueError("Inconsistent lead position or physical calibration.")
        values = path["canonicalMv"]
        mask = flags[lead]
        start, end = _window(path)
        if len(values) != SAMPLES or len(mask) != SAMPLES or any(type(v) is not bool for v in mask):
            raise ValueError("Invalid canonical waveform or review-mask shape.")
        for i, (value, flag) in enumerate(zip(values, mask, strict=True)):
            if value is not None and (not _number(value) or not start <= i < end):
                raise ValueError("Invalid value or filled structural absence.")
            if flag and value is None:
                raise ValueError("Review flag without a returned sample.")
        n = path["sourceColumns"]
        if type(n) is not int or n < 2 or len(sampling[lead]["reviewRequired"]) != n or len(sampling[lead]["sourceChordReviewRequired"]) != n - 1:
            raise ValueError("Incomplete source-sampling review flags.")
        for key in ("originalSourceX", "sourcePathY", "priorValid", "mappedValid", "gapReasons"):
            if len(path[key]) != n:
                raise ValueError("Incomplete source-coordinate evidence.")
        if len(path["connectionValid"]) != n - 1 or path.get("oldGapsRecovered") != 0:
            raise ValueError("Invalid source connections or recovered source gaps.")
        for i in range(n):
            if path["mappedValid"][i]:
                x, y = path["originalSourceX"][i], path["sourcePathY"][i]
                if not path["priorValid"][i] or not _number(x) or not _number(y) or not 0 <= x < image.shape[1] or not 0 <= y < image.shape[0] or path["gapReasons"][i] is not None:
                    raise ValueError("Unsupported source coordinate was exported.")
        seen = set()
        for segment in path["segments"]:
            targets = segment["targetIndices"]
            if any(type(i) is not int or not start <= i < end or values[i] is None or i in seen for i in targets):
                raise ValueError("Invalid or overlapping canonical segment.")
            if any(b != a + 1 for a, b in zip(targets, targets[1:])):
                raise ValueError("Noncontiguous canonical segment.")
            seen.update(targets)
        if seen != {i for i, v in enumerate(values) if v is not None}:
            raise ValueError("Canonical support differs from its source segments.")
    return paths, flags


def _overlay(image: np.ndarray, paths: dict, observation: dict, output: Path) -> None:
    overlay = Image.fromarray(image[:, :, ::-1].copy())
    draw = ImageDraw.Draw(overlay)
    source_flags = {p["lead"]: p for p in observation["raySourcePaths"]["arms"][ARM]}
    for lead, path in paths.items():
        points = list(zip(path["originalSourceX"], path["sourcePathY"], strict=True))
        for i, valid in enumerate(path["mappedValid"]):
            if valid:
                colour = (230, 115, 0) if source_flags[lead]["reviewRequired"][i] else (0, 145, 145)
                draw.point(points[i], fill=colour)
        for i, joined in enumerate(path["connectionValid"]):
            if joined and path["mappedValid"][i] and path["mappedValid"][i + 1]:
                review = source_flags[lead]
                colour = (230, 115, 0) if review["reviewRequired"][i] or review["reviewRequired"][i + 1] or review["sourceChordReviewRequired"][i] else (0, 145, 145)
                draw.line([points[i], points[i + 1]], fill=colour, width=3)
    # Text sits in a separate margin; the source raster's dimensions and pixels are not rescaled.
    canvas = Image.new("RGB", (overlay.width, overlay.height + 44), "white")
    canvas.paste(overlay, (0, 44))
    ImageDraw.Draw(canvas).text((12, 10), "Review required | Teal: returned trace | Orange: qualitative review flag | Gaps remain absent", fill="black", font=ImageFont.load_default(size=20))
    canvas.save(output)


def _paper(paths: dict, flags: dict, output: Path) -> None:
    # 10 pixels/mm gives a 254 dpi render with the stated physical scale.
    ppm = 10
    width, height = 2800, 1640
    image = Image.new("RGB", (width, height), (255, 253, 250))
    draw = ImageDraw.Draw(image)
    for x in range(0, width, ppm):
        draw.line((x, 0, x, height), fill=(238, 168, 168) if x % 50 == 0 else (249, 221, 221))
    for y in range(0, height, ppm):
        draw.line((0, y, width, y), fill=(238, 168, 168) if y % 50 == 0 else (249, 221, 221))
    draw.rectangle((0, 0, width, 105), fill=(255, 253, 250))
    draw.text((180, 20), "25 mm/s  |  10 mm/mV  |  500 Hz output  |  Review required", fill=(25, 25, 25), font=ImageFont.load_default(size=27))
    draw.text((180, 62), "Approximate geometry; lead-median baseline; absolute ST levels not established. Orange = review flag.", fill=(55, 55, 55), font=ImageFont.load_default(size=21))
    for row in range(4):
        baseline = (38 + 32 * row) * ppm
        draw.line([(40, baseline), (50, baseline), (50, baseline - 100), (100, baseline - 100), (100, baseline), (110, baseline)], fill=(30, 30, 30), width=3)
    for lead, path in paths.items():
        start, _ = _window(path)
        baseline = (38 + 32 * path["row"]) * ppm
        left = 180 + (0 if lead == "rhythm_II" else path["column"]) * 625
        draw.text((left + 8, baseline - 180), "II rhythm" if lead == "rhythm_II" else lead, fill=(25, 25, 25), font=ImageFont.load_default(size=30))
        for segment in path["segments"]:
            targets = segment["targetIndices"]
            for i in targets:
                point = (left + (i - start) * 25 * ppm / RATE, baseline - path["canonicalMv"][i] * 10 * ppm)
                draw.point(point, fill=(205, 100, 0) if flags[lead][i] else (25, 25, 25))
            for a, b in zip(targets, targets[1:]):
                points = [(left + (i - start) * 25 * ppm / RATE, baseline - path["canonicalMv"][i] * 10 * ppm) for i in (a, b)]
                draw.line(points, fill=(205, 100, 0) if flags[lead][a] or flags[lead][b] else (25, 25, 25), width=3)
    image.save(output, dpi=(254, 254))


def write_diagnostic_bundle(image: np.ndarray, result: dict, output_dir: Path) -> dict:
    """Write into a new directory; the final candidate.json is the completion marker.

    This accepts an in-memory result from observe_source_photo. It is not an
    authorization to promote a loaded result to production or clinical use.
    """
    paths, flags = validate_export(image, result)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    contract = {k: v for k, v in result.items() if k != "observation"}
    observation = result["observation"]
    if paths:
        columns = [paths["rhythm_II" if lead == "II" else lead]["canonicalMv"] for lead in LEADS]
        masks = [flags["rhythm_II" if lead == "II" else lead] for lead in LEADS]
        _csv(output_dir / "timeseries-canonical-uv.csv", LEADS, ([_uv(col[i]) for col in columns] for i in range(SAMPLES)))
        _csv(output_dir / "qualitative-review-required.csv", LEADS, ([int(col[i]) for col in masks] for i in range(SAMPLES)))
        _csv(output_dir / "primary-II-uv.csv", ["sample", "time_seconds", "primary_II_uv"], ([i, format(i / RATE, ".17g"), _uv(v)] for i, v in enumerate(paths["II"]["canonicalMv"])))
        def segment_rows():
            for lead, path in paths.items():
                start, end = _window(path)
                membership = {i: s for s, segment in enumerate(path["segments"]) for i in segment["targetIndices"]}
                for i in range(start, end):
                    yield [lead, "rhythm" if lead == "rhythm_II" else "primary", i - start, i, format(i / RATE, ".17g"), _uv(path["canonicalMv"][i]), int(flags[lead][i]), membership.get(i, "")]
        _csv(output_dir / "segments-and-review.csv", ["lead", "role", "local_sample", "canonical_sample", "display_time_seconds", "value_uv", "review_required", "connected_segment"], segment_rows())
        _json(output_dir / "source-path-evidence.json", {
            "schemaVersion": 2, "coordinateSpace": "working-raster", "source": result["source"],
            "paths": [{k: v for k, v in p.items() if k != "canonicalMv"} for p in paths.values()],
            "review": observation["waveforms"]["arms"][ARM]["canonicalReview"],
            "sourceSampling": observation["raySourcePaths"]["arms"][ARM],
            "canonicalValuesReference": "timeseries-canonical-uv.csv; short primary II in primary-II-uv.csv",
            "sourceGapsPreserved": True, "quantitativeUncertainty": "unavailable",
        })
        context = observation["sourceGrid"]
        _json(output_dir / "source-context.json", {
            "schemaVersion": 2, "source": result["source"],
            "calibrationAndIdentity": context["sourcePaths"]["calibration"],
            "geometry": {k: context["grid"][k] for k in ("sourceInterval", "field", "mesh", "extension")},
            "rayModel": observation["rays"]["model"], "geometryIsApproximate": True,
        })
        _overlay(image, paths, observation, output_dir / "source-overlay.png")
        _paper(paths, flags, output_dir / "ecg-paper.png")
    else:
        _json(output_dir / "unresolved-evidence.json", observation)
    files = {q.name: {"sha256": _sha(q.read_bytes()), "bytes": q.stat().st_size} for q in sorted(output_dir.iterdir())}
    candidate = {
        **contract, "candidateId": "source-photo-diagnostic-v1", "selectedArm": ARM if paths else None,
        "complete": True, "publicationOutcome": "diagnostic_only" if paths else "unresolved",
        "files": files, "units": "uV" if paths else None, "sampleRateHz": RATE if paths else None,
        "layout": "standard_3x4_with_r1" if paths else None,
        "canonicalLeadOrder": list(LEADS) if paths else [],
        "canonicalII": "rhythm_II" if paths else None,
        "primaryIIPreservedSeparately": bool(paths), "gapFillingApplied": False,
        "clinicalMeaning": "unclassified", "absoluteStLevelsEstablished": False,
        "overlaySourceOffsetPixels": [0, 44] if paths else None,
        "paperPixelsPerMm": 10 if paths else None,
        "branch": observation["branch"],
    }
    _json(output_dir / "candidate.json", candidate)
    return candidate
