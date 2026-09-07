"""Read physical settings locally and reconcile them without inventing units.

Only matched setting tokens and original-image boxes leave the OCR boundary.
Names, dates and unrelated OCR strings are neither retained nor logged here.
Printed values corroborate a pulse/grid inference; they do not establish an
acquisition sample rate or resolve an ambiguous physical grid by themselves.
"""
from __future__ import annotations

import math
import re
import subprocess
from typing import Any, Callable

import cv2
import numpy as np

CONFIDENCE_MINIMUM = 0.85
SETTING_PATTERN = re.compile(
    r"(?<![\w.,+-])(?P<value>[+-]?\d{1,3}(?:[.,]\d{1,3})?)\s*"
    r"mm\s*[/\u2044\u2215]\s*(?P<unit>mV|sec(?:ond)?s?|s)(?!\w)",
    re.IGNORECASE,
)


def decode_source_raster(source_bytes: bytes) -> np.ndarray:
    """Decode stored raster coordinates, matching Pillow admission/preprocessing.

    EXIF orientation is display metadata. Applying it here alone would rotate
    or mirror OCR boxes relative to the admitted original and its transforms.
    The original bytes (including EXIF) remain untouched and hash-bound.
    """
    image = cv2.imdecode(np.frombuffer(source_bytes, dtype=np.uint8),
                         cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("Could not decode the admitted calibration source.")
    return image


def read_printed_settings(
    image: np.ndarray,
    *,
    ocr_runner: Callable[[np.ndarray], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if image.ndim not in (2, 3) or min(image.shape[:2]) < 1:
        raise ValueError("Printed calibration requires a nonempty raster.")
    height, width = image.shape[:2]
    result: dict[str, Any] = {
        "version": 1,
        "state": "unresolved",
        "method": "local-setting-token-ocr-v1",
        "sourceSize": {"width": int(width), "height": int(height)},
        "coordinateSpace": "original",
        "confidenceMinimum": CONFIDENCE_MINIMUM,
        "observations": [],
        "values": {"speed": [], "gain": []},
    }
    if ocr_runner is None:
        from ecg_pipeline.lead_label_identity import read_local_text
        ocr_runner = read_local_text
    scale = min(1.0, 2400.0 / max(height, width))
    working = image if scale == 1 else cv2.resize(
        image, (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    try:
        output = ocr_runner(working)
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired):
        result["reason"] = "local-setting-recognizer-unavailable"
        return result
    if not isinstance(output, dict) or not isinstance(output.get("observations"), list):
        result["reason"] = "invalid-local-setting-recognizer-output"
        return result
    result["engine"] = str(output.get("engine") or "local-ocr")[:120]
    for observation in output["observations"]:
        if not isinstance(observation, dict):
            continue
        try:
            x, y, w, h = (float(observation[key]) for key in ("x", "y", "width", "height"))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not all(math.isfinite(v) for v in (x, y, w, h)) or not (
            0 <= x < 1 and 0 <= y < 1 and w > 0 and h > 0
            and x + w <= 1.000001 and y + h <= 1.000001
        ):
            continue
        for candidate in observation.get("candidates") or []:
            if not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str):
                continue
            confidence = candidate.get("confidence")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not (
                math.isfinite(confidence) and CONFIDENCE_MINIMUM <= confidence <= 1
            ):
                continue
            for match in SETTING_PATTERN.finditer(candidate["text"]):
                value = float(match["value"].replace(",", "."))
                kind = "gain" if match["unit"].casefold() == "mv" else "speed"
                result["observations"].append({
                    "kind": kind, "value": value,
                    "units": "mm/mV" if kind == "gain" else "mm/s",
                    "confidence": float(confidence),
                    "sourceBox": {"left": x * width, "top": max(0.0, (1 - y - h) * height),
                                  "right": min(float(width), (x + w) * width), "bottom": (1 - y) * height},
                })
    values = {kind: sorted({o["value"] for o in result["observations"] if o["kind"] == kind})
              for kind in ("speed", "gain")}
    result["values"] = values
    result["state"] = "conflict" if any(len(v) > 1 for v in values.values()) else (
        "recognized" if any(values.values()) else "unresolved"
    )
    if not any(values.values()):
        result["reason"] = "no-confident-explicit-setting-token"
    return result


def reconcile_printed_settings(calibration: dict[str, Any], printed: dict[str, Any]) -> dict[str, Any]:
    """Preserve pulse/grid alternatives and refuse conflicts across all routes."""
    result = {**calibration, "printedSettings": printed}
    values = printed.get("values") or {"speed": [], "gain": []}
    reasons: list[str] = []
    if printed.get("state") == "conflict":
        reasons.append("conflicting-printed-settings")
    supported = {"speed": {25.0, 50.0}, "gain": {5.0, 10.0, 20.0}}
    unsupported = any(value not in supported[kind] for kind in supported for value in values[kind])
    if unsupported:
        reasons.append("unsupported-printed-setting")
    compared: list[str] = []
    if calibration.get("detected") is True:
        for kind, key in (("speed", "paperSpeedMmPerSecond"), ("gain", "gainMmPerMv")):
            if len(values[kind]) == 1:
                compared.append(kind)
                if not math.isclose(values[kind][0], float(calibration.get(key, 0)), rel_tol=0.01, abs_tol=0):
                    reasons.append(f"printed-{kind}-disagrees-with-pulse-grid")
    state = "unsupported" if unsupported else "conflict" if reasons else (
        "corroborated_inference" if len(compared) == 2 else
        "partially_corroborated" if compared else "unresolved"
    )
    result["reconciliation"] = {
        "version": 1, "state": state, "reasons": reasons,
        "compared": compared, "quantitativeBlocked": bool(reasons),
        "pulseAssumptions": {"durationSeconds": 0.2, "amplitudeMv": 1.0},
        "acquisitionSampleRateHz": None,
    }
    return result
