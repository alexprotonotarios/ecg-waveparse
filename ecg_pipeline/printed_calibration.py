"""Read physical settings locally and reconcile them without inventing units.

Only matched setting tokens and original-image boxes leave the OCR boundary.
Names, dates and unrelated OCR strings are neither retained nor logged here.
Printed values corroborate a pulse/grid inference; they do not establish an
acquisition sample rate or resolve an ambiguous physical grid by themselves.
"""
from __future__ import annotations

import copy
import math
import re
import statistics
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
    """Read source tokens; complete omissions only with agreeing recognizers."""
    if ocr_runner is not None:
        return _read_printed_settings_once(image, ocr_runner=ocr_runner)
    from ecg_pipeline.lead_label_identity import read_local_text, _rapidocr_ocr_runner
    primary = _read_printed_settings_once(image, ocr_runner=read_local_text)
    if primary["state"] == "unresolved":
        return _complete_empty_primary(image, primary, primary_runner=read_local_text,
                                       secondary_runner=_rapidocr_ocr_runner)
    present = [kind for kind in ("speed", "gain") if len(primary["values"][kind]) == 1]
    if (primary["state"] != "recognized" or len(present) != 1
            or not primary.get("engine", "").startswith("apple-vision-")):
        return primary
    secondary = _read_printed_settings_once(image, ocr_runner=_rapidocr_ocr_runner)
    combined = {kind: sorted(set(primary["values"][kind]) | set(secondary["values"][kind]))
                for kind in ("speed", "gain")}
    conflict = any(len(values) > 1 for values in combined.values())
    agreed = (all(len(secondary["values"][kind]) == 1 for kind in ("speed", "gain"))
              and secondary["values"][present[0]] == primary["values"][present[0]])
    outcome = "conflict" if conflict else "completed" if agreed else "no_agreed_completion"
    result = {**primary, "recognizerFallback": {
        "version": 1, "method": "agreement-gated-missing-setting-v1", "outcome": outcome,
        "primary": primary, "secondary": secondary,
    }}
    if conflict or agreed:
        result.update(values=combined, state="conflict" if conflict else "recognized",
                      observations=primary["observations"] + secondary["observations"],
                      engine=primary.get("engine", "local-ocr") + "+" + secondary.get("engine", "local-ocr"))
    return result


def _setting_boxes_agree(left: dict[str, float], right: dict[str, float]) -> bool:
    area = min((left['right']-left['left'])*(left['bottom']-left['top']),
               (right['right']-right['left'])*(right['bottom']-right['top']))
    intersection = max(0., min(left['right'], right['right'])-max(left['left'], right['left'])) * max(
        0., min(left['bottom'], right['bottom'])-max(left['top'], right['top']))
    return area > 0 and intersection / area >= .5


def _complete_empty_primary(
    image: np.ndarray,
    primary: dict[str, Any],
    *,
    primary_runner: Callable[[np.ndarray], dict[str, Any]],
    secondary_runner: Callable[[np.ndarray], dict[str, Any]],
) -> dict[str, Any]:
    """Preserve old observations unless complete independent context proof exists."""
    if (primary['state'] != 'unresolved' or any(primary['values'].values())
            or primary.get('reason') != 'no-confident-explicit-setting-token'
            or not primary.get('engine', '').startswith('apple-vision-')
            or min(image.shape[:2]) < 2):
        return primary
    secondary = _read_printed_settings_once(image, ocr_runner=secondary_runner)
    fallback = {'version': 1, 'method': 'agreement-gated-source-quadrants-v1',
                'outcome': 'no_agreed_completion', 'minimumBoxOverlapFraction': .5,
                'primary': primary, 'secondary': secondary, 'contexts': []}
    result = {**primary, 'recognizerFallback': fallback}
    if secondary['state'] == 'conflict':
        fallback['outcome'] = 'conflict'
        result.update(state='conflict', values=secondary['values'], observations=secondary['observations'])
        result.pop('reason', None)
        return result
    if (secondary['state'] != 'recognized' or not secondary.get('engine', '').startswith('rapidocr-')
            or not all(len(secondary['values'][kind]) == 1 for kind in ('speed', 'gain'))):
        return result
    height, width = image.shape[:2]
    observations = list(secondary['observations'])
    # Observe every quadrant, including those after a successful match, to retain conflicts.
    for name, (left, top, right, bottom) in [
            ('top-left', (0, 0, width//2, height//2)),
            ('top-right', (width//2, 0, width, height//2)),
            ('bottom-left', (0, height//2, width//2, height)),
            ('bottom-right', (width//2, height//2, width, height))]:
        settings = copy.deepcopy(_read_printed_settings_once(
            image[top:bottom, left:right].copy(), ocr_runner=primary_runner))
        for item in settings['observations']:
            for key in ('left', 'right'):
                item['sourceBox'][key] += left
            for key in ('top', 'bottom'):
                item['sourceBox'][key] += top
        settings['sourceSize'] = primary['sourceSize']
        observations.extend(settings['observations'])
        fallback['contexts'].append({'id': name, 'bounds': [left, top, right, bottom], 'settings': settings})
    combined = {kind: sorted({item['value'] for item in observations if item['kind'] == kind})
                for kind in ('speed', 'gain')}
    conflict = any(len(values) > 1 for values in combined.values())
    valid = all(context['settings'].get('engine', '').startswith('apple-vision-') and
                context['settings'].get('reason') in (None, 'no-confident-explicit-setting-token')
                for context in fallback['contexts'])
    agreed = []
    for context in fallback['contexts']:
        settings = context['settings']
        if settings['state'] != 'recognized' or settings['values'] != secondary['values']:
            continue
        if all(any(a['kind'] == b['kind'] == kind and a['value'] == b['value'] and
                       _setting_boxes_agree(a['sourceBox'], b['sourceBox'])
                       for a in settings['observations'] for b in secondary['observations'])
               for kind in ('speed', 'gain')):
            agreed.append(context['id'])
    fallback.update(outcome='conflict' if conflict else 'completed' if valid and agreed else 'no_agreed_completion',
                    allContextsValid=valid, agreeingContexts=agreed)
    if conflict or (valid and agreed):
        result.update(state='conflict' if conflict else 'recognized', values=combined,
                      observations=observations, engine=primary['engine']+'+'+secondary['engine'])
        result.pop('reason', None)
    return result



def _read_printed_settings_once(
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


def printed_speed_resolution(calibration: dict[str, Any], printed: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve a short reference pulse only with separately repeated grid scale.

    A printed speed is observed; 200 ms is merely the legacy pulse convention.
    Keep this narrow path independent of waveform truth and never raise confidence.
    """
    values = printed.get("values") or {}
    gain = calibration.get("gainMmPerMv")
    rows = calibration.get("rowPixelsPerMmX")
    evidence = calibration.get("horizontalScaleEvidence") or {}
    required = ("confidence", "pixelsPerMmX", "pixelsPerMmY", "gridSpacingXPixels",
                "gridSpacingYPixels", "measuredGainMmPerMv", "pulseWidthMm")
    finite = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if (calibration.get("method") != "grid-period-plus-rectangular-pulse-v1"
        or calibration.get("detected") is not True
        or calibration.get("gridScaleDetected") is not True
        or calibration.get("gridScaleAmbiguous") is not False
        or calibration.get("gridScaleMmX") != 1 or calibration.get("gridScaleMmY") != 1
        or calibration.get("paperSpeedMmPerSecond") != 25
        or printed.get("state") != "recognized" or values.get("speed") != [50.0]
        or gain not in (5., 10., 20.) or values.get("gain") != [gain]
        or any(not finite(calibration.get(key)) for key in required)
        or not .35 <= calibration["confidence"] <= 1
        or not isinstance(rows, list) or len(rows) not in (6, 12)
        or any(not finite(value) or not 1 <= value <= 40 for value in rows)):
        return None
    center = float(statistics.median(rows))
    spread = (max(rows) - min(rows)) / center
    x, y = calibration["pixelsPerMmX"], calibration["pixelsPerMmY"]
    prior = evidence.get("priorPixelsPerMmX")
    if (evidence.get("version") != 1 or evidence.get("method") != "consistent-row-grid-median-v1"
        or evidence.get("rowCount") != len(rows) or not finite(evidence.get("relativeRange"))
        or abs(evidence["relativeRange"] - spread) > 1e-9 or spread > .02
        or not finite(prior) or prior <= 0 or abs(center-prior)/max(center, prior) > .15
        or abs(x-center) > 1e-9 or y <= 0
        or abs(x-y)/max(x, y) > .05
        or abs(calibration["gridSpacingXPixels"]-x)/x > .05
        or abs(calibration["gridSpacingYPixels"]-y)/y > .05
        or abs(calibration["measuredGainMmPerMv"]-gain)/gain > .1
        or not 4.5 <= calibration["pulseWidthMm"] <= 5.5):
        return None
    return {"version": 1, "method": "printed-speed-with-row-grid-v1",
            "previousInference": {"method": calibration["method"], "paperSpeedMmPerSecond": 25.0,
                                  "assumedPulseDurationSeconds": .2},
            "printedSpeedMmPerSecond": 50.0,
            "derivedPulseDurationSeconds": calibration["pulseWidthMm"] / 50.0}


def reconcile_printed_settings(calibration: dict[str, Any], printed: dict[str, Any]) -> dict[str, Any]:
    """Preserve pulse/grid alternatives and refuse conflicts across all routes."""
    result = {**calibration, "printedSettings": printed}
    resolution = printed_speed_resolution(calibration, printed)
    if resolution is not None:
        result.update(method=resolution["method"], paperSpeedMmPerSecond=50.0, speedResolution=resolution)
    values = printed.get("values") or {"speed": [], "gain": []}
    reasons: list[str] = []
    if printed.get("state") == "conflict":
        reasons.append("conflicting-printed-settings")
    supported = {"speed": {25.0, 50.0}, "gain": {5.0, 10.0, 20.0}}
    unsupported = any(value not in supported[kind] for kind in supported for value in values[kind])
    if unsupported:
        reasons.append("unsupported-printed-setting")
    compared: list[str] = []
    if result.get("detected") is True:
        for kind, key in (("speed", "paperSpeedMmPerSecond"), ("gain", "gainMmPerMv")):
            if len(values[kind]) == 1:
                compared.append(kind)
                if not math.isclose(values[kind][0], float(result.get(key, 0)), rel_tol=0.01, abs_tol=0):
                    reasons.append(f"printed-{kind}-disagrees-with-pulse-grid")
    state = "unsupported" if unsupported else "conflict" if reasons else (
        "corroborated_inference" if len(compared) == 2 else
        "partially_corroborated" if compared else "unresolved"
    )
    result["reconciliation"] = {
        "version": 2 if resolution is not None else 1, "state": state, "reasons": reasons,
        "compared": compared, "quantitativeBlocked": bool(reasons),
        "pulseAssumptions": {"durationSeconds": None if resolution is not None else .2, "amplitudeMv": 1.0},
        "acquisitionSampleRateHz": None,
    }
    return result
