from __future__ import annotations

import importlib.metadata
import itertools
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


STANDARD_LEADS = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)
STANDARD_THREE_BY_FOUR = (
    ("I", "aVR", "V1", "V4"),
    ("II", "aVL", "V2", "V5"),
    ("III", "aVF", "V3", "V6"),
)
SEMANTIC_METHOD = "source-pixel-glyphs-plus-value-aware-ocr-v2"
LABEL_OCR_ENGINE_ENV = "ECG_DIGITIZER_LABEL_OCR_ENGINE"
DIGIT_SHEET_CELL_SIZE = 128


@dataclass
class LabelSlot:
    expected: str
    row: int
    column: int
    source_box: tuple[int, int, int, int]
    mask: np.ndarray
    spacing: float
    gray: np.ndarray | None = None


@dataclass
class ContactSheet:
    image: np.ndarray
    cells: dict[str, tuple[int, int, int, int]]
    slots: dict[str, LabelSlot]


OcrRunner = Callable[[np.ndarray], dict[str, Any]]


def _connected_components(
    mask: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    binary = np.ascontiguousarray(np.asarray(mask, dtype=bool), dtype=np.uint8)
    if binary.ndim != 2 or binary.size == 0 or 0 in binary.shape:
        return (
            1,
            np.zeros(binary.shape, dtype=np.int32),
            np.zeros((1, 5), dtype=np.int32),
            np.zeros((1, 2), dtype=np.float64),
        )
    return cv2.connectedComponentsWithStatsWithAlgorithm(
        binary,
        8,
        cv2.CV_32S,
        cv2.CCL_SAUF,
    )


def _label_mask(image: np.ndarray, threshold: float = 160.0) -> np.ndarray:
    if image.ndim == 2:
        return (image < threshold).astype(np.uint8)
    gray = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)
    raw = (gray < threshold).astype(np.uint8)
    channels = image[..., :3].astype(np.int16)
    channel_max = channels.max(axis=2)
    channel_min = channels.min(axis=2)
    chroma = channel_max - channel_min
    chromatic_fraction = float(
        np.mean((chroma >= 30) & (channel_max >= 100))
    )
    if chromatic_fraction < 0.002:
        return raw
    neutral = ((channel_max < threshold) & (chroma <= 28)).astype(np.uint8)
    return neutral if np.count_nonzero(neutral) >= 8 else raw


def _bounded_box(
    width: int,
    height: int,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> tuple[int, int, int, int] | None:
    resolved = (
        max(0, min(width, round(left))),
        max(0, min(height, round(top))),
        max(0, min(width, round(right))),
        max(0, min(height, round(bottom))),
    )
    if resolved[2] - resolved[0] < 4 or resolved[3] - resolved[1] < 4:
        return None
    return resolved


def _slot(
    image: np.ndarray,
    expected: str,
    row: int,
    column: int,
    box: tuple[int, int, int, int] | None,
    spacing: float,
) -> LabelSlot | None:
    if box is None:
        return None
    left, top, right, bottom = box
    crop = image[top:bottom, left:right]
    gray = (
        crop.copy()
        if crop.ndim == 2
        else cv2.cvtColor(crop[..., :3], cv2.COLOR_BGR2GRAY)
    )
    return LabelSlot(
        expected=expected,
        row=row,
        column=column,
        source_box=box,
        mask=_label_mask(crop),
        spacing=spacing,
        gray=gray,
    )


def _layout_slots(
    image: np.ndarray,
    geometry: dict[str, Any],
) -> list[LabelSlot]:
    height, width = image.shape[:2]
    layout = str(geometry.get("layoutHint") or "")
    centers = [int(value) for value in geometry.get("rowCenters") or []]
    slots: list[LabelSlot | None] = []

    if layout in {"standard_6x2", "standard_6x2_with_r1_ignored"}:
        if len(centers) not in {6, 7}:
            return []
        centers = centers[:6]
        spacing = float(np.median(np.diff(centers)))
        for row, center in enumerate(centers):
            top = center - spacing * 0.62
            bottom = center - spacing * 0.18
            slots.append(
                _slot(
                    image,
                    STANDARD_LEADS[row],
                    row,
                    0,
                    _bounded_box(
                        width,
                        height,
                        width * 0.01,
                        top,
                        width * 0.18,
                        bottom,
                    ),
                    spacing,
                )
            )
            slots.append(
                _slot(
                    image,
                    STANDARD_LEADS[row + 6],
                    row,
                    1,
                    _bounded_box(
                        width,
                        height,
                        width * 0.35,
                        top,
                        width * 0.62,
                        bottom,
                    ),
                    spacing,
                )
            )
    elif layout in {"standard_3x4", "standard_3x4_with_r1"}:
        if len(centers) not in {3, 4}:
            return []
        centers = centers[:3]
        spacing = float(np.median(np.diff(centers)))
        validation = geometry.get("leadLabelValidation") or {}
        fractions = (
            (0.030, 0.260, 0.495, 0.735)
            if validation.get("anchorHypothesis") == "panel-boundary"
            else (0.055, 0.283, 0.511, 0.739)
        )
        for row, center in enumerate(centers):
            expected_y = center - spacing * 0.33
            for column, expected in enumerate(STANDARD_THREE_BY_FOUR[row]):
                expected_x = width * fractions[column]
                slots.append(
                    _slot(
                        image,
                        expected,
                        row,
                        column,
                        _bounded_box(
                            width,
                            height,
                            expected_x - width * 0.015,
                            expected_y - spacing * 0.10,
                            expected_x + width * 0.045,
                            expected_y + spacing * 0.10,
                        ),
                        spacing,
                    )
                )
    elif layout == "standard_12x1":
        if len(centers) != 12:
            return []
        spacing = float(np.median(np.diff(centers)))
        label_anchored = bool(
            geometry.get("method")
            == "standard-and-precordial-lead-label-anchors-v1"
            and len(geometry.get("traceRowCenters") or []) == 12
        )
        for index, (expected, center) in enumerate(
            zip(STANDARD_LEADS, centers, strict=True)
        ):
            # Label-anchored 12 x 1 geometry reports the printed-label rows in
            # ``rowCenters`` and the waveform baselines separately in
            # ``traceRowCenters``.  The fallback row detector reports only
            # waveform baselines.  Treating both coordinate spaces alike
            # cropped around the trace and missed visibly clear labels on
            # low-resolution/faded pages.  The fallback band mirrors the
            # renderer-independent 6 x 2 label band above each baseline.
            top = (
                center - spacing * 0.25
                if label_anchored
                else center - spacing * 0.68
            )
            bottom = (
                center + spacing * 0.25
                if label_anchored
                else center - spacing * 0.18
            )
            slots.append(
                _slot(
                    image,
                    expected,
                    index % 6,
                    index // 6,
                    _bounded_box(
                        width,
                        height,
                        width * 0.015,
                        top,
                        width * 0.13,
                        bottom,
                    ),
                    spacing,
                )
            )
    return [slot for slot in slots if slot is not None]


def _remove_long_lines(mask: np.ndarray, spacing: float) -> np.ndarray:
    vertical = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(9, round(spacing * 0.24))),
        ),
    )
    horizontal = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(13, round(spacing * 0.40)), 1),
        ),
    )
    vertical = cv2.dilate(
        vertical,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    horizontal = cv2.dilate(
        horizontal,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    cleaned = mask.copy()
    cleaned[(vertical > 0) | (horizontal > 0)] = 0
    return cleaned


def _remove_roman_overlapping_lines(
    mask: np.ndarray,
    spacing: float,
) -> np.ndarray:
    # A QRS clipped by the small Roman-label crop may be shorter than the
    # general line-removal kernel. Cap this Roman-only kernel to the crop and
    # avoid horizontal dilation, which would erase adjacent printed I strokes.
    vertical_length = min(
        max(9, round(spacing * 0.24)),
        max(9, round(mask.shape[0] * 0.70)),
    )
    vertical = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_length)),
    )
    vertical = cv2.dilate(
        vertical,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)),
    )
    cleaned = mask.copy()
    cleaned[vertical > 0] = 0
    return cleaned


def _contact_sheet(slots: list[LabelSlot]) -> ContactSheet:
    rendered: list[tuple[LabelSlot, np.ndarray]] = []
    for slot in slots:
        mask = _remove_long_lines(slot.mask, slot.spacing)
        glyphs = np.where(mask > 0, 0, 255).astype(np.uint8)
        glyphs = cv2.resize(
            glyphs,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_NEAREST,
        )
        rendered.append((slot, glyphs))
    cell_height = max(item.shape[0] for _, item in rendered) + 40
    cell_width = max(item.shape[1] for _, item in rendered) + 40
    row_count = max(slot.row for slot, _ in rendered) + 1
    column_count = max(slot.column for slot, _ in rendered) + 1
    sheet = np.full(
        (row_count * cell_height, column_count * cell_width),
        255,
        dtype=np.uint8,
    )
    cells: dict[str, tuple[int, int, int, int]] = {}
    slot_map: dict[str, LabelSlot] = {}
    for slot, glyphs in rendered:
        cell_left = slot.column * cell_width
        cell_top = slot.row * cell_height
        left = cell_left + 20
        top = cell_top + 20
        sheet[top : top + glyphs.shape[0], left : left + glyphs.shape[1]] = (
            glyphs
        )
        cells[slot.expected] = (
            cell_left,
            cell_top,
            cell_left + cell_width,
            cell_top + cell_height,
        )
        slot_map[slot.expected] = slot
    return ContactSheet(image=sheet, cells=cells, slots=slot_map)


def _apple_vision_ocr_runner(sheet: np.ndarray, *, accurate: bool = False) -> dict[str, Any]:
    if sys.platform != "darwin":
        raise RuntimeError("apple-vision-unavailable")
    osascript = shutil.which("osascript")
    helper = Path(__file__).with_name("recognize_lead_labels.js")
    if not osascript or not helper.is_file():
        raise RuntimeError("apple-vision-unavailable")
    with tempfile.TemporaryDirectory(prefix="ecg-lead-labels-") as temp_dir:
        sheet_path = Path(temp_dir) / "source-labels.png"
        if not cv2.imwrite(str(sheet_path), sheet):
            raise RuntimeError("label-contact-sheet-write-failed")
        completed = subprocess.run(
            [
                osascript,
                "-s",
                "o",
                "-l",
                "JavaScript",
                str(helper),
                str(sheet_path),
                *(["--accurate"] if accurate else []),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise RuntimeError("apple-vision-recognition-failed")
    output = json.loads(completed.stdout)
    if not isinstance(output, dict) or not isinstance(
        output.get("observations"), list
    ):
        raise RuntimeError("apple-vision-invalid-response")
    return output


@lru_cache(maxsize=1)
def _rapidocr_engine() -> Any:
    # Official ONNX Runtime wheels enable cross-platform telemetry by default.
    # Disable it before RapidOCR imports ONNX Runtime so local ECG processing
    # neither uploads runtime events nor creates a persistent device/session ID.
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    try:
        from rapidocr import RapidOCR
    except (ImportError, OSError) as error:
        raise RuntimeError("rapidocr-unavailable") from error
    try:
        return RapidOCR()
    except Exception as error:
        raise RuntimeError("rapidocr-initialization-failed") from error


def _rapidocr_output_to_report(
    output: Any,
    *,
    image_shape: tuple[int, ...],
) -> dict[str, Any]:
    if len(image_shape) < 2 or image_shape[0] <= 0 or image_shape[1] <= 0:
        raise RuntimeError("rapidocr-invalid-image-shape")
    height, width = image_shape[:2]
    boxes = getattr(output, "boxes", None)
    texts = tuple(getattr(output, "txts", ()) or ())
    scores = tuple(getattr(output, "scores", ()) or ())
    if boxes is None:
        box_values: tuple[Any, ...] = ()
    else:
        box_values = tuple(np.asarray(boxes))
    if len(box_values) != len(texts) or len(texts) != len(scores):
        raise RuntimeError("rapidocr-invalid-response")

    observations: list[dict[str, Any]] = []
    for box, value, confidence in zip(
        box_values,
        texts,
        scores,
        strict=True,
    ):
        try:
            points = np.asarray(box, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeError("rapidocr-invalid-box") from error
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(
            np.isfinite(points)
        ):
            raise RuntimeError("rapidocr-invalid-box")
        try:
            score = float(confidence)
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeError("rapidocr-invalid-confidence") from error
        if not np.isfinite(score):
            raise RuntimeError("rapidocr-invalid-confidence")
        left = float(np.clip(np.min(points[:, 0]), 0.0, width))
        right = float(np.clip(np.max(points[:, 0]), 0.0, width))
        top = float(np.clip(np.min(points[:, 1]), 0.0, height))
        bottom = float(np.clip(np.max(points[:, 1]), 0.0, height))
        if right <= left or bottom <= top:
            continue
        observations.append(
            {
                "x": left / width,
                "y": 1.0 - bottom / height,
                "width": (right - left) / width,
                "height": (bottom - top) / height,
                "candidates": [
                    {
                        "text": str(value),
                        "confidence": float(np.clip(score, 0.0, 1.0)),
                    }
                ],
            }
        )
    try:
        rapidocr_version = importlib.metadata.version("rapidocr")
    except importlib.metadata.PackageNotFoundError:
        rapidocr_version = "unknown"
    return {
        "engine": f"rapidocr-{rapidocr_version}-ppocrv6-onnxruntime",
        "revision": 6,
        "observations": observations,
    }


def _rapidocr_ocr_runner(sheet: np.ndarray) -> dict[str, Any]:
    try:
        engine = _rapidocr_engine()
        output = engine(
            sheet,
            use_det=True,
            use_cls=True,
            use_rec=True,
        )
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError("rapidocr-recognition-failed") from error
    report = _rapidocr_output_to_report(output, image_shape=sheet.shape)

    # Direct recognition is intentionally limited to the six-cell digit strip.
    # Applying it to a contact sheet could collapse several independent labels
    # into one observation and attach that value to the wrong source slot.
    height, width = sheet.shape[:2]
    is_digit_strip = (height, width) == (
        DIGIT_SHEET_CELL_SIZE,
        DIGIT_SHEET_CELL_SIZE * 6,
    )
    if not is_digit_strip:
        return report
    try:
        direct_output = engine(
            sheet,
            use_det=False,
            use_cls=False,
            use_rec=True,
        )
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError("rapidocr-recognition-failed") from error
    direct_texts = tuple(getattr(direct_output, "txts", ()) or ())
    direct_scores = tuple(getattr(direct_output, "scores", ()) or ())
    if len(direct_texts) != 1 or len(direct_scores) != 1:
        return report
    direct_text = str(direct_texts[0])
    try:
        direct_confidence = float(direct_scores[0])
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("rapidocr-invalid-confidence") from error
    exact_digits = (
        re.sub(r"\s+", "", direct_text)
        if direct_text.isascii()
        else ""
    )
    if not (
        np.isfinite(direct_confidence)
        and direct_confidence >= 0.5
        and exact_digits == "123456"
    ):
        return report
    return {
        **report,
        "observations": [
            {
                "x": 0.0,
                "y": 0.0,
                "width": 1.0,
                "height": 1.0,
                "candidates": [
                    {
                        "text": direct_text,
                        "confidence": float(
                            np.clip(direct_confidence, 0.0, 1.0)
                        ),
                    }
                ],
            }
        ],
    }


def _default_ocr_runner(sheet: np.ndarray, *, accurate: bool = False) -> dict[str, Any]:
    requested = os.environ.get(LABEL_OCR_ENGINE_ENV, "auto").strip().casefold()
    if requested == "auto":
        requested = "apple-vision" if sys.platform == "darwin" else "rapidocr"
    if requested == "apple-vision":
        return _apple_vision_ocr_runner(sheet, accurate=accurate)
    if requested == "rapidocr":
        return _rapidocr_ocr_runner(sheet)
    raise RuntimeError("unsupported-label-ocr-engine")


def read_local_text(image: np.ndarray) -> dict[str, Any]:
    """Use the same explicitly provisioned local recognizer for source evidence."""
    # Fast Apple Vision observations have coarse confidence (often 0.5).
    # Printed physical settings need its accurate recognizer; lead-label
    # contact sheets retain their existing recognizer and decision semantics.
    return _default_ocr_runner(image, accurate=True)


def _top_observations(output: dict[str, Any]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for item in output.get("observations") or []:
        candidates = item.get("candidates") or []
        if not candidates or not isinstance(candidates[0].get("text"), str):
            continue
        observations.append(item)
    return observations


def _tokens_by_slot(
    sheet: ContactSheet,
    output: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    height, width = sheet.image.shape
    mapped = {expected: [] for expected in sheet.cells}
    for observation in _top_observations(output):
        center_x = (
            float(observation.get("x", 0.0))
            + float(observation.get("width", 0.0)) / 2.0
        ) * width
        center_y = (
            1.0
            - float(observation.get("y", 0.0))
            - float(observation.get("height", 0.0)) / 2.0
        ) * height
        for expected, (left, top, right, bottom) in sheet.cells.items():
            if left <= center_x < right and top <= center_y < bottom:
                mapped[expected].append(observation)
                break
    return mapped


def _merge_mapped_tokens(
    primary: dict[str, list[dict[str, Any]]],
    secondary: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        expected: [
            *primary.get(expected, []),
            *secondary.get(expected, []),
        ]
        for expected in primary.keys() | secondary.keys()
    }


def _normalise_token(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _slot_has_exact_token(
    mapped: dict[str, list[dict[str, Any]]],
    expected: str,
) -> bool:
    normalised = expected.casefold()
    return any(
        _normalise_token(str(observation["candidates"][0]["text"]))
        == normalised
        and float(observation["candidates"][0].get("confidence", 0.0))
        >= 0.5
        for observation in mapped.get(expected, [])
    )


def _output_has_exact_token(
    output: dict[str, Any],
    expected: str,
) -> bool:
    normalised = expected.casefold()
    return any(
        _normalise_token(str(observation["candidates"][0]["text"]))
        == normalised
        and float(observation["candidates"][0].get("confidence", 0.0))
        >= 0.5
        for observation in _top_observations(output)
    )


def _isolated_label_sheet(slot: LabelSlot) -> np.ndarray:
    mask = _remove_long_lines(slot.mask, slot.spacing)
    glyphs = np.where(mask > 0, 0, 255).astype(np.uint8)
    glyphs = cv2.resize(
        glyphs,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC,
    )
    return cv2.copyMakeBorder(
        glyphs,
        40,
        40,
        40,
        40,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def _tight_isolated_label_sheet(
    slot: LabelSlot,
    expected: str,
) -> np.ndarray:
    mask = _remove_roman_overlapping_lines(slot.mask, slot.spacing)
    count, _, stats, centroids = _connected_components(mask)
    minimum_width = max(2, round(slot.spacing * 0.02))
    minimum_height = max(4, round(slot.spacing * 0.04))
    components = sorted(
        (
            index
            for index in range(1, count)
            if stats[index, cv2.CC_STAT_WIDTH] >= minimum_width
            and stats[index, cv2.CC_STAT_HEIGHT] >= minimum_height
        ),
        key=lambda index: int(stats[index, cv2.CC_STAT_LEFT]),
    )
    if not components:
        raise RuntimeError("tight-label-components-unavailable")
    groups: list[list[int]] = []
    maximum_gap = slot.spacing * 0.10
    for index in components:
        left = int(stats[index, cv2.CC_STAT_LEFT])
        if groups:
            previous = groups[-1][-1]
            previous_right = int(
                stats[previous, cv2.CC_STAT_LEFT]
                + stats[previous, cv2.CC_STAT_WIDTH]
            )
        if not groups or left - previous_right > maximum_gap:
            groups.append([index])
        else:
            groups[-1].append(index)
    expected_component_count = len(expected)
    compact_candidates: list[tuple[float, list[int]]] = []
    for group in groups:
        if len(group) < expected_component_count:
            continue
        for start in range(len(group) - expected_component_count + 1):
            candidate = group[start : start + expected_component_count]
            heights = np.asarray(
                [stats[index, cv2.CC_STAT_HEIGHT] for index in candidate],
                dtype=np.float64,
            )
            centers_y = np.asarray(
                [centroids[index][1] for index in candidate],
                dtype=np.float64,
            )
            left = min(
                int(stats[index, cv2.CC_STAT_LEFT]) for index in candidate
            )
            right = max(
                int(
                    stats[index, cv2.CC_STAT_LEFT]
                    + stats[index, cv2.CC_STAT_WIDTH]
                )
                for index in candidate
            )
            if (
                float(np.min(heights) / max(np.max(heights), 1.0)) < 0.65
                or float(np.ptp(centers_y)) > slot.spacing * 0.08
                or right - left > slot.spacing * 0.35
            ):
                continue
            ink_area = sum(
                int(stats[index, cv2.CC_STAT_AREA]) for index in candidate
            )
            score = (
                float(np.min(heights) / max(np.max(heights), 1.0))
                - float(np.ptp(centers_y)) / max(slot.spacing, 1.0)
                + ink_area / max((right - left) * float(np.max(heights)), 1.0)
            )
            compact_candidates.append((score, candidate))
    selected = (
        max(compact_candidates, key=lambda item: item[0])[1]
        if compact_candidates
        else max(
            groups,
            key=lambda group: sum(
                int(stats[index, cv2.CC_STAT_AREA]) for index in group
            ),
        )
    )
    left = min(int(stats[index, cv2.CC_STAT_LEFT]) for index in selected)
    top = min(int(stats[index, cv2.CC_STAT_TOP]) for index in selected)
    right = max(
        int(
            stats[index, cv2.CC_STAT_LEFT]
            + stats[index, cv2.CC_STAT_WIDTH]
        )
        for index in selected
    )
    bottom = max(
        int(
            stats[index, cv2.CC_STAT_TOP]
            + stats[index, cv2.CC_STAT_HEIGHT]
        )
        for index in selected
    )
    crop = mask[top:bottom, left:right]
    glyphs = cv2.resize(
        np.where(crop > 0, 0, 255).astype(np.uint8),
        None,
        fx=8,
        fy=8,
        interpolation=cv2.INTER_CUBIC,
    )
    return cv2.copyMakeBorder(
        glyphs,
        40,
        40,
        40,
        40,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def _tight_slot_has_exact_token(
    slot: LabelSlot,
    expected: str,
    ocr_runner: OcrRunner,
) -> bool:
    try:
        output = ocr_runner(_tight_isolated_label_sheet(slot, expected))
    except (
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
    ):
        return False
    return _output_has_exact_token(output, expected)


def _normalised_component(mask: np.ndarray) -> np.ndarray:
    y_values, x_values = np.nonzero(mask)
    canvas = np.zeros((32, 32), dtype=np.uint8)
    if y_values.size == 0:
        return canvas
    crop = mask[
        int(y_values.min()) : int(y_values.max()) + 1,
        int(x_values.min()) : int(x_values.max()) + 1,
    ]
    scale = min(26.0 / max(crop.shape[1], 1), 26.0 / max(crop.shape[0], 1))
    resized = cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )
    top = (32 - resized.shape[0]) // 2
    left = (32 - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = (
        resized
    )
    return canvas


def _dice(first: np.ndarray, second: np.ndarray) -> float:
    denominator = int(np.count_nonzero(first)) + int(np.count_nonzero(second))
    if denominator == 0:
        return 0.0
    return float(
        2 * np.count_nonzero((first > 0) & (second > 0)) / denominator
    )


def _vertical_occupancy_dice(first: np.ndarray, second: np.ndarray) -> float:
    first_rows = np.any(first > 0, axis=1)
    second_rows = np.any(second > 0, axis=1)
    denominator = int(np.count_nonzero(first_rows)) + int(
        np.count_nonzero(second_rows)
    )
    if denominator == 0:
        return 0.0
    return float(
        2
        * np.count_nonzero(first_rows & second_rows)
        / denominator
    )


def _roman_groups(
    slot: LabelSlot,
    expected_count: int,
) -> list[dict[str, Any]]:
    # Roman labels are often printed over the first QRS. Remove only strokes
    # that span most of this bounded label crop before counting I glyphs; the
    # crop-height cap in _remove_long_lines preserves the shorter printed I's.
    component_mask = _remove_roman_overlapping_lines(
        slot.mask,
        slot.spacing,
    )
    count, labels, stats, centroids = _connected_components(component_mask)
    components: list[dict[str, Any]] = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        aspect = height / max(width, 1)
        if not (
            max(4.0, slot.spacing * 0.035)
            <= height
            <= slot.spacing * 0.17
            and width <= slot.spacing * 0.12
            # A two- or three-stem Roman value can be wider than tall before
            # its antialias bridge is split below. Each resulting glyph must
            # still satisfy the stricter 1.25 vertical aspect ratio.
            and aspect >= 0.70
            and area >= 4
        ):
            continue
        component = (labels[y : y + height, x : x + width] == index).astype(
            np.uint8
        )
        # Antialiasing can join adjacent printed I glyphs through a one-pixel
        # serif bridge.  Preserve the value proof by splitting only components
        # with two or three independently tall vertical cores; the subsequent
        # I/II/III sequence, glyph-similarity, alignment, and ambiguity gates
        # still have to agree across all three source rows.
        column_support = np.count_nonzero(component, axis=0)
        tall_columns = column_support >= max(3, int(np.ceil(height * 0.45)))
        padded = np.pad(tall_columns, (1, 1), constant_values=False)
        transitions = np.diff(padded.astype(np.int8))
        core_starts = np.flatnonzero(transitions == 1)
        core_ends = np.flatnonzero(transitions == -1)
        cores = [
            (int(start), int(end))
            for start, end in zip(core_starts, core_ends, strict=True)
            if end - start <= slot.spacing * 0.06
        ]
        if not 2 <= len(cores) <= 3 and height >= 6:
            # Blur can fill the one-pixel valley between adjacent I stems over
            # most rows even though the independently visible top and bottom
            # endpoints remain separated. Split only when both terminal rows
            # contain the same two/three bounded source-ink cores.
            endpoint_columns = (component[0] > 0) & (component[-1] > 0)
            endpoint_padded = np.pad(
                endpoint_columns,
                (1, 1),
                constant_values=False,
            )
            endpoint_transitions = np.diff(endpoint_padded.astype(np.int8))
            endpoint_starts = np.flatnonzero(endpoint_transitions == 1)
            endpoint_ends = np.flatnonzero(endpoint_transitions == -1)
            endpoint_cores = [
                (int(start), int(end))
                for start, end in zip(
                    endpoint_starts,
                    endpoint_ends,
                    strict=True,
                )
                if end - start <= slot.spacing * 0.06
            ]
            if 2 <= len(endpoint_cores) <= 3:
                cores = endpoint_cores
        partitions: list[tuple[int, int]]
        if 2 <= len(cores) <= 3:
            boundaries = [0]
            for left_core, right_core in zip(cores, cores[1:]):
                boundaries.append(round((left_core[1] + right_core[0]) / 2))
            boundaries.append(int(width))
            partitions = list(zip(boundaries, boundaries[1:]))
        else:
            partitions = [(0, int(width))]

        for partition_left, partition_right in partitions:
            partition = component[:, partition_left:partition_right]
            y_values, x_values = np.nonzero(partition)
            if y_values.size == 0:
                continue
            local_left = int(x_values.min())
            local_right = int(x_values.max()) + 1
            local_top = int(y_values.min())
            local_bottom = int(y_values.max()) + 1
            glyph_mask = partition[
                local_top:local_bottom,
                local_left:local_right,
            ]
            glyph_width = local_right - local_left
            glyph_height = local_bottom - local_top
            glyph_area = int(np.count_nonzero(glyph_mask))
            glyph_aspect = glyph_height / max(glyph_width, 1)
            if not (
                max(4.0, slot.spacing * 0.035)
                <= glyph_height
                <= slot.spacing * 0.17
                and glyph_width <= slot.spacing * 0.12
                and glyph_aspect >= 1.25
                and glyph_area >= 4
            ):
                continue
            components.append(
                {
                    "index": index,
                    "x": int(x + partition_left + local_left),
                    "y": int(y + local_top),
                    "width": glyph_width,
                    "height": glyph_height,
                    "area": glyph_area,
                    "centerX": float(
                        x + partition_left + np.mean(x_values)
                    ),
                    "centerY": float(y + np.mean(y_values)),
                    "glyph": _normalised_component(glyph_mask),
                }
            )
    groups: list[dict[str, Any]] = []
    for combination in itertools.combinations(components, expected_count):
        ordered = sorted(combination, key=lambda item: item["centerX"])
        y_spread = float(
            np.ptp([item["centerY"] for item in ordered])
        )
        if y_spread > slot.spacing * 0.06:
            continue
        # Stems belonging to one printed Roman value share a vertical text
        # band. Page-row spacing is much larger than glyph height, so the
        # broad center-distance guard alone also admits short, vertically
        # displaced QRS fragments. Require common support over most of the
        # tallest glyph, or aligned upper endpoints when lower stems have
        # been shortened by overlap cleanup. This retains cropped/occluded
        # letters without admitting fragments from a different vertical band.
        common_top = max(item["y"] for item in ordered)
        common_bottom = min(item["y"] + item["height"] for item in ordered)
        tallest_height = max(item["height"] for item in ordered)
        top_spread = common_top - min(item["y"] for item in ordered)
        if (
            common_bottom - common_top < tallest_height * 0.60
            and top_spread > tallest_height * 0.25
        ):
            continue
        gaps = np.diff([item["centerX"] for item in ordered])
        if gaps.size and (
            float(np.min(gaps)) < slot.spacing * 0.015
            or float(np.max(gaps)) > slot.spacing * 0.12
        ):
            continue
        if (
            gaps.size > 1
            and float(np.std(gaps) / max(np.mean(gaps), 1e-9)) > 0.35
        ):
            continue
        right = max(item["x"] + item["width"] for item in ordered)
        left = min(item["x"] for item in ordered)
        if right - left > slot.spacing * 0.35:
            continue
        groups.append(
            {
                "components": ordered,
                "left": float(left),
                "right": float(right),
                "centerY": float(
                    np.mean([item["centerY"] for item in ordered])
                ),
            }
        )
    return groups


def _occluded_roman_ii_recovery(
    slots: dict[str, LabelSlot],
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Confirm II when one I stem is visible and the other is under a QRS.

    Low-resolution pages can place the first waveform excursion directly over
    the second printed I. Require intact I and III source sequences, exactly
    one visible II stem, glyph agreement across all visible stems, and a long
    vertical source stroke exactly where III predicts the hidden second stem.
    A visible conflicting II/III group therefore never enters this fallback.
    """
    if groups["II"] or not groups["I"] or not groups["III"]:
        return None
    partial_groups = _roman_groups(slots["II"], 1)
    if len(partial_groups) != 1:
        return None
    partial = partial_groups[0]
    partial_component = partial["components"][0]
    slot = slots["II"]
    slot_height = slot.mask.shape[0]
    vertical_length = min(
        max(9, round(slot.spacing * 0.24)),
        max(9, round(slot_height * 0.70)),
    )
    vertical = cv2.morphologyEx(
        slot.mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_length)),
    )
    scored: list[dict[str, Any]] = []
    for first, third in itertools.product(groups["I"], groups["III"]):
        template = first["components"][0]["glyph"]
        comparison_components = [
            partial_component,
            *third["components"],
        ]
        pixel_similarities = [
            _dice(template, component["glyph"])
            for component in comparison_components
        ]
        stroke_similarities = [
            _vertical_occupancy_dice(template, component["glyph"])
            for component in comparison_components
        ]
        similarities = [
            max(pixel, stroke * 0.75)
            for pixel, stroke in zip(
                pixel_similarities,
                stroke_similarities,
                strict=True,
            )
        ]
        minimum_similarity = min(similarities, default=0.0)
        heights = [
            first["components"][0]["height"],
            partial_component["height"],
            *[
                component["height"]
                for component in third["components"]
            ],
        ]
        height_cv = float(
            np.std(heights) / max(np.mean(heights), 1e-9)
        )
        left_spread = float(
            np.ptp([first["left"], partial["left"], third["left"]])
        )
        third_centers = [
            component["centerX"] for component in third["components"]
        ]
        stem_gap = float(np.median(np.diff(third_centers)))
        predicted_x = float(partial_component["centerX"] + stem_gap)
        band_radius = max(1, round(slot.spacing * 0.025))
        x0 = max(0, round(predicted_x) - band_radius)
        x1 = min(slot.mask.shape[1], round(predicted_x) + band_radius + 1)
        line_span_fraction = float(
            np.max(
                np.count_nonzero(vertical[:, x0:x1], axis=0),
                initial=0,
            )
            / max(slot_height, 1)
        )
        if (
            minimum_similarity < 0.60
            or height_cv > 0.25
            or left_spread > slot.spacing * 0.12
            or not slot.spacing * 0.015 <= stem_gap <= slot.spacing * 0.12
            or line_span_fraction < 0.70
        ):
            continue
        score = (
            float(np.mean(similarities))
            - height_cv * 0.25
            - left_spread / max(slot.spacing, 1.0) * 0.5
            + min(line_span_fraction, 1.0) * 0.05
        )
        scored.append(
            {
                "score": score,
                "minimumGlyphSimilarity": minimum_similarity,
                "minimumPixelSimilarity": min(
                    pixel_similarities,
                    default=0.0,
                ),
                "minimumStrokeSimilarity": min(
                    stroke_similarities,
                    default=0.0,
                ),
                "alignmentSpread": left_spread,
                "heightCv": height_cv,
                "predictedSecondStemX": predicted_x,
                "occludingLineSpanFraction": line_span_fraction,
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    if not scored:
        return None
    margin = scored[0]["score"] - scored[1]["score"] if len(scored) > 1 else 1.0
    if scored[0]["score"] < 0.62 or margin < 0.04:
        return None
    return {**scored[0], "ambiguityMargin": float(margin)}


def _verify_roman_sequence(
    slots: dict[str, LabelSlot],
) -> dict[str, Any]:
    groups = {
        "I": _roman_groups(slots["I"], 1),
        "II": _roman_groups(slots["II"], 2),
        "III": _roman_groups(slots["III"], 3),
    }
    scored: list[dict[str, Any]] = []
    spacing = min(slots[label].spacing for label in ("I", "II", "III"))
    for first, second, third in itertools.product(
        groups["I"], groups["II"], groups["III"]
    ):
        template = first["components"][0]["glyph"]
        pixel_similarities = [
            _dice(template, component["glyph"])
            for group in (second, third)
            for component in group["components"]
        ]
        stroke_similarities = [
            _vertical_occupancy_dice(template, component["glyph"])
            for group in (second, third)
            for component in group["components"]
        ]
        similarities = [
            max(pixel, stroke * 0.75)
            for pixel, stroke in zip(
                pixel_similarities,
                stroke_similarities,
                strict=True,
            )
        ]
        minimum_similarity = min(similarities, default=0.0)
        left_spread = float(
            np.ptp([first["left"], second["left"], third["left"]])
        )
        right_spread = float(
            np.ptp([first["right"], second["right"], third["right"]])
        )
        left_steps = (
            second["left"] - first["left"],
            third["left"] - second["left"],
        )
        right_steps = (
            second["right"] - first["right"],
            third["right"] - second["right"],
        )
        linear_alignments = [
            ("left-linear-shear", abs(left_steps[1] - left_steps[0]), left_steps),
            (
                "right-linear-shear",
                abs(right_steps[1] - right_steps[0]),
                right_steps,
            ),
        ]
        valid_linear_alignments = [
            item
            for item in linear_alignments
            if item[1] <= spacing * 0.03
            and max(abs(step) for step in item[2]) <= spacing * 0.20
        ]
        if min(left_spread, right_spread) <= spacing * 0.12:
            alignment_edge = "left" if left_spread <= right_spread else "right"
            alignment_spread = min(left_spread, right_spread)
            alignment_shear_per_row = 0.0
        elif valid_linear_alignments:
            alignment_edge, alignment_spread, selected_steps = min(
                valid_linear_alignments,
                key=lambda item: item[1],
            )
            alignment_shear_per_row = float(np.mean(selected_steps))
        else:
            alignment_edge = "left" if left_spread <= right_spread else "right"
            alignment_spread = min(left_spread, right_spread)
            alignment_shear_per_row = 0.0
        height_values = [
            component["height"]
            for group in (first, second, third)
            for component in group["components"]
        ]
        height_cv = float(
            np.std(height_values) / max(np.mean(height_values), 1e-9)
        )
        if (
            minimum_similarity < 0.60
            or alignment_spread > spacing * 0.12
            or height_cv > 0.25
        ):
            continue
        score = (
            float(np.mean(similarities))
            - alignment_spread / max(spacing, 1.0) * 0.5
            - abs(alignment_shear_per_row) / max(spacing, 1.0) * 0.2
            - height_cv * 0.25
        )
        scored.append(
            {
                "score": score,
                "minimumGlyphSimilarity": minimum_similarity,
                "minimumPixelSimilarity": min(
                    pixel_similarities,
                    default=0.0,
                ),
                "minimumStrokeSimilarity": min(
                    stroke_similarities,
                    default=0.0,
                ),
                "alignmentEdge": alignment_edge,
                "alignmentSpread": alignment_spread,
                "alignmentShearPerRow": alignment_shear_per_row,
                "leftEdgeSpread": left_spread,
                "rightEdgeSpread": right_spread,
                "heightCv": height_cv,
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    best = scored[0] if scored else None
    occlusion_recovery = (
        _occluded_roman_ii_recovery(slots, groups)
        if best is None
        else None
    )
    if best is None and occlusion_recovery is not None:
        best = {
            **occlusion_recovery,
            "alignmentEdge": "left-with-occluded-ii-stem",
            "alignmentShearPerRow": 0.0,
        }
    margin = (
        float(best["score"] - scored[1]["score"])
        if best is not None and occlusion_recovery is None and len(scored) > 1
        else float(occlusion_recovery["ambiguityMargin"])
        if occlusion_recovery is not None
        else 1.0
    )
    passed = bool(best is not None and best["score"] >= 0.62 and margin >= 0.04)
    return {
        "passed": passed,
        "method": "source-derived-roman-i-glyph-sequence-v1",
        "candidateCounts": {label: len(value) for label, value in groups.items()},
        "score": float(best["score"]) if best is not None else 0.0,
        "minimumGlyphSimilarity": (
            float(best["minimumGlyphSimilarity"]) if best is not None else 0.0
        ),
        "minimumPixelSimilarity": (
            float(best["minimumPixelSimilarity"]) if best is not None else 0.0
        ),
        "minimumStrokeSimilarity": (
            float(best["minimumStrokeSimilarity"]) if best is not None else 0.0
        ),
        "alignmentEdge": best.get("alignmentEdge") if best is not None else None,
        "alignmentSpread": (
            float(best["alignmentSpread"]) if best is not None else 0.0
        ),
        "alignmentShearPerRow": (
            float(best["alignmentShearPerRow"])
            if best is not None
            else 0.0
        ),
        "ambiguityMargin": margin if best is not None else 0.0,
        "occlusionSourceRecovery": (
            {
                "label": "II",
                "predictedSecondStemX": float(
                    occlusion_recovery["predictedSecondStemX"]
                ),
                "occludingLineSpanFraction": float(
                    occlusion_recovery["occludingLineSpanFraction"]
                ),
            }
            if occlusion_recovery is not None
            else None
        ),
    }


def _compact_augmented_glyphs(slot: LabelSlot) -> list[dict[str, Any]] | None:
    """Locate one compact three-glyph ``aV?`` label in source pixels."""
    mask = _remove_roman_overlapping_lines(slot.mask, slot.spacing)
    count, labels, stats, centroids = _connected_components(mask)
    minimum_width = max(2, round(slot.spacing * 0.02))
    minimum_height = max(4, round(slot.spacing * 0.04))
    components: list[dict[str, Any]] = []
    for index in range(1, count):
        x, y, width, height, area = (
            int(value) for value in stats[index]
        )
        if not (
            width >= minimum_width
            and minimum_height <= height <= slot.spacing * 0.18
            and width <= slot.spacing * 0.16
            and area >= 4
        ):
            continue
        components.append(
            {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "area": area,
                "centerY": float(centroids[index][1]),
                "glyph": _normalised_component(
                    (labels[y : y + height, x : x + width] == index).astype(
                        np.uint8
                    )
                ),
            }
        )
    components.sort(key=lambda item: item["x"])
    groups: list[list[dict[str, Any]]] = []
    maximum_gap = slot.spacing * 0.10
    for component in components:
        if groups:
            previous = groups[-1][-1]
            previous_right = previous["x"] + previous["width"]
        if not groups or component["x"] - previous_right > maximum_gap:
            groups.append([component])
        else:
            groups[-1].append(component)

    candidates: list[tuple[float, list[dict[str, Any]]]] = []
    for group in groups:
        if len(group) < 3:
            continue
        for start in range(len(group) - 2):
            candidate = group[start : start + 3]
            heights = np.asarray(
                [item["height"] for item in candidate], dtype=np.float64
            )
            centers_y = np.asarray(
                [item["centerY"] for item in candidate], dtype=np.float64
            )
            left = min(item["x"] for item in candidate)
            right = max(
                item["x"] + item["width"] for item in candidate
            )
            height_ratio = float(
                np.min(heights) / max(np.max(heights), 1.0)
            )
            vertical_spread = float(np.ptp(centers_y))
            if (
                height_ratio < 0.60
                or vertical_spread > slot.spacing * 0.10
                or right - left > slot.spacing * 0.36
            ):
                continue
            ink_area = sum(item["area"] for item in candidate)
            score = (
                height_ratio
                - vertical_spread / max(slot.spacing, 1.0)
                + ink_area
                / max((right - left) * float(np.max(heights)), 1.0)
            )
            candidates.append((score, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if (
        len(candidates) > 1
        and candidates[0][0] - candidates[1][0] < 0.04
    ):
        return None
    return candidates[0][1]


def _compact_augmented_suffix_pair(
    slot: LabelSlot,
) -> list[dict[str, Any]] | None:
    """Locate the intact ``V?`` suffix when ``a`` touches a long stroke."""
    components = sorted(_compact_components(slot), key=lambda item: item["x"])
    candidates: list[tuple[float, list[dict[str, Any]]]] = []
    for first, second in zip(components, components[1:]):
        gap = second["x"] - (first["x"] + first["width"])
        height_ratio = min(first["height"], second["height"]) / max(
            first["height"], second["height"]
        )
        vertical_spread = abs(first["centerY"] - second["centerY"])
        if not (
            -slot.spacing * 0.015 <= gap <= slot.spacing * 0.08
            and height_ratio >= 0.65
            and vertical_spread <= slot.spacing * 0.08
            and 0.25 <= first["centerX"] / slot.mask.shape[1] <= 0.55
            and second["centerX"] / slot.mask.shape[1] <= 0.70
        ):
            continue
        recovered = []
        for component in (first, second):
            recovered.append(
                {
                    **component,
                    "area": int(np.count_nonzero(component["glyph"])),
                    "glyph": _normalised_component(component["glyph"]),
                }
            )
        score = (
            height_ratio
            - vertical_spread / max(slot.spacing, 1.0)
            - max(gap, 0.0) / max(slot.spacing, 1.0) * 0.2
        )
        candidates.append((score, recovered))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.04:
        return None
    return candidates[0][1]


def _verify_augmented_sequence(slots: dict[str, LabelSlot]) -> dict[str, Any]:
    """Verify aVR/aVL/aVF from their repeated source-derived ``aV`` prefix.

    Small OCR engines commonly omit the final R/L/F even when the printed
    labels are visually clear.  The three labels share two independently
    repeated prefix glyphs, while their suffixes have a stable source-shape
    ordering: R is denser than F and F is denser than L.  Requiring the whole
    three-row sequence prevents one fuzzy token from assigning lead identity.
    """
    labels = ("aVR", "aVL", "aVF")
    resolved = {
        label: _compact_augmented_glyphs(slots[label]) for label in labels
    }
    missing_labels = [label for label, value in resolved.items() if value is None]
    partial_source_recovery: dict[str, Any] | None = None
    if len(missing_labels) == 1:
        missing_label = missing_labels[0]
        intact_labels = [label for label in labels if label != missing_label]
        suffix_pair = _compact_augmented_suffix_pair(slots[missing_label])
        anchor = resolved[intact_labels[0]]
        if suffix_pair is not None and anchor is not None:
            target_v, target_suffix = suffix_pair
            a_coverage = _prefix_coverage(
                anchor[0],
                anchor[1],
                slots[missing_label],
                target_v,
            )
            v_similarities = []
            for intact_label in intact_labels:
                intact = resolved[intact_label]
                if intact is None:
                    continue
                pixel = _dice(
                    _normalised_component(intact[1]["glyph"]),
                    _normalised_component(target_v["glyph"]),
                )
                stroke = _vertical_occupancy_dice(
                    _normalised_component(intact[1]["glyph"]),
                    _normalised_component(target_v["glyph"]),
                )
                v_similarities.append(max(pixel, stroke * 0.75))
            minimum_v_similarity = min(v_similarities, default=0.0)
            if a_coverage >= 0.65 and minimum_v_similarity >= 0.65:
                resolved[missing_label] = [
                    anchor[0],
                    target_v,
                    target_suffix,
                ]
                partial_source_recovery = {
                    "label": missing_label,
                    "aPrefixCoverage": a_coverage,
                    "minimumVSimilarity": minimum_v_similarity,
                }
    if any(value is None for value in resolved.values()):
        return {
            "passed": False,
            "reason": "compact-source-glyphs-unavailable",
            "method": "source-derived-avr-avl-avf-glyph-sequence-v1",
        }
    glyphs = [resolved[label] for label in labels]
    assert all(value is not None for value in glyphs)
    concrete = [value for value in glyphs if value is not None]
    prefix_similarities: list[float] = []
    for position in (0, 1):
        for first, second in itertools.combinations(concrete, 2):
            pixel = _dice(
                first[position]["glyph"], second[position]["glyph"]
            )
            stroke = _vertical_occupancy_dice(
                first[position]["glyph"], second[position]["glyph"]
            )
            prefix_similarities.append(max(pixel, stroke * 0.75))
    suffix_fill = [
        float(
            value[2]["area"]
            / max(value[2]["width"] * value[2]["height"], 1)
        )
        for value in concrete
    ]
    suffix_aspect = [
        float(value[2]["width"] / max(value[2]["height"], 1))
        for value in concrete
    ]
    minimum_prefix_similarity = min(prefix_similarities, default=0.0)
    if partial_source_recovery is not None:
        minimum_prefix_similarity = min(
            minimum_prefix_similarity,
            float(partial_source_recovery["aPrefixCoverage"]),
        )
    suffix_f_distinct_from_l = bool(
        suffix_fill[2] >= suffix_fill[1] + 0.02
        or suffix_aspect[2] >= suffix_aspect[1] * 1.20
    )
    passed = bool(
        minimum_prefix_similarity >= 0.65
        and suffix_fill[0] >= max(suffix_fill[1], suffix_fill[2]) + 0.06
        and suffix_f_distinct_from_l
        and suffix_aspect[0] >= suffix_aspect[1] * 0.90
    )
    return {
        "passed": passed,
        "reason": None if passed else "source-glyph-sequence-mismatch",
        "method": "source-derived-avr-avl-avf-glyph-sequence-v1",
        "minimumPrefixSimilarity": minimum_prefix_similarity,
        "suffixFillFractions": {
            label: value for label, value in zip(labels, suffix_fill, strict=True)
        },
        "suffixAspectRatios": {
            label: value
            for label, value in zip(labels, suffix_aspect, strict=True)
        },
        "suffixFDistinctFromL": suffix_f_distinct_from_l,
        "partialSourceRecovery": partial_source_recovery,
    }


def _compact_components(slot: LabelSlot) -> list[dict[str, Any]]:
    # A printed Vn label can touch the first QRS or a grid line. Locate glyph
    # components only after removing those long strokes; otherwise the label
    # becomes one oversized component and a visibly present digit is reported
    # as missing. Coordinates remain in the source crop, and downstream prefix
    # coverage still has to match the untouched source-pixel mask.
    component_mask = _remove_long_lines(slot.mask, slot.spacing)
    count, labels, stats, centroids = _connected_components(component_mask)
    components: list[dict[str, Any]] = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if not (
            max(4.0, slot.spacing * 0.035)
            <= height
            <= slot.spacing * 0.17
            and width <= slot.spacing * 0.16
            and area >= 4
        ):
            continue
        components.append(
            {
                "index": index,
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
                "centerX": float(centroids[index][0]),
                "centerY": float(centroids[index][1]),
                "glyph": (
                    labels[y : y + height, x : x + width] == index
                ).astype(np.uint8),
            }
        )
    return components


def _source_window_digit(
    slot: LabelSlot,
    prefix: dict[str, Any],
    *,
    horizontal_close_fraction: float | None = None,
    vertical_margin: tuple[int, int] = (0, 0),
    retain_prefix_edge: bool = False,
    intensity_threshold: int | None = None,
) -> dict[str, Any] | None:
    """Recover the digit immediately beside a compact V prefix.

    Printed digits occasionally touch a vertical grid/QRS stroke. General
    long-line removal can then erase most of the digit (notably the crossbar
    of ``4``). This bounded window retains the untouched source mask and clips
    it to the height of the independently located V prefix. An optional small
    horizontal close is recognition-only and is reported as reconstructed.
    """
    left = max(0, int(prefix["x"] + prefix["width"] - 1))
    right = min(
        slot.mask.shape[1],
        left
        + max(
            round(slot.spacing * 0.09),
            int(prefix["width"]) + 3,
        ),
    )
    top_margin, bottom_margin = vertical_margin
    top = max(0, int(prefix["y"]) - top_margin)
    bottom = min(
        slot.mask.shape[0],
        int(prefix["y"] + prefix["height"]) + bottom_margin,
    )
    if right - left < 3 or bottom - top < 4:
        return None
    if intensity_threshold is not None:
        if slot.gray is None:
            return None
        source_mask = (slot.gray < intensity_threshold).astype(np.uint8)
    else:
        source_mask = slot.mask
    glyph_window = source_mask[top:bottom, left:right].copy()
    # The first column may be the right-most antialiased pixel of the V. Pair
    # discovery excludes it; a morphology-only OCR variant may retain that
    # one-pixel source edge because it helps bridge a line-occluded digit.
    if not retain_prefix_edge:
        glyph_window[:, 0] = 0
    reconstructed = horizontal_close_fraction is not None
    if horizontal_close_fraction is not None:
        kernel_width = max(
            3,
            round(slot.spacing * horizontal_close_fraction),
        )
        glyph_window = cv2.morphologyEx(
            glyph_window,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1)),
        )
    y_values, x_values = np.nonzero(glyph_window)
    if y_values.size < 4:
        return None
    x0 = int(x_values.min())
    x1 = int(x_values.max()) + 1
    y0 = int(y_values.min())
    y1 = int(y_values.max()) + 1
    glyph = glyph_window[y0:y1, x0:x1]
    if glyph.shape[1] > slot.spacing * 0.12:
        return None
    return {
        "index": -1,
        "x": left + x0,
        "y": top + y0,
        "width": int(glyph.shape[1]),
        "height": int(glyph.shape[0]),
        "centerX": float(left + x0 + (glyph.shape[1] - 1) / 2),
        "centerY": float(top + y0 + (glyph.shape[0] - 1) / 2),
        "glyph": glyph.astype(np.uint8),
        "sourceWindow": True,
        "morphologyReconstructed": reconstructed,
        "intensityThreshold": intensity_threshold,
    }


def _projected_source_window_digit(
    anchor_slot: LabelSlot,
    anchor_prefix: dict[str, Any],
    anchor_digit: dict[str, Any],
    target_slot: LabelSlot,
) -> tuple[dict[str, Any], dict[str, Any], float] | None:
    """Split a low-resolution ``Vn`` label using a source-derived V anchor.

    Lanczos downsampling can join the last pixel of ``V`` diagonally to the
    first pixel of its digit. Connected-component discovery then reports one
    wide glyph even though both values remain visible. Project the independently
    located prefix box from another row in the same printed column and recover
    only the adjacent source-pixel window. The recovered digit must stay aligned
    with the anchor and the untouched target pixels must still cover the anchor
    prefix, so geometry alone cannot manufacture a lead value.
    """
    source_height, source_width = anchor_slot.mask.shape
    target_height, target_width = target_slot.mask.shape
    if min(source_height, source_width, target_height, target_width) <= 0:
        return None
    x_scale = target_width / source_width
    y_scale = target_height / source_height
    projected_prefix = {
        "x": round(anchor_prefix["x"] * x_scale),
        "y": round(anchor_prefix["y"] * y_scale),
        "width": max(1, round(anchor_prefix["width"] * x_scale)),
        "height": max(1, round(anchor_prefix["height"] * y_scale)),
    }
    expected_x = anchor_digit["centerX"] / source_width
    expected_y = anchor_digit["centerY"] / source_height
    candidates: list[tuple[int, float, dict[str, Any]]] = []
    for vertical_margin in ((0, 0), (1, 0), (0, 1), (1, 1), (1, 2)):
        digit = _source_window_digit(
            target_slot,
            projected_prefix,
            vertical_margin=vertical_margin,
        )
        if digit is None:
            continue
        observed_x = digit["centerX"] / target_width
        observed_y = digit["centerY"] / target_height
        if (
            abs(observed_x - expected_x) > 0.04
            or abs(observed_y - expected_y) > 0.12
            or digit["height"] / max(anchor_digit["height"], 1) < 0.50
        ):
            continue
        coverage = _prefix_coverage(
            anchor_prefix,
            anchor_digit,
            target_slot,
            digit,
        )
        if coverage >= 0.60:
            candidates.append((int(digit["height"]), coverage, digit))
    if not candidates:
        return None
    _, coverage, digit = max(candidates, key=lambda item: (item[0], item[1]))
    return projected_prefix, digit, coverage


def _v_anchor_pair(slot: LabelSlot) -> tuple[dict[str, Any], dict[str, Any]] | None:
    components = _compact_components(slot)
    candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    source_digits: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for prefix in components:
        if not (
            prefix["width"] / max(prefix["height"], 1) >= 0.35
            and prefix["height"] <= slot.spacing * 0.12
        ):
            continue
        digit = _source_window_digit(slot, prefix)
        if digit is None:
            continue
        gap = digit["x"] - (prefix["x"] + prefix["width"])
        if -slot.spacing * 0.015 <= gap <= slot.spacing * 0.035:
            source_digits[
                (
                    prefix["x"],
                    prefix["y"],
                    prefix["width"],
                    prefix["height"],
                )
            ] = digit

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for first, second in itertools.combinations(components, 2):
        left, right = sorted((first, second), key=lambda item: item["centerX"])
        key = (left["x"], left["y"], left["width"], left["height"])
        if key in source_digits:
            continue
        pairs.append((left, right))
    pairs.extend(
        (prefix, digit)
        for prefix in components
        if (
            digit := source_digits.get(
                (
                    prefix["x"],
                    prefix["y"],
                    prefix["width"],
                    prefix["height"],
                )
            )
        )
        is not None
    )

    for left, right in pairs:
        gap = right["x"] - (left["x"] + left["width"])
        height_ratio = min(left["height"], right["height"]) / max(
            left["height"], right["height"]
        )
        if not (
            abs(left["centerY"] - right["centerY"]) <= slot.spacing * 0.05
            and -slot.spacing * 0.015 <= gap <= slot.spacing * 0.08
            and height_ratio >= 0.70
            and right["centerX"] <= slot.mask.shape[1] * 0.72
        ):
            continue
        score = (
            height_ratio
            - abs(left["centerY"] - right["centerY"])
            / max(slot.spacing, 1.0)
            - max(0.0, gap) / max(slot.spacing, 1.0) * 0.2
            - left["centerX"] / max(slot.mask.shape[1], 1) * 0.03
        )
        candidates.append((score, left, right))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.03:
        return None
    return candidates[0][1], candidates[0][2]


def _prefix_coverage(
    template_component: dict[str, Any],
    template_digit: dict[str, Any],
    target_slot: LabelSlot,
    target_digit: dict[str, Any],
) -> float:
    scale = target_digit["height"] / max(template_digit["height"], 1)
    left = round(
        target_digit["x"]
        + (template_component["x"] - template_digit["x"]) * scale
    )
    top = round(
        target_digit["y"]
        + (template_component["y"] - template_digit["y"]) * scale
    )
    width = max(1, round(template_component["width"] * scale))
    height = max(1, round(template_component["height"] * scale))
    template = cv2.resize(
        template_component["glyph"],
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    best = 0.0
    for y_shift in range(-2, 3):
        for x_shift in range(-2, 3):
            x0 = left + x_shift
            y0 = top + y_shift
            x1 = x0 + width
            y1 = y0 + height
            if (
                x0 < 0
                or y0 < 0
                or x1 > target_slot.mask.shape[1]
                or y1 > target_slot.mask.shape[0]
            ):
                continue
            target = target_slot.mask[y0:y1, x0:x1]
            coverage = float(
                np.count_nonzero((template > 0) & (target > 0))
                / max(np.count_nonzero(template), 1)
            )
            best = max(best, coverage)
    return best


def _digit_sheet(
    glyphs: list[np.ndarray],
) -> np.ndarray:
    cell_size = DIGIT_SHEET_CELL_SIZE
    sheet = np.full((cell_size, cell_size * len(glyphs)), 255, dtype=np.uint8)
    for index, component in enumerate(glyphs):
        y_values, x_values = np.nonzero(component)
        if y_values.size == 0:
            continue
        crop = component[
            int(y_values.min()) : int(y_values.max()) + 1,
            int(x_values.min()) : int(x_values.max()) + 1,
        ]
        padding = 3
        crop = cv2.copyMakeBorder(
            crop,
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        scale = min(
            (cell_size - 32) / max(crop.shape[1], 1),
            (cell_size - 32) / max(crop.shape[0], 1),
        )
        rendered = cv2.resize(
            np.where(crop > 0, 0, 255).astype(np.uint8),
            None,
            fx=scale,
            fy=scale,
            # Linear resampling avoids staircase artifacts that can make a
            # low-resolution printed "5" resemble "$" to the OCR engine.
            # This contact sheet is recognition-only; source pixels and ECG
            # morphology are never modified.
            interpolation=cv2.INTER_LINEAR,
        )
        top = (cell_size - rendered.shape[0]) // 2
        left = index * cell_size + (cell_size - rendered.shape[1]) // 2
        sheet[top : top + rendered.shape[0], left : left + rendered.shape[1]] = (
            rendered
        )
    return sheet


def _expanded_isolated_digit_sheet(sheet: np.ndarray) -> np.ndarray:
    """Re-render one isolated source digit with a larger OCR-only footprint.

    Apple Vision can return punctuation for a clearly visible six-pixel digit
    after the conservative contact-sheet rendering, while recognizing the
    identical source shape when it occupies more of the same 128-pixel cell.
    Keep the ordinary rendering as the first attempt and use this binary,
    aspect-preserving enlargement only as a second exact-value check.
    """
    mask = np.asarray(sheet < 128, dtype=np.uint8)
    y_values, x_values = np.nonzero(mask)
    if y_values.size == 0:
        return sheet
    crop = mask[
        int(y_values.min()) : int(y_values.max()) + 1,
        int(x_values.min()) : int(x_values.max()) + 1,
    ]
    target_extent = 84
    scale = min(
        target_extent / max(crop.shape[1], 1),
        target_extent / max(crop.shape[0], 1),
    )
    rendered = cv2.resize(
        np.where(crop > 0, 0, 255).astype(np.uint8),
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_LINEAR,
    )
    output = np.full(
        (DIGIT_SHEET_CELL_SIZE, DIGIT_SHEET_CELL_SIZE),
        255,
        dtype=np.uint8,
    )
    top = (output.shape[0] - rendered.shape[0]) // 2
    left = (output.shape[1] - rendered.shape[1]) // 2
    output[top : top + rendered.shape[0], left : left + rendered.shape[1]] = rendered
    return output


def _output_has_isolated_digit_value(
    output: dict[str, Any],
    expected: str,
) -> bool:
    """Accept one exact digit even if OCR attaches a tiny alphabetic spur.

    This is only used for the enlarged, single-glyph recognition sheet. A
    low-resolution ``3`` can retain a detached top-left serif that Vision
    transcribes as ``r3``. Require exactly one numeric character, require it
    to be the expected value, and bound the nonnumeric prefix to two ASCII
    letters. Numeric conflicts therefore continue to fail closed.
    """
    if _output_has_exact_token(output, expected):
        return True
    for observation in _top_observations(output):
        candidates = observation.get("candidates") or []
        if not candidates:
            continue
        candidate = candidates[0]
        if float(candidate.get("confidence", 0.0)) < 0.5:
            continue
        compact = re.sub(r"\s+", "", str(candidate.get("text") or ""))
        if (
            re.fullmatch(rf"[A-Za-z]{{1,2}}{re.escape(expected)}", compact)
            and sum(character.isdigit() for character in compact) == 1
        ):
            return True
    return False


def _vertical_trace_occlusion(
    slot: LabelSlot,
    digit: dict[str, Any],
) -> dict[str, Any]:
    """Measure whether a long waveform stroke covers a printed digit.

    This is source evidence for bounded layout inference, not an attempt to
    reconstruct the hidden value. Ordinary numeral strokes occupy only the
    label-height band; an overprinted QRS spans most of the entire source slot
    and overlaps most of the candidate digit ink.
    """
    slot_height, slot_width = slot.mask.shape
    vertical_length = min(
        max(9, round(slot.spacing * 0.24)),
        max(9, round(slot_height * 0.70)),
    )
    vertical = cv2.morphologyEx(
        slot.mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_length)),
    )
    x0 = max(0, int(digit["x"]) - 1)
    x1 = min(slot_width, int(digit["x"] + digit["width"]) + 1)
    y0 = max(0, int(digit["y"]) - 1)
    y1 = min(slot_height, int(digit["y"] + digit["height"]) + 1)
    source_region = slot.mask[y0:y1, x0:x1] > 0
    vertical_region = vertical[y0:y1, x0:x1] > 0
    overlap_fraction = float(
        np.count_nonzero(source_region & vertical_region)
        / max(np.count_nonzero(source_region), 1)
    )
    line_span_fraction = float(
        np.max(np.count_nonzero(vertical[:, x0:x1], axis=0), initial=0)
        / max(slot_height, 1)
    )
    occluded = bool(
        line_span_fraction >= 0.70 and overlap_fraction >= 0.40
    )
    return {
        "occluded": occluded,
        "lineSpanFraction": line_span_fraction,
        "inkOverlapFraction": overlap_fraction,
    }


def _digit_text(output: dict[str, Any]) -> str:
    ordered = sorted(
        _top_observations(output),
        key=lambda item: float(item.get("x", 0.0)),
    )
    values = [str(item["candidates"][0]["text"]) for item in ordered]
    compact = re.sub(r"\s+", "", "".join(values))
    return compact if compact.isascii() and compact.isdigit() else ""


def _digit_tokens(output: dict[str, Any]) -> list[str]:
    ordered = sorted(
        _top_observations(output),
        key=lambda item: float(item.get("x", 0.0)),
    )
    joined = " ".join(
        str(item["candidates"][0]["text"]) for item in ordered
    )
    separated = joined.split()
    if len(separated) == 6:
        return separated
    compact = re.sub(r"\s+", "", joined)
    if len(compact) == 6 and compact.isascii() and compact.isalnum():
        return list(compact)
    tokens = re.findall(r"[A-Za-z0-9]+", joined)
    return tokens if len(tokens) == 6 else []


def _verify_precordial_values(
    sheet: ContactSheet,
    mapped: dict[str, list[dict[str, Any]]],
    ocr_runner: OcrRunner,
) -> dict[str, Any]:
    slots = sheet.slots
    three_by_four = slots["V1"].column != slots["V4"].column
    groups = (
        (("V1", "V2", "V3"), ("V4", "V5", "V6"))
        if three_by_four
        else (("V1", "V2", "V3", "V4", "V5", "V6"),)
    )
    anchors: dict[
        str,
        tuple[str, tuple[dict[str, Any], dict[str, Any]]],
    ] = {}
    unresolved_groups: dict[str, str] = {}
    for labels in groups:
        exact_labels = [
            label
            for label in labels
            if _slot_has_exact_token(mapped, label)
        ]
        resolved = next(
            (
                (label, pair)
                for label in exact_labels
                if (pair := _v_anchor_pair(slots[label])) is not None
            ),
            None,
        )
        group_key = labels[0]
        if resolved is None:
            unresolved_groups[group_key] = (
                f"{group_key.lower()}-source-glyphs-ambiguous"
                if exact_labels
                else f"{group_key.lower()}-group-anchor-not-recognized"
            )
        else:
            anchors[group_key] = resolved

    cross_group_anchor_recoveries: list[str] = []
    if unresolved_groups and three_by_four and len(anchors) == 1:
        # A 3 x 4 page places V1-V3 and V4-V6 in separate columns. OCR can
        # miss every exact token in one column when a label touches the first
        # QRS, even though the shared printed ``V`` prefix remains visible in
        # the source pixels. Reuse the independently resolved prefix/digit
        # geometry from the other precordial column, then continue to require
        # source-prefix agreement and the exact 1..6 digit sequence below.
        # This cannot promote a swapped sequence: the digits are still read
        # from their own source slots and verified value by value.
        source_anchor = next(iter(anchors.values()))
        for group_key in unresolved_groups:
            anchors[group_key] = source_anchor
            cross_group_anchor_recoveries.append(group_key)
    elif unresolved_groups:
        first_group = next(iter(unresolved_groups))
        return {
            "passed": False,
            "reason": unresolved_groups[first_group],
        }

    selected_digits: list[dict[str, Any]] = []
    selected_prefixes: list[dict[str, Any] | None] = []
    prefix_coverages: list[float] = []
    anchor_labels: list[str] = []
    direct_label_recoveries: list[str] = []
    source_pair_recoveries: list[str] = []
    projected_source_window_recoveries: list[str] = []
    contact_label_confirmations = [
        f"V{index}"
        for index in range(1, 7)
        if _slot_has_exact_token(mapped, f"V{index}")
    ]
    for index in range(1, 7):
        label = f"V{index}"
        slot = slots[label]
        group_key = "V4" if three_by_four and index >= 4 else "V1"
        anchor_label, anchor = anchors[group_key]
        anchor_slot = slots[anchor_label]
        anchor_prefix, anchor_digit = anchor
        source_prefix_coverage: float | None = None
        predicted_x_fraction = anchor_digit["centerX"] / max(
            anchor_slot.mask.shape[1], 1
        )
        predicted_y_fraction = anchor_digit["centerY"] / max(
            anchor_slot.mask.shape[0], 1
        )
        components = _compact_components(slot)
        recovered_pair: tuple[dict[str, Any], dict[str, Any]] | None = None
        projected_recovery = (
            None
            if label == anchor_label
            else _projected_source_window_digit(
                anchor_slot,
                anchor_prefix,
                anchor_digit,
                slot,
            )
        )
        projected_prefix: dict[str, Any] | None = None
        if label == anchor_label:
            digit = anchor_digit
        else:
            target_pair = None
            failure_reason: str | None = None
            candidates = sorted(
                components,
                key=lambda item: (
                    abs(
                        item["centerX"] / max(slot.mask.shape[1], 1)
                        - predicted_x_fraction
                    )
                    + abs(
                        item["centerY"] / max(slot.mask.shape[0], 1)
                        - predicted_y_fraction
                    )
                    * 0.5
                ),
            )
            if not candidates:
                failure_reason = f"{label.lower()}-digit-missing"
            else:
                digit = candidates[0]
                distance = abs(
                    digit["centerX"] / max(slot.mask.shape[1], 1)
                    - predicted_x_fraction
                )
                if distance > 0.06:
                    failure_reason = f"{label.lower()}-digit-unaligned"
                elif len(candidates) > 1:
                    second_distance = abs(
                        candidates[1]["centerX"]
                        / max(slot.mask.shape[1], 1)
                        - predicted_x_fraction
                    )
                    if second_distance - distance < 0.015:
                        failure_reason = f"{label.lower()}-digit-ambiguous"
            if failure_reason is not None:
                if _tight_slot_has_exact_token(slot, label, ocr_runner):
                    recovered_pair = _v_anchor_pair(slot)
                if recovered_pair is None:
                    target_pair = _v_anchor_pair(slot)
                    if target_pair is not None:
                        target_prefix, target_digit = target_pair
                        source_prefix_coverage = _dice(
                            _normalised_component(anchor_prefix["glyph"]),
                            _normalised_component(target_prefix["glyph"]),
                        )
                        if source_prefix_coverage >= 0.60:
                            digit = target_digit
                            source_pair_recoveries.append(label)
                        else:
                            target_pair = None
                if (
                    recovered_pair is None
                    and target_pair is None
                    and projected_recovery is not None
                ):
                    (
                        projected_prefix,
                        digit,
                        source_prefix_coverage,
                    ) = projected_recovery
                    projected_source_window_recoveries.append(label)
                elif recovered_pair is None and target_pair is None:
                    return {"passed": False, "reason": failure_reason}
                if recovered_pair is not None:
                    digit = recovered_pair[1]
                    direct_label_recoveries.append(label)
        prefix_coverage = (
            1.0
            if recovered_pair is not None
            else source_prefix_coverage
            if source_prefix_coverage is not None
            else _prefix_coverage(
                    anchor_prefix,
                    anchor_digit,
                    slot,
                    digit,
                )
        )
        if prefix_coverage < 0.60 and recovered_pair is None:
            target_pair = _v_anchor_pair(slot)
            if target_pair is not None:
                target_prefix, target_digit = target_pair
                pair_coverage = _dice(
                    _normalised_component(anchor_prefix["glyph"]),
                    _normalised_component(target_prefix["glyph"]),
                )
                if pair_coverage >= 0.60:
                    digit = target_digit
                    prefix_coverage = pair_coverage
                    if label not in source_pair_recoveries:
                        source_pair_recoveries.append(label)
            if prefix_coverage < 0.60 and projected_recovery is not None:
                projected_prefix, digit, prefix_coverage = projected_recovery
                if label not in projected_source_window_recoveries:
                    projected_source_window_recoveries.append(label)
        selected_digits.append(digit)
        resolved_pair = _v_anchor_pair(slot)
        selected_prefixes.append(
            resolved_pair[0]
            if resolved_pair is not None
            else projected_prefix
        )
        anchor_labels.append(anchor_label)
        prefix_coverages.append(prefix_coverage)

    minimum_prefix_coverage = min(prefix_coverages)
    if minimum_prefix_coverage < 0.60:
        return {
            "passed": False,
            "reason": "v-prefix-source-pixel-mismatch",
            "minimumPrefixCoverage": minimum_prefix_coverage,
            "prefixCoverages": prefix_coverages,
        }
    digit_output = ocr_runner(
        _digit_sheet([item["glyph"] for item in selected_digits])
    )
    recognised_digits = _digit_text(digit_output)
    sequence_recognized = recognised_digits == "123456"
    sequence_mode = "six-digit-sequence"
    source_sequence_tokens: list[str] = []
    if not sequence_recognized and all(
        prefix is not None for prefix in selected_prefixes
    ):
        source_sequence = [
            _source_window_digit(
                slots[f"V{index}"],
                prefix,
                vertical_margin=(2, 2),
            )
            for index, prefix in enumerate(selected_prefixes, start=1)
            if prefix is not None
        ]
        if len(source_sequence) == 6 and all(
            digit is not None for digit in source_sequence
        ):
            source_sequence_output = ocr_runner(
                _digit_sheet(
                    [
                        digit["glyph"]
                        for digit in source_sequence
                        if digit is not None
                    ]
                )
            )
            source_sequence_text = _digit_text(source_sequence_output)
            source_sequence_tokens = _digit_tokens(source_sequence_output)
            if source_sequence_text == "123456":
                recognised_digits = source_sequence_text
                sequence_recognized = True
                sequence_mode = "source-window-six-digit-sequence"
    individual_digit_confirmations: list[str] = []
    confirmed_labels = {
        *contact_label_confirmations,
        *direct_label_recoveries,
    }
    if not sequence_recognized:
        for index, (digit, prefix) in enumerate(
            zip(selected_digits, selected_prefixes, strict=True),
            start=1,
        ):
            label = f"V{index}"
            if label in confirmed_labels:
                continue
            variants = [digit["glyph"]]
            if prefix is not None:
                for close_fraction, intensity_threshold in (
                    (None, None),
                    (0.045, None),
                    (None, 120),
                    (None, 140),
                    (None, 160),
                ):
                    source_digit = _source_window_digit(
                        slots[label],
                        prefix,
                        horizontal_close_fraction=close_fraction,
                        vertical_margin=(1, 2),
                        retain_prefix_edge=close_fraction is not None,
                        intensity_threshold=intensity_threshold,
                    )
                    if source_digit is not None:
                        variants.append(source_digit["glyph"])
            matched = False
            seen: set[tuple[tuple[int, ...], bytes]] = set()
            for variant in variants:
                key = (variant.shape, variant.tobytes())
                if key in seen:
                    continue
                seen.add(key)
                standard_sheet = _digit_sheet([variant])
                recognition_sheets = (
                    (standard_sheet, False),
                    (_expanded_isolated_digit_sheet(standard_sheet), True),
                )
                for recognition_sheet, enlarged in recognition_sheets:
                    try:
                        output = ocr_runner(recognition_sheet)
                    except (
                        OSError,
                        RuntimeError,
                        subprocess.SubprocessError,
                        json.JSONDecodeError,
                    ):
                        continue
                    if (
                        _output_has_isolated_digit_value(output, str(index))
                        if enlarged
                        else _output_has_exact_token(output, str(index))
                    ):
                        matched = True
                        break
                if matched:
                    break
            if matched:
                confirmed_labels.add(label)
                individual_digit_confirmations.append(label)
    contextual_sequence_recoveries: list[str] = []
    sequence_tokens = _digit_tokens(digit_output)
    if len(sequence_tokens) != 6 and len(source_sequence_tokens) == 6:
        sequence_tokens = source_sequence_tokens
    if (
        not sequence_recognized
        and len(sequence_tokens) == 6
        and len(confirmed_labels) >= 4
    ):
        # At six-pixel source resolution, the numeric shapes remain visible
        # but general OCR can read the tiny printed curves and serifs as
        # similarly shaped letters (for example ``2`` as ``q``, ``4`` as
        # ``P``, or ``6`` as ``e``/``F``). Only resolve those confusions inside
        # the already aligned six-cell V1..V6 sequence, after at least four
        # labels were confirmed independently. A conflicting numeric value is
        # never corrected.
        confusions = {
            "1": {"I", "i", "J", "j", "l"},
            "2": {"Q", "q"},
            "3": {"J", "j"},
            "4": {"P", "p"},
            "5": {"E"},
            "6": {"e", "F"},
        }
        provisional: list[str] = []
        for index, token in enumerate(sequence_tokens, start=1):
            label = f"V{index}"
            if label in confirmed_labels:
                continue
            expected = str(index)
            if token == expected or token in confusions.get(expected, set()):
                provisional.append(label)
                continue
            provisional = []
            break
        for label in provisional:
            confirmed_labels.add(label)
            contextual_sequence_recoveries.append(label)
    occlusion_sequence_recoveries: list[str] = []
    occlusion_evidence: dict[str, dict[str, Any]] = {}
    unresolved_labels = [
        f"V{index}"
        for index in range(1, 7)
        if f"V{index}" not in confirmed_labels
    ]
    if 1 <= len(unresolved_labels) <= 2 and len(confirmed_labels) >= 4:
        explicit_numeric_conflict = False
        for label in unresolved_labels:
            index = int(label[1:]) - 1
            evidence = _vertical_trace_occlusion(
                slots[label],
                selected_digits[index],
            )
            evidence["digitSource"] = "selected-component"
            prefix = selected_prefixes[index]
            if prefix is not None:
                source_digit = _source_window_digit(
                    slots[label],
                    prefix,
                    vertical_margin=(0, 0),
                )
                if source_digit is not None:
                    source_evidence = _vertical_trace_occlusion(
                        slots[label],
                        source_digit,
                    )
                    source_evidence["digitSource"] = "source-window"
                    if (
                        source_evidence["lineSpanFraction"]
                        * source_evidence["inkOverlapFraction"]
                        > evidence["lineSpanFraction"]
                        * evidence["inkOverlapFraction"]
                    ):
                        evidence = source_evidence
            occlusion_evidence[label] = evidence
            if len(sequence_tokens) == 6:
                token = sequence_tokens[index]
                if (
                    len(token) == 1
                    and token.isascii()
                    and token.isdigit()
                    # A fully overprinted vertical QRS is itself commonly
                    # transcribed as ``1``. Other explicit numeric values
                    # remain conflicts even when the source is occluded.
                    and token != "1"
                    and token != str(index + 1)
                ):
                    explicit_numeric_conflict = True
        if all(
            evidence["occluded"]
            for evidence in occlusion_evidence.values()
        ) and not explicit_numeric_conflict:
            for label in unresolved_labels:
                confirmed_labels.add(label)
                occlusion_sequence_recoveries.append(label)
    passed = sequence_recognized or all(
        f"V{index}" in confirmed_labels for index in range(1, 7)
    )
    return {
        "passed": passed,
        "reason": None if passed else "precordial-digit-sequence-mismatch",
        "recognizedDigits": recognised_digits,
        "valueRecognitionMode": (
            sequence_mode
            if sequence_recognized
            else "individually-confirmed-source-values"
            if passed
            else "unconfirmed"
        ),
        "contactLabelConfirmations": contact_label_confirmations,
        "individualDigitConfirmations": individual_digit_confirmations,
        "contextualSequenceRecoveries": contextual_sequence_recoveries,
        "sequenceTokens": sequence_tokens,
        "occlusionSequenceRecoveries": occlusion_sequence_recoveries,
        "occlusionEvidence": occlusion_evidence,
        "anchorLabels": sorted(set(anchor_labels)),
        "directLabelRecoveries": direct_label_recoveries,
        "sourcePairRecoveries": source_pair_recoveries,
        "projectedSourceWindowRecoveries": projected_source_window_recoveries,
        "crossGroupAnchorRecoveries": cross_group_anchor_recoveries,
        "minimumPrefixCoverage": minimum_prefix_coverage,
        "prefixCoverages": prefix_coverages,
        "method": "exact-digits-plus-source-derived-v-prefix-v1",
    }


def recognise_semantic_lead_identity(
    image: np.ndarray,
    geometry: dict[str, Any],
    *,
    ocr_runner: OcrRunner | None = None,
) -> dict[str, Any]:
    slots = _layout_slots(image, geometry)
    if len(slots) != 12 or {slot.expected for slot in slots} != set(
        STANDARD_LEADS
    ):
        return {
            "passed": False,
            "order": None,
            "confidence": 0.0,
            "method": SEMANTIC_METHOD,
            "semanticIdentityConfirmed": False,
            "failureReasons": ["unsupported-or-incomplete-layout-slots"],
        }
    runner = ocr_runner or _default_ocr_runner
    sheet = _contact_sheet(slots)
    try:
        full_output = runner(sheet.image)
    except (OSError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError):
        return {
            "passed": False,
            "order": None,
            "confidence": 0.0,
            "method": SEMANTIC_METHOD,
            "semanticIdentityConfirmed": False,
            "failureReasons": ["value-aware-recognizer-unavailable"],
        }
    mapped = _tokens_by_slot(sheet, full_output)
    secondary_engine: dict[str, Any] | None = None
    # Apple Vision and RapidOCR make different, complementary mistakes on
    # small ECG labels (for example, ``V5`` versus ``VS``). For the packaged
    # runtime, require exact tokens from either recognizer while retaining the
    # same source-cell mapping. Custom/test runners remain single-engine.
    if (
        runner is _default_ocr_runner
        and str(full_output.get("engine") or "").startswith("apple-vision")
    ):
        try:
            secondary_output = _rapidocr_ocr_runner(sheet.image)
            mapped = _merge_mapped_tokens(
                mapped,
                _tokens_by_slot(sheet, secondary_output),
            )
            secondary_engine = {
                "name": secondary_output.get("engine"),
                "revision": secondary_output.get("revision"),
            }
        except (
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            json.JSONDecodeError,
        ):
            secondary_engine = None
    roman = _verify_roman_sequence(sheet.slots)
    augmented = {
        label: _slot_has_exact_token(mapped, label)
        for label in ("aVR", "aVL", "aVF")
    }
    augmented_sources = {
        label: "contact-sheet" for label, passed in augmented.items() if passed
    }
    augmented_labels = ("aVR", "aVL", "aVF")
    normalised_augmented = {
        label.casefold(): label for label in augmented_labels
    }
    augmented_conflicts = {
        expected: sorted(
            {
                normalised_augmented[value]
                for observation in mapped.get(expected, [])
                if (candidates := observation.get("candidates") or [])
                and isinstance(candidates[0].get("text"), str)
                and float(candidates[0].get("confidence", 0.0)) >= 0.5
                and (
                    value := _normalise_token(str(candidates[0]["text"]))
                )
                in normalised_augmented
                and normalised_augmented[value] != expected
            }
        )
        for expected in augmented_labels
    }
    for label, passed in augmented.items():
        if passed:
            continue
        try:
            isolated_output = runner(_isolated_label_sheet(sheet.slots[label]))
        except (
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            json.JSONDecodeError,
        ):
            continue
        if _output_has_exact_token(isolated_output, label):
            augmented[label] = True
            augmented_sources[label] = "isolated-cubic-source-label"
            continue
        if _tight_slot_has_exact_token(sheet.slots[label], label, runner):
            augmented[label] = True
            augmented_sources[label] = "tight-isolated-source-glyphs"
    augmented_sequence = _verify_augmented_sequence(sheet.slots)
    if augmented_sequence.get("passed") and not any(
        augmented_conflicts.values()
    ):
        for label, passed in augmented.items():
            if not passed:
                augmented[label] = True
                augmented_sources[label] = str(augmented_sequence["method"])
    try:
        precordial = _verify_precordial_values(sheet, mapped, runner)
    except (OSError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError):
        precordial = {
            "passed": False,
            "reason": "value-aware-recognizer-unavailable",
        }
    failure_reasons: list[str] = []
    if not roman.get("passed"):
        failure_reasons.append("roman-label-values-unconfirmed")
    for label, passed in augmented.items():
        if not passed:
            failure_reasons.append(f"{label.lower()}-value-unconfirmed")
    if not precordial.get("passed"):
        failure_reasons.append(
            str(precordial.get("reason") or "precordial-values-unconfirmed")
        )
    passed = not failure_reasons
    engine = {
        "name": full_output.get("engine"),
        "revision": full_output.get("revision"),
        "platform": "macOS" if sys.platform == "darwin" else platform.system(),
        "platformVersion": (
            platform.mac_ver()[0]
            if sys.platform == "darwin"
            else platform.release()
        ),
    }
    if secondary_engine is not None:
        engine["secondary"] = secondary_engine

    def label_confirmed(label: str) -> bool:
        if label in {"I", "II", "III"}:
            return bool(roman.get("passed"))
        if label in augmented:
            return augmented[label]
        if label.startswith("V"):
            return bool(precordial.get("passed"))
        return False

    recognized_labels = [
        label for label in STANDARD_LEADS if label_confirmed(label)
    ]
    confidence = (
        min(
            0.95,
            0.50
            + float(roman.get("minimumGlyphSimilarity", 0.0)) * 0.20
            + float(precordial.get("minimumPrefixCoverage", 0.0)) * 0.20,
        )
        if passed
        else 0.0
    )
    return {
        "passed": passed,
        "order": "standard" if passed else None,
        "confidence": confidence,
        "method": SEMANTIC_METHOD,
        "semanticIdentityConfirmed": passed,
        "expectedLabels": list(STANDARD_LEADS),
        "recognizedLabels": recognized_labels,
        "romanValidation": roman,
        "augmentedLeadValidation": augmented,
        "augmentedLeadSources": augmented_sources,
        "augmentedLeadSequenceValidation": augmented_sequence,
        "augmentedLeadConflicts": augmented_conflicts,
        "precordialValidation": precordial,
        "engine": engine,
        "failureReasons": failure_reasons,
    }


def recognise_rhythm_lead_identity(
    image: np.ndarray,
    geometry: dict[str, Any],
    *,
    ocr_runner: OcrRunner | None = None,
    primary_identity_confirmed: bool = False,
) -> dict[str, Any]:
    """Confirm the printed label of a full-width 6 x 2 rhythm strip.

    The waveform row itself cannot establish lead identity.  Keep this proof
    separate from the twelve primary labels so a duplicated ``II`` has its own
    source-pixel crop and value-aware OCR evidence.
    """

    method = "source-pixel-rhythm-label-value-validation-v1"
    layout = str(geometry.get("layoutHint") or "")
    centers = [int(value) for value in geometry.get("rowCenters") or []]
    if layout != "standard_6x2_with_r1_ignored" or len(centers) != 7:
        return {
            "passed": False,
            "lead": None,
            "confidence": 0.0,
            "method": method,
            "semanticIdentityConfirmed": False,
            "expectedLabel": "II",
            "recognizedLabels": [],
            "failureReasons": ["unsupported-or-incomplete-rhythm-slot"],
        }

    height, width = image.shape[:2]
    spacing = float(np.median(np.diff(centers[:6])))
    center = centers[6]
    slot = _slot(
        image,
        "II",
        6,
        0,
        _bounded_box(
            width,
            height,
            width * 0.01,
            center - spacing * 0.62,
            width * 0.18,
            center - spacing * 0.18,
        ),
        spacing,
    )
    if slot is None:
        return {
            "passed": False,
            "lead": None,
            "confidence": 0.0,
            "method": method,
            "semanticIdentityConfirmed": False,
            "expectedLabel": "II",
            "recognizedLabels": [],
            "failureReasons": ["rhythm-label-crop-unavailable"],
        }

    runner = ocr_runner or _default_ocr_runner
    output: dict[str, Any] | None = None
    try:
        output = runner(_isolated_label_sheet(slot))
    except (
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
    ):
        output = None

    exact = [
        observation
        for observation in _top_observations(output or {})
        if _normalise_token(str(observation["candidates"][0]["text"])) == "ii"
        and float(observation["candidates"][0].get("confidence", 0.0)) >= 0.5
    ]
    exact_confidence = (
        max(
            float(observation["candidates"][0].get("confidence", 0.0))
            for observation in exact
        )
        if exact
        else 0.0
    )

    primary_slots = {
        candidate.expected: candidate
        for candidate in _layout_slots(image, geometry)
    }
    primary_groups = (
        _roman_groups(primary_slots["II"], 2)
        if "II" in primary_slots
        else []
    )
    rhythm_groups = _roman_groups(slot, 2)
    source_scores: list[float] = []
    for primary, rhythm in itertools.product(primary_groups, rhythm_groups):
        primary_components = primary["components"]
        rhythm_components = rhythm["components"]
        pixel_similarities = [
            _dice(source["glyph"], repeated["glyph"])
            for source, repeated in zip(
                primary_components,
                rhythm_components,
                strict=True,
            )
        ]
        stroke_similarities = [
            _vertical_occupancy_dice(source["glyph"], repeated["glyph"])
            for source, repeated in zip(
                primary_components,
                rhythm_components,
                strict=True,
            )
        ]
        similarities = [
            max(pixel, stroke * 0.75)
            for pixel, stroke in zip(
                pixel_similarities,
                stroke_similarities,
                strict=True,
            )
        ]
        source_scores.append(min(similarities, default=0.0))
    source_scores.sort(reverse=True)
    source_confidence = source_scores[0] if source_scores else 0.0
    source_margin = (
        source_scores[0] - source_scores[1]
        if len(source_scores) > 1
        else 1.0 if source_scores else 0.0
    )
    source_match = bool(
        source_confidence >= 0.72 and source_margin >= 0.04
    )
    passed = bool(
        primary_identity_confirmed and (exact or source_match)
    )
    confidence = max(exact_confidence, source_confidence if source_match else 0.0)
    validation_source = (
        "exact-ocr"
        if exact
        else "source-derived-primary-ii-glyph-match"
        if source_match
        else None
    )
    failure_reasons = (
        []
        if passed
        else ["primary-lead-identity-unconfirmed"]
        if not primary_identity_confirmed
        else ["rhythm-label-ii-unconfirmed"]
    )
    engine = (
        {
            "name": output.get("engine"),
            "revision": output.get("revision"),
            "platform": "macOS" if sys.platform == "darwin" else platform.system(),
            "platformVersion": (
                platform.mac_ver()[0]
                if sys.platform == "darwin"
                else platform.release()
            ),
        }
        if output is not None
        else None
    )
    semantic_recognition = {
        "passed": passed,
        "order": "standard" if passed else None,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "method": method,
        "semanticIdentityConfirmed": passed,
        "expectedLabels": ["II"],
        "recognizedLabels": ["II"] if passed else [],
        "failureReasons": failure_reasons,
        **({"engine": engine} if engine is not None else {}),
    }
    return {
        "passed": passed,
        "lead": "II" if passed else None,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "method": method,
        "semanticIdentityConfirmed": passed,
        "semanticRecognition": semantic_recognition,
        "expectedLabel": "II",
        "recognizedLabels": ["II"] if passed else [],
        "validationSource": validation_source,
        "sourceGlyphSimilarity": source_confidence,
        "sourceGlyphAmbiguityMargin": source_margin,
        **({"engine": engine} if engine is not None else {}),
        "failureReasons": failure_reasons,
    }


def attach_semantic_lead_identity(
    image: np.ndarray,
    geometry: dict[str, Any],
    *,
    ocr_runner: OcrRunner | None = None,
) -> dict[str, Any]:
    report = recognise_semantic_lead_identity(
        image,
        geometry,
        ocr_runner=ocr_runner,
    )
    result = dict(geometry)
    layout = str(result.get("layoutHint") or "")
    shared = {
        "semanticIdentityConfirmed": bool(report.get("passed")),
        "semanticRecognition": report,
    }
    if report.get("passed"):
        shared.update(
            {
                "passed": True,
                "order": "standard",
                "confidence": float(report.get("confidence", 0.0)),
                "method": str(report.get("method") or SEMANTIC_METHOD),
            }
        )
    if layout in {"standard_6x2", "standard_6x2_with_r1_ignored"}:
        result["limbLabelValidation"] = {
            **(result.get("limbLabelValidation") or {}),
            **shared,
        }
        result["precordialLabelValidation"] = {
            **(result.get("precordialLabelValidation") or {}),
            **shared,
        }
        if layout == "standard_6x2_with_r1_ignored":
            result["rhythmLeadValidation"] = recognise_rhythm_lead_identity(
                image,
                result,
                ocr_runner=ocr_runner,
                primary_identity_confirmed=bool(report.get("passed")),
            )
    else:
        result["leadLabelValidation"] = {
            **(result.get("leadLabelValidation") or {}),
            **shared,
        }
    return result
