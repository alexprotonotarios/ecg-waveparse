from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ecg_pipeline.domain import (
    PAGE_DURATION_SECONDS,
    SAMPLE_RATE_HZ,
    SIX_BY_TWO_PANEL_SAMPLES,
    SIX_BY_TWO_ROW_LEADS,
    THREE_BY_FOUR_PANEL_SAMPLES,
    THREE_BY_FOUR_ROW_LEADS,
    TWELVE_ROW_LEADS,
)


@dataclass(frozen=True)
class NativeFidelityPolicy:
    method: str | None
    minimum_layout_confidence: float
    require_rhythm_anchors: bool
    minimum_evidence_median: float
    minimum_evidence_p10: float
    maximum_crossing_p95_fraction: float
    maximum_crossing_jump_fraction: float
    unsupported_large_jump_fraction: float


@dataclass(frozen=True)
class NativeLayoutStrategy:
    layout: str
    row_leads: tuple[tuple[str, ...], ...]
    primary_centers: tuple[int, ...]
    primary_bounds: tuple[tuple[int, int], ...]
    panel_edges: tuple[int, ...]
    panel_samples: int
    pixels_per_mm: float
    primary_bottom: int
    rhythm_row_index: int | None
    publish_rhythm_trace: bool
    allow_ink_backed_crossings: bool
    row_local_twelve: bool
    label_anchored_twelve: bool
    row_local_labeled_three: bool
    row_local_labeled_six: bool
    row_local_calibration_six: bool
    fidelity: NativeFidelityPolicy

    @property
    def row_local_geometry(self) -> bool:
        return (
            self.row_local_twelve
            or self.row_local_labeled_three
            or self.row_local_labeled_six
            or self.row_local_calibration_six
        )


def row_bounds(row_centers: list[int], height: int) -> list[tuple[int, int]]:
    if len(row_centers) < 7:
        raise ValueError("Seven row centers are required for a 6 x 2 plus rhythm ECG.")
    boundaries = [0]
    boundaries.extend(
        round((row_centers[index] + row_centers[index + 1]) / 2)
        for index in range(5)
    )
    boundaries.append(min(height, round((row_centers[5] + row_centers[6]) / 2)))
    return [
        (max(0, boundaries[index]), min(height, boundaries[index + 1]))
        for index in range(6)
    ]


def six_row_bounds(row_centers: list[int], height: int) -> list[tuple[int, int]]:
    if len(row_centers) < 6:
        raise ValueError("Six row centers are required for a 6 x 2 ECG.")
    median_spacing = float(np.median(np.diff(row_centers[:6])))
    boundaries = [round(row_centers[0] - median_spacing * 0.48)]
    boundaries.extend(
        round((row_centers[index] + row_centers[index + 1]) / 2)
        for index in range(5)
    )
    boundaries.append(round(row_centers[5] + median_spacing * 0.48))
    return [
        (max(0, boundaries[index]), min(height, boundaries[index + 1]))
        for index in range(6)
    ]


def sequential_row_bounds(
    row_centers: list[int],
    height: int,
    *,
    expansion: float = 0.82,
) -> list[tuple[int, int]]:
    if len(row_centers) < 2:
        raise ValueError("At least two row centers are required.")
    spacing = np.diff(row_centers).astype(np.float64)
    bounds: list[tuple[int, int]] = []
    for index, center in enumerate(row_centers):
        upper_spacing = spacing[index - 1] if index else spacing[0]
        lower_spacing = spacing[index] if index < spacing.size else spacing[-1]
        bounds.append(
            (
                max(0, round(center - upper_spacing * expansion)),
                min(height, round(center + lower_spacing * expansion)),
            )
        )
    return bounds


def build_native_layout_strategy(
    geometry: dict[str, Any],
    *,
    width: int,
    height: int,
    paper_speed_mm_per_second: float,
    detected_pixels_per_mm: float | None,
) -> NativeLayoutStrategy:
    layout = str(geometry["layoutHint"])
    row_centers = [int(value) for value in geometry["rowCenters"]]
    row_spacing = float(geometry["medianRowSpacing"])
    pixels_per_mm = detected_pixels_per_mm or (
        width / (paper_speed_mm_per_second * PAGE_DURATION_SECONDS)
    )
    row_local_twelve = layout == "standard_12x1"
    label_anchored_twelve = bool(
        row_local_twelve
        and geometry.get("method") == "standard-and-precordial-lead-label-anchors-v1"
        and (geometry.get("leadLabelValidation") or {}).get("passed")
    )
    labeled_six_method = geometry.get("method") or geometry.get(
        "labelGeometryMethod"
    )
    row_local_labeled_six = bool(
        layout in {"standard_6x2", "standard_6x2_with_r1_ignored"}
        and labeled_six_method
        == "paired-standard-limb-and-v-label-geometry-v1"
        and (geometry.get("precordialLabelValidation") or {}).get("passed")
    )
    row_local_calibration_six = bool(
        layout == "standard_6x2_with_r1_ignored"
        and geometry.get("method") == "repeated-calibration-pulse-row-anchors-v1"
        and int(geometry.get("calibrationRowAnchorCount", 0)) == 7
    )

    if layout in {"standard_6x2", "standard_6x2_with_r1_ignored"}:
        primary_centers = tuple(row_centers[:6])
        primary_bounds = tuple(
            row_bounds(row_centers, height)
            if layout == "standard_6x2_with_r1_ignored"
            else six_row_bounds(row_centers, height)
        )
        rhythm_row_index = 6 if layout == "standard_6x2_with_r1_ignored" else None
        primary_bottom = (
            min(height, round((row_centers[5] + row_centers[6]) / 2))
            if rhythm_row_index is not None
            else height
        )
        fidelity = NativeFidelityPolicy(
            method=(
                "row-local-calibration-anchored-six-by-two-v1"
                if row_local_calibration_six
                else "row-local-labeled-six-by-two-algebra-validated-v1"
                if row_local_labeled_six
                else None
            ),
            minimum_layout_confidence=(
                0.80 if row_local_calibration_six else 0.12 if row_local_labeled_six else 0.45
            ),
            require_rhythm_anchors=not (
                row_local_labeled_six or row_local_calibration_six
            ),
            minimum_evidence_median=0.50 if row_local_labeled_six else 0.70,
            minimum_evidence_p10=(
                0.15 if row_local_calibration_six else 0.0 if row_local_labeled_six else 0.45
            ),
            maximum_crossing_p95_fraction=(
                1.10 if row_local_labeled_six or row_local_calibration_six else 0.45
            ),
            maximum_crossing_jump_fraction=(
                1.55 if row_local_labeled_six or row_local_calibration_six else 1.15
            ),
            unsupported_large_jump_fraction=0.50 if row_local_labeled_six else 0.40,
        )
        return NativeLayoutStrategy(
            layout=layout,
            row_leads=SIX_BY_TWO_ROW_LEADS,
            primary_centers=primary_centers,
            primary_bounds=primary_bounds,
            panel_edges=(0, width // 2, width),
            panel_samples=SIX_BY_TWO_PANEL_SAMPLES,
            pixels_per_mm=pixels_per_mm,
            primary_bottom=primary_bottom,
            rhythm_row_index=rhythm_row_index,
            publish_rhythm_trace=bool(
                layout == "standard_6x2_with_r1_ignored"
                and geometry.get("verifiedRhythmLead") == "II"
            ),
            allow_ink_backed_crossings=(
                row_local_labeled_six or row_local_calibration_six
            ),
            row_local_twelve=False,
            label_anchored_twelve=False,
            row_local_labeled_three=False,
            row_local_labeled_six=row_local_labeled_six,
            row_local_calibration_six=row_local_calibration_six,
            fidelity=fidelity,
        )

    if layout in {"standard_3x4", "standard_3x4_with_r1"}:
        has_rhythm_row = layout == "standard_3x4_with_r1"
        label_validated_three = bool(
            (geometry.get("leadLabelValidation") or {}).get("passed")
            and (geometry.get("leadLabelValidation") or {}).get("order")
            == "standard"
        )
        row_local_labeled_three = bool(
            not has_rhythm_row
            and label_validated_three
        )
        primary_centers = tuple(row_centers[:3])
        primary_bottom = (
            min(height, round((row_centers[2] + row_centers[3]) / 2))
            if has_rhythm_row
            else height
        )
        return NativeLayoutStrategy(
            layout=layout,
            row_leads=THREE_BY_FOUR_ROW_LEADS,
            primary_centers=primary_centers,
            primary_bounds=tuple(
                (
                    max(
                        0,
                        round(
                            center
                            - row_spacing * (2.0 if has_rhythm_row else 1.60)
                        ),
                    ),
                    min(
                        primary_bottom,
                        round(
                            center
                            + row_spacing * (1.5 if has_rhythm_row else 1.60)
                        ),
                    ),
                )
                for center in primary_centers
            ),
            panel_edges=tuple(round(width * index / 4) for index in range(5)),
            panel_samples=THREE_BY_FOUR_PANEL_SAMPLES,
            pixels_per_mm=pixels_per_mm,
            primary_bottom=primary_bottom,
            rhythm_row_index=3 if has_rhythm_row else None,
            publish_rhythm_trace=has_rhythm_row,
            allow_ink_backed_crossings=(
                has_rhythm_row or row_local_labeled_three
            ),
            row_local_twelve=False,
            label_anchored_twelve=False,
            row_local_labeled_three=row_local_labeled_three,
            row_local_labeled_six=False,
            row_local_calibration_six=False,
            fidelity=(
                NativeFidelityPolicy(
                    "rhythm-anchored-label-validated-three-by-four-v1",
                    0.35,
                    True,
                    0.70,
                    0.45,
                    0.45,
                    1.15,
                    0.40,
                )
                if has_rhythm_row and label_validated_three
                else NativeFidelityPolicy(
                    None,
                    0.45,
                    True,
                    0.70,
                    0.45,
                    0.45,
                    1.15,
                    0.40,
                )
                if has_rhythm_row
                else NativeFidelityPolicy(
                    "row-local-label-validated-three-by-four-v1",
                    0.60,
                    False,
                    0.70,
                    0.25,
                    0.45,
                    1.15,
                    0.40,
                )
            ),
        )

    if layout != "standard_12x1":
        raise ValueError(f"Unsupported native layout strategy: {layout}")
    primary_centers = tuple(
        int(value) for value in (geometry.get("traceRowCenters") or row_centers[:12])
    )
    # Sequential traces can contain QRS complexes taller than the distance
    # between adjacent baselines.  Row detection is sufficient to centre the
    # band; OCR is not evidence that a large excursion exists.  Keep the same
    # source-ink reach for labeled and unlabeled 12 x 1 pages so the path is
    # not deterministically clipped before lead identity is adjudicated.
    bounds = sequential_row_bounds(
        list(primary_centers), height, expansion=1.10
    )
    return NativeLayoutStrategy(
        layout=layout,
        row_leads=TWELVE_ROW_LEADS,
        primary_centers=primary_centers,
        primary_bounds=tuple(bounds),
        panel_edges=(0, width),
        panel_samples=SAMPLE_RATE_HZ * PAGE_DURATION_SECONDS,
        pixels_per_mm=pixels_per_mm,
        primary_bottom=height,
        rhythm_row_index=None,
        publish_rhythm_trace=False,
        allow_ink_backed_crossings=True,
        row_local_twelve=True,
        label_anchored_twelve=label_anchored_twelve,
        row_local_labeled_three=False,
        row_local_labeled_six=False,
        row_local_calibration_six=False,
        fidelity=NativeFidelityPolicy(
            "row-local-sequential-label-validated-v3"
            if label_anchored_twelve
            else "row-local-twelve-lead-unverified-v2",
            0.24,
            False,
            0.70,
            0.25,
            0.45,
            1.55,
            0.40,
        ),
    )
