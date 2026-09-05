from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from ecg_pipeline.native_grid_digitizer import (
    LEAD_ORDER,
    allow_labeled_twelve_global_time_fallback,
    artifact_safe_validity,
    calibration_confirms_labeled_twelve_source_timing,
    contiguous_rhythm_time_range,
    cross_lead_event_anchors,
    digitize_native_grid,
    grid_period_to_pixels_per_mm,
    grid_residual_evidence,
    labeled_six_by_two_time_ranges,
    labeled_three_by_four_time_ranges,
    labeled_twelve_row_time_range,
    labeled_twelve_row_time_ranges,
    labeled_twelve_source_consensus_time_range,
    labeled_twelve_source_consensus_time_ranges,
    labeled_twelve_source_timing_disagrees_with_global,
    native_path_to_canonical,
    page_boundary_clipped_validity,
    recover_rejected_intervals,
    sequential_row_bounds,
    source_boundary_excursion_count,
    six_row_bounds,
    timing_scale_pixels_per_mm,
    trace_crossing_path,
    trace_path,
    validate_limb_lead_order,
)
from ecg_pipeline.native_layout_strategies import build_native_layout_strategy


def synthetic_dense_grid_ecg(
    *,
    width: int = 800,
    height: int = 420,
    row_count: int = 7,
) -> tuple[np.ndarray, list[np.ndarray]]:
    image = np.full((height, width, 3), 248, dtype=np.uint8)
    for x in range(0, width, 4):
        shade = 185 if x % 20 == 0 else 220
        cv2.line(image, (x, 0), (x, height - 1), (shade,) * 3, 1)
    for y in range(0, height, 4):
        shade = 185 if y % 20 == 0 else 220
        cv2.line(image, (0, y), (width - 1, y), (shade,) * 3, 1)

    centers = np.linspace(28, height - 35, row_count).round().astype(int)
    x_values = np.arange(width)
    waveforms: list[np.ndarray] = []
    for row_index, center in enumerate(centers):
        phase = row_index * 0.35
        waveform = (
            center
            + 5 * np.sin(x_values / 12 + phase)
            - 21 * np.exp(-((x_values % 88 - 36) / 3.2) ** 2)
            + 12 * np.exp(-((x_values % 88 - 42) / 4.5) ** 2)
        ).round().astype(np.int32)
        waveforms.append(waveform)
        points = np.column_stack((x_values, waveform)).astype(np.int32)
        cv2.polylines(image, [points], False, (18, 18, 18), 2)
    return image, waveforms


def synthetic_three_by_four_crossing_ecg(
    *,
    width: int = 800,
    height: int = 620,
) -> np.ndarray:
    image = np.full((height, width, 3), 248, dtype=np.uint8)
    for x in range(0, width, 4):
        shade = 185 if x % 20 == 0 else 220
        cv2.line(image, (x, 0), (x, height - 1), (shade,) * 3, 1)
    for y in range(0, height, 4):
        shade = 185 if y % 20 == 0 else 220
        cv2.line(image, (0, y), (width - 1, y), (shade,) * 3, 1)

    centers = (70, 230, 390, 550)
    panel_width = width // 4
    for row_index, center in enumerate(centers[:3]):
        for column in range(4):
            panel_start = column * panel_width
            panel_end = width if column == 3 else (column + 1) * panel_width
            x_values = np.arange(panel_start, panel_end)
            phase = (x_values - panel_start) % 62
            amplitude = 34
            if row_index == 1 and column == 2:
                amplitude = 150
            elif row_index == 2 and column == 2:
                amplitude = -145
            waveform = (
                center
                + 4 * np.sin(x_values / 9 + row_index)
                - amplitude * np.exp(-((phase - 28) / 1.3) ** 2)
                + amplitude * 0.35 * np.exp(-((phase - 31) / 1.8) ** 2)
            ).round().astype(np.int32)
            points = np.column_stack((x_values, waveform)).astype(np.int32)
            cv2.polylines(image, [points], False, (18, 18, 18), 2)

    x_values = np.arange(width)
    rhythm_phase = x_values % 62
    rhythm = (
        centers[3]
        + 4 * np.sin(x_values / 10)
        - 34 * np.exp(-((rhythm_phase - 28) / 2.6) ** 2)
        + 12 * np.exp(-((rhythm_phase - 31) / 3.6) ** 2)
    ).round().astype(np.int32)
    cv2.polylines(
        image,
        [np.column_stack((x_values, rhythm)).astype(np.int32)],
        False,
        (18, 18, 18),
        2,
    )
    return image


class NativeGridDigitizerTests(unittest.TestCase):
    def test_page_boundary_clipping_becomes_missing_not_finite_amplitude(
        self,
    ) -> None:
        evidence = np.zeros((50, 20), dtype=np.float32)
        evidence[-2:, 7:10] = 1.0
        columns = np.arange(5, 12, dtype=np.int32)

        validity, metrics = page_boundary_clipped_validity(
            evidence,
            columns,
            boundary="bottom",
            guard_columns=1,
        )

        self.assertEqual(
            validity.tolist(),
            [True, False, False, False, False, False, True],
        )
        self.assertEqual(metrics["rejectedExcursionCount"], 1)
        self.assertEqual(metrics["rejectedSampleCount"], 5)

    def test_internal_row_boundary_does_not_hide_large_excursion(self) -> None:
        evidence = np.zeros((80, 20), dtype=np.float32)
        evidence[48:50, 7:10] = 1.0
        columns = np.arange(5, 12, dtype=np.int32)

        validity, metrics = page_boundary_clipped_validity(
            evidence,
            columns,
            boundary="bottom",
        )

        self.assertTrue(np.all(validity))
        self.assertEqual(metrics["rejectedExcursionCount"], 0)

    def test_source_boundary_excursions_require_multi_pixel_source_ink(
        self,
    ) -> None:
        evidence = np.zeros((100, 200), dtype=np.float32)
        evidence[:4, 60:63] = 1.0
        evidence[-4:, 120:124] = 1.0
        evidence[:4, 80] = 1.0

        self.assertEqual(
            source_boundary_excursion_count(
                evidence,
                x_start=20,
                x_end=100,
                boundary="top",
            ),
            1,
        )
        self.assertEqual(
            source_boundary_excursion_count(
                evidence,
                x_start=100,
                x_end=160,
                boundary="bottom",
            ),
            1,
        )

    def test_labeled_three_by_four_time_ranges_preserve_margins_and_gaps(
        self,
    ) -> None:
        evidence = np.zeros((480, 1_770), dtype=np.float32)
        row_centers = [100, 250, 400]
        expected = ((126, 501), (543, 918), (960, 1_335), (1_377, 1_752))
        for center in row_centers:
            for start, end in expected:
                evidence[center - 2 : center + 3, start:end] = 1.0
            evidence[center - 25 : center + 26, 18:20] = 1.0
            evidence[center - 25 : center + 26, 48:50] = 1.0

        ranges = labeled_three_by_four_time_ranges(
            evidence,
            row_centers,
            pixels_per_mm=6,
            paper_speed_mm_per_second=25,
        )

        self.assertEqual(ranges, expected)

        rhythm_range = contiguous_rhythm_time_range(
            ranges,
            page_width=evidence.shape[1],
        )
        self.assertEqual(rhythm_range, (126, 1_626))

        columns = np.arange(*rhythm_range)
        path = np.rint(100 + 15 * np.sin(columns / 11)).astype(np.int32)
        canonical = native_path_to_canonical(
            columns,
            path,
            panel_start=rhythm_range[0],
            panel_width=rhythm_range[1] - rhythm_range[0],
            panel_samples=5_000,
            pixels_per_mm=6,
        )
        self.assertTrue(np.isfinite(canonical[0]))
        self.assertTrue(np.isfinite(canonical[-1]))

    def test_labeled_three_by_four_time_ranges_align_each_uneven_panel_gap(
        self,
    ) -> None:
        evidence = np.zeros((480, 1_800), dtype=np.float32)
        row_centers = [100, 250, 400]
        expected = ((126, 501), (543, 918), (975, 1_350), (1_397, 1_772))
        for center in row_centers:
            for start, end in expected:
                evidence[center - 2 : center + 3, start:end] = 1.0

        ranges = labeled_three_by_four_time_ranges(
            evidence,
            row_centers,
            pixels_per_mm=6,
            paper_speed_mm_per_second=25,
        )

        self.assertEqual(ranges, expected)
        self.assertEqual(
            contiguous_rhythm_time_range(
                ranges,
                page_width=evidence.shape[1],
            ),
            (126, 1_626),
        )

    def test_labeled_three_by_four_time_ranges_use_validated_label_edges(self) -> None:
        evidence = np.zeros((480, 1_650), dtype=np.float32)
        row_centers = [100, 250, 400]
        true_ranges = ((84, 459), (459, 834), (834, 1_209), (1_209, 1_584))
        # Simulate label/steep-onset occlusion: shared baseline evidence does
        # not become sustained until 13 pixels after each true time origin.
        for center in row_centers:
            for start, end in true_ranges:
                evidence[center - 2 : center + 3, start + 13 : end] = 1.0

        ranges = labeled_three_by_four_time_ranges(
            evidence,
            row_centers,
            pixels_per_mm=6,
            paper_speed_mm_per_second=25,
            label_left_edges=[90.0, 465.0, 840.0, 1_215.0],
        )

        self.assertEqual(ranges, true_ranges)

    def test_labeled_three_by_four_time_ranges_reject_inconsistent_label_edges(
        self,
    ) -> None:
        evidence = np.zeros((480, 1_770), dtype=np.float32)
        row_centers = [100, 250, 400]
        expected = ((126, 501), (543, 918), (960, 1_335), (1_377, 1_752))
        for center in row_centers:
            for start, end in expected:
                evidence[center - 2 : center + 3, start:end] = 1.0

        ranges = labeled_three_by_four_time_ranges(
            evidence,
            row_centers,
            pixels_per_mm=6,
            paper_speed_mm_per_second=25,
            label_left_edges=[132.0, 507.0, 960.0, 1_383.0],
        )

        self.assertEqual(ranges, expected)

    def test_labeled_six_by_two_time_ranges_exclude_page_margins_and_gutter(
        self,
    ) -> None:
        evidence = np.zeros((300, 1_140), dtype=np.float32)
        row_centers = [35, 80, 125, 170, 215, 260]
        for center in row_centers:
            evidence[center - 1 : center + 2, 70:570] = 1.0
            evidence[center - 1 : center + 2, 640:1_140] = 1.0
        # Repeated calibration pulses must not become time zero.
        evidence[np.asarray(row_centers), 15:45] = 1.0

        ranges = labeled_six_by_two_time_ranges(
            evidence,
            row_centers,
            pixels_per_mm=4,
            paper_speed_mm_per_second=25,
        )

        self.assertEqual(ranges, ((70, 570), (640, 1_140)))

    def test_labeled_six_by_two_time_ranges_use_sustained_row_band_ink(self) -> None:
        evidence = np.zeros((420, 1_644), dtype=np.float32)
        row_centers = [45, 105, 165, 225, 285, 345]
        x_values = np.arange(83, 1_584)
        for row_index, center in enumerate(row_centers):
            waveform = np.rint(
                center
                + 12 * np.sin(x_values / (17 + row_index))
                + 8 * np.sin(x_values / (5 + row_index))
            ).astype(np.int32)
            evidence[waveform, x_values] = 1.0
        # A repeated calibration pulse ahead of the trace must remain too
        # short to establish either five-second panel range.
        for center in row_centers:
            evidence[center - 20 : center + 21, 17:50] = 1.0

        ranges = labeled_six_by_two_time_ranges(
            evidence,
            row_centers,
            pixels_per_mm=5.984,
            paper_speed_mm_per_second=25,
        )

        self.assertEqual(ranges, ((83, 831), (836, 1_584)))

    def test_labeled_six_by_two_time_ranges_resolve_small_grid_overlap_from_source_span(
        self,
    ) -> None:
        evidence = np.zeros((420, 1_644), dtype=np.float32)
        row_centers = [45, 105, 165, 225, 285, 345]
        x_values = np.arange(84, 1_585)
        for row_index, center in enumerate(row_centers):
            waveform = np.rint(
                center
                + 12 * np.sin(x_values / (17 + row_index))
                + 8 * np.sin(x_values / (5 + row_index))
            ).astype(np.int32)
            evidence[waveform, x_values] = 1.0

        ranges = labeled_six_by_two_time_ranges(
            evidence,
            row_centers,
            # The detected grid period is 3.3% high, which would otherwise
            # produce two overlapping 775-pixel panel ranges.
            pixels_per_mm=6.2,
            paper_speed_mm_per_second=25,
        )

        self.assertEqual(ranges, ((84, 834), (834, 1_585)))

    def test_labeled_six_by_two_time_ranges_exclude_left_calibration_band(
        self,
    ) -> None:
        evidence = np.zeros((420, 1_620), dtype=np.float32)
        row_centers = [45, 105, 165, 225, 285, 345]
        # Grid residual inside each broad home-row band makes the page look
        # continuously supported, as on a rotated red-grid scan.
        for center in row_centers:
            evidence[center + 20, :] = 1.0
            # Calibration/label ink merges into the first trace span and
            # would otherwise pull the densest duration window to pixel 32.
            evidence[center - 1 : center + 2, 32:806] = 1.0
            evidence[center - 1 : center + 2, 843:1_591] = 1.0

        ranges = labeled_six_by_two_time_ranges(
            evidence,
            row_centers,
            pixels_per_mm=5.984,
            paper_speed_mm_per_second=25,
        )

        self.assertEqual(ranges, ((58, 806), (843, 1_591)))

    def test_labeled_twelve_row_time_range_excludes_page_margins(self) -> None:
        evidence = np.zeros((720, 1_640), dtype=np.float32)
        row_centers = list(range(35, 696, 60))
        for center in row_centers:
            evidence[center - 2 : center + 3, 84:1_584] = 1.0
        # Repeated calibration pulses must not become time zero.
        evidence[np.asarray(row_centers), 15:45] = 1.0
        # Nor should a short shared artifact ahead of the trace onset.
        evidence[np.asarray(row_centers), 75:77] = 1.0

        time_range = labeled_twelve_row_time_range(
            evidence,
            row_centers,
            pixels_per_mm=6,
            paper_speed_mm_per_second=25,
        )

        self.assertEqual(time_range, (84, 1_584))

    def test_labeled_twelve_source_consensus_recovers_uncalibrated_span(
        self,
    ) -> None:
        evidence = np.zeros((760, 1_644), dtype=np.float32)
        row_centers = list(range(45, 706, 60))
        for center in row_centers:
            evidence[center - 2 : center + 3, 83:1_585] = 1.0
            evidence[center - 2 : center + 3, 17:19] = 1.0
            evidence[center - 2 : center + 3, 1_591] = 1.0

        inferred = labeled_twelve_source_consensus_time_range(
            evidence,
            row_centers,
            raw_grid_period_x=30.04,
            paper_speed_mm_per_second=25,
        )

        self.assertIsNotNone(inferred)
        time_range, diagnostics = inferred
        self.assertEqual(time_range, (83, 1_585))
        self.assertAlmostEqual(diagnostics["inferredPixelsPerMm"], 6.008)
        self.assertEqual(diagnostics["inferredGridScaleMm"], 5.0)
        self.assertFalse(diagnostics["quantitativeCalibrationConfirmed"])

    def test_labeled_twelve_source_consensus_rejects_grid_mismatch(
        self,
    ) -> None:
        evidence = np.zeros((760, 1_644), dtype=np.float32)
        row_centers = list(range(45, 706, 60))
        for center in row_centers:
            evidence[center - 2 : center + 3, 83:1_585] = 1.0

        inferred = labeled_twelve_source_consensus_time_range(
            evidence,
            row_centers,
            raw_grid_period_x=21.0,
            paper_speed_mm_per_second=25,
        )

        self.assertIsNone(inferred)

    def test_source_timing_requires_explicit_pulse_calibration(self) -> None:
        calibrated = {
            "detected": True,
            "confidence": 0.41,
            "gridScaleDetected": True,
            "gridScaleMmX": 5.0,
            "pixelsPerMmX": 6.2,
            "pixelsPerMmY": 5.9,
            "paperSpeedMmPerSecond": 25.0,
            "gainMmPerMv": 10.0,
            "pulseStartX": 18,
            "pulseEndX": 50,
        }
        source_timing = {
            "inferredGridScaleMm": 5.0,
            "inferredPixelsPerMm": 6.008,
            "gridScaleErrorFraction": 0.008,
        }

        self.assertTrue(
            calibration_confirms_labeled_twelve_source_timing(
                calibrated,
                source_timing,
            )
        )
        self.assertFalse(
            calibration_confirms_labeled_twelve_source_timing(
                {**calibrated, "detected": False},
                source_timing,
            )
        )
        self.assertFalse(
            calibration_confirms_labeled_twelve_source_timing(
                {**calibrated, "gridScaleDetected": False},
                source_timing,
            )
        )
        self.assertFalse(
            calibration_confirms_labeled_twelve_source_timing(
                {**calibrated, "pulseEndX": 18},
                source_timing,
            )
        )
        self.assertFalse(
            calibration_confirms_labeled_twelve_source_timing(
                calibrated,
                {**source_timing, "inferredPixelsPerMm": 4.8},
            )
        )

    def test_labeled_twelve_source_consensus_tracks_row_local_rotation(
        self,
    ) -> None:
        evidence = np.zeros((760, 1_644), dtype=np.float32)
        row_centers = list(range(45, 706, 60))
        expected = tuple(
            (83 + index * 2, 1_585 + index)
            for index in range(12)
        )
        for center, (start, end) in zip(
            row_centers,
            expected,
            strict=True,
        ):
            evidence[center - 2 : center + 3, start:end] = 1.0

        inferred = labeled_twelve_source_consensus_time_ranges(
            evidence,
            row_centers,
            raw_grid_period_x=30.0,
            paper_speed_mm_per_second=25,
        )

        self.assertIsNotNone(inferred)
        ranges, diagnostics = inferred
        self.assertEqual(ranges, expected)
        self.assertEqual(diagnostics["correctedRowCount"], 0)
        self.assertEqual(diagnostics["inferredGridScaleMm"], 5.0)

    def test_labeled_twelve_source_consensus_rejects_censored_boundaries(
        self,
    ) -> None:
        evidence = np.zeros((760, 1_644), dtype=np.float32)
        row_centers = list(range(45, 706, 60))
        for center in row_centers:
            evidence[center - 2 : center + 3, 60:1_570] = 1.0

        inferred = labeled_twelve_source_consensus_time_ranges(
            evidence,
            row_centers,
            raw_grid_period_x=30.0,
            paper_speed_mm_per_second=25,
        )

        self.assertIsNone(inferred)

    def test_labeled_twelve_source_consensus_excludes_detected_calibration_pulse(
        self,
    ) -> None:
        evidence = np.zeros((1_400, 1_720), dtype=np.float32)
        row_centers = [100 + index * 108 for index in range(12)]
        columns = np.arange(120, 1_620)
        for center in row_centers:
            pulse = np.array(
                [[53, center], [53, center - 60], [85, center - 60], [85, center]],
                dtype=np.int32,
            )
            waveform = np.column_stack(
                (columns, center + 10 * np.sin((columns - 120) * 0.045))
            ).astype(np.int32)
            cv2.polylines(evidence, [pulse], False, 1.0, 1)
            cv2.polylines(evidence, [waveform], False, 1.0, 1)
        untouched = evidence.copy()

        # The pulse tail and waveform together satisfy the old onset window,
        # even though blank source pixels separate the two components.
        contaminated = labeled_twelve_source_consensus_time_ranges(
            evidence,
            row_centers,
            raw_grid_period_x=6.0,
            paper_speed_mm_per_second=25,
        )
        self.assertIsNotNone(contaminated)
        self.assertTrue(any(start < 120 for start, _ in contaminated[0]))

        corrected = labeled_twelve_source_consensus_time_ranges(
            evidence,
            row_centers,
            raw_grid_period_x=6.0,
            paper_speed_mm_per_second=25,
            calibration_exclusion_end=85,
        )
        self.assertIsNotNone(corrected)
        self.assertEqual(corrected[0], ((120, 1_620),) * 12)
        np.testing.assert_array_equal(evidence, untouched)

    def test_labeled_twelve_source_consensus_detects_vertical_onset(
        self,
    ) -> None:
        evidence = np.zeros((1_360, 1_644), dtype=np.float32)
        row_centers = list(range(111, 1_300, 108))
        for center in row_centers:
            evidence[center - 18 : center + 19, 83:85] = 1.0
            evidence[center + 8 : center + 11, 115:1_585] = 1.0

        inferred = labeled_twelve_source_consensus_time_ranges(
            evidence,
            row_centers,
            raw_grid_period_x=5.739,
            paper_speed_mm_per_second=25,
        )

        self.assertIsNotNone(inferred)
        ranges, diagnostics = inferred
        self.assertEqual(ranges, ((83, 1_585),) * 12)
        self.assertEqual(
            diagnostics["method"],
            "row-local-source-edge-plus-grid-ratio-v2",
        )

    def test_labeled_twelve_source_timing_detects_global_conflict(
        self,
    ) -> None:
        source_ranges = ((83, 1_585),) * 12

        self.assertTrue(
            labeled_twelve_source_timing_disagrees_with_global(
                source_ranges,
                (101, 1_536),
                tolerance_pixels=5,
            )
        )
        self.assertFalse(
            labeled_twelve_source_timing_disagrees_with_global(
                source_ranges,
                (84, 1_584),
                tolerance_pixels=5,
            )
        )

    def test_labeled_twelve_row_time_ranges_follow_perspective(self) -> None:
        evidence = np.zeros((760, 1_700), dtype=np.float32)
        row_centers = list(range(45, 706, 60))
        starts = [34 + index * 8 for index in range(12)]
        row_scales = [6.42 - index * 0.035 for index in range(12)]
        expected: list[tuple[int, int]] = []
        for center, start, scale in zip(
            row_centers, starts, row_scales, strict=True
        ):
            end = start + round(scale * 25 * 10)
            evidence[center - 2 : center + 3, start:end] = 1.0
            expected.append((start, end))
        # Narrow calibration bars ahead of the trace must not become onsets.
        evidence[np.asarray(row_centers), 15:29] = 1.0

        ranges = labeled_twelve_row_time_ranges(
            evidence,
            row_centers,
            row_pixels_per_mm=row_scales,
            paper_speed_mm_per_second=25,
        )

        self.assertEqual(ranges, tuple(expected))

    def test_labeled_twelve_row_time_ranges_exclude_full_calibration_pulses(
        self,
    ) -> None:
        evidence = np.zeros((760, 1_700), dtype=np.float32)
        row_centers = list(range(45, 706, 60))
        starts = [42 + index * 5 for index in range(12)]
        scales = [6.0] * 12
        for center, start in zip(row_centers, starts, strict=True):
            evidence[center - 2 : center + 3, start : start + 1_500] = 1.0
            # A complete 5 mm-wide calibration plateau lies inside the narrow
            # home-row support band and would satisfy the old 3 mm onset test.
            evidence[center - 2 : center + 3, 10:40] = 1.0

        ranges = labeled_twelve_row_time_ranges(
            evidence,
            row_centers,
            row_pixels_per_mm=scales,
            paper_speed_mm_per_second=25,
        )

        self.assertEqual(
            ranges,
            tuple((start, start + 1_500) for start in starts),
        )

        explicitly_excluded = labeled_twelve_row_time_ranges(
            evidence,
            row_centers,
            row_pixels_per_mm=scales,
            paper_speed_mm_per_second=25,
            calibration_exclusion_end=40,
        )
        self.assertEqual(explicitly_excluded, ranges)

    def test_labeled_twelve_row_time_ranges_repair_three_artifact_origins(
        self,
    ) -> None:
        evidence = np.zeros((760, 1_700), dtype=np.float32)
        row_centers = list(range(45, 706, 60))
        starts = [80 + index * 2 for index in range(12)]
        artifact_rows = {6, 9, 11}
        for index, (center, start) in enumerate(
            zip(row_centers, starts, strict=True)
        ):
            evidence[center - 2 : center + 3, start : start + 1_500] = 1.0
            if index in artifact_rows:
                evidence[center - 2 : center + 3, 53:start] = 1.0
        diagnostics: dict[str, object] = {}

        ranges = labeled_twelve_row_time_ranges(
            evidence,
            row_centers,
            row_pixels_per_mm=[6.0] * 12,
            paper_speed_mm_per_second=25,
            calibration_exclusion_end=53,
            timing_diagnostics=diagnostics,
        )

        self.assertEqual(
            ranges,
            tuple((start, start + 1_500) for start in starts),
        )
        self.assertTrue(diagnostics["accepted"])
        self.assertTrue(diagnostics["applied"])
        self.assertEqual(diagnostics["correctedRowCount"], 3)
        self.assertEqual(
            [diagnostics["rawStarts"][index] for index in sorted(artifact_rows)],
            [53, 53, 53],
        )

    def test_labeled_twelve_row_time_ranges_reject_four_artifact_origins(
        self,
    ) -> None:
        evidence = np.zeros((760, 1_700), dtype=np.float32)
        row_centers = list(range(45, 706, 60))
        starts = [80 + index * 2 for index in range(12)]
        artifact_rows = {5, 7, 9, 11}
        for index, (center, start) in enumerate(
            zip(row_centers, starts, strict=True)
        ):
            evidence[center - 2 : center + 3, start : start + 1_500] = 1.0
            if index in artifact_rows:
                evidence[center - 2 : center + 3, 53:start] = 1.0
        diagnostics: dict[str, object] = {}

        ranges = labeled_twelve_row_time_ranges(
            evidence,
            row_centers,
            row_pixels_per_mm=[6.0] * 12,
            paper_speed_mm_per_second=25,
            calibration_exclusion_end=53,
            timing_diagnostics=diagnostics,
        )

        self.assertIsNone(ranges)
        self.assertFalse(diagnostics["accepted"])
        self.assertFalse(diagnostics["applied"])
        self.assertEqual(diagnostics["correctedRowCount"], 0)
        self.assertFalse(
            allow_labeled_twelve_global_time_fallback(
                ranges,
                diagnostics,
                grid_scale_detected=True,
                raw_grid_period_scale_mm=5.0,
            )
        )

    def test_labeled_twelve_global_time_fallback_remains_available_without_rejection(
        self,
    ) -> None:
        self.assertTrue(
            allow_labeled_twelve_global_time_fallback(
                None,
                {},
                grid_scale_detected=True,
                raw_grid_period_scale_mm=None,
            )
        )

    def test_grid_period_requires_explicit_physical_grid_scale(self) -> None:
        self.assertEqual(
            grid_period_to_pixels_per_mm(16, period_scale_mm=1.0),
            16,
        )
        self.assertEqual(
            grid_period_to_pixels_per_mm(20, period_scale_mm=5.0),
            4,
        )
        self.assertIsNone(
            grid_period_to_pixels_per_mm(20, period_scale_mm=None)
        )

    def test_twelve_row_timing_does_not_change_six_by_two_calibration(self) -> None:
        common = {
            "row_pixels_per_mm_x": [],
            "calibrated_pixels_per_mm": 5.3,
            "raw_grid_period_x": 5.12,
            "raw_grid_period_scale_mm": 1.0,
        }

        self.assertEqual(
            timing_scale_pixels_per_mm(**common, prefer_raw_grid=False),
            5.3,
        )
        self.assertEqual(
            timing_scale_pixels_per_mm(**common, prefer_raw_grid=True),
            5.12,
        )
        self.assertEqual(
            timing_scale_pixels_per_mm(
                **{
                    **common,
                    "raw_grid_period_x": 16.0,
                    "raw_grid_period_scale_mm": None,
                },
                prefer_raw_grid=True,
            ),
            5.3,
        )

    def test_label_anchored_twelve_row_bounds_use_trace_centers(self) -> None:
        strategy = build_native_layout_strategy(
            {
                "layoutHint": "standard_12x1",
                "rowCenters": list(range(40, 700, 55)),
                "traceRowCenters": list(range(70, 730, 55)),
                "medianRowSpacing": 55,
                "method": "standard-and-precordial-lead-label-anchors-v1",
                "leadLabelValidation": {"passed": True, "order": "standard"},
            },
            width=1_000,
            height=760,
            paper_speed_mm_per_second=25,
            detected_pixels_per_mm=4,
        )

        self.assertEqual(strategy.primary_centers[0], 70)
        self.assertGreater(strategy.primary_bounds[0][1], 100)

    def test_layout_strategies_own_panel_geometry_and_rhythm_policy(self) -> None:
        three_by_four = build_native_layout_strategy(
            {
                "layoutHint": "standard_3x4_with_r1",
                "rowCenters": [80, 200, 320, 440],
                "medianRowSpacing": 120,
            },
            width=1000,
            height=500,
            paper_speed_mm_per_second=25,
            detected_pixels_per_mm=4,
        )
        six_by_two = build_native_layout_strategy(
            {
                "layoutHint": "standard_6x2_with_r1_ignored",
                "rowCenters": [50, 100, 150, 200, 250, 300, 360],
                "medianRowSpacing": 50,
            },
            width=1000,
            height=400,
            paper_speed_mm_per_second=25,
            detected_pixels_per_mm=4,
        )

        self.assertEqual(three_by_four.panel_edges, (0, 250, 500, 750, 1000))
        self.assertEqual(three_by_four.panel_samples, 1250)
        self.assertTrue(three_by_four.publish_rhythm_trace)
        self.assertEqual(six_by_two.panel_edges, (0, 500, 1000))
        self.assertEqual(six_by_two.panel_samples, 2500)
        self.assertFalse(six_by_two.publish_rhythm_trace)

        verified_six_by_two = build_native_layout_strategy(
            {
                "layoutHint": "standard_6x2_with_r1_ignored",
                "rowCenters": [50, 100, 150, 200, 250, 300, 360],
                "medianRowSpacing": 50,
                "verifiedRhythmLead": "II",
            },
            width=1000,
            height=400,
            paper_speed_mm_per_second=25,
            detected_pixels_per_mm=4,
        )
        self.assertTrue(verified_six_by_two.publish_rhythm_trace)

        labeled_rhythm = build_native_layout_strategy(
            {
                "layoutHint": "standard_6x2_with_r1_ignored",
                "rowCenters": [50, 100, 150, 200, 250, 300, 360],
                "medianRowSpacing": 50,
                "labelGeometryMethod": (
                    "paired-standard-limb-and-v-label-geometry-v1"
                ),
                "precordialLabelValidation": {"passed": True},
            },
            width=1000,
            height=400,
            paper_speed_mm_per_second=25,
            detected_pixels_per_mm=4,
        )
        self.assertTrue(labeled_rhythm.row_local_labeled_six)
        self.assertTrue(labeled_rhythm.allow_ink_backed_crossings)
        self.assertEqual(
            labeled_rhythm.fidelity.method,
            "row-local-labeled-six-by-two-algebra-validated-v1",
        )

    def test_plain_three_by_four_strategy_requires_explicit_standard_labels(
        self,
    ) -> None:
        strategy = build_native_layout_strategy(
            {
                "layoutHint": "standard_3x4",
                "rowCenters": [100, 250, 400],
                "medianRowSpacing": 150,
                "method": "three-row-four-panel-geometry-v1",
                "leadLabelValidation": {
                    "passed": True,
                    "order": "standard",
                },
            },
            width=1_000,
            height=500,
            paper_speed_mm_per_second=25,
            detected_pixels_per_mm=4,
        )

        self.assertEqual(strategy.layout, "standard_3x4")
        self.assertEqual(strategy.panel_edges, (0, 250, 500, 750, 1000))
        self.assertIsNone(strategy.rhythm_row_index)
        self.assertFalse(strategy.publish_rhythm_trace)
        self.assertTrue(strategy.allow_ink_backed_crossings)
        self.assertTrue(strategy.row_local_labeled_three)
        self.assertEqual(
            strategy.fidelity.method,
            "row-local-label-validated-three-by-four-v1",
        )

        labeled_rhythm = build_native_layout_strategy(
            {
                "layoutHint": "standard_3x4_with_r1",
                "rowCenters": [100, 250, 400, 575],
                "medianRowSpacing": 150,
                "leadLabelValidation": {
                    "passed": True,
                    "order": "standard",
                },
            },
            width=1_000,
            height=650,
            paper_speed_mm_per_second=25,
            detected_pixels_per_mm=4,
        )
        self.assertFalse(labeled_rhythm.row_local_labeled_three)
        self.assertTrue(labeled_rhythm.fidelity.require_rhythm_anchors)
        self.assertEqual(
            labeled_rhythm.fidelity.method,
            "rhythm-anchored-label-validated-three-by-four-v1",
        )
        self.assertEqual(
            labeled_rhythm.fidelity.minimum_layout_confidence,
            0.35,
        )

    def test_sequential_twelve_row_strategy_requires_label_anchors(self) -> None:
        common = {
            "layoutHint": "standard_12x1",
            "rowCenters": list(range(30, 390, 30)),
            "traceRowCenters": list(range(30, 390, 30)),
            "medianRowSpacing": 30,
        }
        labeled = build_native_layout_strategy(
            {
                **common,
                "method": "standard-and-precordial-lead-label-anchors-v1",
                "leadLabelValidation": {
                    "passed": True,
                    "order": "standard",
                },
            },
            width=1000,
            height=420,
            paper_speed_mm_per_second=25,
            detected_pixels_per_mm=4,
        )
        unlabeled = build_native_layout_strategy(
            common,
            width=1000,
            height=420,
            paper_speed_mm_per_second=25,
            detected_pixels_per_mm=4,
        )

        self.assertEqual(
            labeled.fidelity.method,
            "row-local-sequential-label-validated-v3",
        )
        self.assertTrue(labeled.label_anchored_twelve)
        self.assertEqual(
            unlabeled.fidelity.method,
            "row-local-twelve-lead-unverified-v2",
        )
        self.assertFalse(unlabeled.label_anchored_twelve)

    def test_crossing_path_rewards_only_connected_vertical_ink(self) -> None:
        evidence = np.zeros((140, 100), dtype=np.float32)
        # A lower-confidence centre pixel represents the ambiguity created
        # when a near-vertical QRS stroke crosses the nominal baseline.
        evidence[70, :] = 0.72
        evidence[30:71, 45] = 1.0
        evidence[20:71, 46] = 1.0
        evidence[20:111, 47] = 1.0
        evidence[70:111, 48] = 1.0
        evidence[70, 49:] = 1.0

        _, baseline_path = trace_crossing_path(
            evidence,
            y_start=5,
            y_end=135,
            x_start=2,
            x_end=98,
            row_center=70,
            row_spacing=50,
        )
        _, connected_path = trace_crossing_path(
            evidence,
            y_start=5,
            y_end=135,
            x_start=2,
            x_end=98,
            row_center=70,
            row_spacing=50,
            transition_scale=2.0,
            ink_connected_displacement_reward=0.015,
        )

        self.assertTrue(np.all(baseline_path == 70))
        self.assertGreaterEqual(int(np.max(connected_path)), 78)

        disconnected = np.zeros_like(evidence)
        disconnected[70, :] = 1.0
        disconnected[15:40, 45:49] = 1.0
        _, disconnected_path = trace_crossing_path(
            disconnected,
            y_start=5,
            y_end=135,
            x_start=2,
            x_end=98,
            row_center=70,
            row_spacing=50,
            transition_scale=2.0,
            ink_connected_displacement_reward=0.015,
        )

        self.assertTrue(np.all(disconnected_path == 70))

    def test_cross_lead_events_require_two_independent_nonartifact_peers(
        self,
    ) -> None:
        columns = np.arange(20, 180, dtype=np.int32)
        baseline = np.full(columns.size, 100, dtype=np.int32)

        def path_with_events(*event_columns: int) -> np.ndarray:
            path = baseline.copy()
            for event_column in event_columns:
                index = int(event_column - columns[0])
                path[index : index + 3] = (100, 65, 100)
            return path

        paths = {
            "II": (columns, path_with_events(70, 130), np.ones(columns.size, dtype=bool)),
            "III": (columns, path_with_events(72, 130), np.ones(columns.size, dtype=bool)),
            "aVF": (columns, path_with_events(105), np.ones(columns.size, dtype=bool)),
        }
        artifact_columns = np.zeros(200, dtype=bool)
        artifact_columns[125:136] = True

        anchors = cross_lead_event_anchors(
            paths,
            ["II", "III", "aVF"],
            panel_start=20,
            panel_end=180,
            row_spacing=100,
            tolerance_pixels=4,
            artifact_columns=artifact_columns,
            minimum_other_leads=2,
        )

        self.assertEqual(len(anchors), 1)
        self.assertLessEqual(abs(int(anchors[0]) - 71), 1)

    def test_labeled_six_by_two_rejects_isolated_source_ink_excursion(
        self,
    ) -> None:
        height = 360
        width = 1_100
        row_centers = [35, 92, 149, 206, 263, 320]
        evidence = np.zeros((height, width), dtype=np.uint8)
        panel_ranges = ((50, 550), (600, 1_100))
        for center in row_centers:
            for start, end in panel_ranges:
                evidence[center - 1 : center + 2, start:end] = 255
                for event in range(start + 45, end - 20, 80):
                    evidence[center - 22 : center + 2, event - 1 : event + 2] = 255

        # Remove V2's home-row ink around one second-panel column and replace
        # it with a connected, arrow-like excursion that no other lead shares.
        artifact_x = 730
        v2_center = row_centers[1]
        evidence[
            v2_center - 3 : v2_center + 30,
            artifact_x - 12 : artifact_x + 13,
        ] = 0
        artifact_columns = np.arange(artifact_x - 12, artifact_x + 13)
        artifact_path = np.rint(
            v2_center
            + 27 * np.sin(np.linspace(0, np.pi, artifact_columns.size))
        ).astype(np.int32)
        for column, y_value in zip(
            artifact_columns,
            artifact_path,
            strict=True,
        ):
            evidence[y_value - 1 : y_value + 2, column] = 255

        image = np.full((height, width, 3), 255, dtype=np.uint8)
        image[evidence > 0] = (15, 15, 15)
        geometry = {
            "layoutHint": "standard_6x2",
            "confidence": 0.92,
            "method": "paired-standard-limb-and-v-label-geometry-v1",
            "rowCenters": row_centers,
            "medianRowSpacing": 57,
            "precordialLabelValidation": {
                "passed": True,
                "method": "test-label-sequence",
            },
            "limbLabelValidation": {
                "passed": True,
                "order": "standard",
            },
            "calibration": {
                "detected": True,
                "confidence": 0.9,
                "gridScaleDetected": True,
                "gridScaleConfidence": 0.9,
                "pixelsPerMmX": 4.0,
                "pixelsPerMmY": 4.0,
                "paperSpeedMmPerSecond": 25,
                "gainMmPerMv": 10,
            },
        }
        algebra = {
            "passed": True,
            "selectedOrder": "standard",
            "best": {"order": "standard"},
            "runnerUp": {"order": "cabrera"},
        }
        with (
            patch(
                "ecg_pipeline.native_grid_digitizer.detect_ecg_layout_geometry",
                return_value=geometry,
            ),
            patch(
                "ecg_pipeline.native_grid_digitizer.validate_limb_lead_order",
                return_value=algebra,
            ),
            tempfile.TemporaryDirectory() as directory,
        ):
            result = digitize_native_grid(
                image,
                image.copy(),
                Path(directory),
                trace_evidence=evidence,
            )

        v2 = result["sourceFidelity"]["leadMetrics"]["V2"]
        self.assertGreaterEqual(v2["largeExcursionCount"], 1)
        self.assertGreaterEqual(v2["rejectedExcursionCount"], 1)
        self.assertGreater(v2["rejectedSampleCount"], 0)
        self.assertLess(v2["coverage"], 1.0)

    def test_sequential_row_bounds_overlap_for_large_full_width_qrs(self) -> None:
        bounds = sequential_row_bounds([20, 50, 80], 100)

        self.assertEqual(bounds, [(0, 45), (25, 75), (55, 100)])

    def test_validates_standard_and_cabrera_limb_order_from_lead_algebra(self) -> None:
        x = np.arange(500, dtype=np.float64)
        lead_i = 120 * np.sin(x / 23) + 350 * np.exp(-((x % 95 - 40) / 4) ** 2)
        lead_iii = 80 * np.sin(x / 31 + 0.4) - 180 * np.exp(-((x % 95 - 43) / 5) ** 2)
        lead_ii = lead_i + lead_iii
        avr = -(lead_i + lead_ii) / 2
        avl = lead_i - lead_ii / 2
        avf = lead_ii - lead_i / 2

        standard = validate_limb_lead_order(
            [lead_i, lead_ii, lead_iii, avr, avl, avf]
        )
        cabrera = validate_limb_lead_order(
            [avl, lead_i, -avr, lead_ii, avf, lead_iii]
        )

        self.assertTrue(standard["passed"])
        self.assertEqual(standard["selectedOrder"], "standard")
        self.assertTrue(cabrera["passed"])
        self.assertEqual(cabrera["selectedOrder"], "cabrera")

    def test_plain_six_row_bounds_exclude_page_headers_and_footers(self) -> None:
        bounds = six_row_bounds([51, 84, 114, 148, 177, 211], 239)

        self.assertEqual(bounds[0], (35, 68))
        self.assertEqual(bounds[-1], (194, 227))

    def test_digitizes_a_compact_six_by_two_page_without_a_rhythm_strip(self) -> None:
        image, _ = synthetic_dense_grid_ecg(
            width=360,
            height=420,
            row_count=6,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = digitize_native_grid(
                image,
                image.copy(),
                Path(directory),
            )

        self.assertEqual(result["layout"], "standard_6x2")
        self.assertEqual(
            set(result["sourceFidelity"]["leadMetrics"]),
            set(LEAD_ORDER),
        )
        self.assertGreaterEqual(result["sourceFidelity"]["minimumCoverage"], 0.85)
        self.assertAlmostEqual(
            result["sourceFidelity"]["pixelsPerMm"],
            4.0,
            delta=0.15,
        )

    def test_six_row_waveform_time_excludes_margins_without_label_anchors(self) -> None:
        # Geometry may come from repeated rows rather than recognized labels.
        # Moving the same calibrated trace on the page must not alter its time.
        centers = [70 + row * 90 for row in range(6)]
        geometry = {
            "layoutHint": "standard_6x2",
            "confidence": 0.9,
            "rowCenters": centers,
            "medianRowSpacing": 90,
            "calibration": {
                "detected": True,
                "confidence": 0.9,
                "pixelsPerMmX": 4.0,
                "pixelsPerMmY": 4.0,
                "paperSpeedMmPerSecond": 25.0,
                "gainMmPerMv": 10.0,
            },
        }
        for left_margin in (65, 155):
            with self.subTest(left_margin=left_margin):
                image = np.full((590, left_margin + 1_050, 3), 250, np.uint8)
                native_time = np.arange(1_000) / 100.0
                displacement = 12 * np.sin(2 * np.pi * 1.6 * native_time)
                for center in centers:
                    cv2.polylines(
                        image,
                        [np.column_stack((
                            left_margin + np.arange(1_000),
                            np.rint(center - displacement),
                        )).astype(np.int32)],
                        False, (18, 18, 18), 1,
                    )
                    cv2.polylines(
                        image,
                        [np.array([
                            [10, center], [10, center - 40],
                            [30, center - 40], [30, center],
                        ], dtype=np.int32)],
                        False, (18, 18, 18), 1,
                    )
                with tempfile.TemporaryDirectory() as directory, patch(
                    "ecg_pipeline.native_grid_digitizer.detect_ecg_layout_geometry",
                    return_value=geometry,
                ):
                    result = digitize_native_grid(image, image.copy(), Path(directory))
                    values = np.genfromtxt(
                        result["canonicalPath"], delimiter=",", names=True,
                    )["I"][:2_500]

                expected = 300 * np.sin(2 * np.pi * 1.6 * np.arange(2_500) / 500)
                finite = np.isfinite(values)
                self.assertGreater(float(np.mean(finite)), 0.98)
                self.assertLess(
                    float(np.sqrt(np.mean((values[finite] - expected[finite]) ** 2))),
                    40.0,
                )

    def test_native_path_follows_trace_in_dense_grid(self) -> None:
        image, waveforms = synthetic_dense_grid_ecg()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        evidence = grid_residual_evidence(gray)
        columns, path = trace_path(
            evidence,
            y_start=0,
            y_end=55,
            x_start=20,
            x_end=380,
            row_center=28,
        )
        error = np.abs(path - waveforms[0][columns])
        self.assertLess(float(np.median(error)), 1.5)
        self.assertLess(float(np.quantile(error, 0.95)), 4.0)

    def test_writes_real_overlay_and_truthful_edge_gaps(self) -> None:
        image, _ = synthetic_dense_grid_ecg()
        with tempfile.TemporaryDirectory() as directory:
            result = digitize_native_grid(
                image,
                image.copy(),
                Path(directory),
            )
            self.assertTrue(result["sourceFidelity"]["passed"])
            self.assertEqual(
                set(result["sourceFidelity"]["leadMetrics"]),
                set(LEAD_ORDER),
            )

            overlay = cv2.imread(result["diagnosticPath"], cv2.IMREAD_COLOR)
            self.assertIsNotNone(overlay)
            self.assertGreater(int(np.count_nonzero(overlay != image)), 1_000)

            with Path(result["canonicalPath"]).open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(tuple(rows[0]), LEAD_ORDER)
            self.assertEqual(len(rows) - 1, 5_000)
            for lead_index, lead in enumerate(LEAD_ORDER):
                panel_start = 0 if lead_index < 6 else 2_500
                values = [
                    row[lead_index]
                    for row in rows[1 + panel_start : 1 + panel_start + 2_500]
                ]
                finite = sum(bool(value) for value in values)
                self.assertGreaterEqual(finite, 2_125)
                self.assertLess(finite, 2_475)
                self.assertEqual(values[0], "")

    def test_publishes_a_full_width_six_by_two_rhythm_only_when_verified(self) -> None:
        image, _ = synthetic_dense_grid_ecg()
        with tempfile.TemporaryDirectory() as directory:
            result = digitize_native_grid(
                image,
                image.copy(),
                Path(directory),
                verified_rhythm_lead="II",
            )
            with Path(result["canonicalPath"]).open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.reader(handle))

        lead_ii_index = LEAD_ORDER.index("II")
        rhythm = [row[lead_ii_index] for row in rows[1:]]
        self.assertGreaterEqual(sum(bool(value) for value in rhythm), 4_250)
        self.assertTrue(
            result["sourceFidelity"]["rhythmLeadValidation"]["passed"]
        )

    def test_accepts_a_source_sized_preprocessed_trace_probability(self) -> None:
        image, _ = synthetic_dense_grid_ecg()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        trace_evidence = np.rint(
            grid_residual_evidence(gray) * 255
        ).astype(np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            result = digitize_native_grid(
                image,
                image.copy(),
                Path(directory),
                trace_evidence=trace_evidence,
            )

        self.assertEqual(
            result["sourceFidelity"]["method"],
            "rhythm-anchored-connected-colour-plus-raw-grid-path-v2",
        )
        self.assertTrue(result["sourceFidelity"]["passed"])

    def test_three_by_four_preserves_row_crossing_qrs(self) -> None:
        image = synthetic_three_by_four_crossing_ecg()
        with tempfile.TemporaryDirectory() as directory:
            result = digitize_native_grid(
                image,
                image.copy(),
                Path(directory),
            )
            self.assertEqual(result["layout"], "standard_3x4_with_r1")
            self.assertTrue(result["sourceFidelity"]["passed"])
            self.assertGreaterEqual(
                result["sourceFidelity"]["rhythmAnchorCount"],
                8,
            )
            self.assertGreater(
                result["sourceFidelity"]["maxNativeJumpPixels"],
                25,
            )

            with Path(result["canonicalPath"]).open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(len(rows) - 1, 5_000)
            layout_columns = {
                "I": 0,
                "II": 0,
                "III": 0,
                "aVR": 1,
                "aVL": 1,
                "aVF": 1,
                "V1": 2,
                "V2": 2,
                "V3": 2,
                "V4": 3,
                "V5": 3,
                "V6": 3,
            }
            values_by_lead: dict[str, np.ndarray] = {}
            for lead_index, lead in enumerate(LEAD_ORDER):
                values = np.asarray(
                    [
                        float(row[lead_index]) if row[lead_index] else np.nan
                        for row in rows[1:]
                    ],
                    dtype=np.float64,
                )
                values_by_lead[lead] = values
                if lead == "II":
                    self.assertGreaterEqual(
                        int(np.isfinite(values).sum()),
                        4_250,
                    )
                    continue
                start = layout_columns[lead] * 1_250
                expected = values[start : start + 1_250]
                self.assertGreaterEqual(int(np.isfinite(expected).sum()), 1_062)
                self.assertEqual(int(np.isfinite(values[:start]).sum()), 0)
                self.assertEqual(
                    int(np.isfinite(values[start + 1_250 :]).sum()),
                    0,
                )

            self.assertGreater(
                float(np.nanmax(values_by_lead["V2"]) - np.nanmin(values_by_lead["V2"])),
                2 * float(
                    np.nanmax(values_by_lead["II"])
                    - np.nanmin(values_by_lead["II"])
                ),
            )
            self.assertGreater(
                float(np.nanmax(values_by_lead["V3"]) - np.nanmin(values_by_lead["V3"])),
                2 * float(
                    np.nanmax(values_by_lead["III"])
                    - np.nanmin(values_by_lead["III"])
                ),
            )

    def test_masks_an_unanchored_connected_vertical_excursion(self) -> None:
        height = 240
        width = 180
        row_center = 120
        row_spacing = 100
        columns = np.arange(10, 170, dtype=np.int32)
        path = np.full(columns.size, row_center, dtype=np.int32)
        path[38:42] = np.asarray((90, 45, 45, 90), dtype=np.int32)
        path[98:102] = np.asarray((90, 45, 45, 90), dtype=np.int32)
        evidence = np.zeros((height, width), dtype=np.float32)
        evidence[row_center - 2 : row_center + 3, :] = 1.0
        for start in (38, 98):
            x0 = int(columns[start])
            x1 = int(columns[start + 3])
            evidence[43 : row_center + 3, x0 - 2 : x1 + 3] = 1.0

        validity, safety = artifact_safe_validity(
            evidence,
            columns,
            path,
            row_center=row_center,
            row_spacing=row_spacing,
            qrs_anchors=np.asarray([columns[99]], dtype=np.int32),
            qrs_anchor_shift=0,
            qrs_anchor_matches=1,
            artifact_columns=np.zeros(width, dtype=bool),
            effective_sample_rate_hz=100.0,
        )

        self.assertFalse(bool(np.all(validity[35:46])))
        self.assertTrue(bool(np.all(validity[95:106])))
        self.assertGreaterEqual(safety["rejectedExcursionCount"], 1)
        self.assertGreater(safety["rejectedSampleCount"], 0)

    def test_recovers_only_source_backed_rejected_intervals(self) -> None:
        width = 120
        row_center = 60
        row_spacing = 100
        columns = np.arange(width, dtype=np.int32)
        path = np.full(width, row_center, dtype=np.int32)
        path[20:27] = 15
        path[70:77] = 15
        validity = np.ones(width, dtype=bool)
        validity[20:27] = False
        validity[70:77] = False
        strict_path = np.full(width, row_center, dtype=np.int32)
        evidence = np.zeros((120, width), dtype=np.float32)
        evidence[row_center, 20:27] = 1.0

        recovered_path, recovered_validity, safety = (
            recover_rejected_intervals(
                evidence,
                columns,
                path,
                validity,
                strict_path,
                {
                    "rejectedExcursionCount": 2,
                    "rejectedSampleCount": 14,
                },
                row_center=row_center,
                row_spacing=row_spacing,
            )
        )

        self.assertTrue(bool(np.all(recovered_validity[20:27])))
        self.assertTrue(bool(np.all(recovered_path[20:27] == row_center)))
        self.assertFalse(bool(np.any(recovered_validity[70:77])))
        self.assertEqual(safety["recoveredExcursionCount"], 1)
        self.assertEqual(safety["recoveredSampleCount"], 7)
        self.assertEqual(safety["rejectedExcursionCount"], 1)

    def test_rhythm_trace_ignores_long_strokes_from_primary_panels(self) -> None:
        image = synthetic_three_by_four_crossing_ecg()
        for x in (430, 492, 554):
            cv2.line(image, (x, 150), (x, 555), (18, 18, 18), 3)

        with tempfile.TemporaryDirectory() as directory:
            result = digitize_native_grid(
                image,
                image.copy(),
                Path(directory),
            )
            self.assertTrue(result["sourceFidelity"]["passed"])
            self.assertTrue(result["sourceFidelity"]["rhythmTracePassed"])
            self.assertGreaterEqual(
                result["sourceFidelity"]["rhythmAnchorCount"],
                8,
            )

            with Path(result["canonicalPath"]).open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.reader(handle))
            lead_ii_index = LEAD_ORDER.index("II")
            rhythm = np.asarray(
                [
                    float(row[lead_ii_index]) if row[lead_ii_index] else np.nan
                    for row in rows[1:]
                ],
                dtype=np.float64,
            )
            self.assertGreaterEqual(int(np.isfinite(rhythm).sum()), 4_250)
            self.assertLess(
                float(np.nanmax(rhythm) - np.nanmin(rhythm)),
                2_000.0,
            )


if __name__ == "__main__":
    unittest.main()
