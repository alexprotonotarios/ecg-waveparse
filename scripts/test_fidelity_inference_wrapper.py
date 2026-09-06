from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.nn import functional


ROOT = Path(__file__).resolve().parents[1]
OPEN_ECG_DIR = ROOT / ".external" / "open-ecg-digitizer"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OPEN_ECG_DIR))

from ecg_pipeline.fidelity_inference_wrapper import (
    FidelityInferenceWrapper,
    _GainCalibratedLeadIdentifier,
    dark_ink_probability,
    load_feature_cache,
    save_feature_cache,
)
from src.model.inference_wrapper import InferenceWrapper
from src.model.lead_identifier import LeadIdentifier


class FidelityInferenceWrapperTests(unittest.TestCase):
    def test_nondefault_gain_recovers_voltage_with_the_real_upstream_normalizer(self) -> None:
        identifier = object.__new__(LeadIdentifier)
        identifier.required_valid_samples = 1
        identifier.target_num_samples = 5
        def convert(lines, pixels_per_mm, *, mv_per_mm=.1):
            return identifier.normalize(lines, pixels_per_mm, mv_per_mm)
        for gain in (5., 10., 20.):
            pixels_per_mm = 6.
            # A one-mV source excursion occupies gain * grid spacing pixels.
            source = torch.tensor([[0., 0., gain*pixels_per_mm, 0., 0.]])
            calibrated = _GainCalibratedLeadIdentifier(convert, gain)(source, pixels_per_mm)
            self.assertAlmostEqual(float(calibrated.max()-calibrated.min()), 1000., places=3)
            self.assertEqual(tuple(calibrated.shape), (1, 5))
        for gain in (True, 0., -10., float('nan'), float('inf'), 7.5):
            with self.assertRaises(ValueError): _GainCalibratedLeadIdentifier(convert, gain)
            with self.assertRaises(ValueError): FidelityInferenceWrapper(gain_mm_per_mv=gain)

    def test_resets_candidate_random_state_before_each_inference(self) -> None:
        wrapper = object.__new__(FidelityInferenceWrapper)
        wrapper.forced_layout_substring = None

        with mock.patch.object(
            InferenceWrapper,
            "__call__",
            side_effect=lambda *_args, **_kwargs: {
                "tieBreakers": torch.rand(8)
            },
        ):
            first = wrapper(torch.zeros((1, 3, 2, 2)))["tieBreakers"]
            torch.rand(31)
            second = wrapper(torch.zeros((1, 3, 2, 2)))["tieBreakers"]

        self.assertTrue(torch.equal(first, second))

    def test_separable_support_matches_square_max_pool(self) -> None:
        torch.manual_seed(7)
        image = torch.rand((1, 3, 17, 19), dtype=torch.float32)
        signal = torch.rand((1, 1, 17, 19), dtype=torch.float32)
        grid = torch.rand((1, 1, 17, 19), dtype=torch.float32)
        radius = 4
        expected_support = functional.max_pool2d(
            signal,
            kernel_size=radius * 2 + 1,
            stride=1,
            padding=radius,
        )
        darkness = 1.0 - image.mean(dim=1, keepdim=True)
        expected = (
            torch.clamp(
                (darkness - 0.45) / (1.0 - 0.45), min=0.0, max=1.0
            ).square()
            * (1.0 - grid).square()
            * (expected_support * 2.0).clamp(0.0, 1.0)
            * 0.35
        ).clamp(0.0, 1.0)

        actual = dark_ink_probability(
            image,
            signal,
            grid,
            support_radius=radius,
        )

        self.assertTrue(torch.equal(expected, actual))

    def test_prefers_black_trace_ink_over_a_pale_grid(self) -> None:
        image = torch.ones((1, 3, 5, 5), dtype=torch.float32)
        image[:, :, 2, :] = 0.72
        image[:, :, 1, 2] = 0.0
        signal = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
        signal[:, :, 1, :] = 1.0
        grid = torch.zeros((1, 1, 5, 5), dtype=torch.float32)

        ink = dark_ink_probability(image, signal, grid, support_radius=1)

        self.assertGreater(float(ink[0, 0, 1, 2]), 0.3)
        self.assertEqual(float(ink[0, 0, 2, 0]), 0.0)

    def test_suppresses_pixels_the_model_identifies_as_grid(self) -> None:
        image = torch.zeros((1, 3, 3, 3), dtype=torch.float32)
        signal = torch.ones((1, 1, 3, 3), dtype=torch.float32)
        grid = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
        grid[:, :, 1, 1] = 1.0

        ink = dark_ink_probability(image, signal, grid, support_radius=1)

        self.assertEqual(float(ink[0, 0, 1, 1]), 0.0)
        self.assertGreater(float(ink[0, 0, 0, 0]), 0.3)

    def test_rejects_disconnected_dark_text(self) -> None:
        image = torch.ones((1, 3, 9, 9), dtype=torch.float32)
        image[:, :, 1, 1] = 0.0
        image[:, :, 7, 7] = 0.0
        signal = torch.zeros((1, 1, 9, 9), dtype=torch.float32)
        signal[:, :, 1, 2] = 1.0
        grid = torch.zeros((1, 1, 9, 9), dtype=torch.float32)

        ink = dark_ink_probability(image, signal, grid, support_radius=1)

        self.assertGreater(float(ink[0, 0, 1, 1]), 0.3)
        self.assertEqual(float(ink[0, 0, 7, 7]), 0.0)

    def test_round_trips_valid_feature_maps(self) -> None:
        image = torch.ones((1, 3, 5, 7), dtype=torch.float32)
        feature_maps = tuple(
            torch.full((1, 1, 5, 7), value, dtype=torch.float32)
            for value in (0.1, 0.2, 0.3)
        )
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "features.pt"
            save_feature_cache(cache_path, feature_maps)
            loaded = load_feature_cache(cache_path, image)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        for expected, actual in zip(feature_maps, loaded, strict=True):
            self.assertTrue(torch.equal(expected, actual))

    def test_rejects_cache_with_wrong_image_shape(self) -> None:
        image = torch.ones((1, 3, 5, 7), dtype=torch.float32)
        wrong_shape_maps = tuple(
            torch.ones((1, 1, 4, 7), dtype=torch.float32) for _ in range(3)
        )
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "features.pt"
            save_feature_cache(cache_path, wrong_shape_maps)
            loaded = load_feature_cache(cache_path, image)
            cache_still_exists = cache_path.exists()

        self.assertIsNone(loaded)
        self.assertFalse(cache_still_exists)

    def test_discards_corrupt_cache(self) -> None:
        image = torch.ones((1, 3, 5, 7), dtype=torch.float32)
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "features.pt"
            cache_path.write_bytes(b"not-a-torch-cache")
            loaded = load_feature_cache(cache_path, image)
            cache_still_exists = cache_path.exists()

        self.assertIsNone(loaded)
        self.assertFalse(cache_still_exists)


if __name__ == "__main__":
    unittest.main()
