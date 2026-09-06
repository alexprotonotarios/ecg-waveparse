from __future__ import annotations

import json
import io
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ecg_pipeline.printed_calibration import decode_source_raster, read_printed_settings, reconcile_printed_settings
from scripts.prepare_ecg_input import inspect_source, load_working_image


def observation(text: str, confidence: float = .99):
    return {"x": .1, "y": .8, "width": .7, "height": .1,
            "candidates": [{"text": text, "confidence": confidence}]}


def read(text: str, confidence: float = .99):
    return read_printed_settings(np.full((200, 400, 3), 255, np.uint8), ocr_runner=lambda _: {
        "engine": "fixture", "observations": [observation(text, confidence)],
    })


def calibration(speed: float = 25, gain: float = 10):
    return {"detected": True, "paperSpeedMmPerSecond": speed, "gainMmPerMv": gain,
            "pixelsPerMmX": 4., "pixelsPerMmY": 4., "confidence": .9, "method": "grid-pulse"}


class PrintedCalibrationTests(unittest.TestCase):
    def test_exif_rotations_and_mirrors_keep_admitted_raster_coordinates(self):
        raster = np.zeros((80, 160, 3), dtype=np.uint8)
        raster[:30, :50] = [255, 20, 40]
        raster[40:, 100:] = [30, 200, 60]
        with tempfile.TemporaryDirectory() as directory:
            for orientation in range(1, 9):
                with self.subTest(orientation=orientation):
                    exif = Image.Exif(); exif[274] = orientation
                    stream = io.BytesIO()
                    Image.fromarray(raster).save(stream, 'JPEG', quality=95, exif=exif)
                    original = stream.getvalue()
                    source = Path(directory) / f'orientation-{orientation}.jpg'
                    source.write_bytes(original)
                    admitted = inspect_source(source)
                    working, _ = load_working_image(source, 2400)
                    decoded = decode_source_raster(original)
                    self.assertEqual(decoded.shape[:2], (admitted['height'], admitted['width']))
                    np.testing.assert_array_equal(decoded[..., ::-1], working)
                    settings = read_printed_settings(decoded, ocr_runner=lambda _: {
                        'observations': [observation('25 mm/s 10 mm/mV')]})
                    self.assertEqual(settings['sourceSize'], {'width': 160, 'height': 80})
                    self.assertAlmostEqual(settings['observations'][0]['sourceBox']['left'], 16)
                    self.assertEqual(source.read_bytes(), original)

    def test_units_and_original_boxes_survive_bounded_ocr_without_patient_text(self):
        image = np.full((2000, 4000, 3), 255, np.uint8)
        def runner(working):
            self.assertEqual(working.shape[:2], (1200, 2400))
            return {"engine": "fixture", "observations": [observation("Patient PRIVATE NAME; 25 mm/s 10 mm/mV")]}
        result = read_printed_settings(image, ocr_runner=runner)
        self.assertEqual(result["values"], {"speed": [25.], "gain": [10.]})
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertNotIn("Patient", json.dumps(result))
        box = result["observations"][0]["sourceBox"]
        for key, expected in {"left": 400, "top": 200, "right": 3200, "bottom": 400}.items():
            self.assertAlmostEqual(box[key], expected)

    def test_all_supported_settings_corroborate_without_becoming_acquisition_evidence(self):
        for speed in (25, 50):
            for gain in (5, 10, 20):
                with self.subTest(speed=speed, gain=gain):
                    original = calibration(speed, gain)
                    result = reconcile_printed_settings(original, read(f"{speed}mm/sec {gain}mm/mV"))
                    self.assertEqual(result["reconciliation"]["state"], "corroborated_inference")
                    self.assertFalse(result["reconciliation"]["quantitativeBlocked"])
                    self.assertIsNone(result["reconciliation"]["acquisitionSampleRateHz"])
                    self.assertNotIn("reconciliation", original)

    def test_conflicting_high_confidence_alternatives_refuse_physical_export(self):
        item = observation("25 mm/s 10 mm/mV")
        item["candidates"].append({"text": "50 mm/s 10 mm/mV", "confidence": .95})
        printed = read_printed_settings(np.ones((200, 400), np.uint8), ocr_runner=lambda _: {
            "observations": [item]})
        result = reconcile_printed_settings(calibration(), printed)
        self.assertEqual(result["reconciliation"]["state"], "conflict")
        self.assertTrue(result["reconciliation"]["quantitativeBlocked"])
        self.assertEqual(result["printedSettings"]["values"]["speed"], [25., 50.])

    def test_printed_pulse_disagreement_preserves_both_alternatives(self):
        result = reconcile_printed_settings(calibration(), read("50 mm/s 5 mm/mV"))
        self.assertEqual(result["paperSpeedMmPerSecond"], 25)
        self.assertEqual(result["gainMmPerMv"], 10)
        self.assertEqual(result["reconciliation"]["reasons"], [
            "printed-speed-disagrees-with-pulse-grid", "printed-gain-disagrees-with-pulse-grid"])

    def test_unsupported_decimal_zero_and_negative_settings_are_not_rounded_to_defaults(self):
        for text, value in [("12,5 mm/s", 12.5), ("0 mm/s", 0), ("-25 mm/s", -25), ("100 mm/s", 100)]:
            with self.subTest(text=text):
                result = reconcile_printed_settings(calibration(), read(text))
                self.assertEqual(result["printedSettings"]["values"]["speed"], [value])
                self.assertEqual(result["reconciliation"]["state"], "unsupported")
                self.assertTrue(result["reconciliation"]["quantitativeBlocked"])

    def test_rates_and_low_confidence_text_do_not_manufacture_printed_settings(self):
        for printed in [read("25 bpm 10 mm 50 m/s"), read("50 mm/s 20 mm/mV", .84)]:
            result = reconcile_printed_settings(calibration(), printed)
            self.assertEqual(printed["state"], "unresolved")
            self.assertEqual(result["reconciliation"]["state"], "unresolved")
            self.assertFalse(result["reconciliation"]["quantitativeBlocked"])

    def test_unavailable_recognizer_retains_explicit_absence(self):
        def unavailable(_):
            raise RuntimeError("private diagnostic information")
        result = read_printed_settings(np.ones((200, 400), np.uint8), ocr_runner=unavailable)
        self.assertEqual(result["state"], "unresolved")
        self.assertEqual(result["reason"], "local-setting-recognizer-unavailable")
        self.assertNotIn("private", json.dumps(result))

    def test_recognizer_timeout_retains_absence_without_exception_output(self):
        def timeout(_):
            raise subprocess.TimeoutExpired("private source path", 20, output="private text")
        result = read_printed_settings(np.ones((200, 400), np.uint8), ocr_runner=timeout)
        self.assertEqual(result["reason"], "local-setting-recognizer-unavailable")
        self.assertNotIn("private", json.dumps(result))

    def test_printed_values_alone_do_not_resolve_physical_grid_or_pulse_assumptions(self):
        result = reconcile_printed_settings({"detected": False, "gridScaleAmbiguous": True}, read("25 mm/s 10 mm/mV"))
        self.assertFalse(result["detected"])
        self.assertNotIn("pixelsPerMmX", result)
        self.assertEqual(result["reconciliation"]["state"], "unresolved")


if __name__ == "__main__":
    unittest.main()
