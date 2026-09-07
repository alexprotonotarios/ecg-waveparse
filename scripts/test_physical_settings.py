"""Known full-page scale and duration fixtures, without inference or OCR models."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
from scripts.build_physical_settings_fixtures import build
from scripts.detect_ecg_layout import detect_ecg_layout_geometry, refine_horizontal_grid_consensus


class PhysicalSettingsTests(unittest.TestCase):
    def test_full_pages_recover_known_horizontal_and_vertical_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'fixtures'; build(output)
            protocol = json.loads((output/'protocol.json').read_text())
            for member in protocol['membership']:
                with self.subTest(case=member['caseId']):
                    path = output/member['image']; original = path.read_bytes()
                    report = detect_ecg_layout_geometry(cv2.imread(str(path)))
                    calibration = report['calibration']
                    self.assertTrue(calibration['detected'])
                    self.assertEqual(calibration['paperSpeedMmPerSecond'], member['caseMetadata']['paperSpeedMmPerSecond'])
                    self.assertEqual(calibration['gainMmPerMv'], member['caseMetadata']['gainMmPerMv'])
                    for axis in ('X', 'Y'):
                        self.assertLessEqual(abs(calibration['pixelsPerMm'+axis]/6-1), protocol['maximumScaleRelativeError'])
                    self.assertEqual(path.read_bytes(), original)
            # Deliberate unequal-axis scaling must retain separate horizontal
            # and vertical units. The expected scales follow actual raster sizes.
            image = cv2.imread(str(output/'speed_25_gain_10_segment_5/image.png'))
            height, width = image.shape[:2]
            for scale_x, scale_y in ((.8, 1.2), (1.2, .8)):
                resized = cv2.resize(image, (round(width*scale_x), round(height*scale_y)), interpolation=cv2.INTER_AREA)
                calibration = detect_ecg_layout_geometry(resized)['calibration']
                self.assertTrue(calibration['detected'])
                self.assertEqual((calibration['paperSpeedMmPerSecond'], calibration['gainMmPerMv']), (25, 10))
                for axis, expected in (('X', 6*resized.shape[1]/width), ('Y', 6*resized.shape[0]/height)):
                    self.assertLessEqual(abs(calibration['pixelsPerMm'+axis]/expected-1), protocol['maximumScaleRelativeError'])

    def test_missing_ambiguous_or_varying_geometry_does_not_become_a_scalar_consensus(self):
        calibration = {'detected': True, 'confidence': .9, 'pixelsPerMmX': 6.2, 'gridScaleMmX': 1}
        for scales in ([6]*5, [6,6,6,6,6,float('nan')], [5,5,6,6,7,7], [12]*6):
            self.assertEqual(refine_horizontal_grid_consensus(calibration, scales), calibration)
        for change in ({'detected': False}, {'confidence': .2}, {'gridScaleAmbiguous': True}, {'gridScaleMmX': None}):
            unresolved = {**calibration, **change}
            self.assertEqual(refine_horizontal_grid_consensus(unresolved, [6]*6), unresolved)


if __name__ == '__main__': unittest.main()
