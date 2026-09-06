"""Physical fixture correctness: printed settings and drawn pulses must agree."""
import unittest
import cv2
import numpy as np
from ecg_benchmark.generate import render_waveforms
from ecg_benchmark.paired_render import render_clinical_page
from ecg_benchmark.io import LEADS
from scripts.detect_ecg_layout import detect_ecg_calibration


class RenderedCalibrationTests(unittest.TestCase):
    def test_both_renderers_draw_two_hundred_ms_pulses_at_each_supported_setting(self):
        leads={lead:np.zeros(5000) for lead in LEADS}
        for speed in (25,50):
            for gain in (5,10,20):
                images=[render_waveforms(leads,sample_rate_hz=500,layout_name='standard_6x2',
                            paper_speed_mm_per_second=speed,gain_mm_per_mv=gain,pixels_per_mm=4,trace_width_pixels=2)[0],
                        render_clinical_page(leads,layout='standard_6x2',rhythm_lead=None,paper_speed_mm_per_second=speed,
                            gain_mm_per_mv=gain,pixels_per_mm=4,trace_width_pixels=2)]
                for renderer,image in enumerate(images):
                    with self.subTest(renderer=renderer,speed=speed,gain=gain):
                        report=detect_ecg_calibration(cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR))['calibration']
                        self.assertTrue(report['detected'])
                        self.assertEqual(report['paperSpeedMmPerSecond'],speed)
                        self.assertEqual(report['gainMmPerMv'],gain)


if __name__=='__main__':unittest.main()
