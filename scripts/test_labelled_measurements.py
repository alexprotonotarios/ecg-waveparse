import copy
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ecg_benchmark.coordinates import map_annotations
from ecg_benchmark.io import dump_json
from ecg_benchmark.measurements import score_measurements, source_feature_visibility
from ecg_benchmark.scoring import align_and_score, score_files


def labels():
    common = {'labelMethod': 'controlled', 'visibility': 'visible', 'polarity': 1,
              'baselineRange': {'startSample': 0, 'endSample': 10}}
    return [dict(common, id='half_height_duration', kind='interval', referenceValue=80,
                 levelUv=500, startRange={'startSample': 10, 'endSample': 40},
                 endRange={'startSample': 50, 'endSample': 90}),
            dict(common, id='signed_peak_amplitude', kind='amplitude', referenceValue=1000,
                 peakRange={'startSample': 20, 'endSample': 80})]


def trapezoid():
    return np.interp(np.arange(100), [0, 10, 30, 50, 70, 99], [0, 0, 1000, 1000, 0, 0])


class LabelledMeasurementTests(unittest.TestCase):
    def test_detector_error_is_separate_from_digitizer_error(self):
        truth = trapezoid()
        annotation = labels()
        # At 500 Hz the known 50%-height crossings are sample 20 and 60.
        annotation[0]['endRange'] = {'startSample': 50, 'endSample': 80}
        annotation[1]['referenceValue'] = 990  # A deliberately biased label.
        scored = score_measurements(truth, truth * .9 + 40, annotation, sample_rate_hz=500.)
        amplitude = scored['endpoints'][1]
        self.assertEqual(amplitude['detectorOnTruthError'], 10)
        self.assertEqual(amplitude['reconstructionIncrement'], -100)
        self.assertEqual(amplitude['totalErrorAgainstLabel'], -90)
        self.assertAlmostEqual(scored['endpoints'][0]['truthDetected'], 80)
        self.assertNotEqual(scored['endpoints'][0]['candidateDetected'], 80)

    def test_negative_amplitude_and_missing_or_multiple_crossings(self):
        annotation = labels()[1]
        annotation.update(polarity=-1, referenceValue=-1000)
        result = score_measurements(-trapezoid(), -trapezoid(), [annotation], sample_rate_hz=500.)
        self.assertEqual(result['endpoints'][0]['candidateDetected'], -1000)
        candidate = -trapezoid(); candidate[40] = np.nan
        result = score_measurements(-trapezoid(), candidate, [annotation], sample_rate_hz=500.)
        self.assertEqual(result['completedCount'], 0)
        self.assertEqual(result['annotatedCount'], 1)
        interval = labels()[0]; interval['endRange'] = {'startSample': 50, 'endSample': 80}
        candidate = trapezoid(); candidate[15:19] = 800
        result = score_measurements(trapezoid(), candidate, [interval], sample_rate_hz=500.)
        self.assertIsNone(result['endpoints'][0]['candidateDetected'])

    def test_rate_and_panel_annotation_map_preserves_fractional_windows(self):
        annotation = {'measurements': labels()}
        for measurement in annotation['measurements']:
            for key, value in measurement.items():
                if key.endswith('Range'):
                    for endpoint in value: value[endpoint] += 200
        mapped = map_annotations(annotation, {'truthPanelOffset': 190, 'truthCropOffset': 10, 'truthRateHz': 500}, 1000)
        self.assertEqual(mapped['measurements'][0]['startRange'], {'startSample': 20, 'endSample': 80})
        self.assertEqual(mapped['measurements'][0]['referenceValue'], 80)
        self.assertEqual(annotation['measurements'][0]['startRange']['startSample'], 210)

    def test_all_nan_candidate_keeps_endpoint_denominator_and_no_invented_value(self):
        result = align_and_score(trapezoid(), np.full(100, np.nan), sample_rate=500., max_alignment_ms=40,
                                 annotations={'measurements': labels()})
        self.assertEqual(result['measurements']['unaligned']['annotatedCount'], 2)
        self.assertEqual(result['measurements']['unaligned']['completedCount'], 0)

    def test_raster_visibility_does_not_count_invisible_or_subpixel_events(self):
        source = np.full((20, 20, 3), 255, dtype=np.uint8); other = source.copy()
        source[7:13, 7:13] = 0
        kwargs = dict(region=(0, 0, 20, 20), horizontal_extent_pixels=6., vertical_extent_pixels=6.)
        self.assertEqual(source_feature_visibility(source, other, **kwargs)['state'], 'visible')
        kwargs['horizontal_extent_pixels'] = .5
        self.assertEqual(source_feature_visibility(source, other, **kwargs)['state'], 'ambiguous')
        self.assertEqual(source_feature_visibility(source, source, **kwargs)['state'], 'unrecoverable')
        annotation = labels()[1]; annotation['visibility'] = 'ambiguous'
        scored = score_measurements(trapezoid(), trapezoid(), [annotation], sample_rate_hz=500.)
        self.assertEqual(scored['sourceVisibleCount'], 0)
        self.assertEqual(scored['completedCount'], 0)

    def test_file_scoring_preserves_missing_lead_measurement_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, leads in [('truth', ('I', 'II')), ('candidate', ('I',))]:
                with (root/f'{name}.csv').open('w') as handle:
                    writer = csv.writer(handle); writer.writerow(leads)
                    writer.writerows([value]*len(leads) for value in trapezoid())
            dump_json(root/'annotations.json', {'leads': {lead: {'measurements': labels()} for lead in ('I', 'II')}})
            scored = score_files(root/'truth.csv', root/'candidate.csv', truth_rate=500, candidate_rate=500,
                                 max_alignment_ms=0, annotations_path=root/'annotations.json', expected_leads=('I','II'))
            self.assertEqual(scored['measurementSummary']['unaligned']['annotatedCount'], 4)
            self.assertLessEqual(scored['measurementSummary']['unaligned']['completedCount'], 2)

    def test_invalid_and_duplicate_annotations_fail(self):
        for key, value in [('polarity', True), ('referenceValue', float('inf')), ('labelMethod', 'automated')]:
            annotation = copy.deepcopy(labels()[0]); annotation[key] = value
            with self.assertRaises(ValueError): score_measurements(trapezoid(), trapezoid(), [annotation], sample_rate_hz=500.)
        with self.assertRaises(ValueError): score_measurements(trapezoid(), trapezoid(), [labels()[1]]*2, sample_rate_hz=500.)


if __name__ == '__main__': unittest.main()
