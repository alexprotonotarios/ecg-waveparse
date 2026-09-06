import unittest
import numpy as np
from scripts.build_public_engineering_pilot import read_record
from ecg_benchmark.io import LEADS
from scripts.summarize_public_engineering_pilot import summarize, wilson_interval
from ecg_benchmark.coordinates import contract_for_case, strict_semantics
from ecg_benchmark.paired_render import canonical_visible_truth, layout_column_count


class PublicWaveformSourceTests(unittest.TestCase):
    def fixture(self):
        time = np.arange(21_000) / 1000
        values = np.rint(200 + 2000 * np.sin(2 * np.pi * time))
        digital = np.tile(values[:, None], (1, 12)).astype('<i2')
        lines = ['fixture 15 1000 21000']
        for i, lead in enumerate(LEADS):
            checksum = int(digital[:, i].sum(dtype=np.int64)) % 65536
            lines.append(f'fixture.dat 16 2000 16 0 200 {checksum} 0 {lead.lower()}')
        return '\n'.join(lines), digital.tobytes()

    def test_adc_scale_timing_and_centering_against_known_one_hertz_signal(self):
        header, data = self.fixture()
        result = read_record(header, data)
        self.assertEqual(result.shape, (5000, 12))
        self.assertAlmostEqual(float(result[125, 0]), 1000, delta=2)
        self.assertAlmostEqual(float(result[375, 0]), -1000, delta=2)
        self.assertAlmostEqual(float(np.median(result)), 0, delta=.01)

    def test_checksum_lead_order_and_duration_are_required(self):
        header, data = self.fixture()
        corrupt = bytearray(data); corrupt[0] ^= 1
        with self.assertRaisesRegex(ValueError, 'checksum'): read_record(header, corrupt)
        with self.assertRaisesRegex(ValueError, 'lead order'): read_record(header.replace(' v1', ' v2'), data)
        with self.assertRaisesRegex(ValueError, 'length'): read_record(header, data[:-24])
        with self.assertRaisesRegex(ValueError, 'rate contract'): read_record(header.replace('15 1000', '15 500'), data)

    def test_failed_preparation_and_missing_results_remain_in_the_pilot_denominator(self):
        members=[{'caseId':str(i),'patientId':str(i),'layout':'standard_6x2','artifact':'clean'} for i in range(3)]
        protocol={'membership':members,'gates':{'globalRmseUv':{'maximum':100}},'collection':'frozen_public_engineering_pilot','limitations':[]}
        result=summarize(protocol,{'suiteId':'pilot'},{'denominator':3,'cases':[
            {'id':'pilot__0','outcome':'quantitative_needs_review','semanticStatus':'passed','score':{'globalRmseUv':20}},
            {'id':'pilot__1','outcome':'source_preparation_failure'}]})
        self.assertEqual(result['denominatorScorecard']['quantitativeYield'],1/3)
        self.assertEqual(result['denominatorScorecard']['referenceQualifiedYield'],1/3)
        self.assertEqual(result['denominatorScorecard']['terminalCounts']['missing_result'],1)
        self.assertEqual(result['groupedIntervals']['returnedYield']['groupCount'],3)

    def test_rendered_ten_second_page_has_layout_specific_panel_duration(self):
        leads = {lead: np.sin(np.arange(5000) / 17) for lead in LEADS}
        for layout in ('standard_6x2', 'standard_3x4', 'standard_12x1'):
            with self.subTest(layout=layout):
                case = {'layout': layout, 'segmentDurationSeconds': 10 / layout_column_count(layout),
                        'truthCoordinateFrame': 'canonical_display'}
                contract = contract_for_case(case)
                self.assertEqual(contract['pageDurationSeconds'], 10)
                truth = canonical_visible_truth(leads, layout=layout)
                segments = [{**s, 'identityState': 'verified'} for s in contract['segments']]
                result = strict_semantics(truth, truth, contract, truth_rate=500, candidate_rate=500, candidate_segments=segments)
                self.assertEqual(result['status'], 'passed')

    def test_recording_groups_and_zero_yield_keep_uncertainty(self):
        protocol = {'membership': [
            {'caseId': str(i), 'groupId': f'recording-{i}', 'acquisition': 'scan'} for i in range(8)],
            'gates': {'globalRmseUv': {'maximum': 100}}, 'collection': 'engineering_development',
            'independentUnit': 'recording', 'stratificationKeys': ['acquisition'], 'limitations': []}
        report = {'denominator': 8, 'cases': [
            {'id': f'captures__{i}', 'outcome': 'abstention'} for i in range(8)]}
        result = summarize(protocol, {'suiteId': 'captures'}, report)
        self.assertEqual(result['independentGroups'], 8)
        self.assertEqual(result['strata']['acquisition']['scan']['attempted'], 8)
        interval = result['binomialYieldIntervals']['returnedYield']['interval95']
        self.assertAlmostEqual(interval[0], 0)
        self.assertGreater(interval[1], .3)
        self.assertLess(wilson_interval(8, 8)['interval95'][0], .7)

    def test_repeated_variants_do_not_get_independent_binomial_interval(self):
        protocol = {'membership': [
            {'caseId': str(i), 'groupId': 'same-recording', 'layout': 'standard_6x2', 'artifact': 'clean'} for i in range(2)],
            'gates': {'globalRmseUv': {'maximum': 100}}, 'collection': 'engineering_development', 'limitations': []}
        result = summarize(protocol, {'suiteId': 'captures'}, {'denominator': 2, 'cases': []})
        self.assertEqual(result['independentGroups'], 1)
        self.assertEqual(result['binomialYieldIntervals']['status'], 'unavailable_repeated_groups')
        self.assertIsNone(result['groupedIntervals']['returnedYield']['interval95'])


if __name__ == '__main__': unittest.main()
