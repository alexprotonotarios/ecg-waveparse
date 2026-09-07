import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.platform_evidence import compare, pack


class PlatformEvidenceTests(unittest.TestCase):
    def test_partial_pack_preserves_quantitative_assets_without_satisfying_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); regression = root/'regression'
            run = regression/'workspace/storage/runs/completed'
            run.mkdir(parents=True)
            assets = {}
            for key, name, content in [
                ('canonicalCsv', 'signal.csv', 'I\n0\n100\n'),
                ('uncertaintyCsv', 'uncertainty.csv', 'lead,canonical_sample,status,candidate_count,candidate_spread_uv\nI,0,observed,2,10\n'),
                ('segmentMapJson', 'segments.json', '{"segments":[]}'),
            ]:
                path = run/name; path.write_text(content)
                assets[key] = {'path': str(path.relative_to(regression/'workspace')),
                               'identity': {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}
            (run/'metadata.json').write_text(json.dumps({'sourceIdentity': {'sha256': 'source'}, 'assets': assets}))
            case = {'id': 'completed', 'runId': 'completed', 'sourceSha256': 'source', 'truthSha256': 'truth',
                    'canonicalSha256': assets['canonicalCsv']['identity']['sha256'], 'outcome': 'returned'}
            report = {'denominator': 2, 'cases': [case], 'package': {}, 'scorer': {}, 'platform': {}}
            (regression/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'Incomplete'): pack(regression, root/'strict')
            pack(regression, root/'partial', allow_incomplete=True)
            evidence = json.loads((root/'partial/manifest.json').read_text())
            self.assertEqual(evidence['denominator'], 2)
            self.assertEqual(evidence['completedInputs'], 1)
            self.assertFalse(evidence['complete'])
            for key, asset in evidence['cases'][0]['assets'].items():
                self.assertEqual((root/'partial'/asset['path']).read_bytes(), (regression/'workspace'/assets[key]['path']).read_bytes())
            with self.assertRaisesRegex(ValueError, 'Incomplete'): compare(root/'partial', root/'partial', {'expectedInputs': 2})
            report['cases'] = [case, case]
            (regression/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'duplicate'): pack(regression, root/'duplicate', allow_incomplete=True)

    def test_waveform_missingness_uncertainty_and_tampering_are_independent_failures(self):
        limits = json.loads((Path(__file__).resolve().parents[1]/'benchmark/protocols/platform-parity.v1.json').read_text())
        limits['expectedInputs'] = 1
        with tempfile.TemporaryDirectory() as directory:
            first, second = [Path(directory)/name for name in ('first', 'second')]
            first.mkdir(); second.mkdir()
            def write_case(folder, signal='0\n100\n-100\n0\n', uncertainty_status='observed'):
                files = {'canonicalCsv': ('signal.csv', 'I\n'+signal),
                         'uncertaintyCsv': ('uncertainty.csv', 'lead,canonical_sample,status,candidate_count,candidate_spread_uv\n'+''.join(f'I,{i},{uncertainty_status},2,10\n' for i in range(4))),
                         'segmentMapJson': ('segments.json', '{"segments":[]}')}
                assets = {}
                for key, (name, content) in files.items():
                    path = folder/name; path.write_text(content)
                    assets[key] = {'path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                manifest = {'denominator': 1, 'package': {'payloadManifestSha256': 'payload', 'runtimeManifestSha256': 'runtime'},
                            'scorer': {'version': 7}, 'cases': [{'id': 'case', 'sourceSha256': 'source', 'truthSha256': 'truth',
                            'status': 'needs_review', 'outcome': 'returned', 'publicationDecision': {'reasonCode': 'peer_trusted'},
                            'semanticStatus': 'passed', 'score': {'globalRmseUv': 10, 'macroMeanCorrelation': .99, 'macroMeanCoverage': 1}, 'assets': assets}]}
                (folder/'manifest.json').write_text(json.dumps(manifest))
            write_case(first); write_case(second)
            self.assertTrue(compare(first, second, limits)['passed'])
            write_case(second, signal='0\n200\n-200\n0\n')
            result = compare(first, second, limits)
            self.assertFalse(result['passed']); self.assertFalse(result['cases'][0]['gates']['waveformTolerance'])
            write_case(second, signal='0\n100\nnan\n0\n')
            result = compare(first, second, limits)
            self.assertFalse(result['passed']); self.assertFalse(result['cases'][0]['gates']['missingnessTolerance'])
            write_case(second, uncertainty_status='unavailable')
            result = compare(first, second, limits)
            self.assertFalse(result['passed']); self.assertFalse(result['cases'][0]['gates']['uncertaintyStatusTolerance'])
            write_case(second); (second/'signal.csv').write_text('I\n0\n0\n0\n0\n')
            with self.assertRaises(ValueError): compare(first, second, limits)

    def test_refusals_cannot_hide_missing_cases_different_inputs_or_payloads(self):
        limits = {'expectedInputs': 1}
        manifest = {'denominator': 1, 'package': {'payloadManifestSha256': 'payload', 'runtimeManifestSha256': 'runtime'},
                    'scorer': {'version': 7}, 'cases': [{'id': 'case', 'sourceSha256': 'source', 'truthSha256': 'truth',
                      'status': 'failed', 'outcome': 'abstention', 'publicationDecision': {'reasonCode': 'unsupported'}, 'assets': {}}]}
        with tempfile.TemporaryDirectory() as directory:
            first, second = [Path(directory)/name for name in ('first', 'second')]
            first.mkdir(); second.mkdir()
            (first/'manifest.json').write_text(json.dumps(manifest)); (second/'manifest.json').write_text(json.dumps(manifest))
            result = compare(first, second, limits)
            self.assertTrue(result['passed']); self.assertEqual(result['cases'][0]['comparison'], 'refusal_only')
            changed = copy.deepcopy(manifest); changed['cases'][0]['sourceSha256'] = 'different'
            (second/'manifest.json').write_text(json.dumps(changed)); self.assertFalse(compare(first, second, limits)['passed'])
            changed = copy.deepcopy(manifest); changed['package']['payloadManifestSha256'] = 'different'
            (second/'manifest.json').write_text(json.dumps(changed)); self.assertFalse(compare(first, second, limits)['passed'])
            changed = copy.deepcopy(manifest); changed['cases'] *= 2
            (second/'manifest.json').write_text(json.dumps(changed))
            with self.assertRaises(ValueError): compare(first, second, limits)
            changed['cases'] = []; (second/'manifest.json').write_text(json.dumps(changed))
            with self.assertRaises(ValueError): compare(first, second, limits)


if __name__ == '__main__': unittest.main()
