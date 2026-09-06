import json
import tempfile
import unittest
from pathlib import Path
from scripts.selection_ablation import analyze


class SelectionAblationTests(unittest.TestCase):
    def test_development_choice_does_not_use_validation_or_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest={'cases':[{'caseId':case,'groupId':case,'split':split} for case,split in [('dev','development'),('val','validation')]]}
            rows=[]
            for case in ('dev','val'):
                for candidate in ('default','alternative','selector'):
                    value=10 if candidate in ('default','selector') else 20
                    if case=='val':value=100-value
                    path=f'{case}-{candidate}.json';(root/path).write_text(json.dumps({'summary':{'globalRmseUv':value,'macroMeanCoverage':1,'macroMeanCorrelation':.9}}))
                    rows.append({'caseId':case,'candidateId':candidate,'status':'completed','scorePath':path})
            run={'caseIds':['dev','val'],'results':rows}
            first=analyze(manifest,run,root)
            self.assertEqual(first['bestFixedChosenOnDevelopment'],'default')
            (root/'val-alternative.json').write_text(json.dumps({'summary':{'globalRmseUv':0,'macroMeanCoverage':1,'macroMeanCorrelation':1}}))
            second=analyze(manifest,run,root)
            self.assertEqual(second['bestFixedChosenOnDevelopment'],'default')
            self.assertEqual(second['policies']['offline_truth_oracle']['validation']['meanCaseRmseUv'],0)
            self.assertFalse(second['productionPolicyChanged'])
            run['results'].append(rows[0])
            with self.assertRaises(ValueError):analyze(manifest,run,root)


if __name__=='__main__':unittest.main()
