import unittest
from unittest.mock import patch
from pathlib import Path
from scripts.run_selection_experiment import sample_owned
from scripts.summarize_selection_experiment import targeted_fallback


class CurrentSelectionTests(unittest.TestCase):
    def test_targeted_fallback_uses_observable_qa_not_truth_score(self):
        base={'id':'default','status':'completed','canonicalPath':'base.csv','qa':{'passed':True},'score':{'globalRmseUv':999}}
        other={'id':'other','status':'completed','canonicalPath':'other.csv','score':{'globalRmseUv':0}}
        self.assertIs(targeted_fallback(base,other),base)
        base['qa']['passed']=False
        self.assertIs(targeted_fallback(base,other),other)
        other['score']['globalRmseUv']=9999
        self.assertIs(targeted_fallback(base,other),other)
        other['status']='failed'
        self.assertIs(targeted_fallback(base,other),base)

    def test_memory_sampling_only_attributes_owned_candidate_processes(self):
        processes='100 1 10 node\n200 100 20 python --config /private/tmp/work/storage/runs/run_a/candidates/default/open_ecg_config.yml\n201 200 5 child\n300 1 99 python --config /private/tmp/work/storage/runs/run_a/candidates/foreign/open_ecg_config.yml\n'
        with patch('scripts.run_selection_experiment.subprocess.check_output',return_value=processes):
            result=sample_owned(100,Path('/private/tmp/work'))
        self.assertEqual(result['processTreeRssBytes'],35*1024)
        self.assertEqual(result['candidateProcessTreeRssBytes'],{'default':25*1024})


if __name__=='__main__':unittest.main()
