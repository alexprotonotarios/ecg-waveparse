import copy
import unittest

from ecg_benchmark.evaluation import denominator_scorecard, grouped_mean_interval, reference_qualification, validate_independence


class EvaluationTests(unittest.TestCase):
    def test_aliases_and_duplicate_bytes_cannot_cross_splits(self):
        base = [{"caseId":"a","groupId":"g1","sourceId":"s1","split":"development"}, {"caseId":"b","groupId":"g2","sourceId":"s2","split":"final"}]
        self.assertEqual(validate_independence(base)["splitGroupCounts"], {"development":1,"final":1})
        for key in ("groupId", "sourceId", "patientId", "recordingId", "signalFamilyId"):
            changed=copy.deepcopy(base)
            for case in changed: case[key]="same"
            with self.assertRaisesRegex(ValueError, "cross splits"): validate_independence(changed)
        for key in ("image", "truth"):
            changed=copy.deepcopy(base)
            for case in changed: case[key]={"sha256":"a"*64}
            with self.assertRaises(ValueError): validate_independence(changed)

    def test_final_admission_requires_licence_overlap_and_no_prior_exposure(self):
        case={"caseId":"a","groupId":"g","split":"final", "evaluationProvenance":{"collection":"final_evaluation","licenceEvidence":"controlled-register/1","truthMethod":"paired-acquisition","pretrainedTrainingOverlap":"unknown","previouslyExposed":False,"accessControlReference":"custodian/1"}}
        validate_independence([case],require_provenance=True)
        for field in ("licenceEvidence","pretrainedTrainingOverlap","accessControlReference"):
            changed=copy.deepcopy(case);del changed["evaluationProvenance"][field]
            with self.assertRaises(ValueError):validate_independence([changed],require_provenance=True)
        case["evaluationProvenance"]["previouslyExposed"]=True
        with self.assertRaises(ValueError):validate_independence([case],require_provenance=True)

    def test_all_attempts_remain_in_denominator_and_semantics_is_independent(self):
        rows=[{"caseId":"a","outcome":"quantitative_needs_review","score":{"globalRmseUv":0,"macroMeanCoverage":1},"semanticStatus":"failed"}, {"caseId":"b","outcome":"abstention"}, {"caseId":"c","outcome":"timeout"}]
        gates={"globalRmseUv":{"maximum":30},"macroMeanCoverage":{"minimum":.9}}
        card=denominator_scorecard(["a","b","c","d"],rows,gates=gates)
        self.assertEqual(card["quantitativeYield"],.25);self.assertEqual(card["referenceQualifiedYield"],0)
        self.assertEqual(card["terminalCounts"]["missing_result"],1)
        self.assertEqual(card["conditionalFidelity"]["globalRmseUv"]["n"],1)
        with self.assertRaises(ValueError):denominator_scorecard(["a","b","c","d"],rows+[rows[0]])
        self.assertIsNone(denominator_scorecard(["a","b","c","d"],rows)["referenceQualifiedYield"])
        self.assertEqual(reference_qualification({},semantic_status="passed",gates=gates)["status"],"failed")

    def test_intervals_use_groups_not_image_variants(self):
        self.assertIsNone(grouped_mean_interval([("same",1)]*100)["interval95"])
        a=grouped_mean_interval([("one",1)]*100+[("two",3)])
        b=grouped_mean_interval([("one",1),("two",3)])
        self.assertEqual(a,b);self.assertEqual(a["mean"],2)

    def test_empty_numeric_artifact_is_not_counted_as_a_returned_signal(self):
        rows=[{'id':'empty','outcome':'scoring_failure','score':{'comparedSamples':0,'globalRmseUv':None}},
              {'id':'partial','outcome':'partial','semanticStatus':'failed','score':{'comparedSamples':10,'globalRmseUv':20}}]
        report=denominator_scorecard(['empty','partial'],rows,gates={'globalRmseUv':{'maximum':100}})
        self.assertEqual(report['quantitativeYield'],.5)
        self.assertEqual(report['outcomeCounts'],{'scoring_failure':1,'partial':1})
        self.assertEqual(report['referenceQualifiedYield'],0)
        with self.assertRaises(ValueError):reference_qualification({'globalRmseUv':0},semantic_status='passed',gates={})


if __name__ == "__main__": unittest.main()
