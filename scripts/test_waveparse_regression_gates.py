"""Regression checks must fail if an app silently treats extraction as approval."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_waveparse_regression import check


class WaveParseRegressionGatesTest(unittest.TestCase):
    def test_review_metadata_and_real_quality_failures_cannot_be_hidden(self):
        report = {"cases": [{
            "id": "smoke__low_resolution", "status": "needs_review",
            "outcome": "quantitative_needs_review", "reviewRequired": True,
            "effectiveSampleRateHz": 90,
            "publicationDecision": {"outcome": "needs_review", "reasonCode": "reviewable_constrained"},
            "score": {"complete12Lead": True, "globalRmseUv": 26},
        }]}
        gates = {"version": 1, "suiteId": "smoke", "candidateId": "selector", "cases": {
            "low_resolution": {"maximumGlobalRmseUv": 40, "runSafety": {
                "expectedStatus": "needs_review", "reviewRequired": True,
                "expectedPublicationOutcome": "needs_review",
                "expectedPublicationReasonCode": "reviewable_constrained",
                "maximumEffectiveSampleRateHz": 90,
            }}
        }}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gates.json").write_text(json.dumps(gates))

            def evaluate(candidate):
                (root / "report.json").write_text(json.dumps(candidate))
                return check(root / "report.json", root / "gates.json", root / "result.json")

            self.assertTrue(evaluate(report)["passed"])
            for key, bad_value in (
                ("status", "completed"), ("reviewRequired", False),
                ("reviewRequired", None), ("reviewRequired", "true"),
                ("effectiveSampleRateHz", 500), ("effectiveSampleRateHz", None),
                ("effectiveSampleRateHz", 0), ("effectiveSampleRateHz", float("nan")),
            ):
                with self.subTest(key=key, value=bad_value):
                    bad = deepcopy(report)
                    bad["cases"][0][key] = bad_value
                    self.assertFalse(evaluate(bad)["passed"])
            for key, bad_value in (("outcome", "completed"), ("reasonCode", "source_verified")):
                bad = deepcopy(report)
                bad["cases"][0]["publicationDecision"][key] = bad_value
                self.assertFalse(evaluate(bad)["passed"])
            bad = deepcopy(report)
            bad["cases"][0]["score"]["globalRmseUv"] = 41
            self.assertIn("rmse_limit", {f["code"] for f in evaluate(bad)["failures"]})
            self.assertFalse(evaluate({"cases": []})["passed"])

    def test_global_review_requirements_apply_alongside_case_rules(self):
        from scripts.check_waveparse_regression import check_run_safety
        gates = {"suiteId": "smoke", "runSafety": {"reviewRequired": True},
                 "cases": {"low_resolution": {"runSafety": {"maximumEffectiveSampleRateHz": 90}}}}
        count, failures = check_run_safety({"cases": [{"id": "smoke__low_resolution", "effectiveSampleRateHz": 90}]}, gates)
        self.assertEqual(count, 2)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["code"], "run_safety")


if __name__ == "__main__":
    unittest.main()
