from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path

from scripts.calibrate_candidate_selector import (
    SELECTOR_CALIBRATION_CONTRACT_VERSION,
    artifact_context,
    fit_context_selection,
    validate_grouped_splits,
    scored_rows,
)


class CandidateSelectorCalibrationTests(unittest.TestCase):
    def test_alias_groups_cannot_move_one_source_across_splits(self):
        with self.assertRaisesRegex(ValueError, "Group leakage"):
            validate_grouped_splits([
                {"caseId": "a", "groupId": "alias-a", "sourceId": "same-source", "split": "development"},
                {"caseId": "b", "groupId": "alias-b", "sourceId": "same-source", "split": "heldout"},
            ])

    def test_duplicate_case_ids_are_not_silently_overwritten(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_grouped_splits([{"caseId": "same", "groupId": "one", "split": "development"}] * 2)

    def test_fitting_does_not_open_final_set_scores(self):
        manifest = {"cases": [{"caseId": "reserved", "groupId": "reserved-source", "split": "heldout"}]}
        run = {"results": [{"caseId": "reserved", "candidateId": "base", "status": "completed", "scorePath": "must-not-open.json"}]}
        with patch("scripts.calibrate_candidate_selector.score_rmse", side_effect=AssertionError("Final data opened")):
            self.assertEqual(scored_rows(manifest, run, Path("."), {"development", "validation"}), [])

    def test_calibration_uses_the_current_pipeline_contract(self) -> None:
        self.assertEqual(SELECTOR_CALIBRATION_CONTRACT_VERSION, 3)

    def test_screen_capture_has_a_dedicated_artifact_context(self) -> None:
        self.assertEqual(
            artifact_context({"degradation": {"kind": "screen_capture"}}),
            "screen",
        )

    def test_grouped_split_audit_rejects_source_leakage(self) -> None:
        with self.assertRaisesRegex(ValueError, "Group leakage"):
            validate_grouped_splits(
                [
                    {
                        "caseId": "first",
                        "groupId": "same-source",
                        "split": "development",
                    },
                    {
                        "caseId": "second",
                        "groupId": "same-source",
                        "split": "validation",
                    },
                ]
            )

    def test_grouped_split_audit_counts_disjoint_groups(self) -> None:
        result = validate_grouped_splits(
            [
                {
                    "caseId": "first",
                    "groupId": "source-a",
                    "split": "development",
                },
                {
                    "caseId": "second",
                    "groupId": "source-b",
                    "split": "validation",
                },
            ]
        )

        self.assertTrue(result["leakageChecked"])
        self.assertEqual(
            result["groupsBySplit"],
            {"development": 1, "validation": 1},
        )

    def test_artifact_context_activation_is_scoped_per_layout(self) -> None:
        layouts = {
            "layout-a": {
                "referenceRiskRmseUv": 50.0,
                "candidates": {
                    "base": {"riskRmseUv": 50.0},
                    "special": {"riskRmseUv": 60.0},
                },
                "families": {},
                "contexts": {
                    "screen": {
                        "referenceRiskRmseUv": 50.0,
                        "candidates": {
                            "base": {"riskRmseUv": 60.0},
                            "special": {"riskRmseUv": 40.0},
                        },
                        "families": {},
                    }
                },
            },
            "layout-b": {
                "referenceRiskRmseUv": 50.0,
                "candidates": {
                    "base": {"riskRmseUv": 50.0},
                    "special": {"riskRmseUv": 60.0},
                },
                "families": {},
                "contexts": {
                    "screen": {
                        "referenceRiskRmseUv": 50.0,
                        "candidates": {
                            "base": {"riskRmseUv": 60.0},
                            "special": {"riskRmseUv": 40.0},
                        },
                        "families": {},
                    }
                },
            },
        }
        rows = []
        for layout, base_rmse, special_rmse in (
            ("layout-a", 60.0, 40.0),
            ("layout-b", 40.0, 60.0),
        ):
            for index in range(5):
                for candidate_id, rmse in (
                    ("base", base_rmse),
                    ("special", special_rmse),
                ):
                    rows.append(
                        {
                            "caseId": f"{layout}-{index}",
                            "layout": layout,
                            "context": "screen",
                            "candidateId": candidate_id,
                            "family": candidate_id,
                            "rmseUv": rmse,
                        }
                    )

        selection = fit_context_selection(rows, layouts)

        self.assertTrue(selection["layout-a"]["screen"]["enabled"])
        self.assertFalse(selection["layout-b"]["screen"]["enabled"])
        self.assertTrue(
            layouts["layout-a"]["contexts"]["screen"]["selection"]["enabled"]
        )
        self.assertFalse(
            layouts["layout-b"]["contexts"]["screen"]["selection"]["enabled"]
        )


if __name__ == "__main__":
    unittest.main()
