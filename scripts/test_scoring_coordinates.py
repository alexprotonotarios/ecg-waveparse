from copy import deepcopy
import unittest

import numpy as np

from ecg_benchmark.coordinates import contract_for_case, strict_semantics
from ecg_benchmark.scoring import align_and_score, comparable_signal_frames, signal_metrics


class ScoringCoordinateTests(unittest.TestCase):
    def setUp(self):
        self.truth = 100 * np.sin(np.arange(1250, dtype=float) / 17)
        self.segment = {"segmentId": "I:panel:0", "lead": "I", "panelIndex": 0,
                        "displayStartSeconds": 0, "displayEndSeconds": 2.5, "polarity": 1, "identityState": "verified"}
        self.contract = {"version": 1, "pageDurationSeconds": 10, "segments": [self.segment]}

    def score(self, values, segments=None):
        return strict_semantics({"I": self.truth}, {"I": values}, self.contract,
                                truth_rate=500, candidate_rate=500,
                                candidate_segments=segments if segments is not None else [self.segment])

    def test_wrong_panel_fails_even_with_perfect_lead_local_shape(self):
        for panel in (0, 3):
            page = np.full(5000, np.nan)
            page[panel * 1250:(panel + 1) * 1250] = self.truth
            a, b = comparable_signal_frames(self.truth, page)
            self.assertEqual(float(np.sqrt(np.mean((a - b) ** 2))), 0)
            local = align_and_score(self.truth, page, sample_rate=500, max_alignment_ms=0)
            self.assertEqual(local["frameTransform"]["candidatePanelIndex"], panel)
            self.assertEqual(self.score(page)["status"], "passed" if panel == 0 else "failed")

    def test_duplicate_panel_is_not_hidden_by_local_selection(self):
        page = np.tile(self.truth, 4)
        self.assertEqual(self.score(page)["segments"][0]["outsideSupportSamples"], 3750)

    def test_swapped_labels_and_polarity_fail_independent_segment_identity(self):
        page = np.full(5000, np.nan)
        page[:1250] = self.truth
        for key, value in (("lead", "II"), ("polarity", -1), ("panelIndex", 1)):
            with self.subTest(key=key):
                changed = {**self.segment, key: value}
                self.assertIn("segment_identity_mismatch", self.score(page, [changed])["segments"][0]["reasons"])
        self.assertEqual(self.score(page, [self.segment, self.segment])["status"], "failed")

    def test_headers_without_source_segment_identity_are_unverified(self):
        page = np.full(5000, np.nan)
        page[:1250] = self.truth
        result = self.score(page, [])
        self.assertTrue(result["segments"][0]["placementPassed"])
        self.assertFalse(result["segments"][0]["identityPassed"])

    def test_nan_unequal_partial_single_point_and_missing_metadata(self):
        self.assertEqual(self.score(np.full(5000, np.nan))["status"], "failed")
        self.assertIn("canvas_length_mismatch", self.score(self.truth)["segments"][0]["reasons"])
        page = np.full(5000, np.nan)
        page[100:500] = self.truth[100:500]
        self.assertEqual(self.score(page)["segments"][0]["missingExpectedSamples"], 850)
        self.assertEqual(align_and_score(np.array([1.]), np.array([1.]), sample_rate=500, max_alignment_ms=0)["failureReason"], "no_comparable_samples")
        self.assertEqual(strict_semantics({}, {}, None, truth_rate=500, candidate_rate=500)["status"], "not_evaluated")

    def test_unsupported_leads_versions_and_support_fail_predictably(self):
        for mutation in ({"version": 99}, {"pageDurationSeconds": -1}, {"segments": [{**self.segment, "lead": "XYZ"}]},
                         {"segments": [self.segment, self.segment]}, {"segments": [{**self.segment, "displayEndSeconds": 11}]}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                strict_semantics({}, {}, {**self.contract, **mutation}, truth_rate=500, candidate_rate=500)

    def test_annotated_uncertainty_follows_nonzero_alignment(self):
        truth = 300 * np.sin(np.arange(1000) / 13) + 45 * np.cos(np.arange(1000) / 5)
        candidate = np.full(1000, np.nan)
        candidate[5:] = truth[:-5]
        candidate[305:315] += 200
        statuses = {i: {"status": "uncertain" if 305 <= i < 315 else "observed", "candidateSpreadUv": 200 if 305 <= i < 315 else 0} for i in range(1000)}
        result = align_and_score(truth, candidate, sample_rate=500, max_alignment_ms=40, uncertainty_statuses=statuses)
        self.assertEqual(result["shiftSamples"], 5)
        self.assertEqual(result["uncertainty"]["highErrorRecall"], 1)
        self.assertEqual(result["uncertainty"]["highErrorPrecision"], 1)

    def test_rates_are_applied_after_panel_and_annotation_mapping(self):
        # 2 seconds, 100 Hz truth, 200 Hz candidate in the second panel.
        truth = 100 * np.sin(np.arange(200) / 100 * 3)
        truth[:20] = np.nan
        truth[70:75] = np.nan
        candidate = np.full(800, np.nan)
        candidate[400:800] = 100 * np.sin(np.arange(400) / 200 * 3)
        candidate[400:440] = np.nan
        candidate[540:550] = np.nan
        result = align_and_score(truth, candidate, truth_rate=100, candidate_rate=200, sample_rate=200, max_alignment_ms=0,
                                 annotations={"occludedRanges": [{"startSample": 90, "endSample": 100}]},
                                 uncertainty_statuses={i: {"status": "observed", "candidateSpreadUv": 0} for i in range(400)})
        self.assertEqual(result["frameTransform"]["candidatePanelIndex"], 1)
        self.assertEqual(result["frameTransform"]["truthCropOffset"], 20)
        self.assertEqual(result["frameTransform"]["candidateCropOffset"], 40)
        self.assertLess(result["rmseUv"], 0.02)
        self.assertEqual(result["comparedSamples"] - result["visible"]["comparedSamples"], 20)

    def test_derivatives_do_not_bridge_internal_or_boundary_gaps(self):
        a = np.array([np.nan, 0., 1., np.nan, 100., 101., np.nan])
        b = np.array([np.nan, 0., 1., np.nan, 500., 501., np.nan])
        self.assertEqual(signal_metrics(a, b, truth_sample_count=4)["derivativeRmseUvPerSample"], 0)

    def test_expected_layout_is_taken_from_manifest(self):
        contract = contract_for_case({"layout": "standard_3x4", "segmentDurationSeconds": 2.5})
        v4 = next(s for s in contract["segments"] if s["lead"] == "V4")
        self.assertEqual(v4["panelIndex"], 3)
        self.assertEqual(v4["displayStartSeconds"], 7.5)

    def test_legacy_paired_duration_requires_independent_panel_evidence(self):
        case = {"layout": "standard_3x4", "segmentDurationSeconds": 10, "sampleRateHz": 500,
                "strata": {"truthSupport": "layout_observed_panels", "truthPanelSamples": 1250, "pageCount": 1}}
        contract = contract_for_case(case)
        self.assertEqual(contract["pageDurationSeconds"], 10)
        self.assertEqual(contract["segments"][3]["displayStartSeconds"], 7.5)
        self.assertEqual(contract["durationBasis"], "legacy_canvas_reconciled_with_declared_panel_samples")
        case["strata"].pop("truthPanelSamples")
        with self.assertRaisesRegex(ValueError, "ambiguous_legacy_segment_duration"): contract_for_case(case)
        case["strata"]["truthPanelSamples"] = 5000
        with self.assertRaisesRegex(ValueError, "inconsistent_legacy_panel_duration"): contract_for_case(case)
        case["strata"]["pageCount"] = 2
        with self.assertRaisesRegex(ValueError, "unsupported_multipage"): contract_for_case(case)

    def test_physical_rates_do_not_accept_booleans_strings_or_containers(self):
        for value in (True, "500", None, [], {}, float("nan"), 0, -1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "invalid_sample_rate"):
                contract_for_case({"layout": "standard_6x2", "segmentDurationSeconds": value})

    def test_canonical_truth_uses_the_independent_panel_offset_and_preserves_gaps(self):
        segment = {**self.segment, "segmentId": "I:panel:3", "panelIndex": 3,
                   "displayStartSeconds": 7.5, "displayEndSeconds": 10}
        local = self.truth.copy(); local[100:110] = np.nan
        page = np.full(5000, np.nan); page[3750:] = local
        contract = {**self.contract, "segments": [segment], "truthCoordinateFrame": "canonical_display"}
        result = strict_semantics({"I": page}, {"I": page}, contract, truth_rate=500, candidate_rate=500, candidate_segments=[segment])
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["segments"][0]["expectedSamples"], 1240)
        invented = page.copy(); invented[3850:3860] = 0
        failed = strict_semantics({"I": page}, {"I": invented}, contract, truth_rate=500, candidate_rate=500, candidate_segments=[segment])
        self.assertEqual(failed["segments"][0]["outsideSupportSamples"], 10)
        with self.assertRaisesRegex(ValueError, "truth_coordinate_length_mismatch"):
            strict_semantics({"I": local}, {"I": page}, contract, truth_rate=500, candidate_rate=500, candidate_segments=[segment])

    def test_uncertainty_canonical_indices_follow_later_panels_and_compact_inputs(self):
        local = self.truth.copy()
        local_candidate = local.copy(); local_candidate[300:310] += 200
        truth = np.full(5000, np.nan); truth[3750:] = local
        candidate = np.full(5000, np.nan); candidate[3750:] = local_candidate
        statuses = {i: {"canonicalSample": 3750 + i, "status": "uncertain" if 300 <= i < 310 else "observed",
                        "candidateSpreadUv": 200 if 300 <= i < 310 else 0} for i in range(1250)}
        for a, b in ((truth, candidate), (local, candidate), (local, local_candidate)):
            result = align_and_score(a, b, sample_rate=500, max_alignment_ms=0, uncertainty_statuses=statuses)
            self.assertEqual(result["uncertainty"]["uncertainSamples"], 10)
            self.assertEqual(result["uncertainty"]["highErrorRecall"], 1)
            self.assertEqual(result["uncertainty"]["highErrorPrecision"], 1)


if __name__ == "__main__":
    unittest.main()
