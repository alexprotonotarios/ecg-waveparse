from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ecg_benchmark.io import LEADS, write_leads_csv
from ecg_benchmark.scoring import (
    align_and_score,
    binary_average_precision,
    physionet_snr,
    resample,
    score_files,
    signal_metrics,
)


class DigitizationScoringTests(unittest.TestCase):
    def test_alignment_removes_timing_and_baseline_nuisance(self) -> None:
        truth = np.sin(np.linspace(0, 8 * np.pi, 1000)) * 300
        candidate = np.full(1000, truth[0] + 37)
        candidate[4:] = truth[:-4] + 37

        score = align_and_score(
            truth,
            candidate,
            sample_rate=500,
            max_alignment_ms=40,
        )

        self.assertAlmostEqual(float(score["rmseUv"]), 0, places=8)
        self.assertAlmostEqual(float(score["baselineBiasUv"]), 37, places=8)
        self.assertAlmostEqual(abs(float(score["shiftMs"])), 8, places=8)
        self.assertFalse(score["alignmentAtLimit"])
        self.assertGreater(float(score["correlation"]), 0.999999)

    def test_alignment_reports_when_best_shift_hits_safety_bound(self) -> None:
        truth = np.sin(np.linspace(0, 8 * np.pi, 1000)) * 300
        candidate = np.full(1000, truth[0])
        candidate[20:] = truth[:-20]

        score = align_and_score(
            truth,
            candidate,
            sample_rate=500,
            max_alignment_ms=40,
        )

        self.assertEqual(abs(float(score["shiftMs"])), 40)
        self.assertTrue(score["alignmentAtLimit"])

    def test_resampling_preserves_internal_missing_gap(self) -> None:
        values = np.arange(20, dtype=float)
        values[8:12] = np.nan

        resampled = resample(values, 10, 20)

        self.assertTrue(np.isnan(resampled[16:23]).all())
        self.assertTrue(np.isfinite(resampled[:15]).all())
        self.assertTrue(np.isfinite(resampled[24:]).all())

    def test_canonical_boundary_masks_keep_compact_truth_time_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth_path = root / "truth.csv"
            candidate_path = root / "candidate.csv"
            sample_count = 2_500
            truth_i = (
                np.sin(np.linspace(0, 13 * np.pi, sample_count)) * 180
                + np.sin(np.linspace(0, 47 * np.pi, sample_count)) * 35
            )
            truth = {lead: truth_i.copy() for lead in LEADS}
            write_leads_csv(truth_path, truth, sample_rate_hz=500)

            with candidate_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["I"])
                for sample in range(sample_count * 2):
                    value = (
                        truth_i[sample]
                        if 250 <= sample < 2_425
                        else ""
                    )
                    writer.writerow([value])

            score = score_files(
                truth_path,
                candidate_path,
                truth_rate=500,
                candidate_rate=500,
                max_alignment_ms=40,
                expected_leads=["I"],
            )

        lead = score["leads"]["I"]
        self.assertEqual(lead["shiftSamples"], 0)
        self.assertFalse(lead["alignmentAtLimit"])
        self.assertAlmostEqual(float(lead["rmseUv"]), 0, places=8)
        self.assertAlmostEqual(float(lead["correlation"]), 1, places=8)
        self.assertEqual(lead["comparedSamples"], 2_175)
        self.assertAlmostEqual(float(lead["coverage"]), 0.87, places=8)

    def test_occluded_artificial_wave_is_counted_without_polluting_visible_error(self) -> None:
        truth = np.sin(np.linspace(0, 4 * np.pi, 400)) * 100
        candidate = truth.copy()
        candidate[198:203] += np.array([0, 120, 240, 120, 0])
        annotations = {
            "events": [],
            "occludedRanges": [
                {
                    "startSample": 195,
                    "endSample": 206,
                    "falseDeflectionThresholdUv": 75,
                }
            ],
        }

        score = align_and_score(
            truth,
            candidate,
            sample_rate=500,
            max_alignment_ms=40,
            annotations=annotations,
        )

        self.assertEqual(
            score["morphology"]["falseDeflectionCountInOccludedRanges"], 1
        )
        self.assertGreater(float(score["rmseUv"]), 10)
        self.assertAlmostEqual(float(score["visible"]["rmseUv"]), 0, places=8)
        self.assertGreater(
            float(score["morphology"]["occludedP95AbsoluteErrorUv"]), 100
        )

    def test_derivative_metric_does_not_bridge_missing_gaps(self) -> None:
        truth = np.asarray([0.0, 1.0, np.nan, 100.0, 101.0])
        candidate = np.asarray([0.0, 1.0, np.nan, 200.0, 201.0])

        score = signal_metrics(truth, candidate, truth_sample_count=4)

        self.assertEqual(score["coverage"], 1)
        self.assertEqual(score["derivativeRmseUvPerSample"], 0)

    def test_lost_narrow_peak_fails_morphology_event(self) -> None:
        truth = np.zeros(400)
        truth[196:203] = [0, 40, 150, 260, 145, 35, 0]
        candidate = np.zeros(400)
        annotations = {
            "events": [
                {
                    "id": "narrow_r_prime",
                    "kind": "narrow_r_prime_peak",
                    "sample": 199,
                    "direction": "peak",
                    "windowSamples": 4,
                    "minProminenceUv": 100,
                }
            ],
            "occludedRanges": [],
        }

        score = align_and_score(
            truth,
            candidate,
            sample_rate=500,
            max_alignment_ms=40,
            annotations=annotations,
        )

        self.assertEqual(score["morphology"]["lostEventCount"], 1)
        self.assertEqual(score["morphology"]["eventRecall"], 0)

    def test_monotonic_qrs_slope_does_not_count_as_a_narrow_r_prime(self) -> None:
        truth = np.zeros(400)
        truth[194:206] = [
            0,
            220,
            80,
            -40,
            20,
            145,
            70,
            -80,
            -160,
            -80,
            0,
            0,
        ]
        candidate = np.zeros(400)
        candidate[194:206] = [
            0,
            220,
            190,
            160,
            140,
            120,
            80,
            20,
            -60,
            -80,
            0,
            0,
        ]
        annotations = {
            "events": [
                {
                    "id": "narrow_r_prime",
                    "kind": "narrow_r_prime_peak",
                    "sample": 199,
                    "direction": "peak",
                    "windowSamples": 3,
                    "minProminenceUv": 100,
                    "requiresTurningPoint": True,
                    "prominenceWindowSamples": 6,
                    "minLocalProminenceUv": 60,
                }
            ],
            "occludedRanges": [],
        }

        score = align_and_score(
            truth,
            candidate,
            sample_rate=500,
            max_alignment_ms=0,
            annotations=annotations,
        )

        event = score["morphology"]["events"][0]
        self.assertFalse(event["candidateTurningPoint"])
        self.assertFalse(event["preserved"])

    def test_local_turning_point_and_prominence_preserve_narrow_r_prime(
        self,
    ) -> None:
        truth = np.zeros(400)
        truth[194:206] = [
            0,
            220,
            80,
            -40,
            20,
            145,
            70,
            -80,
            -160,
            -80,
            0,
            0,
        ]
        candidate = truth.copy()
        annotations = {
            "events": [
                {
                    "id": "narrow_r_prime",
                    "kind": "narrow_r_prime_peak",
                    "sample": 199,
                    "direction": "peak",
                    "windowSamples": 3,
                    "minProminenceUv": 100,
                    "requiresTurningPoint": True,
                    "prominenceWindowSamples": 6,
                    "minLocalProminenceUv": 60,
                }
            ],
            "occludedRanges": [],
        }

        score = align_and_score(
            truth,
            candidate,
            sample_rate=500,
            max_alignment_ms=0,
            annotations=annotations,
        )

        event = score["morphology"]["events"][0]
        self.assertTrue(event["candidateTurningPoint"])
        self.assertGreater(float(event["candidateLocalProminenceUv"]), 60)
        self.assertTrue(event["preserved"])

    def test_uncertainty_recall_tracks_high_error_samples(self) -> None:
        truth = np.sin(np.linspace(0, 8 * np.pi, 500)) * 200
        candidate = truth.copy()
        candidate[100:110] += 120
        statuses = {
            sample: {
                "status": (
                    "uncertain_candidate_disagreement"
                    if 100 <= sample < 110
                    else "observed"
                ),
                "candidateSpreadUv": (
                    200.0 if 100 <= sample < 110 else float(sample % 5)
                ),
            }
            for sample in range(500)
        }

        score = align_and_score(
            truth,
            candidate,
            sample_rate=500,
            max_alignment_ms=40,
            uncertainty_statuses=statuses,
            high_error_threshold_uv=75,
        )

        self.assertEqual(score["uncertainty"]["highErrorRecall"], 1)
        self.assertEqual(score["uncertainty"]["highErrorPrecision"], 1)
        self.assertEqual(score["uncertainty"]["uncertaintyScoreAuroc"], 1)
        self.assertEqual(
            score["uncertainty"]["uncertaintyScoreAveragePrecision"], 1
        )
        self.assertEqual(
            score["uncertainty"]["uncertaintyScoreComparedSamples"], 500
        )
        self.assertEqual(len(score["uncertainty"]["calibrationBins"]), 5)
        self.assertEqual(
            [row["targetCoverage"] for row in score["uncertainty"]["selectiveRisk"]],
            [0.5, 0.75, 0.9, 1.0],
        )

    def test_average_precision_treats_tied_scores_as_one_threshold(self) -> None:
        labels = np.asarray([True, False, True, False])
        scores = np.asarray([1.0, 1.0, 0.0, 0.0])

        self.assertEqual(binary_average_precision(labels, scores), 0.5)

    def test_missing_candidate_lead_is_an_explicit_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth_path = root / "truth.csv"
            candidate_path = root / "candidate.csv"
            sample_count = 120
            leads = {
                lead: np.sin(np.linspace(0, 2 * np.pi, sample_count)) * (index + 1)
                for index, lead in enumerate(LEADS)
            }
            write_leads_csv(truth_path, leads, sample_rate_hz=500)
            with candidate_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["sample", *LEADS[:-1]])
                for sample in range(sample_count):
                    writer.writerow(
                        [sample, *[leads[lead][sample] for lead in LEADS[:-1]]]
                    )

            score = score_files(
                truth_path,
                candidate_path,
                truth_rate=500,
                candidate_rate=500,
                max_alignment_ms=40,
            )

        self.assertFalse(score["summary"]["complete12Lead"])
        self.assertEqual(score["summary"]["failedLeadCount"], 1)
        self.assertEqual(score["leads"]["V6"]["failureReason"], "missing_candidate_lead")

    def test_partial_page_scores_only_declared_visible_leads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth_path = root / "truth.csv"
            candidate_path = root / "candidate.csv"
            sample_count = 120
            all_leads = {
                lead: np.sin(np.linspace(0, 2 * np.pi, sample_count)) * (index + 1)
                for index, lead in enumerate(LEADS)
            }
            write_leads_csv(truth_path, all_leads, sample_rate_hz=500)
            visible = list(LEADS[:6])
            with candidate_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["sample", *visible])
                for sample in range(sample_count):
                    writer.writerow([sample, *[all_leads[lead][sample] for lead in visible]])

            score = score_files(
                truth_path,
                candidate_path,
                truth_rate=500,
                candidate_rate=500,
                max_alignment_ms=40,
                expected_leads=visible,
            )

        self.assertTrue(score["summary"]["completeExpectedLeads"])
        self.assertFalse(score["summary"]["complete12Lead"])
        self.assertEqual(score["summary"]["failedLeadCount"], 0)
        self.assertEqual(score["leads"]["V1"]["status"], "not_expected")

    def test_physionet_snr_penalizes_missing_prediction(self) -> None:
        truth = np.sin(np.linspace(0, 2 * np.pi, 500)) * 500
        score = physionet_snr(
            truth,
            np.empty(0),
            sample_rate_hz=500,
        )

        self.assertEqual(score["physionetSnrDb"], 0)
        self.assertFalse(score["physionetPerfectReconstruction"])


if __name__ == "__main__":
    unittest.main()
