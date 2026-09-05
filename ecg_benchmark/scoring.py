from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from .io import LEADS, load_json, read_leads


SCORER_VERSION = 5
PHYSIONET_EVALUATION_REPOSITORY = "https://github.com/physionetchallenges/evaluation-2024"
PHYSIONET_EVALUATION_COMMIT = "1a5135470e7fd9817633f055f3dadebb58fc89ef"


@dataclass
class Alignment:
    shift_samples: int
    baseline_bias_uv: float
    truth_indices: np.ndarray
    truth_values: np.ndarray
    candidate_values: np.ndarray
    candidate_on_truth_grid: np.ndarray
    truth_span_length: int
    candidate_span_length: int


def finite_segment(values: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(np.isfinite(values))
    if indices.size == 0:
        return np.empty(0, dtype=np.float64)
    return values[indices[0] : indices[-1] + 1]


def comparable_signal_frames(
    truth: np.ndarray,
    candidate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Place compact and canonical signals on the same lead-local time axis.

    Benchmark truth is commonly stored as one compact lead segment while a
    digitizer publishes a full-page canonical canvas.  Cropping each array to
    its own finite support discards intentional boundary NaNs from the
    candidate and shifts every remaining sample toward time zero.  When one
    array is an exact multiple of the other, select its most-supported page
    panel first; once the frames have equal lengths, crop both by the truth
    support so candidate gaps retain their original time coordinates.
    """

    truth_frame = np.asarray(truth, dtype=np.float64)
    candidate_frame = np.asarray(candidate, dtype=np.float64)

    def most_supported_panel(values: np.ndarray, panel_size: int) -> np.ndarray:
        if panel_size <= 0 or values.size % panel_size != 0:
            return values
        panel_count = values.size // panel_size
        if panel_count <= 1 or panel_count > len(LEADS):
            return values
        panels = [
            values[index * panel_size : (index + 1) * panel_size]
            for index in range(panel_count)
        ]
        return max(panels, key=lambda panel: int(np.count_nonzero(np.isfinite(panel))))

    if candidate_frame.size > truth_frame.size and truth_frame.size > 0:
        candidate_frame = most_supported_panel(candidate_frame, truth_frame.size)
    elif truth_frame.size > candidate_frame.size and candidate_frame.size > 0:
        truth_frame = most_supported_panel(truth_frame, candidate_frame.size)

    if truth_frame.size == candidate_frame.size:
        truth_indices = np.flatnonzero(np.isfinite(truth_frame))
        if truth_indices.size == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        start = int(truth_indices[0])
        end = int(truth_indices[-1]) + 1
        return truth_frame[start:end], candidate_frame[start:end]

    return finite_segment(truth_frame), finite_segment(candidate_frame)


def contiguous_true_runs(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    return [run for run in np.split(indices, splits) if run.size]


def resample(values: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
    if values.size == 0 or abs(source_rate - target_rate) < 1e-9:
        return values.copy()
    duration = (values.size - 1) / source_rate
    target_count = max(1, int(round(duration * target_rate)) + 1)
    source_time = np.arange(values.size, dtype=np.float64) / source_rate
    target_time = np.arange(target_count, dtype=np.float64) / target_rate
    output = np.full(target_count, np.nan, dtype=np.float64)
    for run in contiguous_true_runs(np.isfinite(values)):
        if run.size == 1:
            nearest = int(round(source_time[run[0]] * target_rate))
            if 0 <= nearest < target_count:
                output[nearest] = values[run[0]]
            continue
        within = (target_time >= source_time[run[0]]) & (target_time <= source_time[run[-1]])
        output[within] = np.interp(
            target_time[within],
            source_time[run],
            values[run],
        )
    return output


def align_series(
    truth: np.ndarray,
    candidate: np.ndarray,
    *,
    sample_rate: float,
    max_alignment_ms: float,
    excluded_truth_mask: np.ndarray | None = None,
) -> Alignment | None:
    if truth.size == 0 or candidate.size == 0:
        return None

    maximum_shift = int(round(max_alignment_ms * sample_rate / 1000.0))
    best: tuple[float, int, float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for shift in range(-maximum_shift, maximum_shift + 1):
        truth_start = max(0, -shift)
        candidate_start = max(0, shift)
        count = min(truth.size - truth_start, candidate.size - candidate_start)
        if count < max(20, int(0.25 * min(truth.size, candidate.size))):
            continue
        truth_window = truth[truth_start : truth_start + count]
        candidate_window = candidate[candidate_start : candidate_start + count]
        finite = np.isfinite(truth_window) & np.isfinite(candidate_window)
        if excluded_truth_mask is not None:
            finite &= ~excluded_truth_mask[truth_start : truth_start + count]
        if np.count_nonzero(finite) < 20:
            continue
        a = truth_window[finite]
        b = candidate_window[finite]
        bias = float(np.median(b - a))
        error = b - bias - a
        rmse = float(np.sqrt(np.mean(error**2)))
        indices = np.arange(truth_start, truth_start + count, dtype=np.int64)[finite]
        if best is None or rmse < best[0]:
            best = (rmse, shift, bias, indices, a, b - bias)

    if best is None:
        return None

    _, shift, bias, truth_indices, truth_values, candidate_values = best
    candidate_on_truth_grid = np.full(truth.size, np.nan, dtype=np.float64)
    for truth_index in range(truth.size):
        candidate_index = truth_index + shift
        if 0 <= candidate_index < candidate.size and np.isfinite(candidate[candidate_index]):
            candidate_on_truth_grid[truth_index] = candidate[candidate_index] - bias

    return Alignment(
        shift_samples=shift,
        baseline_bias_uv=bias,
        truth_indices=truth_indices,
        truth_values=truth_values,
        candidate_values=candidate_values,
        candidate_on_truth_grid=candidate_on_truth_grid,
        truth_span_length=truth.size,
        candidate_span_length=candidate.size,
    )


def signal_metrics(
    truth_values: np.ndarray,
    candidate_values: np.ndarray,
    *,
    truth_sample_count: int,
) -> dict[str, float | int | None]:
    finite = np.isfinite(truth_values) & np.isfinite(candidate_values)
    compared_truth = truth_values[finite]
    compared_candidate = candidate_values[finite]
    if compared_truth.size == 0:
        return {
            "comparedSamples": 0,
            "coverage": 0.0,
            "maeUv": None,
            "rmseUv": None,
            "p95AbsoluteErrorUv": None,
            "correlation": None,
            "derivativeRmseUvPerSample": None,
            "boundedSnrDb": 0.0 if truth_sample_count else None,
        }

    error = compared_candidate - compared_truth
    absolute_error = np.abs(error)
    p_signal = float(np.mean(compared_truth**2))
    p_noise = float(np.mean(error**2))
    coverage = float(compared_truth.size / max(truth_sample_count, 1))
    if p_signal > 0 and p_noise > 0:
        bounded_snr = float(10 * np.log10(p_signal / p_noise) * coverage)
    elif p_signal > 0 and p_noise == 0:
        bounded_snr = None
    else:
        bounded_snr = None

    derivative_rmse: float | None = None
    adjacent = finite[:-1] & finite[1:]
    if np.any(adjacent):
        derivative_error = (
            candidate_values[1:][adjacent]
            - candidate_values[:-1][adjacent]
            - truth_values[1:][adjacent]
            + truth_values[:-1][adjacent]
        )
        derivative_rmse = float(np.sqrt(np.mean(derivative_error**2)))
    truth_std = float(np.std(compared_truth))
    candidate_std = float(np.std(compared_candidate))
    correlation = (
        float(np.corrcoef(compared_truth, compared_candidate)[0, 1])
        if truth_std > 1e-9 and candidate_std > 1e-9
        else None
    )
    return {
        "comparedSamples": int(compared_truth.size),
        "coverage": coverage,
        "maeUv": float(np.mean(absolute_error)),
        "rmseUv": float(np.sqrt(p_noise)),
        "p95AbsoluteErrorUv": float(np.quantile(absolute_error, 0.95)),
        "correlation": correlation,
        "derivativeRmseUvPerSample": derivative_rmse,
        "boundedSnrDb": bounded_snr,
    }


def _convert_signal(
    signal: np.ndarray,
    num_quant_levels: int,
    min_amplitude: float,
    max_amplitude: float,
    maximum_time: int,
) -> np.ndarray:
    finite = np.isfinite(signal)
    time_indices = np.arange(signal.size, dtype=np.int64)[finite]
    matrix = np.zeros((num_quant_levels, maximum_time), dtype=np.float64)
    amplitude_range = max_amplitude - min_amplitude
    if amplitude_range <= 0:
        return matrix
    levels = np.round(
        (num_quant_levels - 1) * (signal[finite] - min_amplitude) / amplitude_range
    ).astype(int)
    levels = np.clip(levels, 0, num_quant_levels - 1)
    matrix[levels, time_indices] = 1
    return matrix


def _fft_correlate(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    return fftconvolve(reference, np.flip(estimate, axis=(0, 1)), mode="full")


def physionet_align_signals(
    truth_mv: np.ndarray,
    candidate_mv: np.ndarray,
    *,
    num_quant_levels: int = 2**8,
) -> tuple[np.ndarray, int, float]:
    """Reproduce the 2024 Challenge helper alignment.

    The implementation follows ``align_signals`` in the official BSD-2-Clause
    evaluation repository at ``PHYSIONET_EVALUATION_COMMIT``.
    """

    if not np.any(np.isfinite(truth_mv)) or not np.any(np.isfinite(candidate_mv)):
        return candidate_mv.copy(), 0, 0.0
    minimum = min(float(np.nanmin(truth_mv)), float(np.nanmin(candidate_mv)))
    maximum = max(float(np.nanmax(truth_mv)), float(np.nanmax(candidate_mv)))
    if maximum <= minimum:
        return candidate_mv.copy(), 0, float(np.nanmedian(candidate_mv - truth_mv))
    maximum_time = max(truth_mv.size, candidate_mv.size)
    reference_matrix = gaussian_filter(
        _convert_signal(truth_mv, num_quant_levels, minimum, maximum, maximum_time),
        0.5,
    )
    candidate_matrix = gaussian_filter(
        _convert_signal(candidate_mv, num_quant_levels, minimum, maximum, maximum_time),
        0.5,
    )
    cross_peak = np.unravel_index(
        np.argmax(_fft_correlate(reference_matrix, candidate_matrix)),
        (num_quant_levels * 2 - 1, maximum_time * 2 - 1),
    )
    auto_peak = np.unravel_index(
        np.argmax(_fft_correlate(reference_matrix, reference_matrix)),
        (num_quant_levels * 2 - 1, maximum_time * 2 - 1),
    )
    horizontal_offset = int(auto_peak[1] - cross_peak[1])
    vertical_offset = float(
        (auto_peak[0] - cross_peak[0])
        / (num_quant_levels - 1)
        * (maximum - minimum)
    )
    if horizontal_offset < 0:
        shifted = np.concatenate(
            (
                np.full(-horizontal_offset, np.nan),
                candidate_mv[:horizontal_offset],
            )
        )
    elif horizontal_offset > 0:
        shifted = np.concatenate(
            (
                candidate_mv[horizontal_offset:],
                np.full(horizontal_offset, np.nan),
            )
        )
    else:
        shifted = candidate_mv.copy()
    return shifted - vertical_offset, horizontal_offset, vertical_offset


def physionet_snr(
    truth_uv: np.ndarray,
    candidate_uv: np.ndarray,
    *,
    sample_rate_hz: float,
) -> dict[str, float | bool | None]:
    truth_mv = finite_segment(truth_uv) / 1000.0
    candidate_mv = finite_segment(candidate_uv) / 1000.0
    if truth_mv.size == 0:
        return {
            "physionetSnrDb": None,
            "physionetPerfectReconstruction": False,
            "physionetShiftMs": None,
            "physionetVerticalShiftMv": None,
        }
    if candidate_mv.size == 0:
        return {
            "physionetSnrDb": 0.0,
            "physionetPerfectReconstruction": False,
            "physionetShiftMs": None,
            "physionetVerticalShiftMv": None,
        }

    shifted, horizontal_offset, vertical_offset = physionet_align_signals(
        truth_mv, candidate_mv
    )
    if (
        abs(horizontal_offset) > round(0.5 * sample_rate_hz)
        or abs(vertical_offset) > 1.0
    ):
        shifted = candidate_mv
        horizontal_offset = 0
        vertical_offset = 0.0

    length = max(truth_mv.size, shifted.size)
    truth_padded = np.full(length, np.nan)
    candidate_padded = np.full(length, np.nan)
    truth_padded[: truth_mv.size] = truth_mv
    candidate_padded[: shifted.size] = shifted
    finite_truth = np.isfinite(truth_padded)
    compared = finite_truth & np.isfinite(candidate_padded)
    if not np.any(compared):
        snr: float | None = 0.0
        perfect = False
    else:
        signal_power = float(np.mean(truth_padded[compared] ** 2))
        noise_power = float(
            np.mean((truth_padded[compared] - candidate_padded[compared]) ** 2)
        )
        coverage = float(np.count_nonzero(compared) / np.count_nonzero(finite_truth))
        perfect = signal_power > 0 and noise_power == 0
        snr = (
            float(10 * np.log10(signal_power / noise_power) * coverage)
            if signal_power > 0 and noise_power > 0
            else None
        )
    return {
        "physionetSnrDb": snr,
        "physionetPerfectReconstruction": perfect,
        "physionetShiftMs": float(horizontal_offset * 1000.0 / sample_rate_hz),
        "physionetVerticalShiftMv": vertical_offset,
    }


def read_uncertainty(path: Path | None) -> dict[str, dict[int, dict[str, Any]]]:
    if path is None:
        return {}
    result: dict[str, dict[int, dict[str, Any]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lead = (row.get("lead") or "").strip()
            try:
                sample = int(row.get("leadSample") or row.get("lead_sample") or "")
            except ValueError:
                continue
            status = (row.get("status") or "").strip()
            raw_spread = (
                row.get("candidateSpreadUv")
                or row.get("candidate_spread_uv")
                or ""
            )
            try:
                spread = float(raw_spread)
            except ValueError:
                spread = np.nan
            result.setdefault(lead, {})[sample] = {
                "status": status,
                "candidateSpreadUv": spread,
            }
    return result


def lead_annotations(annotations: dict[str, Any], lead: str) -> dict[str, Any]:
    leads = annotations.get("leads", {})
    if not isinstance(leads, dict):
        return {}
    value = leads.get(lead, {})
    return value if isinstance(value, dict) else {}


def range_mask(length: int, ranges: Iterable[dict[str, Any]]) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for item in ranges:
        start = max(0, int(item.get("startSample", 0)))
        end = min(length, int(item.get("endSample", start)))
        if end > start:
            mask[start:end] = True
    return mask


def count_threshold_runs(values: np.ndarray, threshold: float) -> int:
    return len(contiguous_true_runs(np.isfinite(values) & (np.abs(values) >= threshold)))


def score_morphology(
    truth: np.ndarray,
    alignment: Alignment,
    annotations: dict[str, Any],
    *,
    sample_rate_hz: float,
) -> dict[str, Any]:
    events = annotations.get("events", [])
    event_results: list[dict[str, Any]] = []
    for event in events if isinstance(events, list) else []:
        center = int(event.get("sample", -1))
        radius = max(1, int(event.get("windowSamples", 8)))
        direction = str(event.get("direction", "peak"))
        start = max(0, center - radius)
        end = min(truth.size, center + radius + 1)
        truth_window = truth[start:end]
        candidate_window = alignment.candidate_on_truth_grid[start:end]
        if truth_window.size == 0:
            continue
        baseline_start = max(0, start - radius * 2)
        baseline_end = min(truth.size, end + radius * 2)
        truth_baseline = float(np.nanmedian(truth[baseline_start:baseline_end]))
        candidate_baseline = float(
            np.nanmedian(alignment.candidate_on_truth_grid[baseline_start:baseline_end])
        )
        if direction == "trough":
            truth_local = int(np.nanargmin(truth_window))
            candidate_local = (
                int(np.nanargmin(candidate_window))
                if np.any(np.isfinite(candidate_window))
                else None
            )
            truth_amplitude = float(truth_window[truth_local] - truth_baseline)
        else:
            truth_local = int(np.nanargmax(truth_window))
            candidate_local = (
                int(np.nanargmax(candidate_window))
                if np.any(np.isfinite(candidate_window))
                else None
            )
            truth_amplitude = float(truth_window[truth_local] - truth_baseline)
        candidate_amplitude = (
            float(candidate_window[candidate_local] - candidate_baseline)
            if candidate_local is not None
            else None
        )
        candidate_sample = (
            start + candidate_local if candidate_local is not None else None
        )
        turning_point = False
        local_prominence: float | None = None
        if candidate_local is not None and candidate_sample is not None:
            finite_local = np.flatnonzero(np.isfinite(candidate_window))
            if finite_local.size >= 3:
                finite_position = np.flatnonzero(
                    finite_local == candidate_local
                )
                if finite_position.size:
                    position = int(finite_position[0])
                    if 0 < position < finite_local.size - 1:
                        previous_value = float(
                            candidate_window[finite_local[position - 1]]
                        )
                        current_value = float(candidate_window[candidate_local])
                        next_value = float(
                            candidate_window[finite_local[position + 1]]
                        )
                        turning_point = (
                            current_value < previous_value
                            and current_value < next_value
                            if direction == "trough"
                            else current_value > previous_value
                            and current_value > next_value
                        )

            prominence_radius = max(
                radius,
                int(event.get("prominenceWindowSamples", radius * 2)),
            )
            prominence_start = max(0, candidate_sample - prominence_radius)
            prominence_end = min(
                alignment.candidate_on_truth_grid.size,
                candidate_sample + prominence_radius + 1,
            )
            left = alignment.candidate_on_truth_grid[
                prominence_start:candidate_sample
            ]
            right = alignment.candidate_on_truth_grid[
                candidate_sample + 1 : prominence_end
            ]
            left = left[np.isfinite(left)]
            right = right[np.isfinite(right)]
            if left.size and right.size:
                current_value = float(
                    alignment.candidate_on_truth_grid[candidate_sample]
                )
                if direction == "trough":
                    local_prominence = float(
                        min(float(np.max(left)), float(np.max(right)))
                        - current_value
                    )
                else:
                    local_prominence = float(
                        current_value
                        - max(float(np.min(left)), float(np.min(right)))
                    )
        minimum_prominence = float(event.get("minProminenceUv", 0.0))
        minimum_local_prominence = float(
            event.get("minLocalProminenceUv", 0.0)
        )
        requires_turning_point = bool(event.get("requiresTurningPoint", False))
        polarity_matches = (
            candidate_amplitude is not None
            and (
                (direction == "trough" and candidate_amplitude < 0)
                or (direction != "trough" and candidate_amplitude > 0)
            )
        )
        preserved = bool(
            polarity_matches
            and candidate_amplitude is not None
            and abs(candidate_amplitude) >= minimum_prominence
            and (
                not requires_turning_point
                or turning_point
            )
            and (
                local_prominence is not None
                and local_prominence >= minimum_local_prominence
                if minimum_local_prominence > 0
                else True
            )
        )
        truth_sample = start + truth_local
        event_results.append(
            {
                "id": str(event.get("id", f"event_{len(event_results)}")),
                "kind": str(event.get("kind", "extremum")),
                "direction": direction,
                "preserved": preserved,
                "truthSample": truth_sample,
                "candidateSample": candidate_sample,
                "timingErrorMs": (
                    float((candidate_sample - truth_sample) * 1000.0 / sample_rate_hz)
                    if candidate_sample is not None
                    else None
                ),
                "truthAmplitudeUv": truth_amplitude,
                "candidateAmplitudeUv": candidate_amplitude,
                "candidateLocalProminenceUv": local_prominence,
                "candidateTurningPoint": turning_point,
                "amplitudeErrorUv": (
                    float(candidate_amplitude - truth_amplitude)
                    if candidate_amplitude is not None
                    else None
                ),
            }
        )

    occlusion_ranges = annotations.get("occludedRanges", [])
    occluded_mask = range_mask(
        truth.size,
        occlusion_ranges if isinstance(occlusion_ranges, list) else [],
    )
    occluded_finite = (
        occluded_mask
        & np.isfinite(truth)
        & np.isfinite(alignment.candidate_on_truth_grid)
    )
    occluded_error = (
        alignment.candidate_on_truth_grid[occluded_finite] - truth[occluded_finite]
    )
    false_deflections = 0
    for region in occlusion_ranges if isinstance(occlusion_ranges, list) else []:
        start = max(0, int(region.get("startSample", 0)))
        end = min(truth.size, int(region.get("endSample", start)))
        threshold = float(region.get("falseDeflectionThresholdUv", 75.0))
        residual = alignment.candidate_on_truth_grid[start:end] - truth[start:end]
        false_deflections += count_threshold_runs(residual, threshold)

    preserved_events = sum(bool(event["preserved"]) for event in event_results)
    timing_errors = [
        abs(float(event["timingErrorMs"]))
        for event in event_results
        if event["timingErrorMs"] is not None and event["preserved"]
    ]
    amplitude_errors = [
        abs(float(event["amplitudeErrorUv"]))
        for event in event_results
        if event["amplitudeErrorUv"] is not None and event["preserved"]
    ]
    return {
        "eventCount": len(event_results),
        "preservedEventCount": preserved_events,
        "lostEventCount": len(event_results) - preserved_events,
        "eventRecall": (
            float(preserved_events / len(event_results)) if event_results else None
        ),
        "meanAbsoluteTimingErrorMs": (
            float(np.mean(timing_errors)) if timing_errors else None
        ),
        "meanAbsoluteAmplitudeErrorUv": (
            float(np.mean(amplitude_errors)) if amplitude_errors else None
        ),
        "falseDeflectionCountInOccludedRanges": false_deflections,
        "occludedComparedSamples": int(occluded_error.size),
        "occludedP95AbsoluteErrorUv": (
            float(np.quantile(np.abs(occluded_error), 0.95))
            if occluded_error.size
            else None
        ),
        "events": event_results,
    }


def score_uncertainty(
    truth: np.ndarray,
    alignment: Alignment,
    statuses: dict[int, Any],
    *,
    high_error_threshold_uv: float,
) -> dict[str, Any] | None:
    if not statuses:
        return None
    compared = np.isfinite(truth) & np.isfinite(alignment.candidate_on_truth_grid)
    errors = np.abs(alignment.candidate_on_truth_grid - truth)
    actual_high_error = compared & (errors >= high_error_threshold_uv)
    status_values: list[str] = []
    uncertainty_scores: list[float] = []
    for sample in range(truth.size):
        value = statuses.get(sample, "observed")
        if isinstance(value, dict):
            status_values.append(str(value.get("status", "observed")))
            try:
                uncertainty_scores.append(float(value.get("candidateSpreadUv", np.nan)))
            except (TypeError, ValueError):
                uncertainty_scores.append(np.nan)
        else:
            status_values.append(str(value))
            uncertainty_scores.append(np.nan)
    predicted_uncertain = np.asarray(
        [status != "observed" for status in status_values],
        dtype=bool,
    )
    uncertainty_score = np.asarray(uncertainty_scores, dtype=np.float64)
    true_positive = int(np.count_nonzero(actual_high_error & predicted_uncertain))
    false_positive = int(np.count_nonzero(~actual_high_error & predicted_uncertain & compared))
    false_negative = int(np.count_nonzero(actual_high_error & ~predicted_uncertain))
    observed_safe = compared & ~predicted_uncertain & ~actual_high_error
    scored = compared & np.isfinite(uncertainty_score)
    score_auc = binary_auroc(actual_high_error[scored], uncertainty_score[scored])
    score_average_precision = binary_average_precision(
        actual_high_error[scored], uncertainty_score[scored]
    )
    calibration_bins: list[dict[str, Any]] = []
    if np.count_nonzero(scored):
        scored_indices = np.flatnonzero(scored)
        order = scored_indices[np.argsort(uncertainty_score[scored_indices])]
        for index, group in enumerate(np.array_split(order, min(5, order.size))):
            if group.size == 0:
                continue
            calibration_bins.append(
                {
                    "bin": index,
                    "sampleCount": int(group.size),
                    "minimumScoreUv": float(np.min(uncertainty_score[group])),
                    "maximumScoreUv": float(np.max(uncertainty_score[group])),
                    "meanScoreUv": float(np.mean(uncertainty_score[group])),
                    "meanAbsoluteErrorUv": float(np.mean(errors[group])),
                    "highErrorFraction": float(np.mean(actual_high_error[group])),
                }
            )
    selective_risk: list[dict[str, Any]] = []
    if np.count_nonzero(scored):
        scored_indices = np.flatnonzero(scored)
        order = scored_indices[np.argsort(uncertainty_score[scored_indices])]
        for target_coverage in (0.5, 0.75, 0.9, 1.0):
            count = max(1, int(np.ceil(order.size * target_coverage)))
            selected = order[:count]
            selective_risk.append(
                {
                    "targetCoverage": target_coverage,
                    "actualCoverage": float(count / max(np.count_nonzero(np.isfinite(truth)), 1)),
                    "sampleCount": count,
                    "rmseUv": float(np.sqrt(np.mean(errors[selected] ** 2))),
                    "highErrorFraction": float(np.mean(actual_high_error[selected])),
                }
            )
    return {
        "highErrorThresholdUv": high_error_threshold_uv,
        "highErrorSamples": int(np.count_nonzero(actual_high_error)),
        "uncertainSamples": int(np.count_nonzero(predicted_uncertain)),
        "highErrorRecall": (
            float(true_positive / (true_positive + false_negative))
            if true_positive + false_negative
            else None
        ),
        "highErrorPrecision": (
            float(true_positive / (true_positive + false_positive))
            if true_positive + false_positive
            else None
        ),
        "safeUsableFraction": float(
            np.count_nonzero(observed_safe) / max(np.count_nonzero(np.isfinite(truth)), 1)
        ),
        "uncertaintyScoreComparedSamples": int(np.count_nonzero(scored)),
        "uncertaintyScoreAuroc": score_auc,
        "uncertaintyScoreAveragePrecision": score_average_precision,
        "calibrationBins": calibration_bins,
        "selectiveRisk": selective_risk,
    }


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.count_nonzero(labels))
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(np.sum(ranks[labels]))
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def binary_average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.count_nonzero(labels))
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    cumulative_positives = np.cumsum(sorted_labels)
    threshold_ends = np.r_[
        np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]),
        sorted_scores.size - 1,
    ]
    true_positives = cumulative_positives[threshold_ends]
    predicted_positives = threshold_ends + 1
    precision = true_positives / predicted_positives
    recall = true_positives / positives
    recall_increase = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increase * precision))


def align_and_score(
    truth: np.ndarray,
    candidate: np.ndarray,
    *,
    sample_rate: float,
    max_alignment_ms: float,
    annotations: dict[str, Any] | None = None,
    uncertainty_statuses: dict[int, Any] | None = None,
    high_error_threshold_uv: float = 75.0,
) -> dict[str, Any]:
    truth, candidate = comparable_signal_frames(truth, candidate)
    annotation_value = annotations or {}
    occluded = range_mask(
        truth.size,
        annotation_value.get("occludedRanges", [])
        if isinstance(annotation_value.get("occludedRanges", []), list)
        else [],
    )
    alignment = align_series(
        truth,
        candidate,
        sample_rate=sample_rate,
        max_alignment_ms=max_alignment_ms,
        excluded_truth_mask=occluded,
    )
    if alignment is None:
        return {
            "status": "failed",
            "failureReason": "no_comparable_samples",
            "comparedSamples": 0,
            "coverage": 0.0,
            "shiftSamples": None,
            "shiftMs": None,
            "alignmentAtLimit": None,
            "baselineBiasUv": None,
            "maeUv": None,
            "rmseUv": None,
            "p95AbsoluteErrorUv": None,
            "correlation": None,
            "derivativeRmseUvPerSample": None,
            "boundedSnrDb": 0.0 if truth.size else None,
            **physionet_snr(truth, candidate, sample_rate_hz=sample_rate),
            "visible": None,
            "morphology": None,
            "uncertainty": None,
        }

    metrics = signal_metrics(
        truth,
        alignment.candidate_on_truth_grid,
        truth_sample_count=int(np.count_nonzero(np.isfinite(truth))),
    )
    raw_count = min(truth.size, candidate.size)
    raw_finite = np.isfinite(truth[:raw_count]) & np.isfinite(candidate[:raw_count])
    raw_error = (
        candidate[:raw_count][raw_finite] - truth[:raw_count][raw_finite]
        if raw_count
        else np.empty(0)
    )
    visible_truth = truth.copy()
    visible_candidate = alignment.candidate_on_truth_grid.copy()
    visible_truth[occluded] = np.nan
    visible_candidate[occluded] = np.nan
    visible_metrics = signal_metrics(
        visible_truth,
        visible_candidate,
        truth_sample_count=int(np.count_nonzero(np.isfinite(truth) & ~occluded)),
    )
    return {
        "status": "completed",
        "failureReason": None,
        **metrics,
        "shiftSamples": alignment.shift_samples,
        "shiftMs": float(alignment.shift_samples * 1000.0 / sample_rate),
        "alignmentAtLimit": (
            abs(alignment.shift_samples)
            >= int(round(max_alignment_ms * sample_rate / 1000.0))
        ),
        "baselineBiasUv": alignment.baseline_bias_uv,
        "rawRmseUv": (
            float(np.sqrt(np.mean(raw_error**2))) if raw_error.size else None
        ),
        **physionet_snr(truth, candidate, sample_rate_hz=sample_rate),
        "visible": visible_metrics,
        "morphology": score_morphology(
            truth,
            alignment,
            annotation_value,
            sample_rate_hz=sample_rate,
        ),
        "uncertainty": score_uncertainty(
            truth,
            alignment,
            uncertainty_statuses or {},
            high_error_threshold_uv=high_error_threshold_uv,
        ),
    }


def _macro_mean(values: Iterable[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def score_files(
    truth_path: Path,
    candidate_path: Path,
    *,
    truth_rate: float,
    candidate_rate: float,
    max_alignment_ms: float,
    annotations_path: Path | None = None,
    uncertainty_path: Path | None = None,
    case_id: str | None = None,
    expected_leads: Iterable[str] | None = None,
) -> dict[str, Any]:
    truth = read_leads(truth_path)
    candidate = read_leads(candidate_path)
    annotations = load_json(annotations_path) if annotations_path else {}
    uncertainty = read_uncertainty(uncertainty_path)
    common_rate = max(truth_rate, candidate_rate)
    per_lead: dict[str, dict[str, Any]] = {}
    missing_leads: list[str] = []
    high_error_threshold = float(annotations.get("highErrorThresholdUv", 75.0))
    expected = tuple(expected_leads or LEADS)
    unknown_expected = sorted(set(expected) - set(LEADS))
    if unknown_expected:
        raise ValueError(
            "Unknown expected lead(s): " + ", ".join(unknown_expected)
        )
    expected_set = set(expected)

    for lead in LEADS:
        if lead not in expected_set:
            per_lead[lead] = {
                "status": "not_expected",
                "failureReason": None,
                "comparedSamples": 0,
                "coverage": None,
            }
            continue
        if lead not in truth or lead not in candidate:
            missing_leads.append(lead)
            per_lead[lead] = {
                "status": "failed",
                "failureReason": (
                    "missing_truth_lead" if lead not in truth else "missing_candidate_lead"
                ),
                "comparedSamples": 0,
                "coverage": 0.0,
            }
            continue
        truth_signal = resample(truth[lead], truth_rate, common_rate)
        candidate_signal = resample(candidate[lead], candidate_rate, common_rate)
        per_lead[lead] = align_and_score(
            truth_signal,
            candidate_signal,
            sample_rate=common_rate,
            max_alignment_ms=max_alignment_ms,
            annotations=lead_annotations(annotations, lead),
            uncertainty_statuses=uncertainty.get(lead),
            high_error_threshold_uv=high_error_threshold,
        )

    completed = [
        per_lead[lead]
        for lead in expected
        if per_lead[lead].get("status") == "completed"
    ]
    total_samples = sum(int(metrics.get("comparedSamples", 0)) for metrics in completed)
    total_squared_error = sum(
        float(metrics["rmseUv"]) ** 2 * int(metrics["comparedSamples"])
        for metrics in completed
        if metrics.get("rmseUv") is not None
    )
    morphology_event_count = sum(
        int((metrics.get("morphology") or {}).get("eventCount", 0))
        for metrics in completed
    )
    preserved_event_count = sum(
        int((metrics.get("morphology") or {}).get("preservedEventCount", 0))
        for metrics in completed
    )
    false_deflections = sum(
        int(
            (metrics.get("morphology") or {}).get(
                "falseDeflectionCountInOccludedRanges", 0
            )
        )
        for metrics in completed
    )
    return {
        "version": SCORER_VERSION,
        "caseId": case_id,
        "truth": str(truth_path),
        "candidate": str(candidate_path),
        "annotations": str(annotations_path) if annotations_path else None,
        "uncertainty": str(uncertainty_path) if uncertainty_path else None,
        "truthSampleRateHz": truth_rate,
        "candidateSampleRateHz": candidate_rate,
        "comparisonSampleRateHz": common_rate,
        "maxAlignmentMs": max_alignment_ms,
        "physionetCompatibility": {
            "repository": PHYSIONET_EVALUATION_REPOSITORY,
            "commit": PHYSIONET_EVALUATION_COMMIT,
            "metric": "mean lead-level SNR after official-style 2D quantized alignment",
        },
        "summary": {
            "expectedLeadCount": len(expected),
            "expectedLeads": list(expected),
            "scoredLeadCount": len(completed),
            "failedLeadCount": len(expected) - len(completed),
            "missingLeads": missing_leads,
            "completeExpectedLeads": len(completed) == len(expected),
            "complete12Lead": (
                len(expected) == len(LEADS) and len(completed) == len(LEADS)
            ),
            "comparedSamples": total_samples,
            "globalRmseUv": (
                float(np.sqrt(total_squared_error / total_samples))
                if total_samples
                else None
            ),
            "macroMeanRmseUv": _macro_mean(
                metrics.get("rmseUv") for metrics in completed
            ),
            "macroMeanVisibleRmseUv": _macro_mean(
                (metrics.get("visible") or {}).get("rmseUv")
                for metrics in completed
            ),
            "macroMeanPhysionetSnrDb": _macro_mean(
                metrics.get("physionetSnrDb") for metrics in completed
            ),
            "macroMeanCoverage": _macro_mean(
                metrics.get("coverage") for metrics in completed
            ),
            "macroMeanCorrelation": _macro_mean(
                metrics.get("correlation") for metrics in completed
            ),
            "alignmentLimitHitCount": sum(
                bool(metrics.get("alignmentAtLimit")) for metrics in completed
            ),
            "morphologyEventCount": morphology_event_count,
            "preservedMorphologyEventCount": preserved_event_count,
            "morphologyEventRecall": (
                float(preserved_event_count / morphology_event_count)
                if morphology_event_count
                else None
            ),
            "falseDeflectionCountInOccludedRanges": false_deflections,
            "macroMeanSafeUsableFraction": _macro_mean(
                (metrics.get("uncertainty") or {}).get("safeUsableFraction")
                for metrics in completed
            ),
            "macroMeanHighErrorRecall": _macro_mean(
                (metrics.get("uncertainty") or {}).get("highErrorRecall")
                for metrics in completed
            ),
            "macroMeanUncertaintyScoreAuroc": _macro_mean(
                (metrics.get("uncertainty") or {}).get("uncertaintyScoreAuroc")
                for metrics in completed
            ),
            "macroMeanUncertaintyScoreAveragePrecision": _macro_mean(
                (metrics.get("uncertainty") or {}).get(
                    "uncertaintyScoreAveragePrecision"
                )
                for metrics in completed
            ),
        },
        "leads": per_lead,
    }
