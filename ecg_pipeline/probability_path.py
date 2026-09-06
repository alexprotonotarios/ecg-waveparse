from __future__ import annotations

import numpy as np


def _column_runs(has_signal: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(has_signal.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def trace_probability_path(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    continuity_weight: float = 0.015,
    jump_weight: float = 0.0001,
    implausible_jump_penalty: float = 8.0,
    max_expected_jump: int = 48,
) -> np.ndarray:
    """Trace a high-probability ordered ridge without averaging full vertical strokes.

    The stock extractor collapses every signal region to the probability-weighted
    mean y-value in each column. That can round narrow R-prime notches because a
    near-vertical stroke contributes many y-values to the same mean. This routine
    instead finds a minimum-cost path through the segmented ridge, then refines the
    selected pixel using only a small local probability neighbourhood.
    """

    height, width = probability.shape
    output = np.full(width, np.nan, dtype=np.float32)
    has_signal = mask.any(axis=0)

    for start, end in _column_runs(has_signal):
        candidates = [
            np.flatnonzero(mask[:, column]).astype(np.int32)
            for column in range(start, end)
        ]
        if not candidates or candidates[0].size == 0:
            continue

        first_y = candidates[0]
        costs = -np.log(np.clip(probability[first_y, start], 1e-6, 1.0))
        back_references: list[np.ndarray] = []

        for offset, current_y in enumerate(candidates[1:], start=1):
            column = start + offset
            previous_y = candidates[offset - 1]
            delta = np.abs(current_y[:, None] - previous_y[None, :]).astype(
                np.float32
            )
            transition = continuity_weight * delta + jump_weight * delta**2
            transition += (delta > max_expected_jump) * implausible_jump_penalty
            total = transition + costs[None, :]
            best_previous = np.argmin(total, axis=1)
            costs = (
                total[np.arange(current_y.size), best_previous]
                - np.log(np.clip(probability[current_y, column], 1e-6, 1.0))
            )
            back_references.append(best_previous.astype(np.int32))

        selected_indices = [int(np.argmin(costs))]
        for back_reference in reversed(back_references):
            selected_indices.append(int(back_reference[selected_indices[-1]]))
        selected_indices.reverse()

        for offset, selected_index in enumerate(selected_indices):
            column = start + offset
            selected_y = int(candidates[offset][selected_index])
            local_y = np.arange(
                max(0, selected_y - 2), min(height, selected_y + 3)
            )
            local_mask = mask[local_y, column]
            local_y = local_y[local_mask]
            if local_y.size == 0:
                output[column] = float(selected_y)
                continue
            weights = probability[local_y, column]
            weight_sum = float(weights.sum())
            output[column] = (
                float(np.dot(local_y, weights) / weight_sum)
                if weight_sum > 1e-8
                else float(selected_y)
            )

    return output

