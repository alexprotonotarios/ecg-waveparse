from __future__ import annotations

import numpy as np
import torch

from ecg_pipeline.reliable_signal_extractor import trace_probability_path


def fixed_row_centers(height: int, row_count: int = 6) -> np.ndarray:
    return (np.arange(row_count, dtype=np.float32) + 0.5) * height / row_count


def _row_bounds(
    center: float,
    spacing: float,
    height: int,
) -> tuple[int, int]:
    # Adjacent ECG rows can have tall complexes, but their extraction bands
    # must never overlap. An overlapping band lets the dynamic path jump onto
    # the neighbouring lead wherever text or a vertical grid line bridges the
    # two traces, creating multi-millivolt artificial spikes.
    radius = spacing * 0.49
    return (
        max(0, int(np.ceil(center - radius))),
        min(height, int(np.floor(center + radius)) + 1),
    )


def adaptive_column_mask(
    band: np.ndarray,
    absolute_threshold: float,
) -> np.ndarray:
    column_max = band.max(axis=0, keepdims=True)
    adaptive_threshold = np.minimum(
        absolute_threshold,
        column_max * 0.65,
    )
    return (column_max > 1e-8) & (band >= adaptive_threshold)


class FixedRowCentroidSignalExtractor:
    """Extract one independent trace per row from a constrained six-row crop."""

    def __init__(self, label_thresh: float = 0.1, row_count: int = 6) -> None:
        self.label_thresh = label_thresh
        self.row_count = row_count

    def __call__(self, feature_map: torch.Tensor) -> torch.Tensor:
        probability = feature_map.detach().cpu().numpy()
        height, width = probability.shape
        spacing = height / self.row_count
        lines: list[np.ndarray] = []

        for center in fixed_row_centers(height, self.row_count):
            top, bottom = _row_bounds(float(center), spacing, height)
            band = probability[top:bottom]
            mask = adaptive_column_mask(band, self.label_thresh)
            weights = np.where(mask, band, 0.0)
            weight_sum = weights.sum(axis=0)
            local_y = np.arange(top, bottom, dtype=np.float32)[:, None]
            line = np.full(width, np.nan, dtype=np.float32)
            supported = weight_sum > 1e-8
            line[supported] = (
                (weights[:, supported] * local_y).sum(axis=0)
                / weight_sum[supported]
            )
            lines.append(line)

        return torch.as_tensor(
            np.stack(lines),
            dtype=feature_map.dtype,
            device="cpu",
        )

class FixedRowPathSignalExtractor:
    """Trace each constrained row independently with morphology-preserving paths."""

    def __init__(self, label_thresh: float = 0.1, row_count: int = 6) -> None:
        self.label_thresh = label_thresh
        self.row_count = row_count

    def __call__(self, feature_map: torch.Tensor) -> torch.Tensor:
        probability = feature_map.detach().cpu().numpy()
        height, _width = probability.shape
        spacing = height / self.row_count
        lines: list[np.ndarray] = []

        for center in fixed_row_centers(height, self.row_count):
            top, bottom = _row_bounds(float(center), spacing, height)
            band = probability[top:bottom]
            path = trace_probability_path(
                band,
                adaptive_column_mask(band, self.label_thresh),
                continuity_weight=0.001,
                jump_weight=0.000005,
                max_expected_jump=max(32, round(spacing * 1.1)),
            )
            path[np.isfinite(path)] += top
            lines.append(path)

        return torch.as_tensor(
            np.stack(lines),
            dtype=feature_map.dtype,
            device="cpu",
        )
