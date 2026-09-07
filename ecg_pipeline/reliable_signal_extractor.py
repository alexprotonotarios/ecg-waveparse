from __future__ import annotations

import numpy as np
import torch

from src.model.signal_extractor import SignalExtractor


from ecg_pipeline.probability_path import _column_runs, trace_probability_path


class ReliableSignalExtractor(SignalExtractor):
    """Open-ECG signal extractor using an ordered probability-ridge path."""

    def _extract_line_from_region(
        self, fmap: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        path = trace_probability_path(
            fmap.detach().cpu().numpy(),
            mask.detach().cpu().numpy().astype(bool),
            continuity_weight=0.001,
            jump_weight=0.000005,
            # A 1 mV QRS spans roughly 10 mm vertically. At the 2200 px
            # inference scale that can exceed 80 pixels between adjacent
            # raster columns, so the stock candidate-span-derived limit
            # incorrectly shortcuts sharp R/R-prime morphology.
            max_expected_jump=max(self.candidate_span * 12, 96),
        )
        return torch.as_tensor(path, dtype=fmap.dtype, device=fmap.device)
