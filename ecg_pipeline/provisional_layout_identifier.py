"""Unverified layout routing for the explicit provisional trace operation.

No source identity, geometry, calibration or waveform fidelity is established.
Merged/normalized upstream decoder values and gaps are retained unchanged.
This upstream-interface extension is offered under CC BY-SA 4.0; see notices.
"""
import torch
from src.model.lead_identifier import LeadIdentifier


class ProvisionalLayoutLeadIdentifier(LeadIdentifier):
    def __call__(self, lines, feature_map, avg_pixel_per_mm, threshold=0.8,
                 mv_per_mm=0.1, layout_should_include_substring=None):
        if (layout_should_include_substring != "standard_3x4_with_r1" or
                self.possibly_flipped or set(self.layouts) != {"standard_3x4_with_r1"}):
            raise ValueError("unsupported_provisional_layout_hypothesis")
        values = -self.normalize(self._merge_nonoverlapping_lines(lines), avg_pixel_per_mm, mv_per_mm)
        if values.ndim != 2 or not 4 <= values.shape[0] <= 6 or values.shape[1] != 5000 or torch.isinf(values).any():
            raise ValueError("unsupported_provisional_observed_rows")
        match = {"layout": "standard_3x4_with_r1", "flip": False, "cost": 1.0,
                 "provisionalLayoutHint": True, "leadIdentityConfirmed": False}
        canonical = self._canonicalize_lines(values.clone(), match)
        return {"rows_in_layout": values.shape[0], "n_detected": 0, **match,
                "canonical_lines": canonical, "lines": values}
