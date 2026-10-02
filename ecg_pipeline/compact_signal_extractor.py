"""Store rejected upstream centroid regions as independent occupied crops.

Adapted from Ahus-AIM/Open-ECG-Digitizer (CC BY-SA 4.0); see
packages/distribution/THIRD_PARTY_NOTICES.md for attribution and modifications.

The upstream region loop is retained verbatim except for compacting each rejected
map. Crops own their storage so later refinement cannot alter earlier snapshots.
"""
from dataclasses import dataclass
import numpy as np
import torch
from skimage.measure import label
from src.model.signal_extractor import SignalExtractor

@dataclass(frozen=True)
class RejectedRegion:
    top: int
    left: int
    values: torch.Tensor


def compact_region(mask_map: torch.Tensor) -> RejectedRegion | None:
    ys, xs = torch.where(mask_map > 0)
    if ys.numel() == 0:
        return None
    y0, y1 = int(ys.min().item()), int(ys.max().item())
    x0, x1 = int(xs.min().item()), int(xs.max().item())
    return RejectedRegion(y0, x0, mask_map[y0:y1 + 1, x0:x1 + 1].clone())


class CroppedCentroidSignalExtractor(SignalExtractor):
    def _extract_candidate_lines(self, fmap):
        lab = self._label_regions(fmap)
        zero_mask = lab == 0
        big_offset = lab.max() + 1000
        for i in range(self.split_num_stripes):
            sl = slice(i * lab.shape[1] // self.split_num_stripes,
                       (i + 1) * lab.shape[1] // self.split_num_stripes)
            relab = label(lab[:, sl] > 0, connectivity=1)
            lab[:, sl] += big_offset * i + relab
        lab[zero_mask] = 0
        good, rejected, rej_maps = [], [], []
        for lid in np.unique(lab):
            if lid == 0:
                continue
            mask = torch.tensor(lab == lid)
            line = self._extract_line_from_region(fmap, mask)
            if self._classify_line(line, mask):
                good.append(line)
            else:
                rejected.append(line)
                rej_maps.append(compact_region(fmap * mask))
        for line in (*good, *rejected):
            line[line < 5] = float('nan')
        return good, rejected, rej_maps

    def _refine_fmap_by_removal(self, fmap, rej_maps):
        for region in rej_maps:
            if region is None:
                continue
            cropped = region.values.cpu().numpy()
            _, new_img = self._trace_horizontal_path(cropped)
            y0, x0 = region.top, region.left
            h, w = cropped.shape
            target = fmap[y0:y0 + h, x0:x0 + w]
            target[cropped > 0] = new_img[cropped > 0]
