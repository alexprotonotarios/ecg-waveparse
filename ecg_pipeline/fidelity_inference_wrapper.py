from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import Tensor

from src.model.inference_wrapper import InferenceWrapper


FEATURE_CACHE_VERSION = 1
CANDIDATE_RANDOM_SEED = 0


def _valid_cached_feature_maps(
    payload: object,
    image: Tensor,
) -> tuple[Tensor, Tensor, Tensor] | None:
    if not isinstance(payload, dict) or payload.get("version") != FEATURE_CACHE_VERSION:
        return None
    tensors = payload.get("feature_maps")
    if not isinstance(tensors, (list, tuple)) or len(tensors) != 3:
        return None
    expected_shape = (image.shape[0], 1, image.shape[2], image.shape[3])
    feature_maps: list[Tensor] = []
    for value in tensors:
        if not isinstance(value, Tensor) or tuple(value.shape) != expected_shape:
            return None
        if not value.is_floating_point() or not torch.isfinite(value).all():
            return None
        feature_maps.append(value)
    return feature_maps[0], feature_maps[1], feature_maps[2]


def load_feature_cache(
    cache_path: Path,
    image: Tensor,
) -> tuple[Tensor, Tensor, Tensor] | None:
    if not cache_path.is_file():
        return None
    try:
        payload = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=True,
        )
        feature_maps = _valid_cached_feature_maps(payload, image)
        if feature_maps is None:
            cache_path.unlink(missing_ok=True)
            return None
        return tuple(value.to(image.device) for value in feature_maps)  # type: ignore[return-value]
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        EOFError,
        pickle.UnpicklingError,
    ):
        cache_path.unlink(missing_ok=True)
        return None


def save_feature_cache(
    cache_path: Path,
    feature_maps: tuple[Tensor, Tensor, Tensor],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_path.name}.",
        suffix=".tmp",
        dir=cache_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(
            {
                "version": FEATURE_CACHE_VERSION,
                "feature_maps": tuple(
                    value.detach().to(device="cpu", dtype=torch.float32)
                    for value in feature_maps
                ),
            },
            temporary_path,
        )
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)


class _ThresholdedLeadIdentifier:
    def __init__(self, identifier: object, threshold: float) -> None:
        self.identifier = identifier
        self.threshold = threshold

    def __call__(self, *args: object, **kwargs: object) -> object:
        kwargs.setdefault("threshold", self.threshold)
        return self.identifier(*args, **kwargs)  # type: ignore[operator]


def dark_ink_probability(
    image: Tensor,
    signal_probability: Tensor,
    grid_probability: Tensor,
    *,
    darkness_threshold: float = 0.45,
    strength: float = 0.35,
    support_radius: int = 8,
) -> Tensor:
    """Return dark-pixel evidence only near the model's existing trace support."""

    darkness = 1.0 - image.mean(dim=1, keepdim=True)
    ink = torch.clamp(
        (darkness - darkness_threshold) / max(1.0 - darkness_threshold, 1e-6),
        min=0.0,
        max=1.0,
    ).square()
    grid_suppression = (1.0 - grid_probability.clamp(0.0, 1.0)).square()
    kernel_size = support_radius * 2 + 1
    # A square max filter is separable. Applying its vertical and horizontal
    # passes independently is exactly equivalent to a kernel_size ×
    # kernel_size max pool, while avoiding the prohibitive CPU cost of a large
    # two-dimensional sliding window at the 2200 px inference scale.
    signal_support = functional.max_pool2d(
        signal_probability.clamp(0.0, 1.0),
        kernel_size=(kernel_size, 1),
        stride=1,
        padding=(support_radius, 0),
    )
    signal_support = functional.max_pool2d(
        signal_support,
        kernel_size=(1, kernel_size),
        stride=1,
        padding=(0, support_radius),
    )
    signal_support = (signal_support * 2.0).clamp(0.0, 1.0)
    return (
        ink * grid_suppression * signal_support * strength
    ).clamp(0.0, 1.0)


class FidelityInferenceWrapper(InferenceWrapper):
    """Open-ECG wrapper that retains source-pixel evidence for narrow trace ink."""

    def __init__(
        self,
        *args: object,
        dark_ink_threshold: float = 0.45,
        dark_ink_strength: float = 0.5,
        dark_ink_support_radius: int = 48,
        forced_layout_substring: str | None = None,
        lead_label_threshold: float = 0.8,
        feature_cache_path: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dark_ink_threshold = dark_ink_threshold
        self.dark_ink_strength = dark_ink_strength
        self.dark_ink_support_radius = dark_ink_support_radius
        self.forced_layout_substring = forced_layout_substring
        self.feature_cache_path = (
            Path(feature_cache_path).resolve() if feature_cache_path else None
        )
        if not 0.5 <= lead_label_threshold <= 0.8:
            raise ValueError("Lead label threshold must be between 0.5 and 0.8.")
        if lead_label_threshold < 0.8:
            self.identifier = _ThresholdedLeadIdentifier(
                self.identifier,
                lead_label_threshold,
            )

    def _get_feature_maps(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.feature_cache_path is not None:
            cached = load_feature_cache(self.feature_cache_path, image)
            if cached is not None:
                self.times["Segmentation cache"] = 0.0
                return cached
        feature_maps = super()._get_feature_maps(image)
        if self.feature_cache_path is not None:
            try:
                save_feature_cache(self.feature_cache_path, feature_maps)
            except OSError:
                # Cache writes are an optimisation and must never prevent
                # deterministic digitisation from completing.
                pass
        return feature_maps

    def __call__(
        self,
        image: Tensor,
        layout_should_include_substring: str | None = None,
    ) -> dict[str, object]:
        # Open-ECG's stock signal extractor adds a very small random value to
        # break equal-cost path ties.  Candidate subprocesses otherwise start
        # from unrelated RNG states, so the same page can cross a structural
        # completeness boundary between runs.  Each subprocess handles a
        # single candidate image; resetting all Torch device generators here
        # makes both extraction and any backend stochasticity reproducible.
        torch.manual_seed(CANDIDATE_RANDOM_SEED)
        return super().__call__(
            image,
            layout_should_include_substring=(
                self.forced_layout_substring
                or layout_should_include_substring
            ),
        )

    def _align_feature_maps(
        self,
        image: Tensor,
        signal_probability: Tensor,
        grid_probability: Tensor,
        text_probability: Tensor,
        source_points: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        (
            aligned_image,
            aligned_signal_probability,
            aligned_grid_probability,
            aligned_text_probability,
        ) = super()._align_feature_maps(
            image,
            signal_probability,
            grid_probability,
            text_probability,
            source_points,
        )
        ink_probability = dark_ink_probability(
            aligned_image,
            aligned_signal_probability,
            aligned_grid_probability,
            darkness_threshold=self.dark_ink_threshold,
            strength=self.dark_ink_strength,
            support_radius=self.dark_ink_support_radius,
        )
        return (
            aligned_image,
            (aligned_signal_probability + ink_probability).clamp(0.0, 1.0),
            aligned_grid_probability,
            aligned_text_probability,
        )
