from __future__ import annotations

import os
import pickle
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import Tensor

from src.model.inference_wrapper import InferenceWrapper
from ecg_pipeline.cpu_convolution import bound_cpu_convolutions
from ecg_pipeline.cpu_unet_lifetime import release_cpu_unet_tensors


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


class _GainCalibratedLeadIdentifier:
    """Supply source gain at the upstream pixel-to-voltage conversion boundary."""

    def __init__(self, identifier: object, gain_mm_per_mv: float) -> None:
        if isinstance(gain_mm_per_mv, bool) or gain_mm_per_mv not in (5.0, 10.0, 20.0):
            raise ValueError("Unsupported source gain; expected 5, 10 or 20 mm/mV.")
        self.identifier = identifier
        self.mv_per_mm = 1.0 / gain_mm_per_mv

    def __call__(self, *args: object, **kwargs: object) -> object:
        kwargs["mv_per_mm"] = self.mv_per_mm
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


def provisional_source_ink_probability(
    image: Tensor, signal_probability: Tensor, grid_probability: Tensor,
) -> Tensor:
    """Add local neutral source contrast to the explicit provisional fallback.

    This fixed setting is an availability hypothesis. It retains the model's
    original probability and never modifies the source image or fills gaps.
    """
    import numpy as np
    from scipy.ndimage import maximum_filter

    if (image.ndim != 4 or image.shape[1] != 3 or
            signal_probability.shape != grid_probability.shape or
            signal_probability.shape != (image.shape[0], 1, *image.shape[2:])):
        raise ValueError("Source ink requires aligned RGB and matching probability maps.")
    source = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    signal = signal_probability.detach().cpu().numpy()
    grid = grid_probability.detach().cpu().numpy()
    enhanced = []
    for rgb, probability, grid_map in zip(source, signal[:, 0], grid[:, 0], strict=True):
        maximum = rgb.max(axis=2)
        chroma = maximum - rgb.min(axis=2)
        background = maximum_filter(maximum, size=15, mode="nearest")
        contrast = np.clip((background - maximum) / .20, 0, 1)
        neutral = np.clip(1 - chroma / .12, 0, 1)
        support = np.clip(maximum_filter(probability, size=97, mode="nearest") * 2, 0, 1)
        ink = (contrast * neutral * support * .4).astype(np.float32)
        ink = ink * (1 - np.clip(grid_map, 0, 1))
        enhanced.append(np.maximum(probability, ink))
    return torch.from_numpy(np.stack(enhanced)[:, None]).to(signal_probability)


class FidelityInferenceWrapper(InferenceWrapper):
    """Open-ECG wrapper that retains source-pixel evidence for narrow trace ink."""

    def _load_signal_extractor(self) -> object:
        extractor = super()._load_signal_extractor()
        if sys.platform == "linux" and torch.device(self.device).type == "cpu":
            from src.model.signal_extractor import SignalExtractor
            from ecg_pipeline.compact_signal_extractor import CroppedCentroidSignalExtractor

            # Preserve specialized extractors and their own vectorization rules.
            if type(extractor) is SignalExtractor:
                return CroppedCentroidSignalExtractor(**self.config.SIGNAL_EXTRACTOR.KWARGS)
        return extractor

    def __init__(
        self,
        *args: object,
        dark_ink_threshold: float = 0.45,
        dark_ink_strength: float = 0.5,
        dark_ink_support_radius: int = 48,
        provisional_source_ink: bool = False,
        forced_layout_substring: str | None = None,
        lead_label_threshold: float = 0.8,
        gain_mm_per_mv: float = 10.0,
        feature_cache_path: str | None = None,
        decoder_coordinate_path: str | None = None,
        decoder_panel_observation_path: str | None = None,
        source_rhythm_context_path: str | None = None,
        **kwargs: object,
    ) -> None:
        if (type(provisional_source_ink) is not bool or
                (provisional_source_ink and (forced_layout_substring != "standard_3x4_with_r1" or dark_ink_strength != 0.0))):
            raise ValueError("Provisional source ink requires its explicit r1 fallback and no dark-ink addition.")
        if decoder_coordinate_path is not None and (
            not isinstance(decoder_coordinate_path, str) or not decoder_coordinate_path
        ):
            raise ValueError("Decoder coordinate path must be a nonempty string.")
        if decoder_panel_observation_path is not None and (
            not isinstance(decoder_panel_observation_path, str) or not decoder_panel_observation_path
        ):
            raise ValueError("Decoder numeric panel observation path must be a nonempty string.")
        if decoder_panel_observation_path is not None and decoder_coordinate_path is not None:
            raise ValueError("Numeric-only panel observation and coordinate capture use separate modes.")
        if source_rhythm_context_path is not None and (
            not isinstance(source_rhythm_context_path, str) or not source_rhythm_context_path or not decoder_coordinate_path
        ):
            raise ValueError("Source rhythm assignment requires a context path and coordinate capture.")
        if isinstance(gain_mm_per_mv, bool) or gain_mm_per_mv not in (5.0, 10.0, 20.0):
            raise ValueError("Unsupported source gain; expected 5, 10 or 20 mm/mV.")
        super().__init__(*args, **kwargs)
        if sys.platform in {"darwin", "linux"} and torch.device(self.device).type == "cpu":
            bound_cpu_convolutions(self.segmentation_model)
            bound_cpu_convolutions(self.identifier.unet)
        if sys.platform == "linux" and torch.device(self.device).type == "cpu":
            release_cpu_unet_tensors(self.segmentation_model)
            release_cpu_unet_tensors(self.identifier.unet)
        elif sys.platform == "darwin" and torch.device(self.device).type == "cpu":
            from ecg_pipeline.cpu_unet_decoder_lifetime import release_cpu_unet_decoder_inputs

            release_cpu_unet_decoder_inputs(self.segmentation_model)
            release_cpu_unet_decoder_inputs(self.identifier.unet)
        self.dark_ink_threshold = dark_ink_threshold
        self.dark_ink_strength = dark_ink_strength
        self.dark_ink_support_radius = dark_ink_support_radius
        self.provisional_source_ink = provisional_source_ink
        self.provisional_source_ink_decision: dict[str, object] | None = None
        self.forced_layout_substring = forced_layout_substring
        self.feature_cache_path = (
            Path(feature_cache_path).resolve() if feature_cache_path else None
        )
        self.decoder_coordinate_path = (
            Path(decoder_coordinate_path).resolve() if decoder_coordinate_path else None
        )
        self.decoder_panel_observation_path = (
            Path(decoder_panel_observation_path).resolve() if decoder_panel_observation_path else None
        )
        self.source_rhythm_context_path = source_rhythm_context_path
        if not 0.5 <= lead_label_threshold <= 0.8:
            raise ValueError("Lead label threshold must be between 0.5 and 0.8.")
        if lead_label_threshold < 0.8:
            self.identifier = _ThresholdedLeadIdentifier(
                self.identifier,
                lead_label_threshold,
            )
        # The upstream default is 0.1 mV/mm. Keep its exact default path, but
        # never label a nondefault-gain waveform after converting it at 10 mm/mV.
        if gain_mm_per_mv != 10.0:
            self.identifier = _GainCalibratedLeadIdentifier(self.identifier, gain_mm_per_mv)

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
        layout = self.forced_layout_substring or layout_should_include_substring
        coordinate_path = getattr(self, "decoder_coordinate_path", None)
        if coordinate_path is None:
            panel_path = getattr(self, "decoder_panel_observation_path", None)
            if panel_path is None:
                return super().__call__(image, layout_should_include_substring=layout)
            if panel_path.exists():
                raise FileExistsError("Refusing to overwrite an earlier numeric panel observation.")
            from ecg_pipeline.decoder_panel_observation import DecoderPanelObservation

            with DecoderPanelObservation(self.identifier) as panels:
                result = super().__call__(image, layout_should_include_substring=layout)
            decision = getattr(self, "provisional_source_ink_decision", None)
            if decision is not None:
                panels.data["sourceInkRecovery"] = decision
            panels.save_if_supported(panel_path)
            return result
        if coordinate_path.exists():
            raise FileExistsError("Refusing to overwrite an earlier decoder-coordinate capture.")
        from ecg_pipeline.decoder_coordinates import DecoderCoordinateCapture

        capture = DecoderCoordinateCapture(self)
        rhythm_context = getattr(self, "source_rhythm_context_path", None)
        if rhythm_context is None:
            with capture:
                result = super().__call__(image, layout_should_include_substring=layout)
        else:
            from ecg_pipeline.source_rhythm_assignment import SourceRhythmAssignment
            # Policy hooks first, observer hooks second: every actual assignment
            # is recorded, and the observer never rewrites a waveform or route.
            with SourceRhythmAssignment(capture, rhythm_context, image) as policy:
                with capture:
                    result = super().__call__(image, layout_should_include_substring=layout)
            capture.data["sourceRhythmAssignment"] = policy.evidence
            # Candidate input folders are compacted. Preserve the exact context
            # bytes as JSON text in the permanent coordinate artifact.
            capture.data["sourceRhythmContextJson"] = policy.context_json
        # Save before the upstream export code modifies canonical tensors for
        # its diagnostic PNG. Failure must not publish a false capture success.
        capture.save_prepared_input(coordinate_path)
        return result

    def _provisional_baseline_coverage(self, probability: Tensor) -> dict[str, object]:
        """Observe original panel coverage without consuming the final decoder RNG.

        Coverage only chooses an unverified recovery route. It establishes no
        source-following morphology, lead identity or measurement accuracy.
        """
        from ecg_pipeline.panel_trace_recovery import PANELS

        extractor = self.signal_extractor
        original_threshold = extractor.threshold_line_in_mask
        original_peaks = extractor.num_peaks
        rng = torch.get_rng_state().clone()
        counts: dict[str, int] = {}
        normalized_rows = 0
        try:
            extractor.threshold_line_in_mask = .95
            with torch.no_grad():
                lines = extractor(probability.squeeze())
                identifier = self.identifier
                while hasattr(identifier, "identifier"):
                    identifier = identifier.identifier
                if lines.ndim == 2 and lines.shape[0] > 0:
                    merged = identifier._merge_nonoverlapping_lines(lines)
                    if 4 <= merged.shape[0] <= 6:
                        # Unit conversion does not affect finite coverage. Use
                        # the actual identifier's crop/interpolation rules.
                        normalized = identifier.normalize(merged, 1.0, .1)
                        normalized_rows = int(normalized.shape[0])
                        if (normalized.ndim == 2 and normalized.shape[1] == 5000 and
                                not torch.isinf(normalized).any()):
                            for row, names in enumerate(PANELS):
                                for column, name in enumerate(names):
                                    part = normalized[row, column * 1250:(column + 1) * 1250]
                                    counts[name] = int(torch.isfinite(part).sum())
        finally:
            extractor.threshold_line_in_mask = original_threshold
            extractor.num_peaks = original_peaks
            torch.set_rng_state(rng)
        complete = len(counts) == 12 and all(value >= 250 for value in counts.values())
        return {"version": 1, "method": "original-provisional-panel-coverage-v1",
                "baselineMembership": .95, "recoveryMembership": .8,
                "minimumFiniteSamplesPerPanel": 250, "baselineNormalizedRows": normalized_rows,
                "baselineFiniteByColumn": counts, "baselineNumericCoverageComplete": complete,
                "sourceInkApplied": not complete, "decoderRngRestored": True,
                "sourceFidelityConfirmed": False, "clinicalValidationUse": False}

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
        if getattr(self, "provisional_source_ink", False):
            decision = self._provisional_baseline_coverage(aligned_signal_probability)
            self.provisional_source_ink_decision = decision
            complete = decision["baselineNumericCoverageComplete"] is True
            self.signal_extractor.threshold_line_in_mask = .95 if complete else .8
            return (
                aligned_image,
                aligned_signal_probability if complete else provisional_source_ink_probability(
                    aligned_image, aligned_signal_probability, aligned_grid_probability),
                aligned_grid_probability,
                aligned_text_probability,
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
