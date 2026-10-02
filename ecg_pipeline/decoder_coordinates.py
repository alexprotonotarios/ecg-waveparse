"""Coordinate observation of Open-ECG decoder operations in a candidate process.

This records geometry and contributor lineage without changing waveform values.
Use only in an isolated single-candidate, single-threaded inference process.
Upstream layout routing follows Ahus-AIM/Open-ECG-Digitizer (CC BY-SA 4.0).
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from threading import Lock
from unittest.mock import patch

import numpy as np
import torch


_CAPTURE_LOCK = Lock()


def array(value):
    return value.detach().cpu().numpy().copy()


def project(points, matrix):
    values = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate((values, np.ones((*values.shape[:-1], 1))), axis=-1)
    mapped = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    with np.errstate(invalid="ignore", divide="ignore"):
        return mapped[..., :2] / mapped[..., 2:]


class DecoderCoordinateCapture:
    """Attach temporary observations; restore every method even on failure.

    Capture before the upstream PNG saver, which mutates canonical tensors.
    The resulting coordinates refer to pixel centers (first center is 0,0).
    Coordinates of a resampled waveform are accompanied by the two original
    decoder nodes and weights. They are not claimed to be directly observed ink.
    """

    def __init__(self, wrapper):
        from src.model.lead_identifier import LeadIdentifier
        from ecg_pipeline.reliable_signal_extractor import ReliableSignalExtractor

        if wrapper.apply_dewarping:
            raise ValueError("dewarping_coordinate_contract_unimplemented")
        if type(wrapper.signal_extractor) is not ReliableSignalExtractor:
            raise ValueError("extractor_coordinate_contract_unimplemented")
        identifier = wrapper.identifier
        while hasattr(identifier, "identifier"):
            identifier = identifier.identifier
        if type(identifier) is not LeadIdentifier:
            raise ValueError("identifier_coordinate_contract_unimplemented")
        self.wrapper, self.identifier = wrapper, identifier
        self.data = {}
        self._active = False

    def _once(self, key, value):
        if key in self.data:
            raise ValueError(f"duplicate_coordinate_stage:{key}")
        self.data[key] = value

    def _observe(self, obj, name, callback):
        original = getattr(obj, name)

        def observed(*args, **kwargs):
            return callback(original, *args, **kwargs)

        self.stack.enter_context(patch.object(obj, name, observed))

    def __enter__(self):
        import torchvision.transforms.functional as tvf
        import src.model.lead_identifier as identifier_module

        if self._active or self.data:
            raise ValueError("coordinate_probe_not_reusable")
        if not _CAPTURE_LOCK.acquire(blocking=False):
            raise ValueError("concurrent_decoder_coordinate_capture")
        self.stack = ExitStack()
        self.stack.callback(_CAPTURE_LOCK.release)
        self._active = True
        try:
            self._observe(self.wrapper, "_resample_image", self._resample)
            self._observe(self.wrapper, "_rotate_on_resample", self._rotate)
            self._observe(self.wrapper, "_crop_y", self._crop_y)
            self._observe(tvf, "_get_perspective_coeffs", self._perspective)
            self._observe(self.wrapper.signal_extractor, "preprocess_lines", self._preprocess)
            self._observe(self.identifier, "normalize", self._normalize)
            self._observe(self.identifier, "_interpolate_lines", self._interpolate)
            self._observe(self.identifier, "_canonicalize_lines", self._canonicalize)
            self._observe(identifier_module, "linear_sum_assignment", self._assignment)
        except BaseException:
            self.stack.close()
            self._active = False
            raise
        return self

    def __exit__(self, *exc):
        self._active = False
        return self.stack.__exit__(*exc)

    def _resample(self, original, image):
        result = original(image)
        self._once("resample", {"inputHW": list(image.shape[-2:]), "outputHW": list(result.shape[-2:]),
                               "mapMeaning": "nominal_pixel_center_geometry_not_filter_weight_centroid",
                               "photometricInterpolationFootprintExported": False})
        return result

    def _perspective(self, original, startpoints, endpoints):
        result = original(startpoints, endpoints)
        value = {"startpoints": deepcopy(startpoints), "endpoints": deepcopy(endpoints), "coefficients": result.copy()}
        prior = self.data.get("perspective")
        if prior is None:
            self.data["perspective"] = value
            self.data["perspectiveCalls"] = 0
        elif prior != value:
            raise ValueError("feature_maps_have_different_perspective")
        self.data["perspectiveCalls"] += 1
        return result

    def _rotate(self, original, *maps):
        result = original(*maps)
        before = list(maps[0].shape[-2:])
        after = list(result[0].shape[-2:])
        self._once("rotation", {"inputHW": before, "outputHW": after, "clockwiseQuarterTurns": int(before[0] > before[1])})
        return result

    def _crop_y(self, original, *maps):
        result = original(*maps)
        offsets = []
        for before, after in zip(maps, result):
            if before.untyped_storage().data_ptr() != after.untyped_storage().data_ptr():
                raise ValueError("unexpected_crop_storage")
            delta = after.storage_offset() - before.storage_offset()
            y, remainder = divmod(delta, before.stride(-2))
            if remainder or after.shape[-1] != before.shape[-1]:
                raise ValueError("unexpected_crop_stride")
            offsets.append(y)
        if len(set(offsets)) != 1:
            raise ValueError("different_feature_map_crops")
        self._once("cropY", {"first": offsets[0], "inputHW": list(maps[0].shape[-2:]), "outputHW": list(result[0].shape[-2:])})
        return result

    def _preprocess(self, original, lines):
        result = original(lines)
        # Upstream clones then slices. Read the actual slice storage offset,
        # then check it against the input; do not infer an offset from a CSV.
        first = result.storage_offset()
        expected = array(lines)
        expected[expected == 0] = np.nan
        observed = array(result)
        if result.stride(-1) != 1 or not np.array_equal(observed, expected[:, first:first + result.shape[1]], equal_nan=True):
            raise ValueError("unexpected_extractor_crop")
        self._once("extractorCrop", {"first": first, "inputWidth": lines.shape[1], "outputWidth": result.shape[1]})
        return result

    def _normalize(self, original, lines, avg_pixel_per_mm, mv_per_mm):
        self._once("mergedPixelRows", array(lines))
        self._once("voltage", {"averagePixelsPerMm": float(avg_pixel_per_mm), "mvPerMm": float(mv_per_mm), "rowMeanY": array(lines.nanmean(dim=1))})
        result = original(lines, avg_pixel_per_mm, mv_per_mm)
        self._once("normalizedRows", array(result))
        return result

    def _interpolate(self, original, lines, target_num_samples):
        first = lines.storage_offset()
        if lines.stride(-1) != 1 or first >= lines.stride(0):
            raise ValueError("unexpected_identifier_crop_storage")
        self._once("identifierCrop", {"first": first, "inputWidth": self.data["mergedPixelRows"].shape[1], "outputWidth": lines.shape[1], "targetSamples": target_num_samples})
        self._once("interpolationInputUv", array(lines))
        return original(lines, target_num_samples)

    def _canonicalize(self, original, lines, match):
        self._once("canonicalInputUv", array(lines))
        self._once("match", deepcopy(match))
        self._once("rhythmAssignments", [])
        self._inside_canonical = True
        try:
            result = original(lines, match)
        finally:
            self._inside_canonical = False
        self._once("canonicalUv", array(result))
        return result

    def _assignment(self, original, *args, **kwargs):
        result = original(*args, **kwargs)
        if getattr(self, "_inside_canonical", False):
            self.data["rhythmAssignments"].append({"rhythm": result[0].tolist(), "canonical": result[1].tolist()})
        return result

    def aligned_to_candidate(self, points):
        data = self.data
        required = {"resample", "perspective", "cropY"}
        if not required <= data.keys() or data.get("perspectiveCalls") != 4:
            raise ValueError("incomplete_decoder_geometry")
        values = np.asarray(points, dtype=np.float64).copy()
        values[..., 1] += data["cropY"]["first"]
        rotation = data.get("rotation")
        if self.wrapper.rotate_on_resample and rotation is None:
            raise ValueError("missing_rotation_observation")
        if rotation and rotation["clockwiseQuarterTurns"]:
            x, y = values[..., 0].copy(), values[..., 1].copy()
            values[..., 0], values[..., 1] = y, rotation["inputHW"][0] - 1 - x
        coefficients = data["perspective"]["coefficients"]
        matrix = np.asarray(coefficients + [1.0]).reshape(3, 3)
        # torchvision evaluates its inverse homography at output x+.5,y+.5.
        # F.interpolate(align_corners=False) has the same edge-center relation.
        values = project(values + 0.5, matrix)
        original_hw, resampled_hw = data["resample"]["inputHW"], data["resample"]["outputHW"]
        values *= np.asarray(original_hw[::-1]) / np.asarray(resampled_hw[::-1])
        return values - 0.5

    def prepared_input_export(self):
        """Export into the raster actually supplied to this decoder invocation.

        The caller must separately transport actual input preparation geometry
        before these points can be interpreted on the untouched original image.
        """
        if "resample" not in self.data:
            raise ValueError("incomplete_decoder_geometry")
        size = self.data["resample"]["inputHW"][::-1]
        result = self.export(source_to_candidate_edges=np.eye(3), original_size_wh=size)
        for name in ["sourceLeftXY", "sourceRightXY"]:
            result[name.replace("source", "candidate", 1)] = result.pop(name)
        result["bothRequiredNodesInsideCandidate"] = result.pop("bothRequiredNodesInsideSource")
        result["coordinateScope"] = "prepared_decoder_input_raster"
        result["preparedInputSizeWh"] = size
        result["originalSourceCorrespondenceVerified"] = False
        merged = self.data["mergedPixelRows"]
        xy = np.stack((np.broadcast_to(np.arange(merged.shape[1]) + self.data["extractorCrop"]["first"], merged.shape), merged), axis=-1)
        result["nativeCandidateXY"] = self.aligned_to_candidate(xy)
        result.update({name: self.data[name] for name in ["normalizedRows", "interpolationInputUv", "canonicalInputUv"]})
        return result

    def save_prepared_input(self, path):
        """Save one invocation exclusively, without pickled objects or gap filling."""
        result = self.prepared_input_export()

        def clean(value):
            if isinstance(value, np.ndarray):
                return clean(value.tolist())
            if isinstance(value, np.generic):
                return clean(value.item())
            if isinstance(value, dict):
                return {str(k): clean(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value

        arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
        metadata = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
        arrays["metadataJsonUtf8"] = np.frombuffer(json.dumps(clean(metadata), allow_nan=False, sort_keys=True).encode("utf-8"), dtype=np.uint8)
        # An existing capture belongs to an earlier attempt and must survive.
        with Path(path).open("xb") as output:
            np.savez_compressed(output, **arrays)

    def export(self, *, source_to_candidate_edges, original_size_wh):
        data = self.data
        required = {"mergedPixelRows", "extractorCrop", "identifierCrop", "canonicalUv", "match", "rhythmAssignments"}
        if not required <= data.keys():
            raise ValueError("incomplete_decoder_lineage")
        layout = self.identifier.layouts[data["match"]["layout"]]
        values = data["canonicalInputUv"]
        row_count, samples = values.shape
        routes = np.full((12, samples), -1, dtype=np.int32)
        indices = np.broadcast_to(np.arange(samples), (12, samples)).copy()
        sign = np.ones((12, samples), dtype=np.int8)
        flip = bool(data["match"].get("flip", False))
        working = values.copy()
        if flip:
            working = working[::-1, ::-1]
            working = np.nanmax(working) - working
            indices = samples - 1 - indices
        expected = np.full_like(data["canonicalUv"], np.nan)
        cols = layout["layout"]["cols"]
        width = samples // cols
        for row, names in enumerate(layout["leads"]):
            if row >= row_count:
                continue
            for column, name in enumerate(names if isinstance(names, list) else [names]):
                lead = name.lstrip("-")
                if lead not in self.identifier.LEAD_CHANNEL_ORDER:
                    continue
                dest = self.identifier.LEAD_CHANNEL_ORDER.index(lead)
                start, end = column * width, (column + 1) * width if column < cols - 1 else samples
                polarity = -1 if name.startswith("-") else 1
                routes[dest, start:end] = row_count - 1 - row if flip else row
                sign[dest, start:end] = polarity
                expected[dest, start:end] = polarity * working[row, start:end]
        rhythms = len(layout.get("rhythm_leads", []))
        if len(data["rhythmAssignments"]) > 1:
            raise ValueError("ambiguous_rhythm_assignment_observation")
        for assignment in data["rhythmAssignments"]:
            for rhythm, dest in zip(assignment["rhythm"], assignment["canonical"]):
                row = row_count - rhythms + rhythm
                routes[dest] = row_count - 1 - row if flip else row
                sign[dest] = 1
                expected[dest] = working[row]
        if not np.array_equal(expected, data["canonicalUv"], equal_nan=True):
            raise ValueError("canonical_route_replay_mismatch")

        crop = data["identifierCrop"]
        native_width = crop["outputWidth"]
        if native_width < 2 or samples < 2:
            raise ValueError("degenerate_interpolation_domain")
        # Reuse the upstream normalized abscissae, including exact-node behavior.
        native_axis = np.linspace(0, 1, native_width)
        query = np.linspace(0, 1, samples)[indices]
        left = np.searchsorted(native_axis, query, side="right") - 1
        left = np.clip(left, 0, native_width - 1)
        right = np.minimum(left + 1, native_width - 1)
        weights = np.zeros(query.shape)
        np.divide(query - native_axis[left], native_axis[right] - native_axis[left], out=weights, where=right != left)
        raw_left, raw_right = left + crop["first"], right + crop["first"]
        merged = data["mergedPixelRows"]
        yl = merged[np.maximum(routes, 0), raw_left].astype(float)
        yr = merged[np.maximum(routes, 0), raw_right].astype(float)
        finite = np.isfinite(data["canonicalUv"])
        support = (routes >= 0) & np.isfinite(yl) & ((weights == 0) | np.isfinite(yr))
        if not np.array_equal(finite, support):
            raise ValueError("coordinate_missingness_mismatch")
        offset = data["extractorCrop"]["first"]
        left_points = np.stack((raw_left + offset, yl), axis=-1)
        right_points = np.stack((raw_right + offset, yr), axis=-1)
        # A zero-weight unavailable right neighbor is not support and is never
        # used to fill a gap. Retain its missing y and zero weight explicitly.
        source_inverse = np.linalg.inv(np.asarray(source_to_candidate_edges, dtype=float))
        source_left = project(self.aligned_to_candidate(left_points) + 0.5, source_inverse) - 0.5
        source_right = project(self.aligned_to_candidate(right_points) + 0.5, source_inverse) - 0.5
        source_left[~finite], source_right[~finite] = np.nan, np.nan
        w, h = original_size_wh
        left_inside = ((source_left[..., 0] >= 0) & (source_left[..., 0] <= w - 1) & (source_left[..., 1] >= 0) & (source_left[..., 1] <= h - 1))
        right_inside = ((source_right[..., 0] >= 0) & (source_right[..., 0] <= w - 1) & (source_right[..., 1] >= 0) & (source_right[..., 1] <= h - 1))
        return {"version": 1, "coordinateConvention": "pixel_centers_zero_based", "nativeNodesAre": "decoder_probability_path_locations_not_verified_source_ink", "waveformChanged": False,
                "sourceInkVerified": False, "physicalTimingVerified": False, "productionSelectionLineageTransported": False,
                "metadata": {k: v for k, v in data.items() if k not in {"mergedPixelRows", "normalizedRows", "interpolationInputUv", "canonicalInputUv", "canonicalUv"}},
                "canonicalUv": data["canonicalUv"], "mergedPixelRows": merged, "rowIndices": routes, "normalizedSampleIndices": indices, "sourcePolarity": sign,
                "nativeLeftIndices": raw_left, "nativeRightIndices": raw_right, "rightWeights": weights,
                "sourceLeftXY": source_left, "sourceRightXY": source_right, "canonicalFinite": finite,
                "bothRequiredNodesInsideSource": finite & left_inside & ((weights == 0) | right_inside)}
