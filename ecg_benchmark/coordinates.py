"""Coordinate contracts for scoring; no candidate-driven semantic ground truth."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .io import LEADS


def contract_for_case(case: dict) -> dict:
    from ecg_pipeline.domain import LAYOUT_ROWS
    layout = case.get("layout")
    if layout not in LAYOUT_ROWS:
        raise ValueError("unsupported_expected_layout")
    rows = LAYOUT_ROWS[layout]
    duration = positive_rate(case.get("segmentDurationSeconds", 0))
    strata = case.get("strata") or {}
    truth_frame = case.get("truthCoordinateFrame", "canonical_display" if strata.get("truthSupport") == "layout_observed_panels" else "lead_local")
    duration_basis = "declared_panel_duration"
    if case.get("truthCoordinateFrame") is None and truth_frame == "canonical_display":
        # Historical paired manifests called the full ten-second truth array a
        # segment. Use their independently declared panel sample count only when
        # it reconciles with the canvas and layout; never guess from a candidate.
        if strata.get("pageCount", 1) != 1:
            raise ValueError("unsupported_multipage_coordinate_contract")
        if "truthPanelSamples" not in strata:
            raise ValueError("ambiguous_legacy_segment_duration")
        panel_duration = positive_rate(strata["truthPanelSamples"]) / positive_rate(case.get("sampleRateHz", 0))
        if abs(panel_duration * len(rows[0]) - duration) > 1e-6:
            raise ValueError("inconsistent_legacy_panel_duration")
        duration = panel_duration
        duration_basis = "legacy_canvas_reconciled_with_declared_panel_samples"
    if truth_frame not in {"canonical_display", "lead_local"}:
        raise ValueError("unsupported_truth_coordinate_frame")
    return {"version": 1, "basis": "independent_benchmark_metadata", "truthCoordinateFrame": truth_frame,
            "durationBasis": duration_basis,
            "pageDurationSeconds": duration * len(rows[0]),
            "segments": [{"segmentId": f"{lead}:panel:{column}", "lead": lead, "panelIndex": column,
                          "displayStartSeconds": column * duration, "displayEndSeconds": (column + 1) * duration,
                          "polarity": 1} for row in rows for column, lead in enumerate(row)]}


def positive_rate(value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value) or value <= 0:
        raise ValueError("invalid_sample_rate: rates must be finite and positive")
    return float(value)


def local_frames(truth: np.ndarray, candidate: np.ndarray, truth_rate: float, candidate_rate: float):
    """Preserve the historical lead-local scope, returning every crop explicitly.

    Panel selection is useful for conditional shape comparison ONLY. Strict
    placement is evaluated separately from independently supplied metadata.
    Rates are applied after panel selection, avoiding N*r endpoint ambiguity.
    """
    positive_rate(truth_rate)
    positive_rate(candidate_rate)
    a, b = np.asarray(truth, dtype=float), np.asarray(candidate, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("invalid_signal_dimensions: expected one-dimensional leads")
    mapping = {"frame": "lead_local", "truthInputSamples": int(a.size), "candidateInputSamples": int(b.size),
               "truthRateHz": truth_rate, "candidateRateHz": candidate_rate,
               "truthPanelIndex": None, "candidatePanelIndex": None,
               "truthPanelOffset": 0, "candidatePanelOffset": 0,
               "truthCropOffset": 0, "candidateCropOffset": 0}

    def panel(values, size, key):
        if size <= 0 or values.size % size or not 1 < values.size // size <= len(LEADS):
            return values
        index = max(range(values.size // size), key=lambda i: np.count_nonzero(np.isfinite(values[i * size:(i + 1) * size])))
        mapping[key + "PanelIndex"] = int(index)
        mapping[key + "PanelOffset"] = int(index * size)
        return values[index * size:(index + 1) * size]

    if b.size / candidate_rate > a.size / truth_rate and a.size:
        b = panel(b, int(round(a.size * candidate_rate / truth_rate)), "candidate")
    elif a.size / truth_rate > b.size / candidate_rate and b.size:
        a = panel(a, int(round(b.size * truth_rate / candidate_rate)), "truth")
    indices = np.flatnonzero(np.isfinite(a))
    if not indices.size:
        return a[:0], b[:0], mapping
    start, end = int(indices[0]), int(indices[-1]) + 1
    if abs(a.size / truth_rate - b.size / candidate_rate) < 1e-8:
        b_start = int(round(start * candidate_rate / truth_rate))
        b_end = min(b.size, int(round(end * candidate_rate / truth_rate)))
    else:
        candidate_indices = np.flatnonzero(np.isfinite(b))
        b_start = int(candidate_indices[0]) if candidate_indices.size else 0
        b_end = int(candidate_indices[-1]) + 1 if candidate_indices.size else 0
    mapping.update(truthCropOffset=start, candidateCropOffset=b_start,
                   truthRetainedSamples=end - start, candidateRetainedSamples=b_end - b_start)
    return a[start:end], b[b_start:b_end], mapping


def map_annotations(annotations: dict, mapping: dict, comparison_rate: float) -> dict:
    result = deepcopy(annotations)
    offset = mapping["truthPanelOffset"] + mapping["truthCropOffset"]
    ratio = comparison_rate / mapping["truthRateHz"]
    for event in result.get("events", []):
        event["sample"] = int(round((event.get("sample", -1) - offset) * ratio))
        for key in ("windowSamples", "prominenceWindowSamples", "maxTimingErrorSamples"):
            if key in event:
                event[key] = int(round(event[key] * ratio))
    for measurement in result.get("measurements", []):
        for key in ("baselineRange", "startRange", "endRange", "peakRange"):
            if key in measurement:
                for endpoint in ("startSample", "endSample"):
                    measurement[key][endpoint] = (measurement[key][endpoint] - offset) * ratio
    for key, entries in result.items():
        if key.endswith("Ranges") and isinstance(entries, list):
            for entry in entries:
                entry["startSample"] = int(np.floor((entry.get("startSample", 0) - offset) * ratio))
                entry["endSample"] = int(np.ceil((entry.get("endSample", 0) - offset) * ratio))
    return result


def map_uncertainty(statuses: dict[int, Any], mapping: dict, count: int, comparison_rate: float) -> dict:
    """Input indices are exported lead_sample, before cropping and alignment.

    An interpolated point inherits both neighbours' limitations; unspecified
    evidence is unknown. Disagreement spread is not a confidence interval.
    """
    if not statuses:
        return {}
    canonical_rows = {v["canonicalSample"]: v for v in statuses.values()
                      if isinstance(v, dict) and "canonicalSample" in v}
    # Current exports carry both lead-local and canonical indices. Use their
    # explicit canonical map when the input spans that canvas; compact inputs
    # use lead_sample. This keeps later panels aligned when truth is canonical.
    use_canonical = bool(canonical_rows) and max(canonical_rows) < mapping["candidateInputSamples"]
    if use_canonical and len(canonical_rows) != len(statuses):
        raise ValueError("incoherent_uncertainty_canonical_indices")
    lookup = canonical_rows if use_canonical else statuses
    offset = mapping["candidateCropOffset"] + (mapping["candidatePanelOffset"] if use_canonical else 0)
    mapping["uncertaintyLookupFrame"] = "canonical_sample" if use_canonical else "lead_sample"
    result = {}
    for sample in range(count):
        source = offset + sample * mapping["candidateRateHz"] / comparison_rate
        neighbours = [lookup.get(i, {"status": "unavailable"}) for i in sorted({int(np.floor(source)), int(np.ceil(source))})]
        rows = [v if isinstance(v, dict) else {"status": v} for v in neighbours]
        status = next((str(v.get("status", "unavailable")) for v in rows if v.get("status") != "observed"), "observed")
        spreads = [v.get("candidateSpreadUv") for v in rows]
        finite_spreads = [float(v) for v in spreads if isinstance(v, (int, float)) and np.isfinite(v)]
        result[sample] = {"status": status, "candidateSpreadUv": max(finite_spreads) if len(finite_spreads) == len(rows) else None}
    return result


def strict_semantics(truth: dict[str, np.ndarray], candidate: dict[str, np.ndarray], contract: dict | None,
                     *, truth_rate: float, candidate_rate: float, candidate_segments: list[dict] | None = None) -> dict:
    """Validate the original canvas, never searching for a more favourable panel.

    Endpoints are half-open. Missing expected samples are reported independently
    of placement. Identity requires an explicit segment map; CSV headers alone
    do not prove source-label recognition.
    """
    if contract is None:
        return {"version": 1, "status": "not_evaluated", "reason": "expected_coordinate_contract_missing", "frame": "canonical_display"}
    if contract.get("version") != 1 or not isinstance(contract.get("segments"), list) or not contract["segments"]:
        raise ValueError("invalid_semantic_contract")
    duration = contract.get("pageDurationSeconds")
    if not isinstance(duration, (int, float)) or not np.isfinite(duration) or duration <= 0:
        raise ValueError("invalid_page_duration")
    canvas_size = int(round(duration * positive_rate(candidate_rate)))
    if not 1 <= canvas_size <= 10_000_000:
        raise ValueError("unsupported_canvas_size")
    positive_rate(truth_rate)
    truth_frame = contract.get("truthCoordinateFrame", "lead_local")
    if truth_frame not in {"canonical_display", "lead_local"}:
        raise ValueError("unsupported_truth_coordinate_frame")
    ids = [s.get("segmentId") for s in contract["segments"]]
    if any(not isinstance(s, str) or not s for s in ids) or len(set(ids)) != len(ids):
        raise ValueError("invalid_segment_identity")
    # Canonical CSV has one segment per named lead. Repeated leads need an
    # explicit multi-segment export and must not silently overwrite each other.
    leads = [s.get("lead") for s in contract["segments"]]
    if len(set(leads)) != len(leads) or any(lead not in LEADS for lead in leads):
        raise ValueError("unsupported_or_duplicate_lead")
    supplied = {s.get("segmentId"): s for s in candidate_segments or []}
    duplicate_ids = len(supplied) != len(candidate_segments or [])
    results = []
    for segment in contract["segments"]:
        lead = segment["lead"]
        start, end = segment.get("displayStartSeconds"), segment.get("displayEndSeconds")
        if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in (start, end)) or not 0 <= start < end <= duration:
            raise ValueError("invalid_segment_support")
        panel = segment.get("panelIndex")
        if not isinstance(panel, int) or isinstance(panel, bool) or panel < 0:
            raise ValueError("invalid_panel_index")
        values = candidate.get(lead, np.empty(0))
        first, last = int(round(start * candidate_rate)), int(round(end * candidate_rate))
        expected_mask = np.zeros(canvas_size, dtype=bool)
        expected_mask[first:last] = True
        # Independently known unavailable truth intervals must stay unavailable.
        local_truth = truth.get(lead, np.empty(0))
        truth_offset = int(round(start * truth_rate)) if truth_frame == "canonical_display" else 0
        local_indices = truth_offset + np.floor(np.arange(last - first) * truth_rate / candidate_rate).astype(int)
        if local_truth.size:
            required_samples = int(round((duration if truth_frame == "canonical_display" else end - start) * truth_rate))
            if local_truth.size != required_samples:
                raise ValueError("truth_coordinate_length_mismatch")
            local_indices = np.minimum(local_indices, local_truth.size - 1)
            expected_mask[first:last] &= np.isfinite(local_truth[local_indices])
        returned = np.zeros(canvas_size, dtype=bool)
        returned[:min(canvas_size, values.size)] = np.isfinite(values[:canvas_size])
        outside = int(np.count_nonzero(returned & ~expected_mask)) + int(np.count_nonzero(np.isfinite(values[canvas_size:])))
        available = int(np.count_nonzero(returned & expected_mask))
        reasons = []
        if not local_truth.size:
            reasons.append("missing_truth_lead")
        if values.size != canvas_size:
            reasons.append("canvas_length_mismatch")
        if outside:
            reasons.append("samples_outside_expected_support")
        if not available:
            reasons.append("no_expected_support")
        identity = supplied.get(segment["segmentId"])
        identity_reasons = []
        if identity is None:
            identity_reasons.append("segment_identity_unverified")
        elif any(identity.get(key) != segment.get(key) for key in ("lead", "panelIndex", "displayStartSeconds", "displayEndSeconds", "polarity")):
            identity_reasons.append("segment_identity_mismatch")
        if identity is not None and identity.get("identityState") != "verified":
            identity_reasons.append("source_label_identity_unverified")
        if duplicate_ids:
            identity_reasons.append("duplicate_segment_identity")
        results.append({"segmentId": segment["segmentId"], "lead": lead, "expectedPanelIndex": panel,
                        "placementPassed": not reasons, "identityPassed": not identity_reasons,
                        "reasons": reasons + identity_reasons, "expectedSamples": int(expected_mask.sum()),
                        "returnedExpectedSamples": available, "outsideSupportSamples": outside,
                        "missingExpectedSamples": int(np.count_nonzero(expected_mask & ~returned))})
    unexpected = sorted(set(candidate) - set(leads))
    return {"version": 1, "status": "passed" if all(not r["reasons"] for r in results) and not unexpected else "failed",
            "frame": "canonical_display", "acquisitionTime": "not_inferred_from_display_position",
            "unexpectedLeads": unexpected, "segments": results}
