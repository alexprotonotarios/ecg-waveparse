"""Source-derived photo waveforms for explicit diagnostic review.

This entry point accepts an already prepared, untouched BGR raster. It does not
alter production source-label acceptance or approve quantitative/clinical use.
Detailed source observations retain labels, inferred timing, geometric limits,
source gaps and qualitative review flags for inspection.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np

MAX_SOURCE_PHOTO_PIXELS = 16_000_000


def _run_chain(image: np.ndarray, source_sha256: str) -> dict[str, Any]:
    from ecg_pipeline import lead_label_identity, printed_calibration
    from ecg_pipeline import source_label_geometry, source_panel_timing
    from scripts import detect_ecg_layout
    from . import _waveforms
    from ._dependencies import PARTS

    return _waveforms.observe(
        image,
        source_label_geometry,
        lead_label_identity,
        **PARTS,
        layout=detect_ecg_layout,
        printed_calibration=printed_calibration,
        maintained_timing=source_panel_timing,
        source_sha=source_sha256,
    )


def observe_source_photo(
    image: np.ndarray, *, source_sha256: str | None = None
) -> dict[str, Any]:
    """Return a versioned diagnostic result, preserving unsupported outcomes.

    ``source_sha256`` identifies the encoded source when supplied by the caller;
    otherwise the decoded raster hash identifies the input. Neither identity is
    an inference feature. Source gaps remain absent in the waveform arrays.
    """
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 200
        or image.shape[0] * image.shape[1] > MAX_SOURCE_PHOTO_PIXELS
    ):
        raise ValueError("Expected a bounded BGR uint8 source image.")
    if source_sha256 is not None and (
        not isinstance(source_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
    ):
        raise ValueError("Source identity must be a lowercase SHA-256 digest.")

    decoded_sha256 = hashlib.sha256(image.tobytes()).hexdigest()
    from ._masks import invocation_masks

    with invocation_masks():
        observation = _run_chain(image, source_sha256 or decoded_sha256)
    if hashlib.sha256(image.tobytes()).hexdigest() != decoded_sha256:
        raise RuntimeError("Source observer changed the input raster.")
    available = bool(observation["candidatePassed"])
    if available and "guarded-rays" not in observation.get("waveforms", {}).get("arms", {}):
        raise RuntimeError("Diagnostic waveform evidence is incomplete.")
    state = (
        "diagnostic_available"
        if available
        else "preferred_timing_available"
        if observation["branch"] == "existing-source-timing-retained"
        else "unresolved"
    )
    return {
        "schemaVersion": 2,
        "method": "source-photo-waveform-v1",
        "state": state,
        "source": {
            "sha256": source_sha256 or decoded_sha256,
            "identityKind": "encoded-source" if source_sha256 else "decoded-raster",
            "decodedRasterSha256": decoded_sha256,
            "imageSize": [image.shape[1], image.shape[0]],
            "unchanged": True,
        },
        "selectedDiagnosticArm": "guarded-rays" if available else None,
        "sourceIdentityState": "diagnostic_source_supported" if available else "unresolved",
        "productionEligible": False,
        "semanticIdentityConfirmed": False,
        "requiresReview": True,
        "quantitativeUseApproved": False,
        "clinicalMeaning": "unclassified",
        "quantitativeUncertainty": "unavailable",
        "reviewFlagsAreQualitative": True,
        "zeroReviewFlagDoesNotMeanSafe": True,
        "sourceGapsMayNotBeFilled": True,
        "observation": observation,
    }


__all__ = ["observe_source_photo"]
