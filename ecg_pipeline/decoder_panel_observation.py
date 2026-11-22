"""Observe numeric decoder panels without claiming source coordinates.

The observation precedes rhythm assignment and the upstream PNG writer. It
never changes a decoder value, route, gap, or exported canonical signal.
This upstream-interface extension is offered under CC BY-SA 4.0; see notices.
"""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from threading import Lock
from unittest.mock import patch

import numpy as np


_OBSERVATION_LOCK = Lock()


class DecoderPanelObservation:
    def __init__(self, identifier):
        while hasattr(identifier, "identifier"):
            identifier = identifier.identifier
        self.identifier = identifier
        self.data = {}
        self._inside_canonical = False

    def __enter__(self):
        import src.model.lead_identifier as module

        if self.data or not _OBSERVATION_LOCK.acquire(blocking=False):
            raise ValueError("numeric_panel_observer_not_reusable_or_concurrent")
        self.stack = ExitStack()
        self.stack.callback(_OBSERVATION_LOCK.release)
        try:
            canonicalize = self.identifier._canonicalize_lines
            assignment = module.linear_sum_assignment

            def observed_assignment(*args, **kwargs):
                result = assignment(*args, **kwargs)
                if self._inside_canonical:
                    self.data["rhythmAssignments"].append({
                        "rhythm": result[0].tolist(), "canonical": result[1].tolist(),
                    })
                return result

            def observed_canonicalize(lines, match):
                if self.data:
                    raise ValueError("duplicate_numeric_panel_observation")
                self.data.update(canonicalInputUv=lines.detach().cpu().numpy().copy(),
                                 match=deepcopy(match), rhythmAssignments=[])
                self._inside_canonical = True
                try:
                    result = canonicalize(lines, match)
                finally:
                    self._inside_canonical = False
                self.data["canonicalUv"] = result.detach().cpu().numpy().copy()
                return result

            self.stack.enter_context(patch.object(module, "linear_sum_assignment", observed_assignment))
            self.stack.enter_context(patch.object(self.identifier, "_canonicalize_lines", observed_canonicalize))
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def save_if_supported(self, path):
        """An unsupported geometry leaves normal inference unchanged."""
        match = self.data.get("match", {})
        if match.get("layout") != "standard_3x4_with_r1" or match.get("flip") is not False:
            return False
        values, canonical = self.data.get("canonicalInputUv"), self.data.get("canonicalUv")
        provisional = match.get("provisionalLayoutHint") is True and match.get("leadIdentityConfirmed") is False
        supported_rows = (isinstance(values, np.ndarray) and values.ndim == 2 and values.shape[1] == 5000 and
                          (4 <= values.shape[0] <= 6 if provisional else values.shape[0] == 4))
        if (not supported_rows or
                not isinstance(canonical, np.ndarray) or canonical.shape != (12, 5000) or
                not np.issubdtype(values.dtype, np.floating) or
                not np.issubdtype(canonical.dtype, np.floating) or
                np.isinf(values).any() or np.isinf(canonical).any()):
            return False

        def clean(value):
            if isinstance(value, np.generic):
                return clean(value.item())
            if isinstance(value, dict):
                return {str(key): clean(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(item) for item in value]
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value

        metadata = {"version": 1, "method": "observed-decoder-numeric-panels-v1",
                    "coordinateEvidenceExported": False, "waveformChanged": False,
                    "quantitativeUseApproved": False, "clinicalValidationUse": False,
                    "metadata": {key: self.data[key] for key in ["match", "rhythmAssignments"]}}
        if "sourceInkRecovery" in self.data:
            metadata["metadata"]["sourceInkRecovery"] = self.data["sourceInkRecovery"]
        encoded = json.dumps(clean(metadata), allow_nan=False, sort_keys=True).encode("utf-8")
        with Path(path).open("xb") as handle:
            np.savez_compressed(handle, canonicalInputUv=values, canonicalUv=canonical,
                                metadataJsonUtf8=np.frombuffer(encoded, dtype=np.uint8))
        return True
