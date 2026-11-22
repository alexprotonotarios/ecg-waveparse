"""Retain observed short panels before destructive rhythm-column assignment.

This reads a fresh decoder observation. It changes neither the canonical export
nor the source; it performs no interpolation, gap filling or signal smoothing.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
PANELS = (("I", "aVR", "V1", "V4"), ("II", "aVL", "V2", "V5"), ("III", "aVF", "V3", "V6"))
METHOD = "observed-decoder-panel-preservation-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observed_panels(capture):
    capture = Path(capture)
    if capture.stat().st_size > 64 * 1024 ** 2:
        raise ValueError("decoder_capture_size_bound")
    with zipfile.ZipFile(capture) as archive:
        if sum(info.file_size for info in archive.infolist()) > 128 * 1024 ** 2:
            raise ValueError("decoder_capture_expansion_bound")
    with np.load(capture, allow_pickle=False) as arrays:
        envelope = json.loads(arrays["metadataJsonUtf8"].tobytes())
        metadata = envelope["metadata"]
        values = arrays["canonicalInputUv"].copy()
        canonical = arrays["canonicalUv"].copy()
    match = metadata["match"]
    provisional = match.get("provisionalLayoutHint") is True and match.get("leadIdentityConfirmed") is False
    supported_rows = (values.ndim == 2 and values.shape[1] == 5000 and
                      (4 <= values.shape[0] <= 6 if provisional else values.shape[0] == 4))
    if (match.get("layout") != "standard_3x4_with_r1" or match.get("flip") is not False or
            not supported_rows or canonical.shape != (12, 5000) or
            not np.issubdtype(values.dtype, np.floating) or not np.issubdtype(canonical.dtype, np.floating) or
            np.isinf(values).any() or np.isinf(canonical).any() or not np.isfinite(values[:3]).any()):
        raise ValueError("unsupported_observed_panel_geometry")
    panels = np.full_like(canonical, np.nan)
    for row, names in enumerate(PANELS):
        for column, name in enumerate(names):
            start, end = column * 1250, (column + 1) * 1250
            panels[LEADS.index(name), start:end] = values[row, start:end]
    replay = panels.copy()
    assignments = metadata["rhythmAssignments"]
    if len(assignments) != 1 or assignments[0].get("rhythm") != [0] or len(assignments[0].get("canonical", [])) != 1:
        raise ValueError("unambiguous_observed_rhythm_assignment_required")
    destination = assignments[0]["canonical"][0]
    if type(destination) is not int or not 0 <= destination < 12:
        raise ValueError("invalid_observed_rhythm_destination")
    replay[destination] = values[-1]
    if not np.array_equal(replay, canonical, equal_nan=True):
        raise ValueError("original_canonical_routes_do_not_replay")
    return panels, values.copy(), LEADS[destination]


def write_csv(path, names, values):
    with Path(path).open("x", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(names)
        writer.writerows([float(value) if np.isfinite(value) else "NaN" for value in row] for row in values.T)


def recover(capture, source, output, *, capture_sha256, source_sha256):
    if digest(capture) != capture_sha256 or digest(source) != source_sha256:
        raise ValueError("panel_recovery_source_or_capture_changed")
    panels, observed_rows, replaced_lead = observed_panels(capture)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    primary = output / "candidate-signal.csv"
    extra = output / "candidate-additional-traces.csv"
    write_csv(primary, LEADS, panels)
    write_csv(extra, [f"observed_decoder_row_{index}" for index in range(len(observed_rows))], observed_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    preview = output / "candidate-preview.png"
    figure = plt.figure(figsize=(15, 8), constrained_layout=True)
    axes = figure.subplots(1, 2)
    with Image.open(source) as image:
        image.thumbnail((1600, 1600))
        axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Untouched source")
    spacing = max(500.0, float(np.nanpercentile(np.abs(panels), 98)) * 2.5)
    for row, names in enumerate(PANELS):
        for column, name in enumerate(names):
            values = panels[LEADS.index(name)]
            axes[1].plot(values - row * spacing, linewidth=0.8, label=name)
            finite = np.flatnonzero(np.isfinite(values))
            if finite.size:
                axes[1].text(int(finite[0]), -row * spacing + spacing * 0.2, name + " (inferred)", fontsize=7)
    for index in range(3, len(observed_rows)):
        axes[1].plot(observed_rows[index] - index * spacing, color="gray", linewidth=0.8)
        axes[1].text(0, -index * spacing + spacing * 0.2,
                     "Additional observed row; identity unverified" if len(observed_rows) != 4 else
                     "Additional rhythm row; identity unverified", fontsize=7)
    axes[1].set(xlabel="Decoder sample index; timing unverified", ylabel="Decoder value; gain unverified",
                title="Preserved short panels and separate rhythm; gaps unchanged")
    figure.savefig(preview, dpi=130)
    plt.close(figure)
    if digest(capture) != capture_sha256 or digest(source) != source_sha256:
        raise ValueError("panel_recovery_input_changed_during_export")
    with np.load(capture, allow_pickle=False) as arrays:
        metadata = json.loads(arrays["metadataJsonUtf8"].tobytes())["metadata"]
        match = metadata["match"]
    provisional = match.get("provisionalLayoutHint") is True
    return {"version": 1, "method": METHOD, "state": "recovered", "sourceSha256": source_sha256,
            "captureSha256": capture_sha256, "originalRhythmDestination": replaced_lead,
            "canonicalExportChanged": False, "waveformChanged": False, "gapsPreserved": True,
            "leadIdentityConfirmed": False, "calibrationVerified": False, "quantitativeUseApproved": False,
            "clinicalValidationUse": False, "csvPath": str(primary), "previewPath": str(preview),
            "additionalTracesPath": str(extra), "signalSha256": digest(primary), "previewSha256": digest(preview),
            "additionalTracesContainAllObservedDecoderRows": True,
            "additionalTracesSha256": digest(extra),
            **({"provisionalLayoutHint": True, "observedDecoderRows": len(observed_rows)} if provisional else {}),
            **({"sourceInkRecovery": metadata["sourceInkRecovery"]} if "sourceInkRecovery" in metadata else {}),
            "finiteByColumn": {name: int(np.isfinite(panels[i]).sum()) for i, name in enumerate(LEADS)}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ["capture", "source", "output", "capture-sha256", "source-sha256"]:
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args.capture, args.source, args.output,
                             capture_sha256=args.capture_sha256, source_sha256=args.source_sha256)))
