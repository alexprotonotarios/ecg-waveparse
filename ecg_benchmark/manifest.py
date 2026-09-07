from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator, FormatChecker
from PIL import Image

from .generate import validate_annotations, validate_manifest
from .io import LEADS, dump_json, load_json, read_leads, sha256_file


LAYOUT_COLUMN_COUNTS = {
    "standard_3x4": 4,
    "standard_6x2": 2,
    "standard_12x1": 1,
}


def expected_finite_truth_samples(
    case: dict[str, Any], lead: str, expected_samples: int
) -> int:
    """Return the source-observed truth support declared by a benchmark case.

    Synthetic benchmark cases contain a full duration for every lead. A rendered
    multi-column clinical ECG only displays one time panel per lead, with an
    optional full-width rhythm lead. Those paired cases keep the undisplayed
    intervals explicitly missing instead of inventing waveform truth.
    """
    strata = case.get("strata", {})
    if strata.get("truthSupport") != "layout_observed_panels":
        return expected_samples
    visible_leads = {
        value.strip()
        for value in str(strata.get("visibleLeads", "")).split(",")
        if value.strip()
    }
    if visible_leads and lead not in visible_leads:
        return 0
    if strata.get("rhythmLead") == lead:
        return expected_samples
    declared_panel_samples = strata.get("truthPanelSamples")
    if isinstance(declared_panel_samples, (int, float)):
        panel_samples = int(declared_panel_samples)
        if panel_samples < 1 or panel_samples > expected_samples:
            raise ValueError(
                f"{case.get('caseId', 'case')} has invalid truthPanelSamples."
            )
        return panel_samples
    column_count = LAYOUT_COLUMN_COUNTS.get(str(case.get("layout")))
    if not column_count or expected_samples % column_count:
        raise ValueError(
            f"{case.get('caseId', 'case')} has invalid layout-observed truth support."
        )
    return expected_samples // column_count


def resolve_catalog_path(catalog_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (catalog_path.parent / path).resolve()


def relative_path(output_path: Path, artifact_path: Path) -> str:
    return Path(os.path.relpath(artifact_path, start=output_path.parent)).as_posix()


def validate_catalog(catalog: dict[str, Any], schema_path: Path) -> None:
    Draft202012Validator(
        load_json(schema_path),
        format_checker=FormatChecker(),
    ).validate(catalog)


def validate_benchmark_manifest(
    manifest_path: Path,
    *,
    manifest_schema_path: Path,
    annotations_schema_path: Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    validate_manifest(manifest, manifest_schema_path)
    for case in manifest["cases"]:
        resolved: dict[str, Path] = {}
        for label in ("image", "truth", "annotations"):
            artifact = case[label]
            path = resolve_catalog_path(manifest_path, str(artifact["path"]))
            if not path.is_file():
                raise FileNotFoundError(
                    f"{case['caseId']} {label} file does not exist: {path}"
                )
            digest = sha256_file(path)
            if digest != artifact["sha256"]:
                raise ValueError(
                    f"{case['caseId']} {label} hash mismatch: "
                    f"expected {artifact['sha256']}, found {digest}."
                )
            resolved[label] = path

        with Image.open(resolved["image"]) as image:
            width, height = image.size
        render = case["render"]
        if (width, height) != (
            int(render["widthPixels"]),
            int(render["heightPixels"]),
        ):
            raise ValueError(
                f"{case['caseId']} image dimensions {(width, height)} do not "
                "match its manifest."
            )

        truth = read_leads(resolved["truth"])
        missing_leads = sorted(set(LEADS) - set(truth))
        if missing_leads:
            raise ValueError(
                f"{case['caseId']} truth is missing lead(s): "
                + ", ".join(missing_leads)
            )
        expected_samples = int(
            round(float(case["sampleRateHz"]) * float(case["segmentDurationSeconds"]))
        )
        for lead in LEADS:
            finite_samples = int(np.count_nonzero(np.isfinite(truth[lead])))
            expected_finite_samples = expected_finite_truth_samples(
                case, lead, expected_samples
            )
            if abs(finite_samples - expected_finite_samples) > 1:
                raise ValueError(
                    f"{case['caseId']} {lead} has {finite_samples} finite truth "
                    f"samples; expected approximately {expected_finite_samples}."
                )

        annotations = load_json(resolved["annotations"])
        validate_annotations(annotations, annotations_schema_path)
        if annotations["caseId"] != case["caseId"]:
            raise ValueError(
                f"{case['caseId']} annotation caseId is {annotations['caseId']}."
            )
        for lead, lead_annotations in annotations["leads"].items():
            for event in lead_annotations["events"]:
                if int(event["sample"]) >= expected_samples:
                    raise ValueError(
                        f"{case['caseId']} {lead} event {event['id']} is outside "
                        "the truth segment."
                    )
            for region in lead_annotations["occludedRanges"]:
                start = int(region["startSample"])
                end = int(region["endSample"])
                if end <= start or end > expected_samples:
                    raise ValueError(
                        f"{case['caseId']} {lead} occlusion [{start}, {end}) is "
                        "outside the truth segment."
                    )
        if not case["provenance"]["quantitativeTruth"]:
            raise ValueError(
                f"{case['caseId']} lacks quantitative waveform truth and cannot "
                "enter the scored benchmark."
            )
    return manifest


def build_paired_manifest(
    catalog_path: Path,
    output_path: Path,
    *,
    catalog_schema_path: Path,
    manifest_schema_path: Path,
    annotations_schema_path: Path,
) -> dict[str, Any]:
    catalog = load_json(catalog_path)
    validate_catalog(catalog, catalog_schema_path)
    manifest_cases: list[dict[str, Any]] = []
    for case in catalog["cases"]:
        image_path = resolve_catalog_path(catalog_path, str(case["imagePath"]))
        truth_path = resolve_catalog_path(catalog_path, str(case["truthPath"]))
        annotations_path = resolve_catalog_path(
            catalog_path, str(case["annotationsPath"])
        )
        for label, path in (
            ("image", image_path),
            ("truth", truth_path),
            ("annotations", annotations_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"{case['caseId']} {label} file does not exist: {path}"
                )
        truth_leads = read_leads(truth_path)
        missing_leads = sorted(set(LEADS) - set(truth_leads))
        if missing_leads:
            raise ValueError(
                f"{case['caseId']} truth is missing lead(s): "
                + ", ".join(missing_leads)
            )
        annotations = load_json(annotations_path)
        validate_annotations(annotations, annotations_schema_path)
        if annotations["caseId"] != case["caseId"]:
            raise ValueError(
                f"{case['caseId']} annotation caseId is {annotations['caseId']}."
            )
        with Image.open(image_path) as image:
            width, height = image.size
        provenance = dict(case["provenance"])
        if not provenance.get("quantitativeTruth"):
            raise ValueError(
                f"{case['caseId']} is not declared to have quantitative waveform truth."
            )
        manifest_cases.append(
            {
                "caseId": case["caseId"],
                "sourceId": case["sourceId"],
                "groupId": case["groupId"],
                "split": case["split"],
                "image": {
                    "path": relative_path(output_path, image_path),
                    "sha256": sha256_file(image_path),
                },
                "truth": {
                    "path": relative_path(output_path, truth_path),
                    "sha256": sha256_file(truth_path),
                },
                "annotations": {
                    "path": relative_path(output_path, annotations_path),
                    "sha256": sha256_file(annotations_path),
                },
                "sampleRateHz": case["sampleRateHz"],
                "units": "uV",
                "layout": case["layout"],
                "paperSpeedMmPerSecond": case["paperSpeedMmPerSecond"],
                "gainMmPerMv": case["gainMmPerMv"],
                "segmentDurationSeconds": case["segmentDurationSeconds"],
                "render": {
                    "widthPixels": width,
                    "heightPixels": height,
                    "pixelsPerMm": case["pixelsPerMm"],
                    "traceWidthPixels": case["traceWidthPixels"],
                    "horizontalPixelsPerSecond": (
                        case["paperSpeedMmPerSecond"] * case["pixelsPerMm"]
                    ),
                    "waveformSamplesPerHorizontalPixel": (
                        case["sampleRateHz"]
                        / (
                            case["paperSpeedMmPerSecond"]
                            * case["pixelsPerMm"]
                        )
                    ),
                    "microvoltsPerVerticalPixel": (
                        1000.0 / (case["gainMmPerMv"] * case["pixelsPerMm"])
                    ),
                },
                "degradation": case["degradation"],
                "strata": case["strata"],
                "provenance": provenance,
            }
        )
    manifest = {
        "version": 1,
        "suiteId": catalog["suiteId"],
        "generator": {
            "name": "paired-corpus-manifest-builder",
            "version": 1,
            "suiteSha256": sha256_file(catalog_path),
        },
        "cases": manifest_cases,
    }
    if "clinicalValidation" in catalog:
        manifest["clinicalValidation"] = dict(catalog["clinicalValidation"])
    validate_manifest(manifest, manifest_schema_path)
    dump_json(output_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hash and validate an existing paired ECG image/waveform corpus."
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--catalog-schema",
        type=Path,
        default=Path("benchmark/schemas/paired-catalog.schema.json"),
    )
    parser.add_argument(
        "--manifest-schema",
        type=Path,
        default=Path("benchmark/schemas/case-manifest.schema.json"),
    )
    parser.add_argument(
        "--annotations-schema",
        type=Path,
        default=Path("benchmark/schemas/annotations.schema.json"),
    )
    args = parser.parse_args()
    build_paired_manifest(
        args.catalog.resolve(),
        args.output.resolve(),
        catalog_schema_path=args.catalog_schema.resolve(),
        manifest_schema_path=args.manifest_schema.resolve(),
        annotations_schema_path=args.annotations_schema.resolve(),
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
