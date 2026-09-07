from __future__ import annotations

import argparse
import binascii
import csv
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
from PIL import Image
if TYPE_CHECKING:
    from remotezip import RemoteZip

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg_benchmark.io import LEADS, dump_json, sha256_file, write_leads_csv
from ecg_benchmark.manifest import build_paired_manifest, validate_benchmark_manifest
from ecg_benchmark.paired_render import (
    canonical_page_segment_truth,
    visible_leads_for_page,
)


ROOT = Path(__file__).resolve().parents[1]
METADATA_MEMBER = "final_data/metadata.csv"
LEADS_MEMBER = "final_data/data/leads.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the external PMcardio paired image/waveform benchmark."
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=ROOT / "benchmark" / "suites" / "pmcardio-v1.json",
    )
    parser.add_argument(
        "--cache", type=Path, default=ROOT / ".external" / "pmcardio-v1"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark" / "generated" / "pmcardio-core-v1",
    )
    parser.add_argument("--profile", choices=("core", "extended"), default="core")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--request-pause-ms", type=int, default=250)
    return parser.parse_args([item for item in sys.argv[1:] if item != "--"])


def safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)


def recording_identity(value: str) -> str:
    """Noise-added signals remain variants of the same underlying recording."""
    return re.sub(r"^(?:high|low)_freq_noise_(?:large|middle|low)_", "", value)


def source_acquisition(category: str) -> str:
    if category.startswith("digital_data"):
        return "digital_export"
    if category.startswith("augmentation_"):
        return "digital_augmentation"
    return "scan" if category == "photos_scans" else "photograph"


def load_cache_index(cache: Path) -> dict[str, Any]:
    path = cache / "selective-cache.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "version": 1,
        "members": {},
    }


def cache_valid(path: Path, member: dict[str, Any] | None) -> bool:
    return bool(
        member
        and path.is_file()
        and path.stat().st_size == member.get("size")
        and sha256_file(path) == member.get("sha256")
    )


def read_member_with_retry(remote: RemoteZip, member: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            return remote.read(member)
        except Exception as error:  # Remote servers may transiently rate-limit Range reads.
            last_error = error
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"Could not read {member} after retries: {last_error}")


def cache_member(
    remote: RemoteZip | None,
    *,
    member: str,
    destination: Path,
    cache_index: dict[str, Any],
    offline: bool,
) -> None:
    recorded = cache_index["members"].get(member)
    if cache_valid(destination, recorded):
        return
    if offline:
        raise FileNotFoundError(f"Missing hash-valid cached PMcardio member {member}.")
    if remote is None:
        raise RuntimeError("Remote archive is not open.")
    payload = read_member_with_retry(remote, member)
    info = remote.getinfo(member)
    crc32 = binascii.crc32(payload) & 0xFFFFFFFF
    if len(payload) != info.file_size or crc32 != info.CRC:
        raise ValueError(f"ZIP size/CRC mismatch for {member}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.download")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    cache_index["members"][member] = {
        "size": len(payload),
        "crc32": f"{crc32:08x}",
        "sha256": sha256_file(destination),
    }


def archive_image_index(names: list[str]) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    prefix = "final_data/visual_data/"
    for member in names:
        if not member.startswith(prefix) or member.endswith("/"):
            continue
        relative = member[len(prefix) :]
        category, _, filename = relative.partition("/")
        if category and filename:
            index[(category, Path(filename).stem)] = member
    return index


def layout_name(format_name: str) -> str:
    if format_name.startswith("3x4"):
        return "standard_3x4"
    if format_name.startswith("6x2"):
        return "standard_6x2"
    if format_name.startswith("12x1"):
        return "standard_12x1"
    raise ValueError(f"Unsupported PMcardio layout {format_name}.")


def waveform_key(category: str, ecg_id: str, available: set[str]) -> str:
    if ecg_id in available and recording_identity(ecg_id) != ecg_id:
        candidate = ecg_id
    elif category.startswith("digital_data_high_freq_noise_"):
        candidate = f"high_freq_noise_{category.rsplit('_', 1)[1]}_{ecg_id}"
    elif category.startswith("digital_data_low_freq_noise_"):
        candidate = f"low_freq_noise_{category.rsplit('_', 1)[1]}_{ecg_id}"
    else:
        candidate = ecg_id
    if candidate not in available:
        raise KeyError(f"No PMcardio waveform {candidate} for {category}/{ecg_id}.")
    return candidate


def image_pixels_per_mm(
    *, image_width: int, layout: str, columns_per_page: int
) -> float:
    del layout, columns_per_page
    estimated_paper_width_mm = 24.0 + 10.0 * 25.0
    return image_width / estimated_paper_width_mm


def build_case(
    row: dict[str, str],
    *,
    category: str,
    member: str,
    image_path: Path,
    leads: np.lib.npyio.NpzFile,
    suite: dict[str, Any],
    output: Path,
    cache_index: dict[str, Any],
) -> dict[str, Any]:
    layout = layout_name(row["ECG format"])
    page = int(row["Image page"])
    columns_per_page = int(row["ECG number of columns per page"])
    visible_leads = visible_leads_for_page(
        layout=layout, page=page, columns_per_page=columns_per_page
    )
    key = waveform_key(category, row["ECG ID"], set(leads.files))
    segments = np.asarray(leads[key], dtype=np.float64)
    truth = canonical_page_segment_truth(
        segments,
        layout=layout,
        visible_leads=visible_leads,
        page=page,
        columns_per_page=columns_per_page,
    )
    case_id = safe_id(
        f"pm_{category}_{row['Image ID']}_page_{row['Image page']}"
    )
    case_output = output / "cases" / case_id
    case_output.mkdir(parents=True, exist_ok=True)
    truth_path = case_output / "truth.csv"
    annotations_path = case_output / "annotations.json"
    write_leads_csv(truth_path, truth, sample_rate_hz=int(suite["sampleRateHz"]))
    dump_json(
        annotations_path,
        {"version": 1, "caseId": case_id, "highErrorThresholdUv": 75, "leads": {}},
    )
    with Image.open(image_path) as image:
        width, _ = image.size
    ppm = image_pixels_per_mm(
        image_width=width, layout=layout, columns_per_page=columns_per_page
    )
    has_rhythm = int(row["ECG number of rhythm leads"]) > 0
    return {
        "caseId": case_id,
        "sourceId": f"pmcardio_{row['ECG ID']}_{category}_page_{page}",
        "groupId": f"pmcardio_{recording_identity(row['ECG ID'])}",
        "split": "heldout",
        "imagePath": str(image_path),
        "truthPath": str(truth_path),
        "annotationsPath": str(annotations_path),
        "sampleRateHz": suite["sampleRateHz"],
        "layout": layout,
        "paperSpeedMmPerSecond": suite["paperSpeedMmPerSecond"],
        "gainMmPerMv": suite["gainMmPerMv"],
        "segmentDurationSeconds": 10,
        "pixelsPerMm": ppm,
        "traceWidthPixels": max(1, round(ppm * 0.35)),
        "degradation": {
            "kind": category,
            "seed": int(row["Image ID"]),
            "parameters": {
                "archiveMember": member,
                "page": page,
                "columnsPerPage": columns_per_page,
            },
        },
        "strata": {
            "benchmarkTier": "external_artifact",
            "artifactFamily": category,
            "truthSupport": "layout_observed_panels",
            "truthPanelSamples": int(segments.shape[0]),
            "visibleLeads": ",".join(visible_leads),
            "rhythmLead": "",
            "rhythmRowsExcludedFromTruth": has_rhythm,
            "ecgFormat": row["ECG format"],
            "page": page,
            "pageCount": int(row["ECG number of pages"]),
            "sourceWaveformKey": key,
            "sourceLeadsMemberSha256": cache_index["members"][LEADS_MEMBER]["sha256"],
            "sourceMetadataMemberSha256": cache_index["members"][METADATA_MEMBER]["sha256"],
            "imageSha256": cache_index["members"][member]["sha256"],
            "lockedSplit": True,
        },
        "provenance": {
            "waveformOrigin": f"PMcardio archive source segments {key}",
            "quantitativeTruth": True,
            "clinicalValidationUse": False,
            "licence": suite["licence"],
            "licenceUrl": suite["licenceUrl"],
            "dataset": suite["dataset"],
            "datasetVersion": suite["datasetVersion"],
            "acquisition": source_acquisition(category),
        },
    }


def main() -> None:
    args = parse_args()
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    cache = args.cache.resolve()
    output = args.output.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    cache_index = load_cache_index(cache)
    cache_index.update(
        {
            "archiveUrl": suite["archiveUrl"],
            "archiveBytes": suite["archiveBytes"],
            "archiveMd5": suite["archiveMd5"],
            "metadataMemberSha256": cache_index["members"][METADATA_MEMBER]["sha256"],
            "leadsMemberSha256": cache_index["members"][LEADS_MEMBER]["sha256"],
        }
    )
    metadata_path = cache / "metadata.csv"
    leads_path = cache / "leads.npz"

    remote: RemoteZip | None = None
    try:
        if not args.offline:
            from remotezip import RemoteZip
            remote = RemoteZip(suite["archiveUrl"])
        cache_member(
            remote,
            member=METADATA_MEMBER,
            destination=metadata_path,
            cache_index=cache_index,
            offline=args.offline,
        )
        cache_member(
            remote,
            member=LEADS_MEMBER,
            destination=leads_path,
            cache_index=cache_index,
            offline=args.offline,
        )
        if remote is None:
            archive_index = {
                (entry["category"], entry["stem"]): member
                for member, entry in cache_index["members"].items()
                if "category" in entry
            }
        else:
            archive_index = archive_image_index(remote.namelist())

        with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        core = set(suite["coreCategories"])
        selected = [
            row
            for row in rows
            if args.profile == "extended"
            or row["Image relative path"].split("/", 1)[0] in core
        ]
        selected.sort(
            key=lambda row: (
                row["Image relative path"].split("/", 1)[0],
                int(row["Image ID"]),
                int(row["Image page"]),
            )
        )
        if args.limit:
            selected = selected[: args.limit]

        member_rows: list[tuple[dict[str, str], str, str]] = []
        for row in selected:
            category = row["Image relative path"].split("/", 1)[0]
            stem = Path(row["Image name"]).stem
            member = archive_index.get((category, stem))
            if not member:
                raise KeyError(f"Archive image is missing for {category}/{stem}.")
            member_rows.append((row, category, member))

        for index, (_, category, member) in enumerate(member_rows):
            destination = cache / "images" / category / Path(member).name
            cache_member(
                remote,
                member=member,
                destination=destination,
                cache_index=cache_index,
                offline=args.offline,
            )
            cache_index["members"][member].update(
                {"category": category, "stem": Path(member).stem}
            )
            if not args.offline and args.request_pause_ms:
                time.sleep(args.request_pause_ms / 1000.0)
            if (index + 1) % 25 == 0 or index + 1 == len(member_rows):
                dump_json(cache / "selective-cache.json", cache_index)
                print(f"cached {index + 1}/{len(member_rows)} images", flush=True)
    finally:
        if remote is not None:
            remote.close()
    dump_json(cache / "selective-cache.json", cache_index)

    cases: list[dict[str, Any]] = []
    with np.load(leads_path, allow_pickle=False) as leads:
        for index, (row, category, member) in enumerate(member_rows):
            image_path = cache / "images" / category / Path(member).name
            cases.append(
                build_case(
                    row,
                    category=category,
                    member=member,
                    image_path=image_path,
                    leads=leads,
                    suite=suite,
                    output=output,
                    cache_index=cache_index,
                )
            )
            if (index + 1) % 50 == 0 or index + 1 == len(member_rows):
                print(f"built {index + 1}/{len(member_rows)} cases", flush=True)

    suite_id = suite["suiteId"] if args.profile == "extended" else "pmcardio_core_v1"
    catalog_path = output / "paired-catalog.json"
    dump_json(catalog_path, {"version": 1, "suiteId": suite_id, "cases": cases})
    dump_json(
        output / "source-provenance.json",
        {
            "version": 1,
            "suitePath": str(args.suite.resolve()),
            "suiteSha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
            "archiveUrl": suite["archiveUrl"],
            "archiveBytes": suite["archiveBytes"],
            "archiveMd5": suite["archiveMd5"],
            "selectiveRangeExtraction": True,
            "profile": args.profile,
            "caseCount": len(cases),
            "clinicalValidationUse": False,
        },
    )
    manifest_path = output / "manifest.json"
    build_paired_manifest(
        catalog_path,
        manifest_path,
        catalog_schema_path=ROOT / "benchmark" / "schemas" / "paired-catalog.schema.json",
        manifest_schema_path=ROOT / "benchmark" / "schemas" / "case-manifest.schema.json",
        annotations_schema_path=ROOT / "benchmark" / "schemas" / "annotations.schema.json",
    )
    validate_benchmark_manifest(
        manifest_path,
        manifest_schema_path=ROOT / "benchmark" / "schemas" / "case-manifest.schema.json",
        annotations_schema_path=ROOT / "benchmark" / "schemas" / "annotations.schema.json",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
