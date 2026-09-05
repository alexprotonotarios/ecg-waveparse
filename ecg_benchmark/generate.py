from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .io import LEADS, dump_json, load_json, sha256_file, write_leads_csv


GENERATOR_VERSION = 1
LAYOUTS: dict[str, dict[str, Any]] = {
    "standard_6x2": {
        "rows": [
            ("I", "V1"),
            ("II", "V2"),
            ("III", "V3"),
            ("aVR", "V4"),
            ("aVL", "V5"),
            ("aVF", "V6"),
        ],
        "segmentSeconds": 5.0,
        "rowStepMm": 27.0,
        "columnGapMm": 9.0,
        "topMarginMm": 22.0,
        "bottomMarginMm": 16.0,
    },
    "standard_3x4": {
        "rows": [
            ("I", "aVR", "V1", "V4"),
            ("II", "aVL", "V2", "V5"),
            ("III", "aVF", "V3", "V6"),
        ],
        "segmentSeconds": 2.5,
        "rowStepMm": 36.0,
        "columnGapMm": 7.0,
        "topMarginMm": 28.0,
        "bottomMarginMm": 20.0,
    },
    "standard_12x1": {
        "rows": [(lead,) for lead in LEADS],
        "segmentSeconds": 10.0,
        "rowStepMm": 22.0,
        "columnGapMm": 0.0,
        "topMarginMm": 22.0,
        "bottomMarginMm": 16.0,
    },
}


def gaussian(time: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((time - center) / width) ** 2)


def engineering_waveforms(
    *,
    sample_rate_hz: float,
    duration_seconds: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Create deterministic morphology fixtures, not simulated clinical cases."""

    sample_count = int(round(sample_rate_hz * duration_seconds))
    time = np.arange(sample_count, dtype=np.float64) / sample_rate_hz
    beat_times = np.arange(0.62, duration_seconds - 0.35, 0.9)
    profiles: dict[str, tuple[float, float, float, float]] = {
        "I": (90, 620, -180, 190),
        "II": (130, 880, -240, 260),
        "III": (80, 720, -260, 160),
        "aVR": (-70, -620, 180, -150),
        "aVL": (55, 410, -130, 120),
        "aVF": (105, 760, -220, 210),
        "V1": (65, 320, -760, -190),
        "V2": (75, 640, -620, 120),
        "V3": (95, 920, -360, 230),
        "V4": (105, 1050, -230, 290),
        "V5": (100, 850, -150, 300),
        "V6": (85, 620, -110, 250),
    }
    leads: dict[str, np.ndarray] = {}
    for lead, (p_amplitude, r_amplitude, s_amplitude, t_amplitude) in profiles.items():
        signal = 16 * np.sin(2 * np.pi * 0.24 * time + LEADS.index(lead) * 0.07)
        for beat_index, beat in enumerate(beat_times):
            scale = 1.0 + 0.025 * math.sin(beat_index + LEADS.index(lead))
            signal += gaussian(time, beat - 0.18, 0.035, p_amplitude * scale)
            signal += gaussian(time, beat - 0.018, 0.009, -0.18 * r_amplitude)
            signal += gaussian(time, beat, 0.010, r_amplitude * scale)
            signal += gaussian(time, beat + 0.034, 0.012, s_amplitude * scale)
            signal += gaussian(time, beat + 0.28, 0.075, t_amplitude * scale)
            if lead == "V1":
                signal += gaussian(time, beat + 0.064, 0.008, 510 * scale)
                signal += gaussian(time, beat + 0.049, 0.006, -170 * scale)
            if lead == "V3":
                # Two close positive peaks separated by a narrow downward notch.
                signal += gaussian(time, beat + 0.026, 0.006, 410 * scale)
                signal += gaussian(time, beat + 0.014, 0.004, -330 * scale)
        leads[lead] = signal.astype(np.float64)
    # The 3x4 fixture has a 2.5-second panel and only two complete beats.
    # Keep annotations on a beat actually present in that panel.
    notch_beat = beat_times[min(2, len(beat_times) - 1)]
    return leads, {
        "v1RPrimeSeconds": float(beat_times[1] + 0.064),
        "v3FirstPeakSeconds": float(notch_beat),
        "v3SecondPeakSeconds": float(notch_beat + 0.026),
    }


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_grid(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    pixels_per_mm: float,
) -> None:
    step = max(1, int(round(pixels_per_mm)))
    for x in range(0, width, step):
        major = (x // step) % 5 == 0
        draw.line(
            ((x, 0), (x, height)),
            fill=(255, 151, 151) if major else (255, 220, 220),
            width=2 if major else 1,
        )
    for y in range(0, height, step):
        major = (y // step) % 5 == 0
        draw.line(
            ((0, y), (width, y)),
            fill=(255, 151, 151) if major else (255, 220, 220),
            width=2 if major else 1,
        )


def render_waveforms(
    leads: dict[str, np.ndarray],
    *,
    sample_rate_hz: float,
    layout_name: str,
    paper_speed_mm_per_second: float,
    gain_mm_per_mv: float,
    pixels_per_mm: float,
    trace_width_pixels: int,
) -> tuple[Image.Image, dict[str, dict[str, float]]]:
    layout = LAYOUTS[layout_name]
    rows = layout["rows"]
    segment_seconds = float(layout["segmentSeconds"])
    segment_width_mm = segment_seconds * paper_speed_mm_per_second
    left_margin_mm = 14.0
    right_margin_mm = 10.0
    signal_inset_mm = 7.0
    column_gap_mm = float(layout["columnGapMm"])
    top_margin_mm = float(layout["topMarginMm"])
    bottom_margin_mm = float(layout["bottomMarginMm"])
    row_step_mm = float(layout["rowStepMm"])
    column_count = max(len(row) for row in rows)
    width_mm = (
        left_margin_mm
        + column_count * segment_width_mm
        + (column_count - 1) * column_gap_mm
        + right_margin_mm
    )
    height_mm = top_margin_mm + row_step_mm * (len(rows) - 1) + bottom_margin_mm
    width = int(round(width_mm * pixels_per_mm))
    height = int(round(height_mm * pixels_per_mm))
    image = Image.new("RGB", (width, height), (255, 253, 250))
    draw = ImageDraw.Draw(image)
    _draw_grid(draw, width, height, pixels_per_mm)
    label_font = _font(max(12, int(round(3.0 * pixels_per_mm))), bold=True)
    header_font = _font(max(10, int(round(1.8 * pixels_per_mm))))
    draw.text(
        (int(8 * pixels_per_mm), int(3 * pixels_per_mm)),
        f"{gain_mm_per_mv:g} mm/mV   {paper_speed_mm_per_second:g} mm/s",
        fill=(25, 25, 25),
        font=header_font,
    )
    panel_geometry: dict[str, dict[str, float]] = {}
    sample_count = min(
        min(values.size for values in leads.values()),
        int(round(segment_seconds * sample_rate_hz)),
    )
    time = np.arange(sample_count, dtype=np.float64) / sample_rate_hz
    for row_index, row in enumerate(rows):
        baseline_mm = top_margin_mm + row_index * row_step_mm
        pulse_x = 4.0 * pixels_per_mm
        pulse_y = baseline_mm * pixels_per_mm
        pulse_height = gain_mm_per_mv * pixels_per_mm
        pulse_width = 5.0 * pixels_per_mm
        draw.line(
            (
                (pulse_x - 2 * pixels_per_mm, pulse_y),
                (pulse_x, pulse_y),
                (pulse_x, pulse_y - pulse_height),
                (pulse_x + pulse_width, pulse_y - pulse_height),
                (pulse_x + pulse_width, pulse_y),
                (pulse_x + pulse_width + 2 * pixels_per_mm, pulse_y),
            ),
            fill=(18, 18, 18),
            width=trace_width_pixels,
        )
        for column_index, lead in enumerate(row):
            column_x_mm = left_margin_mm + column_index * (
                segment_width_mm + column_gap_mm
            )
            waveform_x_mm = column_x_mm + signal_inset_mm
            draw.text(
                (
                    int((column_x_mm + 1.0) * pixels_per_mm),
                    int((baseline_mm - 13.0) * pixels_per_mm),
                ),
                lead,
                fill=(18, 18, 18),
                font=label_font,
            )
            signal = leads[lead][:sample_count]
            x = (
                waveform_x_mm + time * paper_speed_mm_per_second
            ) * pixels_per_mm
            y = (
                baseline_mm - signal / 1000.0 * gain_mm_per_mv
            ) * pixels_per_mm
            points = [(float(x_value), float(y_value)) for x_value, y_value in zip(x, y)]
            draw.line(
                points,
                fill=(18, 18, 18),
                width=trace_width_pixels,
                joint="curve",
            )
            panel_geometry[lead] = {
                "x0Pixels": waveform_x_mm * pixels_per_mm,
                "baselineYPixels": baseline_mm * pixels_per_mm,
                "pixelsPerSecond": paper_speed_mm_per_second * pixels_per_mm,
                "pixelsPerMv": gain_mm_per_mv * pixels_per_mm,
            }
    return image, panel_geometry


def apply_degradation(
    image: Image.Image,
    *,
    kind: str,
    parameters: dict[str, Any],
    panel_geometry: dict[str, dict[str, float]],
    leads: dict[str, np.ndarray],
    sample_rate_hz: float,
    feature_times: dict[str, float],
) -> Image.Image:
    if kind == "none":
        return image
    if kind == "blue_arrow":
        output = image.copy()
        draw = ImageDraw.Draw(output)
        lead = str(parameters.get("lead", "V1"))
        geometry = panel_geometry[lead]
        time_seconds = feature_times["v1RPrimeSeconds"]
        sample = min(
            leads[lead].size - 1, int(round(time_seconds * sample_rate_hz))
        )
        x = geometry["x0Pixels"] + geometry["pixelsPerSecond"] * time_seconds
        y = geometry["baselineYPixels"] - (
            leads[lead][sample] / 1000.0 * geometry["pixelsPerMv"]
        )
        arrow_top = max(4.0, y - 135.0)
        draw.line(
            ((x, arrow_top), (x, y - 5.0)),
            fill=(0, 35, 230),
            width=6,
        )
        draw.polygon(
            ((x - 13, y - 25), (x + 13, y - 25), (x, y + 7)),
            fill=(0, 35, 230),
        )
        return output
    if kind in {"notch_resolution", "low_resolution"}:
        target_width = int(parameters["targetWidthPixels"])
        target_height = max(1, int(round(image.height * target_width / image.width)))
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    if kind == "blur":
        return image.filter(
            ImageFilter.GaussianBlur(radius=float(parameters.get("radiusPixels", 1.4)))
        )
    if kind == "perspective":
        horizontal = float(parameters.get("horizontalInsetFraction", 0.025))
        vertical = float(parameters.get("verticalInsetFraction", 0.018))
        inset_x = image.width * horizontal
        inset_y = image.height * vertical
        return image.transform(
            image.size,
            Image.Transform.QUAD,
            (
                inset_x,
                inset_y,
                0,
                image.height,
                image.width,
                image.height - inset_y,
                image.width - inset_x,
                0,
            ),
            resample=Image.Resampling.BICUBIC,
            fillcolor=(255, 253, 250),
        )
    raise ValueError(f"Unsupported degradation kind: {kind}")


def annotations_for_case(
    *,
    case_id: str,
    kind: str,
    parameters: dict[str, Any],
    sample_rate_hz: float,
    feature_times: dict[str, float],
) -> dict[str, Any]:
    v1_sample = int(round(feature_times["v1RPrimeSeconds"] * sample_rate_hz))
    v3_first_sample = int(
        round(feature_times["v3FirstPeakSeconds"] * sample_rate_hz)
    )
    v3_second_sample = int(
        round(feature_times["v3SecondPeakSeconds"] * sample_rate_hz)
    )
    v1_ranges: list[dict[str, Any]] = []
    if kind == "blue_arrow":
        half_width = int(parameters.get("occlusionHalfWidthSamples", 14))
        v1_ranges.append(
            {
                "startSample": v1_sample - half_width,
                "endSample": v1_sample + half_width + 1,
                "reason": "blue_arrow_overlap",
                "falseDeflectionThresholdUv": float(
                    parameters.get("falseDeflectionThresholdUv", 75)
                ),
            }
        )
    return {
        "version": 1,
        "caseId": case_id,
        "highErrorThresholdUv": 75.0,
        "leads": {
            "V1": {
                "events": [
                    {
                        "id": "v1_r_prime",
                        "kind": "r_prime_peak",
                        "sample": v1_sample,
                        "direction": "peak",
                        "windowSamples": 6,
                        "minProminenceUv": 120.0,
                    }
                ],
                "occludedRanges": v1_ranges,
            },
            "V3": {
                "events": [
                    {
                        "id": "v3_r_peak",
                        "kind": "fragmented_qrs_peak",
                        "sample": v3_first_sample,
                        "direction": "peak",
                        "windowSamples": 4,
                        "minProminenceUv": 220.0,
                    },
                    {
                        "id": "v3_r_prime_notch",
                        "kind": "narrow_r_prime_peak",
                        "sample": v3_second_sample,
                        "direction": "peak",
                        "windowSamples": 4,
                        "minProminenceUv": 100.0,
                        "requiresTurningPoint": True,
                        "prominenceWindowSamples": 8,
                        "minLocalProminenceUv": 80.0,
                    },
                ],
                "occludedRanges": [],
            },
        },
    }


def validate_manifest(manifest: dict[str, Any], schema_path: Path) -> None:
    schema = load_json(schema_path)
    Draft202012Validator(schema).validate(manifest)
    case_ids = [str(case["caseId"]) for case in manifest["cases"]]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Benchmark case IDs must be unique.")
    group_splits: dict[str, set[str]] = {}
    for case in manifest["cases"]:
        group_splits.setdefault(str(case["groupId"]), set()).add(str(case["split"]))
    leaked = [group for group, splits in group_splits.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"Groups span multiple splits: {', '.join(leaked)}")


def validate_annotations(annotations: dict[str, Any], schema_path: Path) -> None:
    Draft202012Validator(load_json(schema_path)).validate(annotations)


def generate_suite(
    suite_path: Path,
    output_dir: Path,
    *,
    schema_path: Path,
) -> Path:
    suite = load_json(suite_path)
    if suite.get("version") != 1:
        raise ValueError("Unsupported benchmark suite version.")
    suite_id = str(suite["suiteId"])
    sample_rate_hz = float(suite["sampleRateHz"])
    duration_seconds = float(suite["durationSeconds"])
    layout_name = str(suite["layout"])
    layout_duration = float(LAYOUTS[layout_name]["segmentSeconds"])
    if abs(duration_seconds - layout_duration) > 1e-9:
        raise ValueError(
            f"Suite duration {duration_seconds} does not match {layout_name} "
            f"segment duration {layout_duration}."
        )
    paper_speed = float(suite["paperSpeedMmPerSecond"])
    gain = float(suite["gainMmPerMv"])
    pixels_per_mm = float(suite["render"]["pixelsPerMm"])
    trace_width = int(suite["render"]["traceWidthPixels"])
    leads, feature_times = engineering_waveforms(
        sample_rate_hz=sample_rate_hz,
        duration_seconds=duration_seconds,
    )
    base_image, panel_geometry = render_waveforms(
        leads,
        sample_rate_hz=sample_rate_hz,
        layout_name=layout_name,
        paper_speed_mm_per_second=paper_speed,
        gain_mm_per_mv=gain,
        pixels_per_mm=pixels_per_mm,
        trace_width_pixels=trace_width,
    )
    manifest_cases: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for case_definition in suite["cases"]:
        case_id = str(case_definition["caseId"])
        kind = str(case_definition["kind"])
        seed = int(case_definition["seed"])
        parameters = dict(case_definition.get("parameters", {}))
        case_dir = output_dir / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        truth_path = case_dir / "truth.csv"
        image_path = case_dir / "image.png"
        annotations_path = case_dir / "annotations.json"
        write_leads_csv(truth_path, leads, sample_rate_hz=sample_rate_hz)
        annotations = annotations_for_case(
            case_id=case_id,
            kind=kind,
            parameters=parameters,
            sample_rate_hz=sample_rate_hz,
            feature_times=feature_times,
        )
        validate_annotations(
            annotations,
            schema_path.parent / "annotations.schema.json",
        )
        dump_json(annotations_path, annotations)
        image = apply_degradation(
            base_image,
            kind=kind,
            parameters=parameters,
            panel_geometry=panel_geometry,
            leads=leads,
            sample_rate_hz=sample_rate_hz,
            feature_times=feature_times,
        )
        image.save(image_path, format="PNG", optimize=False)
        raster_scale = image.width / base_image.width
        effective_pixels_per_mm = pixels_per_mm * raster_scale
        effective_trace_width = max(1, int(round(trace_width * raster_scale)))
        horizontal_pixels_per_second = paper_speed * effective_pixels_per_mm
        manifest_cases.append(
            {
                "caseId": case_id,
                "sourceId": "engineering_morphology_fixture_v1",
                "groupId": "engineering_morphology_fixture_v1",
                "split": "smoke",
                "image": {
                    "path": str(image_path.relative_to(output_dir)),
                    "sha256": sha256_file(image_path),
                },
                "truth": {
                    "path": str(truth_path.relative_to(output_dir)),
                    "sha256": sha256_file(truth_path),
                },
                "annotations": {
                    "path": str(annotations_path.relative_to(output_dir)),
                    "sha256": sha256_file(annotations_path),
                },
                "sampleRateHz": sample_rate_hz,
                "units": "uV",
                "layout": layout_name,
                "paperSpeedMmPerSecond": paper_speed,
                "gainMmPerMv": gain,
                "segmentDurationSeconds": duration_seconds,
                "render": {
                    "widthPixels": image.width,
                    "heightPixels": image.height,
                    "pixelsPerMm": effective_pixels_per_mm,
                    "traceWidthPixels": effective_trace_width,
                    "horizontalPixelsPerSecond": horizontal_pixels_per_second,
                    "waveformSamplesPerHorizontalPixel": (
                        sample_rate_hz / horizontal_pixels_per_second
                    ),
                    "microvoltsPerVerticalPixel": (
                        1000.0 / (gain * effective_pixels_per_mm)
                    ),
                },
                "degradation": {
                    "kind": kind,
                    "seed": seed,
                    "parameters": parameters,
                },
                "strata": {
                    "artifact": kind,
                    "hasOcclusion": kind == "blue_arrow",
                    "morphologyFocus": (
                        "V1_r_prime"
                        if kind == "blue_arrow"
                        else "V3_narrow_notch"
                        if kind == "notch_resolution"
                        else "all"
                    ),
                    "widthPixels": image.width,
                },
                "provenance": {
                    "waveformOrigin": "deterministic engineering fixture",
                    "quantitativeTruth": True,
                    "clinicalValidationUse": False,
                    "licence": "repository-internal test fixture",
                    "dataset": "ecg-digitizer smoke fixtures",
                    "datasetVersion": "1",
                    "acquisition": "synthetic_render",
                },
            }
        )
    manifest = {
        "version": 1,
        "suiteId": suite_id,
        "generator": {
            "name": "ecg-benchmark-generator",
            "version": GENERATOR_VERSION,
            "suiteSha256": sha256_file(suite_path),
        },
        "cases": manifest_cases,
    }
    validate_manifest(manifest, schema_path)
    manifest_path = output_dir / "manifest.json"
    dump_json(manifest_path, manifest)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic paired ECG digitization benchmark suite."
    )
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("benchmark/schemas/case-manifest.schema.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = generate_suite(
        args.suite.resolve(),
        args.output.resolve(),
        schema_path=args.schema.resolve(),
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
