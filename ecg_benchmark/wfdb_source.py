from __future__ import annotations

import hashlib
import math
import re
import urllib.request
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import loadmat

from .io import LEADS, sha256_file


GAIN_PATTERN = re.compile(
    r"^(?P<gain>[0-9.eE+-]+)(?:\((?P<baseline>-?[0-9]+)\))?/(?P<unit>[^ ]+)$"
)
LEAD_ALIASES = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}


@dataclass(frozen=True)
class WfdbHeader:
    record_id: str
    signal_file: str
    sample_rate_hz: float
    sample_count: int
    leads: tuple[str, ...]
    gains_per_mv: tuple[float, ...]
    baselines: tuple[int, ...]
    diagnoses: tuple[str, ...]


def download_verified(
    destination: Path,
    *,
    url: str,
    expected_sha256: str,
    offline: bool,
) -> None:
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        return
    if offline:
        raise FileNotFoundError(f"Missing hash-valid cached source {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.download")
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "ecg-digitizer-benchmark/1"})
    with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as target:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            target.write(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Downloaded hash mismatch for {url}: {actual}.")
    temporary.replace(destination)


def parse_checksum_index(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.strip().split(maxsplit=1)
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError(f"Invalid SHA-256 entry in {path}: {line}")
        checksums[relative] = digest
    return checksums


def parse_wfdb_header(text: str) -> WfdbHeader:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    identity = lines[0].split()
    if len(identity) < 4:
        raise ValueError("WFDB identity line is incomplete.")
    record_id = identity[0]
    channel_count = int(identity[1])
    sample_rate_hz = float(identity[2].split("/")[0])
    sample_count = int(identity[3])
    if channel_count != 12:
        raise ValueError(f"Expected 12 channels, received {channel_count}.")
    if len(lines) < channel_count + 1:
        raise ValueError("WFDB header is missing signal lines.")

    signal_file = ""
    leads: list[str] = []
    gains: list[float] = []
    baselines: list[int] = []
    for line in lines[1 : channel_count + 1]:
        fields = line.split()
        if len(fields) < 9:
            raise ValueError(f"Malformed WFDB signal line: {line}")
        signal_file = signal_file or fields[0]
        if fields[0] != signal_file:
            raise ValueError("Multi-file WFDB records are not supported.")
        match = GAIN_PATTERN.match(fields[2])
        if not match or match.group("unit").lower() != "mv":
            raise ValueError(f"Unsupported WFDB calibration token {fields[2]}.")
        raw_lead = fields[-1]
        lead = LEAD_ALIASES.get(raw_lead.upper(), raw_lead)
        leads.append(lead)
        gains.append(float(match.group("gain")))
        baselines.append(
            int(match.group("baseline"))
            if match.group("baseline") is not None
            else int(fields[4])
        )
    if tuple(leads) != tuple(LEADS):
        raise ValueError(f"Unexpected lead order {leads}.")
    diagnoses: tuple[str, ...] = ()
    for line in lines[channel_count + 1 :]:
        if line.lower().startswith("# dx:"):
            diagnoses = tuple(
                value.strip() for value in line.split(":", 1)[1].split(",") if value.strip()
            )
            break
    return WfdbHeader(
        record_id=record_id,
        signal_file=signal_file,
        sample_rate_hz=sample_rate_hz,
        sample_count=sample_count,
        leads=tuple(leads),
        gains_per_mv=tuple(gains),
        baselines=tuple(baselines),
        diagnoses=diagnoses,
    )


def read_mat_waveform(path: Path, header: WfdbHeader) -> np.ndarray:
    contents = loadmat(path)
    if "val" not in contents:
        raise ValueError(f"{path} does not contain a WFDB val matrix.")
    digital = np.asarray(contents["val"], dtype=np.float64)
    if digital.shape == (12, header.sample_count):
        digital = digital.T
    if digital.shape != (header.sample_count, 12):
        raise ValueError(
            f"{path} has shape {digital.shape}; expected {(header.sample_count, 12)}."
        )
    return (
        (digital - np.asarray(header.baselines)[None, :])
        / np.asarray(header.gains_per_mv)[None, :]
        * 1000.0
    )


def ten_second_segment(
    microvolts: np.ndarray,
    *,
    source_rate_hz: float,
    seed: int,
    output_rate_hz: int = 500,
) -> dict[str, np.ndarray]:
    source_count = int(round(10.0 * source_rate_hz))
    if microvolts.shape[0] < source_count:
        raise ValueError("Waveform is shorter than ten seconds.")
    maximum_start = microvolts.shape[0] - source_count
    start = seed % (maximum_start + 1) if maximum_start else 0
    segment = microvolts[start : start + source_count]
    if not math.isclose(source_rate_hz, output_rate_hz):
        ratio = Fraction(output_rate_hz / source_rate_hz).limit_denominator(10_000)
        segment = signal.resample_poly(
            segment,
            up=ratio.numerator,
            down=ratio.denominator,
            axis=0,
            padtype="line",
        )
    expected = output_rate_hz * 10
    if segment.shape[0] < expected:
        segment = np.pad(segment, ((0, expected - segment.shape[0]), (0, 0)), mode="edge")
    segment = segment[:expected]
    return {lead: segment[:, index].copy() for index, lead in enumerate(LEADS)}
