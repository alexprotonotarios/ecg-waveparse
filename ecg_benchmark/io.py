from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ecg_pipeline.domain import LEAD_ORDER


LEADS = LEAD_ORDER


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_leads(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns: dict[str, list[float]] = {
            lead: [] for lead in LEADS if lead in (reader.fieldnames or ())
        }
        for row in reader:
            for lead in columns:
                raw = (row.get(lead) or "").strip()
                try:
                    value = float(raw)
                except ValueError:
                    value = np.nan
                columns[lead].append(value if np.isfinite(value) else np.nan)
    return {lead: np.asarray(values, dtype=np.float64) for lead, values in columns.items()}


def write_leads_csv(
    path: Path,
    leads: dict[str, np.ndarray],
    *,
    sample_rate_hz: float,
) -> None:
    lengths = {values.size for values in leads.values()}
    if len(lengths) != 1:
        raise ValueError("All lead arrays must have the same length.")
    sample_count = lengths.pop() if lengths else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample", "t_s_at_500hz", *LEADS])
        for sample in range(sample_count):
            writer.writerow(
                [
                    sample,
                    f"{sample / sample_rate_hz:.9f}",
                    *[
                        f"{float(leads[lead][sample]):.9f}"
                        if np.isfinite(leads[lead][sample])
                        else ""
                        for lead in LEADS
                    ],
                ]
            )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(json_ready(value), indent=2, sort_keys=True)}\n", encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
