from __future__ import annotations

import json
from pathlib import Path
from typing import Final


_DOMAIN_PATH = Path(__file__).resolve().parents[1] / "config" / "ecg-domain.v1.json"
_DOMAIN = json.loads(_DOMAIN_PATH.read_text(encoding="utf-8"))
if _DOMAIN.get("version") != 1:
    raise RuntimeError(f"Unsupported ECG domain version in {_DOMAIN_PATH}")

SAMPLE_RATE_HZ: Final[int] = int(_DOMAIN["sampleRateHz"])
PAGE_DURATION_SECONDS: Final[int] = int(_DOMAIN["pageDurationSeconds"])
LEAD_ORDER: Final[tuple[str, ...]] = tuple(_DOMAIN["leads"])
if LEAD_ORDER != (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
):
    raise RuntimeError("The ECG domain does not contain the canonical lead order.")

LAYOUT_ROWS: Final[dict[str, tuple[tuple[str, ...], ...]]] = {
    name: tuple(tuple(row) for row in spec["rows"])
    for name, spec in _DOMAIN["layouts"].items()
}
SIX_BY_TWO_ROW_LEADS: Final[tuple[tuple[str, ...], ...]] = LAYOUT_ROWS[
    "standard_6x2"
]
THREE_BY_FOUR_ROW_LEADS: Final[tuple[tuple[str, ...], ...]] = LAYOUT_ROWS[
    "standard_3x4"
]
TWELVE_ROW_LEADS: Final[tuple[tuple[str, ...], ...]] = LAYOUT_ROWS[
    "standard_12x1"
]
SIX_BY_TWO_PANEL_SAMPLES: Final[int] = (
    SAMPLE_RATE_HZ * PAGE_DURATION_SECONDS // 2
)
THREE_BY_FOUR_PANEL_SAMPLES: Final[int] = (
    SAMPLE_RATE_HZ * PAGE_DURATION_SECONDS // 4
)
