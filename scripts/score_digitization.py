from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg_benchmark.io import dump_json, load_json
from ecg_benchmark.coordinates import contract_for_case
from ecg_benchmark.scoring import (
    LEADS,
    align_and_score,
    finite_segment,
    read_leads,
    resample,
    score_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score a digitized ECG against paired waveform ground truth. "
            "Inputs may be canonical or compact CSVs with standard lead headers."
        )
    )
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--truth-rate", type=float, default=500.0)
    parser.add_argument("--candidate-rate", type=float, default=500.0)
    parser.add_argument("--max-alignment-ms", type=float, default=40.0)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--uncertainty", type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--coordinate-contract", type=Path, help="Independent expected segment/display support, version 1.")
    parser.add_argument("--case-metadata", type=Path, help="Independent benchmark case metadata defining the expected layout and duration.")
    parser.add_argument("--candidate-segments", type=Path, help="Exported segment map; never used as the expected placement.")
    parser.add_argument(
        "--expected-leads",
        help="Comma-separated leads that are visibly present in the source image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = score_files(
        args.truth,
        args.candidate,
        truth_rate=args.truth_rate,
        candidate_rate=args.candidate_rate,
        max_alignment_ms=args.max_alignment_ms,
        annotations_path=args.annotations,
        uncertainty_path=args.uncertainty,
        case_id=args.case_id,
        coordinate_contract=(load_json(args.coordinate_contract) if args.coordinate_contract else
                             contract_for_case(load_json(args.case_metadata)) if args.case_metadata else None),
        candidate_segments=load_json(args.candidate_segments)["segments"] if args.candidate_segments else None,
        expected_leads=(
            [lead.strip() for lead in args.expected_leads.split(",") if lead.strip()]
            if args.expected_leads
            else None
        ),
    )
    dump_json(args.output, report)


if __name__ == "__main__":
    main()
