"""Apply existing quality gates to an installed-package engineering regression."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_benchmark.gates import evaluate_benchmark_gates


def check_run_safety(report: dict, gates: dict) -> tuple[int, list[dict]]:
    """Keep application review state separate from benchmark completion status."""
    cases = {case["id"]: case for case in report["cases"]}
    failures = []
    count = 0
    for case_id, case_gates in gates["cases"].items():
        rules = {**gates.get("runSafety", {}), **case_gates.get("runSafety", {})}
        if not rules:
            continue
        case = cases.get(f"{gates['suiteId']}__{case_id}", {})
        decision = case.get("publicationDecision") or {}
        for key, actual in (
            ("expectedStatus", case.get("status")),
            ("expectedPublicationOutcome", decision.get("outcome")),
            ("expectedPublicationReasonCode", decision.get("reasonCode")),
            ("reviewRequired", case.get("reviewRequired")),
        ):
            if key not in rules:
                continue
            count += 1
            expected = rules[key]
            # Missing review evidence or truthy non-booleans must not pass.
            if type(actual) is not type(expected) or actual != expected:
                failures.append({"code": "run_safety", "caseId": case_id,
                    "message": f"{case_id} {key} is {actual!r}; expected {expected!r}."})
        if "maximumEffectiveSampleRateHz" in rules:
            count += 1
            actual = case.get("effectiveSampleRateHz")
            maximum = rules["maximumEffectiveSampleRateHz"]
            if (type(actual) not in (int, float) or not math.isfinite(actual)
                    or not 0 < actual <= maximum):
                failures.append({"code": "effective_sample_rate", "caseId": case_id,
                    "message": f"{case_id} effective sample rate is {actual!r}; "
                    f"expected a positive finite rate no greater than {maximum} Hz."})
    return count, failures


def check(report_path: Path, gates_path: Path, output_path: Path) -> dict:
    report = json.loads(report_path.read_text())
    gates = json.loads(gates_path.read_text())
    prefix = gates["suiteId"] + "__"
    rows = []
    for case in report["cases"]:
        if not case["id"].startswith(prefix):
            continue
        rows.append({
            "caseId": case["id"][len(prefix):], "candidateId": "selector",
            "status": "completed" if case["outcome"] == "quantitative_needs_review" else "failed",
            "failureKind": "publication_abstention" if case["outcome"] == "abstention" else case["outcome"],
            "parameters": {"publicationReasonCode": (case.get("publicationDecision") or {}).get("reasonCode")},
            "summary": case.get("score"),
            "scorePath": str((report_path.parent / f"{case['id']}.score.json").resolve()),
        })
    summary = {"suiteId": gates["suiteId"], "adapter": "digitizer", "caseResults": rows,
               "candidateSummaries": {"selector": {
                   "failureCount": sum(row["status"] == "failed" for row in rows),
                   "falseDeflectionCountInOccludedRanges": sum((row["summary"] or {}).get("falseDeflectionCountInOccludedRanges", 0) for row in rows),
               }}}
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    result = evaluate_benchmark_gates(summary_path, gates_path)
    safety_count, safety_failures = check_run_safety(report, gates)
    result["checkCount"] += safety_count
    result["failures"].extend(safety_failures)
    result["failureCount"] = len(result["failures"])
    result["passed"] = not result["failures"]
    result["regressionReportSha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    result["gateConfigurationSha256"] = hashlib.sha256(gates_path.read_bytes()).hexdigest()
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.report, args.gates, args.output)
    print(json.dumps({key: result[key] for key in ["passed", "checkCount", "failureCount", "failures"]}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
