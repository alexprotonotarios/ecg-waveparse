from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json


GATE_RESULT_VERSION = 1


def resolve_latest_gate_summary(
    gates_path: Path,
    results_root: Path,
) -> Path:
    gates = load_json(gates_path.resolve())
    suite_id = str(gates.get("suiteId", "")).strip()
    adapter = str(gates.get("adapter", "digitizer")).strip()
    candidate_id = str(gates.get("candidateId", "selector"))
    required_case_ids = set((gates.get("cases") or {}).keys())
    if not suite_id:
        raise ValueError(
            "Gate configuration must declare suiteId when --summary is omitted."
        )

    matches: list[tuple[str, str, Path]] = []
    for summary_path in results_root.resolve().glob("*/summary.json"):
        try:
            summary = load_json(summary_path)
        except (OSError, ValueError):
            continue
        if (
            str(summary.get("suiteId", "")) != suite_id
            or str(summary.get("adapter", "")) != adapter
        ):
            continue
        available_case_ids = {
            str(row.get("caseId"))
            for row in summary.get("caseResults", [])
            if str(row.get("candidateId")) == candidate_id
        }
        if not required_case_ids.issubset(available_case_ids):
            continue
        matches.append(
            (
                str(summary.get("createdAt", "")),
                str(summary.get("runId", "")),
                summary_path,
            )
        )

    if not matches:
        raise FileNotFoundError(
            f"No complete {adapter!r} summary for suite {suite_id!r} and "
            f"candidate {candidate_id!r} exists under {results_root.resolve()}."
        )
    return max(matches, key=lambda item: (item[0], item[1]))[2]


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return None


def evaluate_benchmark_gates(
    summary_path: Path,
    gates_path: Path,
) -> dict[str, Any]:
    summary_path = summary_path.resolve()
    gates_path = gates_path.resolve()
    summary = load_json(summary_path)
    gates = load_json(gates_path)
    if gates.get("version") != 1:
        raise ValueError("Benchmark gate configuration version must be 1.")

    candidate_id = str(gates.get("candidateId", "selector"))
    failures: list[dict[str, Any]] = []
    check_count = 0

    def fail(code: str, message: str, *, case_id: str | None = None) -> None:
        failures.append(
            {
                "code": code,
                "message": message,
                **({"caseId": case_id} if case_id else {}),
            }
        )

    candidate_summary = (summary.get("candidateSummaries") or {}).get(
        candidate_id
    )
    if not isinstance(candidate_summary, dict):
        fail(
            "candidate_summary_missing",
            f"Summary does not contain candidate {candidate_id!r}.",
        )
        candidate_summary = {}

    rows_by_case = {
        str(row.get("caseId")): row
        for row in summary.get("caseResults", [])
        if row.get("candidateId") == candidate_id
    }

    def expected_failure_matches(case_id: str, case_gates: dict[str, Any]) -> bool:
        if case_gates.get("expectedStatus", "completed") != "failed":
            return False
        row = rows_by_case.get(case_id)
        if not row or row.get("status") != "failed":
            return False
        expected_kind = case_gates.get("expectedFailureKind")
        if expected_kind is not None and row.get("failureKind") != expected_kind:
            return False
        expected_reason = case_gates.get("expectedPublicationReasonCode")
        actual_reason = (row.get("parameters") or {}).get(
            "publicationReasonCode"
        )
        return expected_reason is None or actual_reason == expected_reason

    global_gates = gates.get("global", {})
    maximum_failure_count = int(global_gates.get("maximumFailureCount", 0))
    check_count += 1
    failure_count = int(candidate_summary.get("failureCount", 0))
    expected_failure_count = sum(
        expected_failure_matches(str(case_id), case_gates)
        for case_id, case_gates in gates.get("cases", {}).items()
    )
    unexpected_failure_count = max(0, failure_count - expected_failure_count)
    if unexpected_failure_count > maximum_failure_count:
        fail(
            "candidate_failures",
            f"{candidate_id} has {unexpected_failure_count} unexpected "
            f"failure(s) after {expected_failure_count} explicitly expected "
            f"failure(s); maximum is {maximum_failure_count}.",
        )

    maximum_false_deflections = int(
        global_gates.get("maximumFalseDeflectionCount", 0)
    )
    check_count += 1
    false_deflections = int(
        candidate_summary.get(
            "falseDeflectionCountInOccludedRanges",
            0,
        )
    )
    if false_deflections > maximum_false_deflections:
        fail(
            "false_deflections",
            f"{candidate_id} has {false_deflections} annotation-region false "
            f"deflection(s); maximum is {maximum_false_deflections}.",
        )

    for case_id, case_gates in gates.get("cases", {}).items():
        row = rows_by_case.get(case_id)
        check_count += 1
        if not row:
            fail(
                "case_missing",
                f"No {candidate_id} result exists for required case {case_id}.",
                case_id=case_id,
            )
            continue

        expected_status = str(case_gates.get("expectedStatus", "completed"))
        check_count += 1
        if row.get("status") != expected_status:
            fail(
                "case_status",
                f"Required case {case_id} has status {row.get('status')!r}; "
                f"expected {expected_status!r}.",
                case_id=case_id,
            )
            continue
        if expected_status == "failed":
            expected_kind = case_gates.get("expectedFailureKind")
            if expected_kind is not None:
                check_count += 1
                if row.get("failureKind") != expected_kind:
                    fail(
                        "failure_kind",
                        f"{case_id} failure kind is {row.get('failureKind')!r}; "
                        f"expected {expected_kind!r}.",
                        case_id=case_id,
                    )
            expected_reason = case_gates.get(
                "expectedPublicationReasonCode"
            )
            if expected_reason is not None:
                check_count += 1
                actual_reason = (row.get("parameters") or {}).get(
                    "publicationReasonCode"
                )
                if actual_reason != expected_reason:
                    fail(
                        "publication_reason",
                        f"{case_id} publication reason is {actual_reason!r}; "
                        f"expected {expected_reason!r}.",
                        case_id=case_id,
                    )
            continue

        case_summary = row.get("summary") or {}
        check_count += 1
        if case_gates.get("requireComplete12Lead", True) and not case_summary.get(
            "complete12Lead"
        ):
            fail(
                "incomplete_12_lead",
                f"Required case {case_id} is not a complete 12-lead result.",
                case_id=case_id,
            )

        maximum_rmse = _finite_number(case_gates.get("maximumGlobalRmseUv"))
        if maximum_rmse is not None:
            check_count += 1
            actual_rmse = _finite_number(case_summary.get("globalRmseUv"))
            if actual_rmse is None or actual_rmse > maximum_rmse:
                fail(
                    "rmse_limit",
                    f"{case_id} global RMSE is {actual_rmse}; maximum is "
                    f"{maximum_rmse} µV.",
                    case_id=case_id,
                )

        minimum_auroc = _finite_number(
            case_gates.get("minimumUncertaintyAuroc")
        )
        if minimum_auroc is not None:
            check_count += 1
            actual_auroc = _finite_number(
                case_summary.get("macroMeanUncertaintyScoreAuroc")
            )
            if actual_auroc is None or actual_auroc < minimum_auroc:
                fail(
                    "uncertainty_auroc",
                    f"{case_id} uncertainty AUROC is {actual_auroc}; minimum is "
                    f"{minimum_auroc}.",
                    case_id=case_id,
                )

        minimum_safe_fraction = _finite_number(
            case_gates.get("minimumSafeUsableFraction")
        )
        if minimum_safe_fraction is not None:
            check_count += 1
            actual_safe_fraction = _finite_number(
                case_summary.get("macroMeanSafeUsableFraction")
            )
            if (
                actual_safe_fraction is None
                or actual_safe_fraction < minimum_safe_fraction
            ):
                fail(
                    "safe_usable_fraction",
                    f"{case_id} safe usable fraction is {actual_safe_fraction}; "
                    f"minimum is {minimum_safe_fraction}.",
                    case_id=case_id,
                )

        maximum_case_false_deflections = case_gates.get(
            "maximumFalseDeflectionCount"
        )
        if maximum_case_false_deflections is not None:
            check_count += 1
            actual_false_deflections = int(
                case_summary.get(
                    "falseDeflectionCountInOccludedRanges",
                    0,
                )
            )
            if actual_false_deflections > int(maximum_case_false_deflections):
                fail(
                    "case_false_deflections",
                    f"{case_id} has {actual_false_deflections} false "
                    f"deflection(s); maximum is "
                    f"{maximum_case_false_deflections}.",
                    case_id=case_id,
                )

        required_events = case_gates.get("requiredEvents", [])
        if not required_events:
            continue
        score_path_value = row.get("scorePath")
        if not score_path_value:
            fail(
                "score_missing",
                f"{case_id} has required morphology events but no score file.",
                case_id=case_id,
            )
            continue
        score_path = (
            Path(score_path_value)
            if Path(score_path_value).is_absolute()
            else (summary_path.parent / score_path_value).resolve()
        )
        score = load_json(score_path)
        for event_gate in required_events:
            check_count += 1
            lead = str(event_gate["lead"])
            event_id = str(event_gate["id"])
            events = (
                score.get("leads", {})
                .get(lead, {})
                .get("morphology", {})
                .get("events", [])
            )
            event = next(
                (
                    item
                    for item in events
                    if str(item.get("id")) == event_id
                ),
                None,
            )
            if event is None:
                fail(
                    "event_missing",
                    f"{case_id} is missing required {lead} event {event_id}.",
                    case_id=case_id,
                )
                continue
            if event_gate.get("requirePreserved", True) and not event.get(
                "preserved"
            ):
                fail(
                    "event_not_preserved",
                    f"{case_id} {lead} event {event_id} was not preserved.",
                    case_id=case_id,
                )
            if event_gate.get("requireTurningPoint", False) and not event.get(
                "candidateTurningPoint"
            ):
                fail(
                    "event_turning_point",
                    f"{case_id} {lead} event {event_id} is not a local turning "
                    "point.",
                    case_id=case_id,
                )
            minimum_prominence = _finite_number(
                event_gate.get("minimumLocalProminenceUv")
            )
            actual_prominence = _finite_number(
                event.get("candidateLocalProminenceUv")
            )
            if minimum_prominence is not None and (
                actual_prominence is None
                or actual_prominence < minimum_prominence
            ):
                fail(
                    "event_prominence",
                    f"{case_id} {lead} event {event_id} local prominence is "
                    f"{actual_prominence}; minimum is {minimum_prominence} µV.",
                    case_id=case_id,
                )

    return {
        "version": GATE_RESULT_VERSION,
        "summary": str(summary_path),
        "gates": str(gates_path),
        "candidateId": candidate_id,
        "checkCount": check_count,
        "passed": not failures,
        "failureCount": len(failures),
        "failures": failures,
    }
