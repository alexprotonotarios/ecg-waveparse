"""Independent-unit accounting. Thresholds are supplied by a frozen protocol, never fitted here."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np


def validate_independence(cases: list[dict], *, require_provenance: bool = False) -> dict:
    ids: set[str] = set()
    membership: dict[tuple[str, str], set[str]] = defaultdict(set)
    groups: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        case_id, split = case.get("caseId"), case.get("split")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError("Case IDs must be nonempty and unique.")
        ids.add(case_id)
        if not isinstance(split, str) or not split or not case.get("groupId"):
            raise ValueError("Every case requires a split and independent group identity.")
        groups[split].add(case["groupId"])
        for key in ("groupId", "sourceId", "recordingId", "patientId", "signalFamilyId"):
            if case.get(key):
                membership[(key, str(case[key]))].add(split)
        for key in ("image", "truth"):
            digest = (case.get(key) or {}).get("sha256")
            if digest:
                membership[(f"{key}Sha256", digest)].add(split)
        if require_provenance:
            provenance = case.get("evaluationProvenance") or {}
            if provenance.get("collection") not in {"regression", "development", "final_evaluation"}:
                raise ValueError("Declare the evaluation collection.")
            if provenance.get("pretrainedTrainingOverlap") not in {"known", "none_documented", "unknown"}:
                raise ValueError("Declare pretrained-model training overlap, including unknown.")
            if not provenance.get("licenceEvidence") or not provenance.get("truthMethod"):
                raise ValueError("Licence evidence and truth provenance are required.")
            if provenance["collection"] == "final_evaluation" and (provenance.get("previouslyExposed") is not False or not provenance.get("accessControlReference")):
                raise ValueError("Final evaluation requires an unexposed, access-controlled collection.")
    leaked = [key for key, splits in membership.items() if len(splits) > 1]
    if leaked:
        # Do not disclose patient/source identifiers in a default exception.
        raise ValueError(f"Independent identities cross splits ({len(leaked)} collisions).")
    digest = hashlib.sha256(json.dumps(sorted((case["caseId"], case["groupId"], case["split"]) for case in cases), separators=(",", ":")).encode()).hexdigest()
    return {"version": 1, "caseCount": len(cases), "splitCaseCounts": dict(Counter(case["split"] for case in cases)),
            "splitGroupCounts": {split: len(group) for split, group in sorted(groups.items())}, "membershipSha256": digest}


def reference_qualification(score: dict | None, *, semantic_status: str | None, gates: dict | None) -> dict:
    if gates is None:
        return {"status": "not_evaluated", "reasons": ["no_prespecified_reference_gates"]}
    if not isinstance(gates, dict) or not gates:
        raise ValueError("Reference qualification requires nonempty prespecified endpoint gates.")
    reasons = []
    if semantic_status != "passed":
        reasons.append("semantic_evidence_not_passed")
    summary = (score or {}).get("summary", score or {})
    for key, rule in gates.items():
        if not isinstance(rule, dict) or not set(rule) <= {"minimum", "maximum", "equals"} or not rule:
            raise ValueError("Invalid prespecified endpoint rule.")
        value = summary.get(key)
        if value is None:
            reasons.append(f"{key}:unavailable")
            continue
        if "equals" in rule and value != rule["equals"]:
            reasons.append(f"{key}:mismatch")
        for op in ("minimum", "maximum"):
            if op in rule:
                bound = rule[op]
                if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not np.isfinite(bound):
                    raise ValueError("Endpoint bounds must be finite numbers.")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or (value < bound if op == "minimum" else value > bound):
                    reasons.append(f"{key}:{op}")
    return {"status": "failed" if reasons else "passed", "reasons": reasons}


def denominator_scorecard(expected_case_ids: list[str], rows: list[dict], *, gates: dict | None = None) -> dict:
    expected = set(expected_case_ids)
    if len(expected) != len(expected_case_ids) or not expected:
        raise ValueError("Evaluation denominator must contain unique attempted case IDs.")
    by_id = {}
    for row in rows:
        case_id = row.get("caseId", row.get("id"))
        if case_id not in expected or case_id in by_id:
            raise ValueError("Duplicate or unexpected result cannot enter the denominator.")
        by_id[case_id] = row
    counts: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    decisions = {}
    returned = []
    for case_id in expected_case_ids:
        row = by_id.get(case_id)
        if row is None:
            counts["missing_result"] += 1
            outcomes["missing_result"] += 1
            decisions[case_id] = {"status": "failed", "reasons": ["missing_result"]}
            continue
        outcome = row.get("outcome") or row.get("failureKind") or row.get("status", "unknown")
        outcomes[str(outcome)] += 1
        score = row.get("score")
        summary = (score or {}).get("summary", score or {})
        compared = summary.get("comparedSamples")
        rmse = summary.get("globalRmseUv")
        finite_comparison = (isinstance(compared, (int, float)) and not isinstance(compared, bool) and np.isfinite(compared) and compared > 0) if compared is not None else (isinstance(rmse, (int, float)) and not isinstance(rmse, bool) and np.isfinite(rmse))
        has_signal = bool(score) and finite_comparison and outcome in {"completed", "quantitative_needs_review", "partial", "scoring_failure"}
        counts["returned_signal" if has_signal else str(outcome)] += 1
        if has_signal:
            returned.append(row)
        decisions[case_id] = reference_qualification(score, semantic_status=row.get("semanticStatus", (score or {}).get("semantics", {}).get("status")), gates=gates) if has_signal else {"status": "failed", "reasons": ["no_scored_signal"]}
    passed = sum(decision["status"] == "passed" for decision in decisions.values())
    def finite_values(key):
        values = [(row["score"].get("summary", row["score"])).get(key) for row in returned]
        return [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)]
    return {"version": 1, "attempted": len(expected), "reported": len(rows), "terminalCounts": dict(counts),
            "outcomeCounts": dict(outcomes),
            "scoredSignalCaseIds": [row.get("caseId", row.get("id")) for row in returned],
            "quantitativeYield": len(returned)/len(expected), "referenceQualifiedCount": passed if gates is not None else None,
            "referenceQualifiedYield": passed/len(expected) if gates is not None else None,
            "referenceDecisions": decisions,
            "conditionalFidelity": {key: {"n": len(values), "mean": float(np.mean(values)) if values else None} for key in ("globalRmseUv", "macroMeanCorrelation", "macroMeanCoverage") for values in [finite_values(key)]},
            "clinicalValidationUse": False}


def grouped_mean_interval(values: list[tuple[str, float]], *, seed: int = 0, repetitions: int = 2000) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    for group, value in values:
        if np.isfinite(value):
            groups[group].append(float(value))
    means = np.asarray([np.mean(group) for group in groups.values()])
    # A single waveform with eight degradations is still one independent unit.
    if len(means) < 2:
        return {"method": "independent-group-bootstrap", "groupCount": len(means), "interval95": None, "reason": "fewer_than_two_independent_groups"}
    if repetitions < 100:
        raise ValueError("At least 100 bootstrap repetitions are required.")
    rng = np.random.default_rng(seed)
    draws = np.mean(rng.choice(means, size=(repetitions, len(means)), replace=True), axis=1)
    return {"method": "independent-group-bootstrap", "groupCount": len(means), "mean": float(np.mean(means)),
            "interval95": np.quantile(draws, [.025, .975]).tolist(), "seed": seed, "repetitions": repetitions}
