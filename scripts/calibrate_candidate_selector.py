from __future__ import annotations

import argparse
import json
import hashlib
import inspect
import platform
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT / "benchmark/generated/multisource-truth-v1/exhaustive-100-manifest.json"
)
DEFAULT_RUN = ROOT / "benchmark/results/current-contract3-semantic-exhaustive-100-c4-20260820/run.json"
DEFAULT_OUTPUT = ROOT / "benchmark/results/selector-calibration-draft/profile.json"
DEFAULT_HELDOUT_MANIFEST = (
    ROOT / "benchmark/generated/pmcardio-core-v1/exhaustive-100-manifest.json"
)
DEFAULT_HELDOUT_RUN = ROOT / "benchmark/results/pmcardio-exhaustive-mps-v1/run.json"
TRAINING_SPLIT = "development"
VALIDATION_SPLIT = "validation"
SHRINKAGE_CASES = 5
MIN_CONTEXT_VALIDATION_CASES = 5
MIN_CONTEXT_MEAN_IMPROVEMENT_UV = 1.0
SELECTOR_CALIBRATION_CONTRACT_VERSION = int(
    json.loads(
        (ROOT / "config/digitizer-policy.v2.json").read_text(encoding="utf-8")
    )["selectorCalibrationContractVersion"]
)
PARAMETER_CONTRACT = json.loads((ROOT / "config/selector-parameter-contract.v1.json").read_text())


def parameter_signature(parameters: dict) -> str:
    return json.dumps({key: parameters.get(key) for key in PARAMETER_CONTRACT["keys"]}, sort_keys=True, separators=(",", ":"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit transparent layout/candidate RMSE priors from paired ECG truth. "
            "Only the development split may influence the saved calibration."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--heldout-manifest",
        type=Path,
        help=f"Optional locked external manifest (for example {DEFAULT_HELDOUT_MANIFEST}).",
    )
    parser.add_argument(
        "--heldout-run",
        type=Path,
        help=f"Matching locked external run (for example {DEFAULT_HELDOUT_RUN}).",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def candidate_family(parameters: dict[str, Any]) -> str:
    kind = str(parameters.get("kind") or "source-model")
    vectorizer = str(parameters.get("vectorizer") or "unknown")
    input_variant = str(parameters.get("inputVariant") or "original")
    return f"{kind}:{vectorizer}:{input_variant}"


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def score_rmse(run_directory: Path, result: dict[str, Any]) -> float | None:
    score_path = result.get("scorePath")
    if not isinstance(score_path, str):
        return None
    resolved = (run_directory / score_path).resolve()
    if not resolved.is_relative_to(run_directory.resolve()):
        raise ValueError("Score path escapes the benchmark run.")
    score = load_json(resolved)
    summary = score.get("summary")
    if not isinstance(summary, dict) or summary.get("completeExpectedLeads") is not True:
        return None
    result = finite_number(summary.get("macroMeanRmseUv"))
    return result if result is not None and result >= 0 else None


def shrunk_median(values: list[float], reference: float) -> float:
    observed = statistics.median(values)
    return (
        observed * len(values) + reference * SHRINKAGE_CASES
    ) / (len(values) + SHRINKAGE_CASES)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction + 0.999999) - 1))
    return ordered[index]


def robust_risk(values: list[float]) -> float:
    return statistics.median(values) * 0.75 + percentile(values, 0.75) * 0.25


def artifact_context(case: dict[str, Any]) -> str:
    degradation = case.get("degradation")
    kind = degradation.get("kind") if isinstance(degradation, dict) else None
    if kind in {"rotation", "perspective"}:
        return "geometry"
    if kind == "low_resolution" or str(kind).startswith(
        "augmentation_resolution_scale_down_factor_"
    ):
        return "low-resolution"
    if kind == "annotation":
        return "annotation"
    if kind == "blur":
        return "blur"
    if kind in {"screen", "screen_capture", "moire", "photos_screens"}:
        return "screen"
    if kind in {"faded", "jpeg_compression", "digital_data_opacity_1"}:
        return "degraded-raster"
    if kind in {"photos_iphone", "photos_crumbles", "photos_scans"}:
        return "photo"
    return "clean"


def validate_grouped_splits(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Prove that a source/patient group cannot influence two fit splits."""

    split_by_group: dict[str, str] = {}
    identity_splits: dict[tuple[str, str], str] = {}
    seen_cases: set[str] = set()
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        case_id = case.get("caseId")
        group_id = case.get("groupId") or case.get("sourceId")
        split = case.get("split")
        if not isinstance(case_id, str) or not isinstance(group_id, str):
            raise ValueError(
                "Every calibration case must declare caseId and groupId/sourceId."
            )
        if not isinstance(split, str):
            raise ValueError(f"Calibration case {case_id} has no split.")
        if case_id in seen_cases:
            raise ValueError(f"Duplicate calibration case: {case_id}.")
        seen_cases.add(case_id)
        for key in ("sourceId", "patientId"):
            identity = case.get(key)
            if isinstance(identity, str) and identity:
                previous_identity_split = identity_splits.setdefault((key, identity), split)
                if previous_identity_split != split:
                    raise ValueError(f"Group leakage: {key} identity appears in both {previous_identity_split} and {split}.")
        previous = split_by_group.setdefault(group_id, split)
        if previous != split:
            raise ValueError(
                f"Group leakage: {group_id} appears in both {previous} and {split}."
            )
        groups_by_split[split].add(group_id)
    return {
        "groupKey": "groupId with sourceId fallback",
        "leakageChecked": True,
        "groupsBySplit": {
            split: len(groups)
            for split, groups in sorted(groups_by_split.items())
        },
    }


def rounded(value: float) -> float:
    return round(value, 3)


def scored_rows(
    manifest: dict[str, Any],
    run: dict[str, Any],
    run_directory: Path,
    allowed_splits: set[str] | None = None,
) -> list[dict[str, Any]]:
    cases = {
        case["caseId"]: case
        for case in manifest.get("cases", [])
        if isinstance(case, dict) and isinstance(case.get("caseId"), str)
    }
    rows: list[dict[str, Any]] = []
    for result in run.get("results", []):
        if (
            not isinstance(result, dict)
            or result.get("candidateId") == "selector"
            or result.get("status") != "completed"
        ):
            continue
        case = cases.get(result.get("caseId"))
        if not case or (allowed_splits is not None and case.get("split") not in allowed_splits):
            continue
        layout = case.get("layout")
        candidate_id = result.get("candidateId")
        parameters = result.get("parameters")
        rmse_uv = score_rmse(run_directory, result)
        if (
            not isinstance(layout, str)
            or not isinstance(candidate_id, str)
            or not isinstance(parameters, dict)
            or rmse_uv is None
        ):
            continue
        rows.append(
            {
                "caseId": case["caseId"],
                "groupId": case.get("groupId") or case.get("sourceId"),
                "split": case.get("split"),
                "layout": layout,
                "candidateId": candidate_id,
                "family": candidate_family(parameters),
                "context": artifact_context(case),
                "rmseUv": rmse_uv,
            }
        )
    return rows


def fit_calibration(
    manifest: dict[str, Any],
    run: dict[str, Any],
    run_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_cases = [
        case for case in manifest.get("cases", []) if isinstance(case, dict)
    ]
    grouped_split_audit = validate_grouped_splits(manifest_cases)
    layout_values: dict[str, list[float]] = defaultdict(list)
    candidate_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    family_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    context_layout_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    context_candidate_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    context_family_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    validation_rows: list[dict[str, Any]] = []

    # Never even open a final-set score while fitting or choosing contexts.
    for row in scored_rows(manifest, run, run_directory, {TRAINING_SPLIT, VALIDATION_SPLIT}):
        split = row["split"]
        layout = row["layout"]
        candidate_id = row["candidateId"]
        rmse_uv = row["rmseUv"]
        if split == VALIDATION_SPLIT:
            validation_rows.append(row)
        if split != TRAINING_SPLIT:
            continue
        layout_values[layout].append(rmse_uv)
        candidate_values[(layout, candidate_id)].append(rmse_uv)
        family_values[(layout, row["family"])].append(rmse_uv)
        context_layout_values[(layout, row["context"])].append(rmse_uv)
        context_candidate_values[(layout, row["context"], candidate_id)].append(
            rmse_uv
        )
        context_family_values[(layout, row["context"], row["family"])].append(
            rmse_uv
        )

    if not layout_values:
        raise ValueError("No scoreable development cases were found.")

    layouts: dict[str, Any] = {}
    for layout, values in sorted(layout_values.items()):
        layouts[layout] = calibration_bucket(
            values,
            {
                candidate_id: candidate_scores
                for (candidate_layout, candidate_id), candidate_scores in candidate_values.items()
                if candidate_layout == layout
            },
            {
                family: family_scores
                for (family_layout, family), family_scores in family_values.items()
                if family_layout == layout
            },
        )
        layouts[layout]["contexts"] = {
            context: calibration_bucket(
                context_values,
                {
                    candidate_id: scores
                    for (
                        candidate_layout,
                        candidate_context,
                        candidate_id,
                    ), scores in context_candidate_values.items()
                    if candidate_layout == layout and candidate_context == context
                },
                {
                    family: scores
                    for (
                        family_layout,
                        family_context,
                        family,
                    ), scores in context_family_values.items()
                    if family_layout == layout and family_context == context
                },
            )
            for (context_layout, context), context_values in sorted(
                context_layout_values.items()
            )
            if context_layout == layout
        }

    context_selection = fit_context_selection(validation_rows, layouts)
    calibration = {
        "version": 2,
        "profileId": "paired-truth-layout-artifact-risk-priors-v2",
        "training": {
            "suiteId": run.get("suiteId"),
            "benchmarkRunId": run.get("runId"),
            "pipelineCommit": run.get("gitCommit"),
            "pipelineContractVersion": SELECTOR_CALIBRATION_CONTRACT_VERSION,
            "split": TRAINING_SPLIT,
            "splitMethod": "manifest-grouped-source-split",
            "groupedSplitAudit": grouped_split_audit,
            "heldoutUsed": False,
            "shrinkageReferenceCases": SHRINKAGE_CASES,
            "objective": "macro mean 12-lead RMSE in microvolts",
            "riskStatistic": "75% median RMSE + 25% p75 RMSE",
            "artifactContextMethod": "manifest-degradation-family-v1",
            "contextSelectionRule": (
                "At least five validation cases and at least 1 uV lower "
                "mean RMSE than the layout-only prior"
            ),
            "clinicalValidationUse": False,
        },
        "contextSelection": context_selection,
        "layouts": layouts,
    }
    validation = {
        **validate_prior_ranking(validation_rows, layouts),
        "groupedSplitAudit": grouped_split_audit,
    }
    return calibration, validation


def calibration_bucket(
    values: list[float],
    candidate_values: dict[str, list[float]],
    family_values: dict[str, list[float]],
) -> dict[str, Any]:
    reference_median = statistics.median(values)
    reference_risk = robust_risk(values)

    def entry(scores: list[float]) -> dict[str, Any]:
        observed_risk = robust_risk(scores)
        shrunk_risk = (
            observed_risk * len(scores) + reference_risk * SHRINKAGE_CASES
        ) / (len(scores) + SHRINKAGE_CASES)
        return {
            "caseCount": len(scores),
            "medianRmseUv": rounded(statistics.median(scores)),
            "p75RmseUv": rounded(percentile(scores, 0.75)),
            "shrunkRmseUv": rounded(shrunk_median(scores, reference_median)),
            "riskRmseUv": rounded(shrunk_risk),
        }

    return {
        "referenceMedianRmseUv": rounded(reference_median),
        "referenceRiskRmseUv": rounded(reference_risk),
        "scoreableCandidateCount": len(values),
        "candidates": {
            candidate_id: entry(scores)
            for candidate_id, scores in sorted(candidate_values.items())
        },
        "families": {
            family: entry(scores)
            for family, scores in sorted(family_values.items())
        },
    }


def prior_for_row(row: dict[str, Any], calibration: dict[str, Any]) -> float:
    candidate = calibration.get("candidates", {}).get(row["candidateId"])
    if candidate:
        return float(candidate["riskRmseUv"])
    family = calibration.get("families", {}).get(row["family"])
    if family:
        return float(family["riskRmseUv"])
    return float(calibration.get("referenceRiskRmseUv", 1e9))


def selected_row(
    rows: list[dict[str, Any]],
    layouts: dict[str, Any],
    *,
    use_context: bool,
) -> dict[str, Any]:
    layout = rows[0]["layout"]
    calibration = layouts.get(layout, {})
    context_calibration = (calibration.get("contexts") or {}).get(
        rows[0].get("context")
    )
    active_calibration = (
        context_calibration
        if use_context and context_calibration
        else calibration
    )
    return min(
        rows,
        key=lambda row: (
            prior_for_row(row, active_calibration),
            row["candidateId"],
        ),
    )


def fit_context_selection(
    rows: list[dict[str, Any]],
    layouts: dict[str, Any],
) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[row["caseId"]].append(row)
    by_layout_context: dict[
        tuple[str, str], list[list[dict[str, Any]]]
    ] = defaultdict(list)
    for case_rows in by_case.values():
        key = (case_rows[0]["layout"], case_rows[0]["context"])
        by_layout_context[key].append(case_rows)

    selection: dict[str, dict[str, Any]] = defaultdict(dict)
    for (layout, context), context_cases in sorted(by_layout_context.items()):
        layout_selected = [
            float(selected_row(case_rows, layouts, use_context=False)["rmseUv"])
            for case_rows in context_cases
        ]
        context_selected = [
            float(selected_row(case_rows, layouts, use_context=True)["rmseUv"])
            for case_rows in context_cases
        ]
        layout_mean = statistics.mean(layout_selected)
        context_mean = statistics.mean(context_selected)
        improvement = layout_mean - context_mean
        enabled = bool(
            len(context_cases) >= MIN_CONTEXT_VALIDATION_CASES
            and improvement >= MIN_CONTEXT_MEAN_IMPROVEMENT_UV
        )
        evidence = {
            "enabled": enabled,
            "validationCaseCount": len(context_cases),
            "layoutPriorMeanRmseUv": rounded(layout_mean),
            "contextPriorMeanRmseUv": rounded(context_mean),
            "meanImprovementUv": rounded(improvement),
        }
        selection[layout][context] = evidence
        context_calibration = (
            (layouts.get(layout, {}).get("contexts") or {}).get(context)
        )
        if context_calibration is not None:
            context_calibration["selection"] = evidence
    return {
        layout: dict(sorted(contexts.items()))
        for layout, contexts in sorted(selection.items())
    }


def validate_prior_ranking(
    rows: list[dict[str, Any]],
    layouts: dict[str, Any],
    *,
    split_label: str = VALIDATION_SPLIT,
) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[row["caseId"]].append(row)

    selected_rmse: list[float] = []
    layout_only_rmse: list[float] = []
    oracle_rmse: list[float] = []
    selected_ids: dict[str, int] = defaultdict(int)
    for case_rows in by_case.values():
        layout = case_rows[0]["layout"]
        calibration = layouts.get(layout, {})
        context_calibration = (calibration.get("contexts") or {}).get(
            case_rows[0].get("context")
        )
        use_context = bool(
            context_calibration
            and (context_calibration.get("selection") or {}).get("enabled")
        )
        selected = selected_row(
            case_rows,
            layouts,
            use_context=use_context,
        )
        layout_only = selected_row(
            case_rows,
            layouts,
            use_context=False,
        )
        oracle = min(case_rows, key=lambda row: row["rmseUv"])
        selected_rmse.append(float(selected["rmseUv"]))
        layout_only_rmse.append(float(layout_only["rmseUv"]))
        oracle_rmse.append(float(oracle["rmseUv"]))
        selected_ids[selected["candidateId"]] += 1

    return {
        "split": split_label,
        "caseCount": len(selected_rmse),
        "priorOnlyMedianRmseUv": (
            rounded(statistics.median(selected_rmse)) if selected_rmse else None
        ),
        "priorOnlyMeanRmseUv": (
            rounded(statistics.mean(selected_rmse)) if selected_rmse else None
        ),
        "layoutOnlyMedianRmseUv": (
            rounded(statistics.median(layout_only_rmse))
            if layout_only_rmse
            else None
        ),
        "layoutOnlyMeanRmseUv": (
            rounded(statistics.mean(layout_only_rmse))
            if layout_only_rmse
            else None
        ),
        "oracleMedianRmseUv": (
            rounded(statistics.median(oracle_rmse)) if oracle_rmse else None
        ),
        "selectedCandidateCounts": dict(sorted(selected_ids.items())),
        "note": (
            "Candidate priors use development truth; the validation split only "
            "activates artifact contexts. This is tuning evidence, not a locked "
            "heldout estimate. Production still applies QA, agreement, and "
            "publication gates."
        ),
    }


def validate_locked_holdout(
    *,
    training_manifest: dict[str, Any],
    heldout_manifest: dict[str, Any],
    heldout_run: dict[str, Any],
    heldout_run_directory: Path,
    layouts: dict[str, Any],
) -> dict[str, Any]:
    training_cases = [
        case
        for case in training_manifest.get("cases", [])
        if isinstance(case, dict)
    ]
    heldout_cases = [
        case
        for case in heldout_manifest.get("cases", [])
        if isinstance(case, dict)
    ]
    validate_grouped_splits(heldout_cases)
    if any(case.get("split") != "heldout" for case in heldout_cases):
        raise ValueError("Every external evaluation case must use split=heldout.")
    training_groups = {
        case.get("groupId") or case.get("sourceId") for case in training_cases
    }
    heldout_groups = {
        case.get("groupId") or case.get("sourceId") for case in heldout_cases
    }
    overlap = sorted(
        str(group) for group in training_groups.intersection(heldout_groups)
    )
    if overlap:
        raise ValueError(
            "External holdout group leakage: " + ", ".join(overlap[:5])
        )
    rows = scored_rows(
        heldout_manifest,
        heldout_run,
        heldout_run_directory,
    )
    result = validate_prior_ranking(
        rows,
        layouts,
        split_label="locked-heldout",
    )
    return {
        **result,
        "suiteId": heldout_run.get("suiteId"),
        "benchmarkRunId": heldout_run.get("runId"),
        "pipelineCommit": heldout_run.get("gitCommit"),
        "usedForFitOrActivation": False,
        "groupOverlapCount": 0,
        "note": (
            "Locked external holdout metrics are reported only after fitting "
            "and context activation; they do not alter the saved priors."
        ),
        "artifactContextCounts": dict(
            sorted(
                {
                    context: len(
                        {
                            row["caseId"]
                            for row in rows
                            if row["context"] == context
                        }
                    )
                    for context in {row["context"] for row in rows}
                }.items()
            )
        ),
    }


def fitting_provenance(manifest: dict, run: dict, run_directory: Path) -> dict:
    """Bind fitting to exact score inputs without copying private membership."""
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    cases = {case["caseId"]: case for case in manifest["cases"] if case.get("split") in {TRAINING_SPLIT, VALIDATION_SPLIT}}
    score_hashes = {}
    parameter_signatures: dict[str, set[str]] = defaultdict(set)
    for result in run.get("results", []):
        if result.get("caseId") not in cases or result.get("candidateId") == "selector" or result.get("status") != "completed":
            continue
        raw_path = result.get("scorePath")
        if not isinstance(raw_path, str):
            continue
        score_path = (run_directory / raw_path).resolve()
        if not score_path.is_relative_to(run_directory.resolve()):
            raise ValueError("Score path escapes the benchmark run.")
        score_hashes[raw_path] = hashlib.sha256(score_path.read_bytes()).hexdigest()
        case = cases[result["caseId"]]
        parameter_signatures[f'{case.get("layout")}/{result.get("candidateId")}'].add(parameter_signature(result.get("parameters") or {}))
    memberships = {split: sorted((case["caseId"], case.get("groupId") or case.get("sourceId")) for case in cases.values() if case["split"] == split) for split in (TRAINING_SPLIT, VALIDATION_SPLIT)}
    source_files = ["scripts/calibrate_candidate_selector.py", "src/lib/digitizer/selector-calibration.ts", "src/lib/digitizer/candidate-capabilities.ts", "config/digitizer-policy.v2.json", "ecg_benchmark/scoring.py", "ecg_benchmark/coordinates.py"]
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files}
    return {"version": 1, "fittingMethod": "deterministic-grouped-robust-risk-v1", "randomSeed": None,
            "python": platform.python_version(), "manifestSha256": digest(manifest), "runSha256": digest(run),
            "parameterContractVersion": PARAMETER_CONTRACT["version"], "parameterKeys": PARAMETER_CONTRACT["keys"],
            "parameterSignatures": {key: sorted(values) for key, values in sorted(parameter_signatures.items())},
            "scoreFilesSha256": digest(score_hashes), "scoreFileCount": len(score_hashes),
            "sourceFileHashes": source_hashes, "fittingSourceSha256": digest(source_hashes),
            "featureDefinitionsSha256": hashlib.sha256((inspect.getsource(candidate_family) + inspect.getsource(artifact_context)).encode()).hexdigest(),
            "splitMembershipSha256": {split: digest(rows) for split, rows in memberships.items()},
            "caseCounts": {split: len(rows) for split, rows in memberships.items()},
            "independentGroupCounts": {split: len({group for _, group in rows}) for split, rows in memberships.items()},
            "hyperparameters": {"shrinkageCases": SHRINKAGE_CASES, "minimumContextValidationCases": MIN_CONTEXT_VALIDATION_CASES, "minimumContextImprovementUv": MIN_CONTEXT_MEAN_IMPROVEMENT_UV},
            "extractionPipelineCommit": run.get("gitCommit"), "extractionSourceSnapshotSha256": run.get("sourceTreeSha256"),
            "extractionReconstructible": bool(run.get("sourceTreeSha256")),
            "finalEvaluationScoresReadForFitting": False}


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    run_path = args.run.resolve()
    output_path = args.output.resolve()
    if output_path == (ROOT / "config/candidate-selector-calibration.v2.json").resolve():
        raise ValueError("Write a calibration draft and evaluate it before replacing the historical production profile.")
    calibration, validation = fit_calibration(
        load_json(manifest_path),
        load_json(run_path),
        run_path.parent,
    )
    recipe = fitting_provenance(load_json(manifest_path), load_json(run_path), run_path.parent)
    recipe["manifestFileSha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    recipe["runFileSha256"] = hashlib.sha256(run_path.read_bytes()).hexdigest()
    calibration["training"]["reproducibility"] = recipe
    if bool(args.heldout_manifest) != bool(args.heldout_run):
        raise ValueError(
            "--heldout-manifest and --heldout-run must be provided together."
        )
    if args.heldout_manifest and args.heldout_run:
        heldout_run_path = args.heldout_run.resolve()
        validation["lockedHeldout"] = validate_locked_holdout(
            training_manifest=load_json(manifest_path),
            heldout_manifest=load_json(args.heldout_manifest.resolve()),
            heldout_run=load_json(heldout_run_path),
            heldout_run_directory=heldout_run_path.parent,
            layouts=calibration["layouts"],
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_path.with_suffix(".recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_path), "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
