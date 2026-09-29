#!/usr/bin/env python3
"""Compose transferred StepGuard decisions with frozen sequential utility gates."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

from nontransitive_safety.provenance import canonical_json_bytes, sha256_file
from option_set_instability.qwen3guard_sequential_v1 import (
    analyze_qwen3guard_sequential,
)
from option_set_instability.sequential_analysis_v1 import (
    aggregate_utility_scores,
    load_jsonl,
    run_evidence_sha256,
)
from option_set_instability.sequential_comparators_v1 import fresh_utility_threshold
from option_set_instability.stepguard_schema_v2 import (
    relabel_stepguard_update_metrics,
)
from option_set_instability.utility_risk_calibration_v1 import (
    family_nonimprovement_maxima,
    threshold_for_false_update_budget,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard-run", required=True, type=Path)
    parser.add_argument("--utility-run", required=True, type=Path)
    parser.add_argument("--utility-extension-run", required=True, type=Path)
    parser.add_argument("--extension-truth", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _nonimproving_sequence_count(rows) -> int:
    return sum(
        int(row.get("nonimproving_safe_updates_accepted", 0)) > 0 for row in rows
    )


def main() -> None:
    args = parse_args()
    guard_run = args.guard_run.resolve()
    utility_run = args.utility_run.resolve()
    extension_run = args.utility_extension_run.resolve()
    extension_truth = args.extension_truth.resolve()
    data_dir = args.data_dir.resolve()
    completion = guard_run / "completion.json"
    if not completion.is_file():
        raise RuntimeError("StepGuard transfer evidence is incomplete")
    completion_value = json.loads(completion.read_text(encoding="utf-8"))
    predictions_path = guard_run / "predictions.jsonl"
    if completion_value.get("predictions_sha256") != sha256_file(predictions_path):
        raise RuntimeError("StepGuard prediction receipt differs")
    if completion_value.get("rows") != 360:
        raise RuntimeError("StepGuard receipt row count differs")

    utility_scores = aggregate_utility_scores(
        load_jsonl(utility_run / "anchorrc_v1.call_plan.jsonl"),
        load_jsonl(utility_run / "anchorrc_v1.jsonl"),
    )
    extension_scores = aggregate_utility_scores(
        load_jsonl(extension_run / "anchorrc_v1.call_plan.jsonl"),
        load_jsonl(extension_run / "anchorrc_v1.jsonl"),
    )
    if set(utility_scores) & set(extension_scores):
        raise RuntimeError("base and extension utility scores overlap")
    utility_scores = {**utility_scores, **extension_scores}

    calibration_truth = load_jsonl(data_dir / "calibration_truth.jsonl")
    strict_threshold = fresh_utility_threshold(calibration_truth, utility_scores)
    balanced_threshold, balanced_audit = threshold_for_false_update_budget(
        family_nonimprovement_maxima(calibration_truth, utility_scores),
        Fraction(1, 20),
    )
    sequences = load_jsonl(data_dir / "sequences_truth.jsonl")
    extension_by_family = {
        row["family_id"]: row for row in load_jsonl(extension_truth)
    }
    merged = []
    for sequence in sequences:
        extension = extension_by_family.get(sequence["family_id"])
        if extension is None or extension["test_index"] != sequence["test_index"]:
            raise RuntimeError("utility extension family binding differs")
        mapping = {
            role: dict(references)
            for role, references in sequence[
                "utility_task_id_by_role_and_incumbent"
            ].items()
        }
        for role, references in extension[
            "utility_task_id_by_role_and_incumbent"
        ].items():
            if set(mapping[role]) & set(references):
                raise RuntimeError("utility extension duplicates base comparison")
            mapping[role].update(references)
        merged.append({**sequence, "utility_task_id_by_role_and_incumbent": mapping})

    adapters = load_jsonl(data_dir / "tsguard_adaptation.jsonl")
    predictions = load_jsonl(predictions_path)
    strict = analyze_qwen3guard_sequential(
        merged,
        adapters,
        predictions,
        utility_scores,
        strict_threshold,
        decision_mode="strict",
    )
    balanced = analyze_qwen3guard_sequential(
        merged,
        adapters,
        predictions,
        utility_scores,
        balanced_threshold,
        decision_mode="strict",
    )
    output = {
        "schema_version": "1.0.0",
        "artifact_type": "stepguard_sequential_comparison_v1",
        "status": "post_hoc_comparator_under_frozen_transfer_protocol",
        "data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "guard_completion_sha256": sha256_file(completion),
        "utility_run_evidence_sha256": run_evidence_sha256(utility_run),
        "utility_extension_run_evidence_sha256": run_evidence_sha256(extension_run),
        "extension_truth_sha256": sha256_file(extension_truth),
        "stepguard_safety_only": strict["qwen3guard_safety_only"],
        "stepguard_plus_strict_sequence_utility": strict[
            "qwen3guard_plus_fresh_utility"
        ],
        "stepguard_plus_balanced_per_update_utility": balanced[
            "qwen3guard_plus_fresh_utility"
        ],
        "strict_nonimproving_safe_replacement_sequences": (
            _nonimproving_sequence_count(strict["sequence_rows"])
        ),
        "balanced_nonimproving_safe_replacement_sequences": (
            _nonimproving_sequence_count(balanced["sequence_rows"])
        ),
        "strict_update_metrics": relabel_stepguard_update_metrics(
            strict["update_metrics"]
        ),
        "balanced_update_metrics": relabel_stepguard_update_metrics(
            balanced["update_metrics"]
        ),
        "strict_utility_threshold_hex": strict_threshold.hex(),
        "balanced_utility_calibration": balanced_audit,
        "transfer_scope": "pre_execution_action_guard_applied_to_complete_workflow",
        "old_action_resurrection": "not_applicable_no_persistent_ledger",
        "human_annotations_collected": False,
    }
    payload = canonical_json_bytes(output) + b"\n"
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != payload:
        raise RuntimeError(f"existing StepGuard comparison differs: {target}")
    if not target.exists():
        target.write_bytes(payload)
    print(output["stepguard_safety_only"])
    print(output["stepguard_plus_strict_sequence_utility"])
    print(output["stepguard_plus_balanced_per_update_utility"])


if __name__ == "__main__":
    main()
