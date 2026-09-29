#!/usr/bin/env python3
"""Analyze official TS-Guard on the frozen Sequential AnchorRC streams."""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

from nontransitive_safety.provenance import canonical_json_bytes, sha256_file
from option_set_instability.sequential_analysis_v1 import (
    aggregate_utility_scores,
    load_jsonl,
    run_evidence_sha256,
)
from option_set_instability.sequential_comparators_v1 import (
    analyze_tsguard_sequential,
    fresh_utility_threshold,
)
from option_set_instability.utility_risk_calibration_v1 import (
    family_nonimprovement_maxima,
    threshold_for_false_update_budget,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsguard-run", required=True, type=Path)
    parser.add_argument("--utility-run", required=True, type=Path)
    parser.add_argument("--utility-extension-run", type=Path)
    parser.add_argument("--extension-truth", type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tsguard_run = args.tsguard_run.resolve()
    utility_run = args.utility_run.resolve()
    data_dir = args.data_dir.resolve()
    utility_scores = aggregate_utility_scores(
        load_jsonl(utility_run / "anchorrc_v1.call_plan.jsonl"),
        load_jsonl(utility_run / "anchorrc_v1.jsonl"),
    )
    extension_evidence = None
    if (args.utility_extension_run is None) != (args.extension_truth is None):
        raise ValueError("utility extension run and truth must be supplied together")
    if args.utility_extension_run is not None:
        extension_run = args.utility_extension_run.resolve()
        extension_scores = aggregate_utility_scores(
            load_jsonl(extension_run / "anchorrc_v1.call_plan.jsonl"),
            load_jsonl(extension_run / "anchorrc_v1.jsonl"),
        )
        if set(utility_scores) & set(extension_scores):
            raise RuntimeError("base and extension utility score keys overlap")
        utility_scores = {**utility_scores, **extension_scores}
        extension_evidence = run_evidence_sha256(extension_run)
    calibration_truth = load_jsonl(data_dir / "calibration_truth.jsonl")
    strict_threshold = fresh_utility_threshold(calibration_truth, utility_scores)
    balanced_threshold, balanced_audit = threshold_for_false_update_budget(
        family_nonimprovement_maxima(calibration_truth, utility_scores),
        Fraction(1, 20),
    )
    sequences = load_jsonl(data_dir / "sequences_truth.jsonl")
    if args.extension_truth is not None:
        extension_rows = load_jsonl(args.extension_truth.resolve())
        extension_by_family = {row["family_id"]: row for row in extension_rows}
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
                    raise RuntimeError("utility extension duplicates a base comparison")
                mapping[role].update(references)
            merged.append(
                {**sequence, "utility_task_id_by_role_and_incumbent": mapping}
            )
        sequences = merged
    strict = analyze_tsguard_sequential(
        sequences,
        load_jsonl(data_dir / "tsguard_adaptation.jsonl"),
        load_jsonl(tsguard_run / "predictions.jsonl"),
        utility_scores,
        strict_threshold,
    )
    balanced = analyze_tsguard_sequential(
        sequences,
        load_jsonl(data_dir / "tsguard_adaptation.jsonl"),
        load_jsonl(tsguard_run / "predictions.jsonl"),
        utility_scores,
        balanced_threshold,
    )
    completion = tsguard_run / "completion.json"
    if not completion.is_file():
        raise RuntimeError("TS-Guard evidence is incomplete")
    output = {
        "schema_version": "1.0.0",
        "artifact_type": "tsguard_sequential_comparison_v1",
        "status": "completed_frozen_stream_comparison",
        "data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "utility_run_evidence_sha256": run_evidence_sha256(utility_run),
        "tsguard_completion_sha256": sha256_file(completion),
        "tsguard_decision_rule": strict["tsguard_decision_rule"],
        "invalid_parse_policy": strict["invalid_parse_policy"],
        "tsguard_safety_only": strict["tsguard_safety_only"],
        "tsguard_plus_strict_sequence_utility": strict[
            "tsguard_plus_fresh_utility"
        ],
        "tsguard_plus_balanced_per_update_utility": balanced[
            "tsguard_plus_fresh_utility"
        ],
        "strict_update_metrics": strict["update_metrics"],
        "balanced_update_metrics": balanced["update_metrics"],
        "strict_utility_threshold_hex": strict_threshold.hex(),
        "balanced_utility_calibration": balanced_audit,
        "human_annotations_collected": False,
    }
    if extension_evidence is not None:
        output["utility_extension"] = {
            "run_evidence_sha256": extension_evidence,
            "truth_sha256": sha256_file(args.extension_truth.resolve()),
            "thresholds_changed": False,
        }
    payload = canonical_json_bytes(output) + b"\n"
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != payload:
        raise RuntimeError(f"existing comparison differs: {target}")
    target.write_bytes(payload)
    print(output["tsguard_safety_only"])
    print(output["tsguard_plus_strict_sequence_utility"])
    print(output["tsguard_plus_balanced_per_update_utility"])


if __name__ == "__main__":
    main()
