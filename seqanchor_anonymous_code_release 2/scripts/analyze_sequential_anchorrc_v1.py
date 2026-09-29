#!/usr/bin/env python3
"""Analyze completed fresh-calibration Sequential AnchorRC evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from nontransitive_safety.provenance import canonical_json_bytes, sha256_file
from option_set_instability.sequential_analysis_v1 import (
    aggregate_safety_scores,
    aggregate_utility_scores,
    analyze_gate_ablations,
    analyze_sequential_anchorrc,
    build_fresh_calibrations,
    load_jsonl,
    run_evidence_sha256,
)


def _write_once(path: Path, value) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"existing analysis differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safety-run", type=Path, required=True)
    parser.add_argument("--utility-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    safety_run = args.safety_run.resolve()
    utility_run = args.utility_run.resolve()
    data_dir = args.data_dir.resolve()
    safety_scores = aggregate_safety_scores(
        load_jsonl(safety_run / "anchorrc_v1.call_plan.jsonl"),
        load_jsonl(safety_run / "anchorrc_v1.jsonl"),
    )
    utility_scores = aggregate_utility_scores(
        load_jsonl(utility_run / "anchorrc_v1.call_plan.jsonl"),
        load_jsonl(utility_run / "anchorrc_v1.jsonl"),
    )
    calibration_truth = load_jsonl(data_dir / "calibration_truth.jsonl")
    sequences = load_jsonl(data_dir / "sequences_truth.jsonl")
    safety_evidence_sha256 = run_evidence_sha256(safety_run)
    calibrations, utility_threshold, calibration_audit = build_fresh_calibrations(
        calibration_truth,
        safety_scores,
        utility_scores,
        safety_scorer_sha256=safety_evidence_sha256,
    )
    analysis = analyze_sequential_anchorrc(
        sequences,
        safety_scores,
        utility_scores,
        calibrations,
        utility_threshold,
    )
    ablations = analyze_gate_ablations(
        sequences,
        safety_scores,
        utility_scores,
        calibrations,
        utility_threshold,
    )
    output = {
        "schema_version": "1.0.0",
        "artifact_type": "sequential_anchorrc_analysis_v1",
        "status": "completed_fresh_family_disjoint_calibration_and_test",
        "data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "safety_run_evidence_sha256": safety_evidence_sha256,
        "utility_run_evidence_sha256": run_evidence_sha256(utility_run),
        "calibration": calibration_audit,
        **analysis,
        "gate_ablations": ablations,
        "human_annotations_collected": False,
    }
    _write_once(args.output.resolve(), output)
    print(output["sequence_metrics"])
    print(output["update_metrics"])


if __name__ == "__main__":
    main()
