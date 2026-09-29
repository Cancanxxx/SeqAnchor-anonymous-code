#!/usr/bin/env python3
"""Add exact paired-task uncertainty to native AgentHarm expansion results."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from scipy.stats import beta

from nontransitive_safety.provenance import canonical_json_bytes, loads_json_strict, sha256_file


def exact_interval(events: int, total: int, confidence: float = 0.95):
    if not (isinstance(events, int) and isinstance(total, int) and 0 <= events <= total):
        raise ValueError("invalid binomial counts")
    if total == 0:
        return None
    tail = (1.0 - confidence) / 2
    lower = 0.0 if events == 0 else float(beta.ppf(tail, events, total - events + 1))
    upper = 1.0 if events == total else float(beta.ppf(1 - tail, events + 1, total - events))
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise RuntimeError("nonfinite exact interval")
    return {"confidence": confidence, "lower": lower, "upper": upper}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    completion_path = args.completion.resolve()
    completion = loads_json_strict(
        completion_path.read_text(encoding="utf-8"), source=str(completion_path)
    )
    if completion.get("artifact_type") != "agentharm_native_expansion_completion_v1":
        raise RuntimeError("unexpected native AgentHarm completion")

    def rate(events: int, total: int):
        return {
            "events": events,
            "denominator": total,
            "rate": None if total == 0 else events / total,
            "exact_95pct": exact_interval(events, total),
        }

    def summarize(row):
        tasks = int(row["tasks"])
        refusals = int(row["base_explicit_refusals"])
        base_target = int(row["base_action_classes"].get("target_tool", 0))
        base_non_target = tasks - base_target
        activations = int(row["target_tool_activations"])
        return {
            "tasks": tasks,
            "first_action_change": rate(int(row["first_action_changes"]), tasks),
            "target_tool_activation": rate(activations, tasks),
            "target_tool_activation_given_base_non_target": rate(
                activations, base_non_target
            ),
            "base_explicit_refusal": rate(refusals, tasks),
            "explicit_refusal_to_target_given_base_refusal": rate(
                int(row["explicit_refusal_to_target_tool"]), refusals
            ),
            "new_safe_tool_selected": rate(int(row["new_safe_tool_selected"]), tasks),
            "base_action_classes": row["base_action_classes"],
            "expanded_action_classes": row["expanded_action_classes"],
        }

    metrics = completion["metrics"]
    output = {
        "schema_version": "1.0.0",
        "artifact_type": "agentharm_native_expansion_uncertainty_v1",
        "completion_sha256": sha256_file(completion_path),
        "snapshot_id": completion["snapshot_id"],
        "aggregate": summarize(metrics["aggregate"]),
        "by_category": {
            category: summarize(row)
            for category, row in sorted(metrics["by_category"].items())
        },
        "unit": "official AgentHarm public-test task / paired first-action update",
        "interval": "two-sided exact Clopper-Pearson 95%",
        "scope": completion["scope"],
        "human_annotations_collected": False,
    }
    payload = canonical_json_bytes(output) + b"\n"
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != payload:
        raise RuntimeError(f"existing native uncertainty result differs: {target}")
    target.write_bytes(payload)
    print(output["aggregate"])


if __name__ == "__main__":
    main()
