#!/usr/bin/env python3
"""Add exact paired-unit uncertainty to an external catalog completion."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from scipy.stats import beta

from nontransitive_safety.provenance import canonical_json_bytes, loads_json_strict, sha256_file


def clopper_pearson(events: int, total: int, confidence: float = 0.95):
    if isinstance(events, bool) or isinstance(total, bool):
        raise ValueError("binomial counts must be integers")
    if not (isinstance(events, int) and isinstance(total, int) and 0 <= events <= total):
        raise ValueError("invalid binomial counts")
    if total == 0:
        return None
    alpha = 1.0 - confidence
    lower = 0.0 if events == 0 else float(beta.ppf(alpha / 2, events, total - events + 1))
    upper = (
        1.0
        if events == total
        else float(beta.ppf(1 - alpha / 2, events + 1, total - events))
    )
    if not all(math.isfinite(value) for value in (lower, upper)):
        raise RuntimeError("exact binomial interval is nonfinite")
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
    if not isinstance(completion, dict) or completion.get("artifact_type") != (
        "external_catalog_expansion_completion_v1"
    ):
        raise RuntimeError("unexpected external catalog completion")
    metrics = completion["metrics"]

    def summarize(row):
        resolved = int(row["resolved_pairs"])
        base_safe = int(row["base_safe"])
        revivals = int(row["safe_to_old_unsafe_revivals"])
        changed = int(row["decision_changes"])
        new_selected = int(row["safe_addition_selected"])
        return {
            "tasks": int(row["tasks"]),
            "resolved_pairs": resolved,
            "base_safe": {
                "events": base_safe,
                "denominator": resolved,
                "rate": None if resolved == 0 else base_safe / resolved,
                "exact_95pct": clopper_pearson(base_safe, resolved),
            },
            "safe_to_old_unsafe_revival_given_base_safe": {
                "events": revivals,
                "denominator": base_safe,
                "rate": None if base_safe == 0 else revivals / base_safe,
                "exact_95pct": clopper_pearson(revivals, base_safe),
            },
            "decision_change": {
                "events": changed,
                "denominator": resolved,
                "rate": None if resolved == 0 else changed / resolved,
                "exact_95pct": clopper_pearson(changed, resolved),
            },
            "safe_addition_selected": {
                "events": new_selected,
                "denominator": resolved,
                "rate": None if resolved == 0 else new_selected / resolved,
                "exact_95pct": clopper_pearson(new_selected, resolved),
            },
            "winner_transitions": row["winner_transitions"],
        }

    output = {
        "schema_version": "1.0.0",
        "artifact_type": "external_catalog_expansion_uncertainty_v1",
        "completion_sha256": sha256_file(completion_path),
        "snapshot_id": completion["snapshot_id"],
        "aggregate": summarize(metrics["aggregate"]),
        "by_benchmark": {
            name: summarize(row) for name, row in sorted(metrics["by_benchmark"].items())
        },
        "unit": "source benchmark row / paired catalog update",
        "interval": "two-sided exact Clopper-Pearson 95%",
        "human_annotations_collected": False,
    }
    payload = canonical_json_bytes(output) + b"\n"
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != payload:
        raise RuntimeError(f"existing external uncertainty result differs: {target}")
    target.write_bytes(payload)
    print(output["aggregate"])
    for name, row in output["by_benchmark"].items():
        print(name, row["safe_to_old_unsafe_revival_given_base_safe"])


if __name__ == "__main__":
    main()
