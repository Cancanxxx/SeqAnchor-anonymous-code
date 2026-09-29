#!/usr/bin/env python3
"""Cluster-aware uncertainty analysis for exact-throughput guard comparisons."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest

from nontransitive_safety.provenance import canonical_json_bytes, sha256_file

PRIMARY = "SeqAnchor + exact throughput"
STEPGUARD = "StepGuard + exact throughput"
COMPARATORS = (
    STEPGUARD,
    "TS-Guard + exact throughput",
    "Qwen3Guard + exact throughput",
)
UNSAFE_ROLES = ("C_GR", "C_GA", "C_RA")
SAFE_ROLES = ("S1", "E", "N")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _wilson(events: int, total: int, z: float = 1.959963984540054) -> Mapping[str, float]:
    _require(0 <= events <= total and total > 0, "invalid binomial counts")
    rate = events / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return {
        "events": events,
        "total": total,
        "rate": rate,
        "lower": center - half,
        "upper": center + half,
    }


def _by_family(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Mapping[str, Any]]:
    output = {str(row["family_id"]): row for row in rows}
    _require(len(output) == len(rows), "family IDs collide")
    return output


def _family_unit_wilson(
    rows: Sequence[Mapping[str, Any]], outcome: str
) -> Mapping[str, Any]:
    """Wilson interval when each row represents one distinct workflow family."""

    _require(bool(rows), "family-unit rows are empty")
    _by_family(rows)
    value = dict(_wilson(sum(bool(row[outcome]) for row in rows), len(rows)))
    value.update(
        {
            "analysis_unit": "workflow_family",
            "rows_per_family": 1,
            "interval": "two_sided_95pct_Wilson",
        }
    )
    return value


def _clustered_descriptive_rate(
    rows: Sequence[Mapping[str, Any]], outcome: str
) -> Mapping[str, Any]:
    """Point estimate for pooled updates without treating clustered rows as IID."""

    _require(bool(rows), "pooled candidate rows are empty")
    family_sizes = Counter(str(row["family_id"]) for row in rows)
    _require(bool(family_sizes), "pooled candidate families are absent")
    events = sum(bool(row[outcome]) for row in rows)
    total = len(rows)
    sizes = tuple(family_sizes.values())
    return {
        "events": events,
        "total": total,
        "rate": events / total,
        "denominator_unit": "candidate_update",
        "cluster_unit": "workflow_family",
        "family_count": len(family_sizes),
        "rows_per_family": {"minimum": min(sizes), "maximum": max(sizes)},
        "confidence_interval": None,
        "inference": "descriptive_point_estimate_only",
        "dependence_note": (
            "candidate updates within a workflow family are clustered and are not "
            "treated as independent binomial observations"
        ),
    }


def _paired_endpoint(
    primary: Sequence[bool],
    comparator: Sequence[bool],
    *,
    seed: int,
    replicates: int,
) -> Mapping[str, Any]:
    _require(len(primary) == len(comparator) and bool(primary), "paired rows differ")
    a = np.asarray(primary, dtype=np.float64)
    b = np.asarray(comparator, dtype=np.float64)
    difference = float(np.mean(a - b))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), size=(replicates, len(a)))
    draws = np.mean(a[indices] - b[indices], axis=1)
    primary_only = int(np.sum((a == 1) & (b == 0)))
    comparator_only = int(np.sum((a == 0) & (b == 1)))
    discordant = primary_only + comparator_only
    pvalue = (
        1.0
        if discordant == 0
        else float(
            binomtest(
                min(primary_only, comparator_only),
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    )
    return {
        "primary_minus_comparator": difference,
        "paired_bootstrap_95pct": {
            "lower": float(np.quantile(draws, 0.025)),
            "upper": float(np.quantile(draws, 0.975)),
            "replicates": replicates,
            "seed": seed,
            "resampling_unit": "workflow_family",
        },
        "mcnemar_exact": {
            "primary_only": primary_only,
            "comparator_only": comparator_only,
            "discordant": discordant,
            "two_sided_p": pvalue,
        },
    }


def _method_summary(value: Mapping[str, Any]) -> Mapping[str, Any]:
    sequence_rows = value["sequence_rows"]
    update_rows = value["update_rows"]
    _require(len(sequence_rows) == 60 and len(update_rows) == 360, "panel size differs")
    _by_family(sequence_rows)
    by_role = {}
    for role in (*SAFE_ROLES, *UNSAFE_ROLES):
        rows = [row for row in update_rows if row["candidate_role"] == role]
        _require(len(rows) == 60, f"role coverage differs: {role}")
        by_role[role] = {
            "safety_pass": _family_unit_wilson(rows, "safety_pass"),
            "admitted": _family_unit_wilson(rows, "accepted"),
        }
    unsafe_rows = [row for row in update_rows if row["candidate_role"] in UNSAFE_ROLES]
    useful_rows = [row for row in update_rows if row["candidate_role"] == "S1"]
    _require(len(unsafe_rows) == 180, "pooled unsafe candidate coverage differs")
    return {
        "unsafe_candidate_safety_pass": _clustered_descriptive_rate(
            unsafe_rows, "safety_pass"
        ),
        "unsafe_candidate_admitted": _clustered_descriptive_rate(
            unsafe_rows, "accepted"
        ),
        "useful_safe_candidate_safety_pass": _family_unit_wilson(
            useful_rows, "safety_pass"
        ),
        "unsafe_sequence": _family_unit_wilson(
            sequence_rows, "any_unsafe_admission"
        ),
        "useful_safe_adoption": _family_unit_wilson(
            sequence_rows, "useful_safe_improvement_adopted"
        ),
        "exact_sequence": _family_unit_wilson(
            sequence_rows, "exact_sequence_correct"
        ),
        "by_candidate_role": by_role,
    }


def _selection_scope() -> Mapping[str, Any]:
    return {
        "stepguard": (
            "post_hoc_comparator_extension_selected_after_the_original_comparator_results"
        ),
        "other_methods": "carried_forward_from_the_preexisting_comparator_analysis",
        "pairwise_inference": "descriptive_exploratory_only",
        "confirmatory_inference": False,
        "multiplicity_adjusted": False,
    }


def _comparison_scope(name: str) -> Mapping[str, Any]:
    return {
        "status": "descriptive_exploratory",
        "comparator_selection": (
            "post_hoc_after_original_comparator_results"
            if name == STEPGUARD
            else "carried_forward_from_preexisting_comparator_analysis"
        ),
        "multiplicity_adjusted": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-isolation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_918)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    source = args.exact_isolation.resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    _require(
        value.get("audit", {}).get("stepguard_comparator_timing")
        == "selected_after_original_comparator_results",
        "source does not identify StepGuard as post-hoc",
    )
    methods = value["methods"]
    _require(PRIMARY in methods, "primary method is absent")
    _require(all(name in methods for name in COMPARATORS), "comparator is absent")
    summaries = {
        name: _method_summary(methods[name]) for name in (PRIMARY, *COMPARATORS)
    }
    primary_rows = _by_family(methods[PRIMARY]["sequence_rows"])
    comparisons = {}
    for offset, name in enumerate(COMPARATORS):
        other_rows = _by_family(methods[name]["sequence_rows"])
        _require(set(primary_rows) == set(other_rows), "paired family set differs")
        families = sorted(primary_rows)
        comparisons[name] = {
            "inference_scope": _comparison_scope(name),
            "safe_sequence_advantage": _paired_endpoint(
                [not bool(primary_rows[f]["any_unsafe_admission"]) for f in families],
                [not bool(other_rows[f]["any_unsafe_admission"]) for f in families],
                seed=args.seed + offset * 10,
                replicates=args.bootstrap_replicates,
            ),
            "useful_adoption_advantage": _paired_endpoint(
                [bool(primary_rows[f]["useful_safe_improvement_adopted"]) for f in families],
                [bool(other_rows[f]["useful_safe_improvement_adopted"]) for f in families],
                seed=args.seed + offset * 10 + 1,
                replicates=args.bootstrap_replicates,
            ),
            "exact_sequence_advantage": _paired_endpoint(
                [bool(primary_rows[f]["exact_sequence_correct"]) for f in families],
                [bool(other_rows[f]["exact_sequence_correct"]) for f in families],
                seed=args.seed + offset * 10 + 2,
                replicates=args.bootstrap_replicates,
            ),
        }
    output = {
        "schema_version": "2.0.0",
        "artifact_type": "guard_comparator_statistics_v2",
        "source_sha256": sha256_file(source),
        "uncertainty_protocol": {
            "role_specific_and_sequence_endpoints": (
                "two_sided_95pct_Wilson_with_one_observation_per_workflow_family"
            ),
            "pooled_candidate_endpoints": (
                "descriptive_point_estimates_only_due_to_within_family_clustering"
            ),
            "paired_method_differences": (
                "workflow_family_paired_bootstrap_and_exact_McNemar"
            ),
        },
        "selection_scope": _selection_scope(),
        "methods": summaries,
        "paired_comparisons": comparisons,
        "multiplicity_note": "descriptive_exploratory_unadjusted_pairwise_inference",
        "human_annotations_collected": False,
    }
    payload = canonical_json_bytes(output) + b"\n"
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != payload:
        raise RuntimeError(f"existing statistics artifact differs: {target}")
    if not target.exists():
        target.write_bytes(payload)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
