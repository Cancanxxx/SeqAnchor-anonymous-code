"""Component-matched calibration ablations for Sequential AnchorRC.

This module performs no model inference.  It reconstructs alternative safety
gates from one frozen scorer run, then replays the same local incumbent update
rule with simulator-exact strict throughput.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scipy.stats import beta

from nontransitive_safety.provenance import canonical_json_bytes, sha256_bytes, sha256_file
from option_set_instability.anchor_rc_v1 import SAFETY_AXES
from option_set_instability.sequential_analysis_v1 import (
    aggregate_safety_scores,
    load_jsonl,
    run_evidence_sha256,
)
from option_set_instability.sequential_oracle_utility_v1 import (
    exact_completed_items_by_family,
)
from option_set_instability.state_expansion_180 import load_workflow_specs

VARIANT_IDS = (
    "family_max_sequence_spending",
    "uncalibrated_zero_margin",
    "per_variant_pseudoreplication",
    "fixed_C_GRA_proposal_unmatched",
    "family_max_no_sequence_spending",
)
SAFE_USEFUL_ROLE = "S1"
SAFE_NONIMPROVING_ROLES = frozenset({"E", "N"})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact_interval(events: int, total: int, confidence: float = 0.95) -> list[float]:
    _require(0 <= events <= total and total > 0, "invalid binomial count")
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if events == 0 else float(beta.ppf(tail, events, total - events + 1))
    upper = 1.0 if events == total else float(beta.ppf(1.0 - tail, events + 1, total - events))
    return [lower, upper]


def _count(events: int, total: int) -> Mapping[str, Any]:
    return {
        "events": int(events),
        "total": int(total),
        "rate": float(events / total),
        "exact_95pct": _exact_interval(events, total),
    }


def build_calibration_panels(
    calibration_truth: Sequence[Mapping[str, Any]],
    safety_scores: Mapping[str, Mapping[str, float]],
) -> Mapping[str, Mapping[str, tuple[float, ...]]]:
    """Build family-max, pooled-variant, and fixed-prototype panels."""

    _require(len(calibration_truth) == 120, "calibration requires 120 workflow families")
    family_max: dict[str, list[float]] = {axis: [] for axis in SAFETY_AXES}
    per_variant: dict[str, list[float]] = {axis: [] for axis in SAFETY_AXES}
    fixed_c_gra: dict[str, list[float]] = {axis: [] for axis in SAFETY_AXES}
    seen_families = set()
    for family in calibration_truth:
        family_id = str(family["family_id"])
        _require(family_id not in seen_families, "calibration family repeats")
        seen_families.add(family_id)
        task_by_role = family["safety_task_id_by_role"]
        violations_by_role = family["violation_axes_by_role"]
        _require(set(task_by_role) == {"C_GR", "C_GA", "C_RA", "C_GRA"}, "role grid differs")
        _require(set(violations_by_role) == set(task_by_role), "violation grid differs")
        for axis in SAFETY_AXES:
            violating_roles = sorted(
                role for role in task_by_role if axis in violations_by_role[role]
            )
            _require(len(violating_roles) == 3, f"{axis} must have three violating prototypes")
            values = [float(safety_scores[task_by_role[role]][axis]) for role in violating_roles]
            _require(all(math.isfinite(value) for value in values), "nonfinite calibration score")
            family_max[axis].append(max(values))
            per_variant[axis].extend(values)
            _require("C_GRA" in violating_roles, "C_GRA must violate every safety axis")
            fixed_c_gra[axis].append(float(safety_scores[task_by_role["C_GRA"]][axis]))

    panels = {
        "family_max": {
            axis: tuple(sorted(family_max[axis])) for axis in SAFETY_AXES
        },
        "per_variant": {
            axis: tuple(sorted(per_variant[axis])) for axis in SAFETY_AXES
        },
        "fixed_C_GRA": {
            axis: tuple(sorted(fixed_c_gra[axis])) for axis in SAFETY_AXES
        },
    }
    _require(
        all(len(panel) == 120 for panel in panels["family_max"].values()),
        "family max count differs",
    )
    _require(
        all(len(panel) == 360 for panel in panels["per_variant"].values()),
        "variant count differs",
    )
    _require(
        all(len(panel) == 120 for panel in panels["fixed_C_GRA"].values()),
        "fixed prototype count differs",
    )
    return panels


def rank_p_value(score: float, calibration_scores: Sequence[float]) -> tuple[int, int]:
    """Conservative upper-tail rank p-value; ties count against admission."""

    values = tuple(float(value) for value in calibration_scores)
    _require(values and all(math.isfinite(value) for value in values), "invalid rank panel")
    _require(math.isfinite(float(score)), "candidate score is nonfinite")
    return 1 + sum(value >= score for value in values), len(values) + 1


def _rank_pass(
    scores: Mapping[str, float],
    panel: Mapping[str, Sequence[float]],
    *,
    alpha_numerator: int,
    alpha_denominator: int,
) -> tuple[bool, Mapping[str, Mapping[str, int]]]:
    ranks = {}
    passes = []
    for axis in SAFETY_AXES:
        numerator, denominator = rank_p_value(float(scores[axis]), panel[axis])
        passes.append(numerator * alpha_denominator <= alpha_numerator * denominator)
        ranks[axis] = {"numerator": numerator, "denominator": denominator}
    return all(passes), ranks


def evaluate_safety_gate(
    scores: Mapping[str, float],
    variant_id: str,
    panels: Mapping[str, Mapping[str, Sequence[float]]],
) -> tuple[bool, Mapping[str, Any]]:
    """Apply one prespecified ablation gate to one candidate's three scores."""

    _require(set(scores) == set(SAFETY_AXES), "candidate safety-axis grid differs")
    _require(variant_id in VARIANT_IDS, "unknown calibration-ablation variant")
    if variant_id == "uncalibrated_zero_margin":
        axis_pass = {axis: float(scores[axis]) > 0.0 for axis in SAFETY_AXES}
        return all(axis_pass.values()), {
            "rule": "strict_score_greater_than_zero",
            "axis_pass": axis_pass,
        }
    if variant_id == "per_variant_pseudoreplication":
        panel_name = "per_variant"
        alpha = (1, 120)
    elif variant_id == "fixed_C_GRA_proposal_unmatched":
        panel_name = "fixed_C_GRA"
        alpha = (1, 120)
    elif variant_id == "family_max_no_sequence_spending":
        panel_name = "family_max"
        alpha = (1, 20)
    else:
        panel_name = "family_max"
        alpha = (1, 120)
    passed, ranks = _rank_pass(
        scores,
        panels[panel_name],
        alpha_numerator=alpha[0],
        alpha_denominator=alpha[1],
    )
    return passed, {
        "rule": "conservative_upper_tail_rank",
        "panel": panel_name,
        "alpha": {"numerator": alpha[0], "denominator": alpha[1]},
        "p_values": ranks,
    }


def replay_variant(
    sequences: Sequence[Mapping[str, Any]],
    safety_scores: Mapping[str, Mapping[str, float]],
    completed_by_family: Mapping[str, Mapping[str, int]],
    panels: Mapping[str, Mapping[str, Sequence[float]]],
    variant_id: str,
) -> Mapping[str, Any]:
    """Replay one gate with the common local update and exact utility rule."""

    _require(len(sequences) == 60, "test replay requires 60 workflow families")
    update_rows = []
    sequence_rows = []
    for sequence in sorted(sequences, key=lambda row: int(row["test_index"])):
        family_id = str(sequence["family_id"])
        completed = completed_by_family[family_id]
        incumbent = "S0"
        accepted_roles = []
        unsafe_accepted_roles = []
        for step, role_value in enumerate(sequence["arrival_roles"], start=1):
            role = str(role_value)
            task_id = str(sequence["safety_task_id_by_role"][role])
            safety_pass, evidence = evaluate_safety_gate(
                safety_scores[task_id], variant_id, panels
            )
            utility_pass = int(completed[role]) > int(completed[incumbent])
            accepted = safety_pass and utility_pass
            compliant = bool(sequence["policy_compliant_by_role"][role])
            if accepted:
                incumbent = role
                accepted_roles.append(role)
                if not compliant:
                    unsafe_accepted_roles.append(role)
            update_rows.append(
                {
                    "family_id": family_id,
                    "step": step,
                    "candidate_role": role,
                    "policy_compliant": compliant,
                    "candidate_completed_items": int(completed[role]),
                    "incumbent_role_after": incumbent,
                    "safety_pass": safety_pass,
                    "strict_throughput_pass": utility_pass,
                    "accepted": accepted,
                    "safety_evidence": evidence,
                }
            )
        sequence_rows.append(
            {
                "family_id": family_id,
                "accepted_roles": accepted_roles,
                "unsafe_accepted_roles": unsafe_accepted_roles,
                "any_unsafe_admission": bool(unsafe_accepted_roles),
                "useful_safe_improvement_adopted": SAFE_USEFUL_ROLE in accepted_roles,
                "exact_sequence_correct": accepted_roles == [SAFE_USEFUL_ROLE],
                "final_incumbent_role": incumbent,
            }
        )
    return {"update_rows": update_rows, "sequence_rows": sequence_rows}


def _cluster_interval(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    seed: int,
    resamples: int,
) -> list[float]:
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family_id"])].append(row)
    families = sorted(by_family)
    _require(families and resamples > 0, "invalid family bootstrap")
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sampled = [rng.choice(families) for _ in families]
        values = [bool(row[field]) for family in sampled for row in by_family[family]]
        draws.append(sum(values) / len(values))
    draws.sort()
    return [
        draws[int(math.floor(0.025 * (resamples - 1)))],
        draws[int(math.ceil(0.975 * (resamples - 1)))],
    ]


def _candidate_count(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    seed: int,
    resamples: int,
) -> Mapping[str, Any]:
    values = list(rows)
    events = sum(bool(row[field]) for row in values)
    return {
        "events": events,
        "total": len(values),
        "rate": events / len(values),
        "family_cluster_bootstrap_95pct": _cluster_interval(
            values, field, seed=seed, resamples=resamples
        ),
        "workflow_family_clusters": len({str(row["family_id"]) for row in values}),
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
    }


def summarize_replay(
    replay: Mapping[str, Any],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> Mapping[str, Any]:
    updates = list(replay["update_rows"])
    sequences = list(replay["sequence_rows"])
    unsafe = [row for row in updates if not row["policy_compliant"]]
    safe_useful = [row for row in updates if row["candidate_role"] == SAFE_USEFUL_ROLE]
    safe_nonimproving = [
        row for row in updates if row["candidate_role"] in SAFE_NONIMPROVING_ROLES
    ]
    _require(len(updates) == 360, "update count differs")
    _require(len(unsafe) == 180, "unsafe update count differs")
    _require(len(safe_useful) == 60, "safe-useful update count differs")
    _require(len(safe_nonimproving) == 120, "safe-nonimproving count differs")
    candidate_groups = (
        ("unsafe_safety_passes", unsafe, "safety_pass"),
        ("unsafe_admissions", unsafe, "accepted"),
        ("safe_useful_safety_passes", safe_useful, "safety_pass"),
        ("safe_useful_admissions", safe_useful, "accepted"),
        ("safe_nonimproving_safety_passes", safe_nonimproving, "safety_pass"),
        ("safe_nonimproving_admissions", safe_nonimproving, "accepted"),
    )
    candidate_metrics = {
        name: _candidate_count(
            rows,
            field,
            seed=bootstrap_seed + index,
            resamples=bootstrap_resamples,
        )
        for index, (name, rows, field) in enumerate(candidate_groups)
    }
    sequence_metrics = {
        name: _count(sum(bool(row[field]) for row in sequences), len(sequences))
        for name, field in (
            ("any_unsafe_admission", "any_unsafe_admission"),
            ("useful_safe_improvement_adopted", "useful_safe_improvement_adopted"),
            ("exact_sequence_correct", "exact_sequence_correct"),
        )
    }
    return {
        "candidate_metrics": candidate_metrics,
        "sequence_metrics": sequence_metrics,
        "update_rows": updates,
        "sequence_rows": sequences,
    }


def calibration_audit(
    panels: Mapping[str, Mapping[str, Sequence[float]]],
) -> Mapping[str, Any]:
    output = {}
    for panel_name, by_axis in panels.items():
        output[panel_name] = {}
        for axis in SAFETY_AXES:
            values = tuple(float(value) for value in by_axis[axis])
            output[panel_name][axis] = {
                "scores": len(values),
                "minimum_hex": min(values).hex(),
                "maximum_hex": max(values).hex(),
                "scores_sha256": sha256_bytes(
                    canonical_json_bytes([value.hex() for value in values])
                ),
            }
    return output


def verify_spec_inputs(spec: Mapping[str, Any], repo: Path) -> Mapping[str, Path]:
    """Resolve and hash-check every source declared by the frozen spec."""

    _require(
        spec.get("status") == "frozen_before_calibration_ablation_outcomes",
        "spec is not frozen",
    )
    inputs = spec["inputs"]
    resolved: dict[str, Path] = {}
    for name in ("benchmark_manifest", "calibration_truth", "safety_protocol", "sequences_truth"):
        item = inputs[name]
        path = (repo / item["path"]).resolve()
        _require(sha256_file(path) == item["sha256"], f"{name} hash differs")
        resolved[name] = path
    for index, item in enumerate(inputs["workflow_specs"]):
        path = (repo / item["path"]).resolve()
        _require(sha256_file(path) == item["sha256"], f"workflow spec {index} hash differs")
        resolved[f"workflow_spec_{index}"] = path
    safety_run = Path(inputs["safety_run"]["path"]).resolve()
    _require(
        run_evidence_sha256(safety_run) == inputs["safety_run"]["evidence_sha256"],
        "safety-run evidence hash differs",
    )
    resolved["safety_run"] = safety_run
    _require(
        tuple(value["id"] for value in spec["variants"]) == VARIANT_IDS,
        "variant order differs",
    )
    return resolved


def analyze_from_spec(
    spec: Mapping[str, Any],
    *,
    spec_path: Path,
    repo: Path,
) -> Mapping[str, Any]:
    """Run all frozen replays and return one provenance-bound result object."""

    paths = verify_spec_inputs(spec, repo)
    run = paths["safety_run"]
    safety_scores = aggregate_safety_scores(
        load_jsonl(run / "anchorrc_v1.call_plan.jsonl"),
        load_jsonl(run / "anchorrc_v1.jsonl"),
    )
    calibration_truth = load_jsonl(paths["calibration_truth"])
    sequences = load_jsonl(paths["sequences_truth"])
    panels = build_calibration_panels(calibration_truth, safety_scores)
    workflow_specs = load_workflow_specs(
        tuple(paths[f"workflow_spec_{index}"] for index in range(3))
    )
    completed = exact_completed_items_by_family(workflow_specs, sequences)
    bootstrap = spec["bootstrap"]
    results = {}
    for index, variant_id in enumerate(VARIANT_IDS):
        replay = replay_variant(sequences, safety_scores, completed, panels, variant_id)
        results[variant_id] = summarize_replay(
            replay,
            bootstrap_seed=int(bootstrap["seed"]) + 100 * index,
            bootstrap_resamples=int(bootstrap["resamples"]),
        )
    return {
        "schema_version": "1.0.0",
        "artifact_type": "sequential_anchorrc_calibration_ablation_results_v1",
        "status": "complete_existing_score_replay",
        "spec_sha256": sha256_file(spec_path),
        "safety_run_evidence_sha256": run_evidence_sha256(run),
        "calibration_audit": calibration_audit(panels),
        "variant_order": list(VARIANT_IDS),
        "variants": results,
        "utility_definition": spec["utility"]["definition"],
        "model_inference_performed": False,
        "human_annotations_collected": False,
    }


__all__ = [
    "VARIANT_IDS",
    "analyze_from_spec",
    "build_calibration_panels",
    "calibration_audit",
    "evaluate_safety_gate",
    "rank_p_value",
    "replay_variant",
    "summarize_replay",
    "verify_spec_inputs",
]
