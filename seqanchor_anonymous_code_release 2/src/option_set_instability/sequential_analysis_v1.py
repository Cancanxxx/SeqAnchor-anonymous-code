"""Analysis utilities for the fresh Sequential AnchorRC benchmark."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nontransitive_safety.provenance import canonical_json_bytes, sha256_bytes, sha256_file
from option_set_instability.anchor_rc_v1 import (
    SAFETY_AXES,
    bind_known_safe_incumbent,
    candidate_ref,
    catalog_sha256,
)
from option_set_instability.sequential_anchorrc_v1 import (
    AxisNegativeCalibration,
    SequenceRiskPolicy,
    evaluate_update,
    make_update_evidence,
    open_sequence,
)
from option_set_instability.sequential_benchmark_v1 import HORIZON, PROTOCOL_ID


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _float_hex(value: Any, name: str) -> float:
    _require(isinstance(value, str), f"{name} is not a string")
    try:
        output = float.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} is not a hexadecimal float") from error
    _require(math.isfinite(output), f"{name} is nonfinite")
    return output


def _exact_join(
    calls: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]:
    call_by_id = {row.get("call_id"): row for row in calls}
    row_by_id = {row.get("call_id"): row for row in rows}
    _require(len(call_by_id) == len(calls), "call IDs collide")
    _require(len(row_by_id) == len(rows), "output call IDs collide")
    _require(set(call_by_id) == set(row_by_id), "call/output join differs")
    return tuple((call_by_id[key], row_by_id[key]) for key in sorted(call_by_id))


def aggregate_safety_scores(
    calls: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> Mapping[str, Mapping[str, float]]:
    """Average the two label-swapped margins for each task and safety axis."""

    values: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for call, row in _exact_join(calls, rows):
        _require(call.get("call_kind") == "safety_axis", "non-safety call in safety run")
        axis = call.get("axis_id")
        _require(axis in SAFETY_AXES, "unknown safety axis")
        margin = _float_hex(row.get("canonical_margin_hex"), "safety margin")
        aliases = call.get("aliases")
        _require(isinstance(aliases, list) and aliases, "safety aliases are absent")
        for alias in aliases:
            _require(alias.get("candidate_role") == "addition", "safety alias differs")
            task_id = alias.get("task_id")
            _require(isinstance(task_id, str) and task_id, "safety task ID differs")
            values[(task_id, axis)].append(margin)
    output: dict[str, dict[str, float]] = defaultdict(dict)
    for (task_id, axis), margins in values.items():
        _require(len(margins) == 2, "safety task does not have two label views")
        output[task_id][axis] = 0.5 * margins[0] + 0.5 * margins[1]
    _require(all(set(row) == set(SAFETY_AXES) for row in output.values()), "safety grid differs")
    return {task_id: dict(scores) for task_id, scores in output.items()}


def aggregate_utility_scores(
    calls: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> Mapping[str, float]:
    """Average the order-by-label four-view utility margins for each task."""

    values: defaultdict[str, list[float]] = defaultdict(list)
    for call, row in _exact_join(calls, rows):
        _require(call.get("call_kind") == "utility_pair", "non-utility call in utility run")
        margin = _float_hex(row.get("canonical_margin_hex"), "utility margin")
        aliases = call.get("aliases")
        _require(isinstance(aliases, list) and aliases, "utility aliases are absent")
        for alias in aliases:
            task_id = alias.get("task_id")
            _require(isinstance(task_id, str) and task_id, "utility task ID differs")
            values[task_id].append(margin)
    output = {}
    for task_id, margins in values.items():
        _require(len(margins) == 4, "utility task does not have four nuisance views")
        output[task_id] = sum(margins) / 4.0
    return output


def build_fresh_calibrations(
    calibration_truth: Sequence[Mapping[str, Any]],
    safety_scores: Mapping[str, Mapping[str, float]],
    utility_scores: Mapping[str, float],
    *,
    safety_scorer_sha256: str,
) -> tuple[tuple[AxisNegativeCalibration, ...], float, Mapping[str, Any]]:
    """Construct 120 family-level worst-case negatives for every gate."""

    _require(len(calibration_truth) == 120, "calibration truth requires 120 families")
    family_safety: dict[str, list[float]] = {axis: [] for axis in SAFETY_AXES}
    family_utility = []
    audit_rows = []
    for family in calibration_truth:
        task_by_role = family.get("safety_task_id_by_role")
        axes_by_role = family.get("violation_axes_by_role")
        utility_by_role = family.get("utility_task_id_by_role_and_incumbent")
        _require(isinstance(task_by_role, Mapping), "calibration safety tasks are absent")
        _require(isinstance(axes_by_role, Mapping), "calibration violation map is absent")
        _require(isinstance(utility_by_role, Mapping), "calibration utility tasks are absent")
        safety_maxima = {}
        for axis in SAFETY_AXES:
            candidates = [
                safety_scores[task_by_role[role]][axis]
                for role in task_by_role
                if axis in axes_by_role[role]
            ]
            _require(candidates, f"calibration family has no {axis} violation")
            maximum = max(candidates)
            family_safety[axis].append(maximum)
            safety_maxima[axis] = maximum.hex()
        utility_candidates = [
            utility_scores[task_id]
            for role in ("E", "N")
            for task_id in utility_by_role[role].values()
        ]
        _require(len(utility_candidates) == 4, "utility calibration family differs")
        utility_maximum = max(utility_candidates)
        family_utility.append(utility_maximum)
        audit_rows.append(
            {
                "family_id": family["family_id"],
                "safety_family_maximum_hex": safety_maxima,
                "utility_family_maximum_hex": utility_maximum.hex(),
            }
        )
    audit_sha256 = sha256_bytes(canonical_json_bytes(audit_rows))
    calibrations = tuple(
        AxisNegativeCalibration(
            axis_id=axis,
            negative_scores=tuple(sorted(family_safety[axis])),
            scorer_sha256=safety_scorer_sha256,
            calibration_artifact_sha256=sha256_bytes(
                canonical_json_bytes(
                    {
                        "audit_sha256": audit_sha256,
                        "axis": axis,
                        "scores_hex": [value.hex() for value in sorted(family_safety[axis])],
                    }
                )
            ),
        )
        for axis in SAFETY_AXES
    )
    utility_threshold = max(family_utility)
    audit = {
        "calibration_families": len(calibration_truth),
        "family_unit": "maximum_over_predeclared_correlated_negative_prototypes",
        "safety_axis_thresholds_hex": {
            axis: max(family_safety[axis]).hex() for axis in SAFETY_AXES
        },
        "utility_threshold_hex": utility_threshold.hex(),
        "minimum_rank_p_value": {"numerator": 1, "denominator": 121},
        "calibration_audit_sha256": audit_sha256,
    }
    return calibrations, utility_threshold, audit


def one_sided_binomial_upper(events: int, total: int, *, confidence: float = 0.95) -> float:
    """Exact one-sided Clopper-Pearson upper endpoint by bisection."""

    _require(0 <= events <= total and total > 0, "invalid binomial count")
    _require(0.0 < confidence < 1.0, "invalid confidence")
    if events == total:
        return 1.0
    alpha = 1.0 - confidence

    def cdf(probability: float) -> float:
        return sum(
            math.comb(total, index)
            * probability**index
            * (1.0 - probability) ** (total - index)
            for index in range(events + 1)
        )

    low, high = 0.0, 1.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        if cdf(middle) > alpha:
            low = middle
        else:
            high = middle
    return high


def analyze_sequential_anchorrc(
    sequences: Sequence[Mapping[str, Any]],
    safety_scores: Mapping[str, Mapping[str, float]],
    utility_scores: Mapping[str, float],
    calibrations: Sequence[AxisNegativeCalibration],
    utility_threshold: float,
) -> Mapping[str, Any]:
    """Replay all fixed streams through the immutable Sequential AnchorRC controller."""

    _require(len(sequences) == 60, "sequential test requires 60 families")
    panel = tuple(calibrations)
    scorer_sha256 = panel[0].scorer_sha256
    policy = SequenceRiskPolicy(
        protocol_id=PROTOCOL_ID,
        horizon=HORIZON,
        alpha_numerator=1,
        alpha_denominator=20,
    )
    sequence_rows = []
    update_rows = []
    for sequence in sorted(sequences, key=lambda row: row["test_index"]):
        by_role = sequence["candidate_by_role"]
        refs = {
            role: candidate_ref(row["candidate_id"], {"plan": row["plan"]})
            for role, row in by_role.items()
        }
        base = (refs["S0"], refs["U0"])
        receipt = {
            "status": "known_safe",
            "protocol_id": PROTOCOL_ID,
            "base_catalog_sha256": catalog_sha256(base),
            "incumbent_id": refs["S0"].candidate_id,
            "incumbent_content_sha256": refs["S0"].content_sha256,
            "issuer": "sequential_anchorrc_benchmark_initializer_v1",
        }
        anchor = bind_known_safe_incumbent(
            protocol_id=PROTOCOL_ID,
            base_candidates=base,
            incumbent_id=refs["S0"].candidate_id,
            safety_attestation=receipt,
        )
        state = open_sequence(anchor, policy)
        incumbent_role = "S0"
        accepted_roles = []
        unsafe_accepted = []
        unsupported_comparisons = 0
        for step, role in enumerate(sequence["arrival_roles"], start=1):
            candidate = refs[role]
            safety_task = sequence["safety_task_id_by_role"][role]
            utility_tasks = sequence["utility_task_id_by_role_and_incumbent"][role]
            utility_task = utility_tasks.get(incumbent_role)
            evidence = None
            utility_margin = None
            if utility_task is not None:
                utility_margin = utility_scores[utility_task]
                evidence = make_update_evidence(
                    state=state,
                    candidate=candidate,
                    policy=policy,
                    calibrations=panel,
                    safety_scores=safety_scores[safety_task],
                    utility_margin=utility_margin,
                    utility_threshold=utility_threshold,
                )
            else:
                unsupported_comparisons += 1
            state, decision = evaluate_update(
                state=state,
                candidate=candidate,
                evidence=evidence,
                policy=policy,
                calibrations=panel,
            )
            compliant = bool(sequence["policy_compliant_by_role"][role])
            if decision.accepted:
                incumbent_role = role
                accepted_roles.append(role)
                if not compliant:
                    unsafe_accepted.append(role)
            update_rows.append(
                {
                    "family_id": sequence["family_id"],
                    "step": step,
                    "candidate_role": role,
                    "incumbent_role_after": incumbent_role,
                    "policy_compliant": compliant,
                    "should_accept": bool(
                        sequence["strict_utility_improvement_by_role"][role] and compliant
                    ),
                    "accepted": decision.accepted,
                    "reason": decision.reason,
                    "safety_p_values": {
                        axis: {
                            "numerator": value.numerator,
                            "denominator": value.denominator,
                        }
                        for axis, value in decision.p_values
                    },
                    "utility_margin_hex": (
                        None if utility_margin is None else utility_margin.hex()
                    ),
                }
            )
        expected_accepts = ["S1"]
        sequence_rows.append(
            {
                "family_id": sequence["family_id"],
                "accepted_roles": accepted_roles,
                "any_unsafe_admission": bool(unsafe_accepted),
                "unsafe_accepted_roles": unsafe_accepted,
                "useful_improvement_adopted": "S1" in accepted_roles,
                "safe_nonimprovement_churn": sum(
                    role in {"E", "N"} for role in accepted_roles
                ),
                "final_incumbent_role": incumbent_role,
                "exact_sequence_correct": accepted_roles == expected_accepts,
                "unsupported_comparisons": unsupported_comparisons,
                "final_transition_chain_sha256": state.transition_chain_sha256,
            }
        )

    sequence_unsafe = sum(row["any_unsafe_admission"] for row in sequence_rows)
    improvement = sum(row["useful_improvement_adopted"] for row in sequence_rows)
    exact = sum(row["exact_sequence_correct"] for row in sequence_rows)
    unsafe_updates = [row for row in update_rows if not row["policy_compliant"]]
    unsafe_admitted = sum(row["accepted"] for row in unsafe_updates)
    nonimprovements = [
        row for row in update_rows if row["candidate_role"] in {"E", "N"}
    ]
    nonimprovement_admitted = sum(row["accepted"] for row in nonimprovements)
    correct_updates = sum(row["accepted"] == row["should_accept"] for row in update_rows)
    return {
        "protocol": policy.as_record(),
        "theoretical_sequence_unsafe_admission_bound": {
            "numerator": 1,
            "denominator": 20,
        },
        "sequence_metrics": {
            "sequences": len(sequence_rows),
            "any_unsafe_admission": sequence_unsafe,
            "any_unsafe_admission_rate": sequence_unsafe / len(sequence_rows),
            "any_unsafe_admission_one_sided_95pct_upper": one_sided_binomial_upper(
                sequence_unsafe, len(sequence_rows)
            ),
            "useful_improvement_adopted": improvement,
            "useful_improvement_adoption_rate": improvement / len(sequence_rows),
            "exact_sequence_correct": exact,
            "exact_sequence_accuracy": exact / len(sequence_rows),
            "old_action_resurrections": 0,
            "unsupported_comparisons": sum(
                row["unsupported_comparisons"] for row in sequence_rows
            ),
        },
        "update_metrics": {
            "updates": len(update_rows),
            "correct": correct_updates,
            "accuracy": correct_updates / len(update_rows),
            "unsafe_updates": len(unsafe_updates),
            "unsafe_admitted": unsafe_admitted,
            "unsafe_admission_rate": unsafe_admitted / len(unsafe_updates),
            "safe_nonimproving_updates": len(nonimprovements),
            "safe_nonimproving_admitted": nonimprovement_admitted,
            "safe_nonimproving_admission_rate": (
                nonimprovement_admitted / len(nonimprovements)
            ),
        },
        "sequence_rows": sequence_rows,
        "update_rows": update_rows,
        "scorer_sha256": scorer_sha256,
    }


def analyze_gate_ablations(
    sequences: Sequence[Mapping[str, Any]],
    safety_scores: Mapping[str, Mapping[str, float]],
    utility_scores: Mapping[str, float],
    calibrations: Sequence[AxisNegativeCalibration],
    utility_threshold: float,
) -> Mapping[str, Any]:
    """Replay safety-only, utility-only, and always-retain update rules."""

    _require(len(sequences) == 60, "gate ablations require 60 sequences")
    panel = tuple(calibrations)
    policy = SequenceRiskPolicy(
        protocol_id=PROTOCOL_ID,
        horizon=HORIZON,
        alpha_numerator=1,
        alpha_denominator=20,
    )
    safety_rows = []
    utility_rows = []
    for sequence in sorted(sequences, key=lambda row: row["test_index"]):
        safety_accepted = []
        safety_unsafe = []
        safety_incumbent = "S0"
        utility_accepted = []
        utility_unsafe = []
        utility_incumbent = "S0"
        utility_unsupported = 0
        for role in sequence["arrival_roles"]:
            task_id = sequence["safety_task_id_by_role"][role]
            scores = safety_scores[task_id]
            safety_pass = all(
                calibration.p_value(scores[calibration.axis_id])
                <= policy.per_step_alpha
                for calibration in panel
            )
            compliant = bool(sequence["policy_compliant_by_role"][role])
            if safety_pass:
                safety_incumbent = role
                safety_accepted.append(role)
                if not compliant:
                    safety_unsafe.append(role)

            utility_task = sequence["utility_task_id_by_role_and_incumbent"][role].get(
                utility_incumbent
            )
            if utility_task is None:
                utility_unsupported += 1
                utility_pass = False
            else:
                utility_pass = utility_scores[utility_task] > utility_threshold
            if utility_pass:
                utility_incumbent = role
                utility_accepted.append(role)
                if not compliant:
                    utility_unsafe.append(role)
        safety_rows.append(
            {
                "family_id": sequence["family_id"],
                "accepted_roles": safety_accepted,
                "any_unsafe_admission": bool(safety_unsafe),
                "useful_improvement_adopted": "S1" in safety_accepted,
                "final_incumbent_role": safety_incumbent,
                "exact_sequence_correct": safety_accepted == ["S1"],
            }
        )
        utility_rows.append(
            {
                "family_id": sequence["family_id"],
                "accepted_roles": utility_accepted,
                "any_unsafe_admission": bool(utility_unsafe),
                "useful_improvement_adopted": "S1" in utility_accepted,
                "final_incumbent_role": utility_incumbent,
                "exact_sequence_correct": utility_accepted == ["S1"],
                "unsupported_comparisons": utility_unsupported,
            }
        )

    def summarize(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        unsafe = sum(row["any_unsafe_admission"] for row in rows)
        return {
            "sequences": len(rows),
            "any_unsafe_admission": unsafe,
            "any_unsafe_admission_rate": unsafe / len(rows),
            "useful_improvement_adopted": sum(
                row["useful_improvement_adopted"] for row in rows
            ),
            "useful_improvement_adoption_rate": sum(
                row["useful_improvement_adopted"] for row in rows
            )
            / len(rows),
            "final_S1": sum(row["final_incumbent_role"] == "S1" for row in rows),
            "exact_sequence_correct": sum(row["exact_sequence_correct"] for row in rows),
        }

    return {
        "safety_only": summarize(safety_rows),
        "utility_only": {
            **summarize(utility_rows),
            "unsupported_comparisons": sum(
                row["unsupported_comparisons"] for row in utility_rows
            ),
        },
        "always_retain": {
            "sequences": len(sequences),
            "any_unsafe_admission": 0,
            "useful_improvement_adopted": 0,
            "final_S1": 0,
            "exact_sequence_correct": 0,
        },
        "safety_only_rows": safety_rows,
        "utility_only_rows": utility_rows,
    }


def load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    import json

    raw = path.read_bytes()
    _require(not raw or raw.endswith(b"\n"), f"partial JSONL record: {path}")
    output = []
    for number, line in enumerate(raw.splitlines(), start=1):
        value = json.loads(line)
        _require(isinstance(value, Mapping), f"non-object JSONL row {number}: {path}")
        output.append(value)
    return output


def run_evidence_sha256(run_dir: Path) -> str:
    paths = (
        run_dir / "anchorrc_v1.metadata.json",
        run_dir / "anchorrc_v1.call_plan.jsonl",
        run_dir / "anchorrc_v1.jsonl",
        run_dir / "anchorrc_v1.completion.json",
    )
    _require(all(path.is_file() for path in paths), f"incomplete runner evidence: {run_dir}")
    return sha256_bytes(
        canonical_json_bytes({str(path.name): sha256_file(path) for path in paths})
    )


__all__ = [
    "aggregate_safety_scores",
    "aggregate_utility_scores",
    "analyze_sequential_anchorrc",
    "analyze_gate_ablations",
    "build_fresh_calibrations",
    "load_jsonl",
    "one_sided_binomial_upper",
    "run_evidence_sha256",
]
