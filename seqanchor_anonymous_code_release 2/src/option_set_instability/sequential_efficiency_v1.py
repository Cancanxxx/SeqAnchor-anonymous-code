"""Frozen update panels and summaries for Sequential AnchorRC efficiency tests."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence


TEST_ROLES = ("S1", "E", "N", "C_GR", "C_GA", "C_RA")
SAFETY_AXES = (
    "gate_before_execution",
    "records_by_deadline",
    "listed_resources_only",
)


def first_arrival_panel(
    sequences: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Select each test sequence's first update, balanced by construction."""

    panel = []
    seen_families = set()
    for sequence in sorted(sequences, key=lambda row: int(row["test_index"])):
        family_id = str(sequence["family_id"])
        if family_id in seen_families:
            raise ValueError("sequential efficiency panel contains a duplicate family")
        seen_families.add(family_id)
        roles = sequence["arrival_roles"]
        if not isinstance(roles, list) or len(roles) != len(TEST_ROLES):
            raise ValueError("sequence has an invalid arrival schedule")
        role = str(roles[0])
        if role not in TEST_ROLES:
            raise ValueError("sequence has an unknown first-arrival role")
        if sequence.get("initial_incumbent_role") != "S0":
            raise ValueError("efficiency panel requires the frozen S0 incumbent")
        utility_tasks = sequence["utility_task_id_by_role_and_incumbent"]
        panel.append(
            {
                "test_index": int(sequence["test_index"]),
                "family_id": family_id,
                "sector": str(sequence["sector"]),
                "arrival_position": 1,
                "candidate_role": role,
                "candidate_id": str(sequence["candidate_by_role"][role]["candidate_id"]),
                "safety_task_id": str(sequence["safety_task_id_by_role"][role]),
                "utility_task_id": str(utility_tasks[role]["S0"]),
            }
        )
    if len(panel) != 60:
        raise ValueError("sequential efficiency panel must contain exactly 60 updates")
    role_counts = Counter(row["candidate_role"] for row in panel)
    if role_counts != Counter({role: 10 for role in TEST_ROLES}):
        raise ValueError("first-arrival panel is not balanced 10 per candidate role")
    return tuple(panel)


def _calls_by_task(
    calls: Sequence[Mapping[str, Any]],
) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for call in calls:
        aliases = call.get("aliases")
        if not isinstance(aliases, list) or len(aliases) != 1:
            raise ValueError("efficiency call plan requires one task alias per call")
        task_id = aliases[0].get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("efficiency call has an invalid task alias")
        grouped[task_id].append(call)
    return {
        task_id: tuple(sorted(values, key=lambda row: str(row["call_id"])))
        for task_id, values in grouped.items()
    }


def build_update_call_groups(
    panel: Sequence[Mapping[str, Any]],
    safety_calls: Sequence[Mapping[str, Any]],
    utility_calls: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Bind each frozen update to six safety and four utility scorer calls."""

    safety_by_task = _calls_by_task(safety_calls)
    utility_by_task = _calls_by_task(utility_calls)
    groups = []
    for update in panel:
        safety = safety_by_task.get(str(update["safety_task_id"]), ())
        utility = utility_by_task.get(str(update["utility_task_id"]), ())
        if len(safety) != 6 or len(utility) != 4:
            raise ValueError("each efficiency update requires six safety and four utility calls")
        if any(call.get("call_kind") != "safety_axis" for call in safety):
            raise ValueError("safety efficiency group contains a non-safety call")
        if any(call.get("call_kind") != "utility_pair" for call in utility):
            raise ValueError("utility efficiency group contains a non-utility call")
        axes = Counter(str(call.get("axis_id")) for call in safety)
        if axes != Counter({axis: 2 for axis in SAFETY_AXES}):
            raise ValueError("safety efficiency group does not cover every axis twice")
        if Counter(int(call.get("label_swap")) for call in safety) != Counter({0: 3, 1: 3}):
            raise ValueError("safety efficiency group has invalid label swaps")
        if Counter(int(call.get("label_swap")) for call in utility) != Counter({0: 2, 1: 2}):
            raise ValueError("utility efficiency group has invalid label swaps")
        snapshots = {str(call.get("snapshot_id")) for call in (*safety, *utility)}
        if len(snapshots) != 1:
            raise ValueError("efficiency update spans multiple model snapshots")
        groups.append(
            {
                **dict(update),
                "snapshot_id": snapshots.pop(),
                "safety_calls": safety,
                "utility_calls": utility,
            }
        )
    return tuple(groups)


def select_tsguard_latency_rows(
    panel: Sequence[Mapping[str, Any]],
    adaptation_rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Select the same 60 first-arrival candidates for TS-Guard latency."""

    indexed = {}
    for row in adaptation_rows:
        key = (str(row.get("family_id")), str(row.get("candidate_role")))
        if key in indexed:
            raise ValueError("TS-Guard adaptation duplicates a family/role pair")
        indexed[key] = row
    selected = []
    for update in panel:
        key = (str(update["family_id"]), str(update["candidate_role"]))
        if key not in indexed:
            raise ValueError("TS-Guard adaptation is missing a latency-panel candidate")
        row = dict(indexed[key])
        row["latency_panel_test_index"] = int(update["test_index"])
        row["latency_panel_arrival_position"] = 1
        selected.append(row)
    if len(selected) != 60 or len({row["row_id"] for row in selected}) != 60:
        raise ValueError("TS-Guard latency panel must contain 60 unique rows")
    return tuple(selected)


def linear_quantile(values: Sequence[float], probability: float) -> float:
    """Return a deterministic type-7 sample quantile."""

    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("quantile values must be finite")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(values: Sequence[float]) -> Mapping[str, float]:
    """Summarize a fixed latency panel without hiding its tail."""

    numeric = tuple(float(value) for value in values)
    return {
        "mean": sum(numeric) / len(numeric),
        "median": linear_quantile(numeric, 0.5),
        "p90": linear_quantile(numeric, 0.9),
        "p95": linear_quantile(numeric, 0.95),
        "minimum": min(numeric),
        "maximum": max(numeric),
    }
