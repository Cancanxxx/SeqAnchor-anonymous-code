from __future__ import annotations

from collections import Counter
from pathlib import Path

from option_set_instability.anchorrc_v1 import normalize_public_task
from option_set_instability.sequential_benchmark_v1 import (
    CALIBRATION_UNSAFE_ROLES,
    TEST_ROLES,
    build_complete_test_utility_extension,
    build_sequential_benchmark,
    family_partition,
)
from option_set_instability.state_expansion_180 import load_workflow_specs

REPO = Path(__file__).resolve().parents[1]
SPEC_PATHS = tuple(
    REPO / "data" / "workflow_specs" / name
    for name in ("part_a.json", "part_b.json", "part_c.json")
)


def _bank():
    return build_sequential_benchmark(load_workflow_specs(SPEC_PATHS))


def test_family_split_is_disjoint_and_sector_blocked() -> None:
    calibration, test = family_partition(load_workflow_specs(SPEC_PATHS))
    assert len(calibration) == 120
    assert len(test) == 60
    assert not ({row.slug for row in calibration} & {row.slug for row in test})
    assert Counter(Counter(row.sector for row in calibration).values()) == Counter({10: 12})
    assert Counter(Counter(row.sector for row in test).values()) == Counter({5: 12})


def test_public_tasks_are_valid_singleton_anchorrc_updates() -> None:
    bank = _bank()
    assert len(bank["safety_tasks"]) == 840
    assert len(bank["utility_tasks"]) == 1140
    for task in (*bank["safety_tasks"], *bank["utility_tasks"]):
        normalized = normalize_public_task(task)
        assert normalized["known_safe_incumbent"]["protocol_id"] == bank["manifest"]["protocol_id"]


def test_calibration_and_test_candidate_grids_are_exact() -> None:
    bank = _bank()
    assert len(bank["calibration_truth"]) == 120
    for row in bank["calibration_truth"]:
        assert set(row["safety_task_id_by_role"]) == set(CALIBRATION_UNSAFE_ROLES)
        assert set(row["utility_task_id_by_role_and_incumbent"]) == {"E", "N"}
    assert len(bank["sequences"]) == 60
    assert len(bank["tsguard_rows"]) == 360
    assert Counter(row["truth_label"] for row in bank["tsguard_rows"]) == {
        0.0: 180,
        1.0: 180,
    }
    for row in bank["sequences"]:
        assert set(row["arrival_roles"]) == set(TEST_ROLES)
        assert len(row["arrival_roles"]) == 6
        assert sum(row["strict_utility_improvement_by_role"].values()) == 1


def test_every_role_occupies_every_arrival_position_ten_times() -> None:
    positions = Counter()
    for row in _bank()["sequences"]:
        for position, role in enumerate(row["arrival_roles"]):
            positions[(role, position)] += 1
    assert len(positions) == 36
    assert set(positions.values()) == {10}


def test_build_is_byte_deterministic() -> None:
    first = _bank()
    second = _bank()
    assert first == second


def test_complete_utility_extension_covers_e_and_n_incumbents() -> None:
    extension = build_complete_test_utility_extension(load_workflow_specs(SPEC_PATHS))
    assert len(extension["utility_tasks"]) == 600
    assert len(extension["extension_truth"]) == 60
    assert extension["manifest"]["thresholds_changed"] is False
    for row in extension["extension_truth"]:
        mapping = row["utility_task_id_by_role_and_incumbent"]
        assert set(mapping) == set(TEST_ROLES)
        assert set(mapping["S1"]) == {"E", "N"}
        assert set(mapping["E"]) == {"N"}
        assert set(mapping["N"]) == {"E"}
        for role in ("C_GR", "C_GA", "C_RA"):
            assert set(mapping[role]) == {"E", "N"}
