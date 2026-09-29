from __future__ import annotations

import itertools

from option_set_instability.expansion_mechanism_analysis_v1 import (
    aggregate_independent_group,
    aggregate_joint_group,
    clustered_binary_count,
    paired_margin_shift_contrast,
    presentation_view_features,
    summarize_presentation_grid,
)


def _joint_rows(roles, semantic_scores, menu_kind="expanded", addition_role="P"):
    labels = tuple("ABC"[: len(roles)])
    rows = []
    for display_order in itertools.permutations(roles):
        for label_roles in itertools.permutations(roles):
            role_to_label = dict(zip(label_roles, labels))
            logits = {role_to_label[role]: float(semantic_scores[role]) for role in roles}
            rows.append(
                {
                    "call_id": f"call-{len(rows)}",
                    "family_id": "family",
                    "sector": "sector",
                    "menu_kind": menu_kind,
                    "addition_role": None if menu_kind == "base" else addition_role,
                    "source_task_id": None,
                    "display_order": list(display_order),
                    "role_to_label": role_to_label,
                    "allowed_label_logits": logits,
                }
            )
    return rows


def test_joint_aggregation_removes_label_and_display_assignment() -> None:
    rows = _joint_rows(("S0", "U0", "P"), {"S0": 2.0, "U0": 3.0, "P": 0.0})
    result = aggregate_joint_group(rows)
    assert len(rows) == 36
    assert result["winner_role"] == "U0"
    assert result["margin_S0_minus_U0"] == -1.0
    assert result["display_order_mean_range"] == 0.0
    assert result["answer_label_mean_range"] == 0.0


def test_base_aggregation_uses_full_four_view_crossing() -> None:
    rows = _joint_rows(("S0", "U0"), {"S0": 4.0, "U0": 1.0}, menu_kind="base", addition_role=None)
    result = aggregate_joint_group(rows)
    assert len(rows) == 4
    assert result["winner_role"] == "S0"
    assert result["margin_S0_minus_U0"] == 3.0


def test_independent_label_swap_recovers_semantic_margin() -> None:
    rows = [
        {
            "family_id": "family",
            "sector": "sector",
            "axis": "policy_compliance",
            "candidate_role": "S0",
            "positive_label": "A",
            "negative_label": "B",
            "allowed_label_logits": {"A": 5.0, "B": 2.0},
        },
        {
            "family_id": "family",
            "sector": "sector",
            "axis": "policy_compliance",
            "candidate_role": "S0",
            "positive_label": "B",
            "negative_label": "A",
            "allowed_label_logits": {"A": 2.0, "B": 5.0},
        },
    ]
    result = aggregate_independent_group(rows)
    assert result["positive_minus_negative_margin"] == 3.0
    assert result["label_swap_range"] == 0.0


def test_combined_update_interval_resamples_workflow_families() -> None:
    rows = [
        {"family_id": family_id, "event": family_id == "family-0"}
        for family_id in ("family-0", "family-1", "family-2", "family-3")
        for _ in range(4)
    ]
    result = clustered_binary_count(rows, "event", seed=7, resamples=1000)
    assert result["events"] == 4
    assert result["total"] == 16
    assert result["rate"] == 0.25
    assert result["workflow_family_clusters"] == 4
    assert result["inference_unit"] == "workflow_family_cluster"
    assert "exact_95pct" not in result


def test_semantic_competition_contrast_is_paired_within_family() -> None:
    rows = [
        {"family_id": "a", "addition_role": "P", "margin_shift": -3.0},
        {"family_id": "a", "addition_role": "N", "margin_shift": -1.0},
        {"family_id": "b", "addition_role": "P", "margin_shift": 2.0},
        {"family_id": "b", "addition_role": "N", "margin_shift": 1.0},
    ]
    result = paired_margin_shift_contrast(rows, "P", "N", seed=11, resamples=1000)
    assert result["definition"] == "margin_shift(P) - margin_shift(N)"
    assert result["workflow_families"] == 2
    assert result["mean"] == -0.5
    assert result["left_shifts_more_toward_U0"] == 1
    assert result["right_shifts_more_toward_U0"] == 1


def test_presentation_grid_keeps_views_inside_workflow_family() -> None:
    aggregate = aggregate_joint_group(
        _joint_rows(("S0", "U0", "P"), {"S0": 2.0, "U0": 3.0, "P": 0.0})
    )
    features = presentation_view_features(aggregate["views"], "P")
    assert len(features["normalized_display_grid"]) == 6
    assert all(
        cell["mean_margin_S0_minus_U0"] == -1.0 and cell["unsafe_selection_fraction"] == 1.0
        for cell in features["normalized_display_grid"].values()
    )
    rows = [
        {
            "family_id": family_id,
            "presentation_normalized_display_grid": features["normalized_display_grid"],
            "presentation_by_addition_position": features["by_addition_position"],
            "presentation_by_relative_order": features["by_relative_order"],
        }
        for family_id in ("family-a", "family-b")
    ]
    summary = summarize_presentation_grid(rows, seed=19, resamples=100)
    first_cell = summary["normalized_display_grid"]["C>S0>U0"]
    assert first_cell["mean_margin_S0_minus_U0"]["family_bootstrap_95pct"] == [
        -1.0,
        -1.0,
    ]
    assert summary["paired_relative_order_contrast"]["mean_margin_S0_minus_U0"][
        "family_bootstrap_95pct"
    ] == [0.0, 0.0]
    assert summary["multiplicity_adjustment"] is None
