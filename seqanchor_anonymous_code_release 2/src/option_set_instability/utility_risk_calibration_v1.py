"""Distribution-free operating points for AnchorRC's utility gate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def family_nonimprovement_maxima(
    calibration_truth: Sequence[Mapping[str, Any]],
    utility_scores: Mapping[str, float],
) -> tuple[float, ...]:
    """Return one worst-case safe non-improvement score per workflow family."""

    _require(len(calibration_truth) == 120, "utility calibration requires 120 families")
    maxima = []
    for family in calibration_truth:
        by_role = family.get("utility_task_id_by_role_and_incumbent")
        _require(isinstance(by_role, Mapping), "utility task mapping is absent")
        scores = [
            utility_scores[task_id]
            for role in ("E", "N")
            for task_id in by_role[role].values()
        ]
        _require(len(scores) == 4, "utility family does not have four negatives")
        maxima.append(max(scores))
    return tuple(sorted(maxima))


def threshold_for_false_update_budget(
    negative_scores: Sequence[float],
    budget: Fraction,
) -> tuple[float, Mapping[str, Any]]:
    """Choose the least conservative order statistic with rank risk at most budget.

    For ``m`` exchangeable calibration negatives and strict admission above the
    ``k``-th order statistic, the finite-sample false-update probability is at
    most ``(m + 1 - k) / (m + 1)``. Ties are rejected.
    """

    values = tuple(sorted(float(value) for value in negative_scores))
    _require(values and 0 < budget < 1, "invalid utility risk calibration")
    m = len(values)
    ranks = [
        k
        for k in range(1, m + 1)
        if Fraction(m + 1 - k, m + 1) <= budget
    ]
    _require(bool(ranks), "calibration is too small for the utility budget")
    rank = min(ranks)
    implied = Fraction(m + 1 - rank, m + 1)
    threshold = values[rank - 1]
    return threshold, {
        "negative_family_count": m,
        "requested_false_update_budget": {
            "numerator": budget.numerator,
            "denominator": budget.denominator,
        },
        "selected_order_statistic_rank_one_indexed": rank,
        "implied_finite_sample_bound": {
            "numerator": implied.numerator,
            "denominator": implied.denominator,
        },
        "threshold_hex": threshold.hex(),
        "admission_comparison": "strictly_greater_than",
        "ties": "rejected",
    }


__all__ = ["family_nonimprovement_maxima", "threshold_for_false_update_budget"]
