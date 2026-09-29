from __future__ import annotations

from fractions import Fraction

import pytest

from option_set_instability.anchor_rc_v1 import (
    bind_known_safe_incumbent,
    candidate_ref,
    catalog_sha256,
)
from option_set_instability.sequential_anchorrc_v1 import (
    AxisNegativeCalibration,
    SequenceRiskPolicy,
    SequentialAnchorRCValidationError,
    evaluate_update,
    familywise_risk_bound,
    make_update_evidence,
    open_sequence,
    validate_calibration_panel,
)


def _setup():
    policy = SequenceRiskPolicy(
        protocol_id="seq-anchor-test-v1",
        horizon=8,
        alpha_numerator=1,
        alpha_denominator=20,
    )
    incumbent = candidate_ref("safe_anchor", {"plan": "verified serial work"})
    old_unsafe = candidate_ref("old_unsafe", {"plan": "execute before approval"})
    old_irrelevant = candidate_ref("old_irrelevant", {"plan": "unrelated work"})
    base = [incumbent, old_unsafe, old_irrelevant]
    receipt = {
        "status": "known_safe",
        "protocol_id": policy.protocol_id,
        "base_catalog_sha256": catalog_sha256(base),
        "incumbent_id": incumbent.candidate_id,
        "incumbent_content_sha256": incumbent.content_sha256,
        "issuer": "test_fixture",
    }
    anchor = bind_known_safe_incumbent(
        protocol_id=policy.protocol_id,
        base_candidates=base,
        incumbent_id=incumbent.candidate_id,
        safety_attestation=receipt,
    )
    scorer = "a" * 64
    calibrations = tuple(
        AxisNegativeCalibration(
            axis_id=axis,
            negative_scores=tuple(float(value) for value in range(180)),
            scorer_sha256=scorer,
            calibration_artifact_sha256=character * 64,
        )
        for axis, character in (("authorization", "b"), ("operations", "c"), ("welfare", "d"))
    )
    return policy, anchor, incumbent, old_unsafe, old_irrelevant, calibrations


def _evidence(state, candidate, policy, calibrations, *, score=1000.0, utility=1.0):
    return make_update_evidence(
        state=state,
        candidate=candidate,
        policy=policy,
        calibrations=calibrations,
        safety_scores={calibration.axis_id: score for calibration in calibrations},
        utility_margin=utility,
        utility_threshold=0.0,
    )


def test_exact_equal_spending_and_calibration_resolution() -> None:
    policy, _, _, _, _, calibrations = _setup()
    panel = validate_calibration_panel(calibrations, policy)
    assert policy.per_step_alpha == Fraction(1, 160)
    assert panel[0].minimum_p_value == Fraction(1, 181)
    assert familywise_risk_bound(policy) == Fraction(1, 20)
    assert familywise_risk_bound(policy, evaluated_steps=3) == Fraction(3, 160)


def test_calibration_ties_count_against_admission() -> None:
    _, _, _, _, _, calibrations = _setup()
    calibration = calibrations[0]
    assert calibration.p_value(1000.0) == Fraction(1, 181)
    assert calibration.p_value(179.0) == Fraction(2, 181)
    assert calibration.p_value(-1.0) == Fraction(181, 181)


def test_initial_old_options_are_excluded_without_scoring() -> None:
    policy, anchor, incumbent, old_unsafe, old_irrelevant, calibrations = _setup()
    state = open_sequence(anchor, policy)
    assert set(state.excluded_content_sha256) == {
        old_unsafe.content_sha256,
        old_irrelevant.content_sha256,
    }
    renamed_old_unsafe = candidate_ref("renamed_attack", {"plan": "execute before approval"})
    next_state, decision = evaluate_update(
        state=state,
        candidate=renamed_old_unsafe,
        evidence=None,
        policy=policy,
        calibrations=calibrations,
    )
    assert decision.selected == incumbent
    assert decision.reason == "content_blocked_by_exclusion_ledger"
    assert decision.p_values == ()
    assert next_state.incumbent == incumbent


def test_safe_improvement_becomes_anchor_and_replaced_anchor_is_retired() -> None:
    policy, anchor, incumbent, _, _, calibrations = _setup()
    state = open_sequence(anchor, policy)
    improvement = candidate_ref("safe_pipeline", {"plan": "verified pipelined work"})
    evidence = _evidence(state, improvement, policy, calibrations)
    next_state, decision = evaluate_update(
        state=state,
        candidate=improvement,
        evidence=evidence,
        policy=policy,
        calibrations=calibrations,
    )
    assert decision.accepted
    assert decision.selected == improvement
    assert next_state.incumbent == improvement
    assert incumbent.content_sha256 in next_state.excluded_content_sha256
    assert improvement.content_sha256 not in next_state.excluded_content_sha256
    rollback = candidate_ref("renamed_rollback", {"plan": "verified serial work"})
    final_state, rollback_decision = evaluate_update(
        state=next_state,
        candidate=rollback,
        evidence=None,
        policy=policy,
        calibrations=calibrations,
    )
    assert not rollback_decision.accepted
    assert rollback_decision.reason == "content_blocked_by_exclusion_ledger"
    assert final_state.incumbent == improvement


def test_unsafe_or_nonimproving_candidate_is_rejected_and_ledgered() -> None:
    policy, anchor, incumbent, _, _, calibrations = _setup()
    state = open_sequence(anchor, policy)
    unsafe = candidate_ref("unsafe_new", {"plan": "late records"})
    unsafe_evidence = _evidence(state, unsafe, policy, calibrations, score=100.0)
    state, decision = evaluate_update(
        state=state,
        candidate=unsafe,
        evidence=unsafe_evidence,
        policy=policy,
        calibrations=calibrations,
    )
    assert not decision.accepted
    assert decision.reason.startswith("unsafe_null_not_rejected")
    assert state.incumbent == incumbent
    assert unsafe.content_sha256 in state.excluded_content_sha256

    equivalent = candidate_ref("equivalent", {"plan": "another safe serial order"})
    equivalent_evidence = _evidence(
        state,
        equivalent,
        policy,
        calibrations,
        score=1000.0,
        utility=0.0,
    )
    state, decision = evaluate_update(
        state=state,
        candidate=equivalent,
        evidence=equivalent_evidence,
        policy=policy,
        calibrations=calibrations,
    )
    assert not decision.accepted
    assert decision.reason == "utility_margin_not_strictly_above_threshold"
    assert equivalent.content_sha256 in state.excluded_content_sha256


def test_evidence_is_bound_to_exact_state_and_candidate() -> None:
    policy, anchor, incumbent, _, _, calibrations = _setup()
    state = open_sequence(anchor, policy)
    first = candidate_ref("first", {"plan": "safe improvement"})
    stale = _evidence(state, first, policy, calibrations)
    other = candidate_ref("other", {"plan": "different improvement"})
    next_state, decision = evaluate_update(
        state=state,
        candidate=other,
        evidence=stale,
        policy=policy,
        calibrations=calibrations,
    )
    assert not decision.accepted
    assert decision.fail_closed
    assert decision.reason == "invalid_or_unbound_evidence"
    assert next_state.incumbent == incumbent


def test_identifier_reuse_and_horizon_overflow_fail_closed() -> None:
    policy, anchor, incumbent, _, _, calibrations = _setup()
    state = open_sequence(anchor, policy)
    collision = candidate_ref(incumbent.candidate_id, {"plan": "changed content"})
    state, decision = evaluate_update(
        state=state,
        candidate=collision,
        evidence=None,
        policy=policy,
        calibrations=calibrations,
    )
    assert decision.fail_closed
    assert decision.reason == "candidate_id_already_seen"
    for index in range(1, policy.horizon):
        candidate = candidate_ref(f"candidate_{index}", {"plan": f"blocked {index}"})
        state, _ = evaluate_update(
            state=state,
            candidate=candidate,
            evidence=None,
            policy=policy,
            calibrations=calibrations,
        )
    assert state.step == policy.horizon
    with pytest.raises(SequentialAnchorRCValidationError, match="sequence_horizon_exhausted"):
        evaluate_update(
            state=state,
            candidate=candidate_ref("overflow", {"plan": "overflow"}),
            evidence=None,
            policy=policy,
            calibrations=calibrations,
        )


def test_too_small_calibration_cannot_claim_requested_budget() -> None:
    policy, _, _, _, _, _ = _setup()
    tiny = AxisNegativeCalibration(
        axis_id="welfare",
        negative_scores=tuple(float(value) for value in range(10)),
        scorer_sha256="a" * 64,
        calibration_artifact_sha256="b" * 64,
    )
    with pytest.raises(
        SequentialAnchorRCValidationError,
        match="calibration_too_small_for_risk_budget",
    ):
        validate_calibration_panel((tiny,), policy)
