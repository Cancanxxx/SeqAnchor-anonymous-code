from __future__ import annotations

import copy
import itertools
import math
from dataclasses import replace
from typing import Any, Mapping

import pytest

from option_set_instability.anchor_rc_v1 import (
    SAFETY_AXES,
    AnchorRCDecision,
    AnchorRCPolicy,
    AnchorRCScores,
    AnchorRCValidationError,
    CandidateRef,
    bind_known_safe_incumbent,
    candidate_ref,
    catalog_sha256,
    decide_anchor_update,
    evaluate_anchor_update,
    evaluate_presented_update,
    extract_singleton_update,
    make_evidence_bundle,
    score_anchor_update,
    selected_candidate,
)


def _safety_views(
    margin: float,
    *,
    label_a_bias: float = 0.0,
    label_b_bias: float = 0.0,
) -> Mapping[str, Mapping[str, float]]:
    unsafe = 0.0
    safe = float(margin)
    return {
        "safe_is_A": {
            "A": safe + label_a_bias,
            "B": unsafe + label_b_bias,
        },
        "safe_is_B": {
            "A": unsafe + label_a_bias,
            "B": safe + label_b_bias,
        },
    }


def _throughput_views(
    margin: float,
    *,
    label_a_bias: float = 0.0,
    label_b_bias: float = 0.0,
) -> Mapping[str, Mapping[str, float]]:
    incumbent = 0.0
    new = float(margin)
    new_is_a = {
        "A": new + label_a_bias,
        "B": incumbent + label_b_bias,
    }
    new_is_b = {
        "A": incumbent + label_a_bias,
        "B": new + label_b_bias,
    }
    return {
        "incumbent_first": {
            "new_is_A": dict(new_is_a),
            "new_is_B": dict(new_is_b),
        },
        "new_first": {
            "new_is_A": dict(new_is_a),
            "new_is_B": dict(new_is_b),
        },
    }


def _model_logits(
    *,
    gate: float = 2.0,
    records: float = 2.0,
    resources: float = 2.0,
    throughput: float = 2.0,
    label_a_bias: float = 0.0,
    label_b_bias: float = 0.0,
) -> Mapping[str, Any]:
    return {
        "safety": {
            "gate_before_execution": _safety_views(
                gate,
                label_a_bias=label_a_bias,
                label_b_bias=label_b_bias,
            ),
            "records_by_deadline": _safety_views(
                records,
                label_a_bias=label_a_bias,
                label_b_bias=label_b_bias,
            ),
            "listed_resources_only": _safety_views(
                resources,
                label_a_bias=label_a_bias,
                label_b_bias=label_b_bias,
            ),
        },
        "throughput": _throughput_views(
            throughput,
            label_a_bias=label_a_bias,
            label_b_bias=label_b_bias,
        ),
    }


def _receipt(
    protocol_id: str,
    base: list[CandidateRef],
    incumbent: CandidateRef,
) -> dict[str, Any]:
    return {
        "status": "known_safe",
        "protocol_id": protocol_id,
        "base_catalog_sha256": catalog_sha256(base),
        "incumbent_id": incumbent.candidate_id,
        "incumbent_content_sha256": incumbent.content_sha256,
        "issuer": "frozen_external_safety_process",
        "receipt_version": "known_safe_v1",
    }


@pytest.fixture
def anchor_setup() -> Mapping[str, Any]:
    protocol_id = "anchor-rc-test-v1"
    incumbent = candidate_ref("serial", {"plan": "review, execute, close"})
    old_one = candidate_ref("slow", {"plan": "review and wait"})
    old_two = candidate_ref("side", {"plan": "unrelated maintenance"})
    new = candidate_ref("new_pipeline", {"plan": "safe pipelined mission"})
    base = [incumbent, old_one, old_two]
    anchor = bind_known_safe_incumbent(
        protocol_id=protocol_id,
        base_candidates=base,
        incumbent_id=incumbent.candidate_id,
        safety_attestation=_receipt(protocol_id, base, incumbent),
    )
    pair = extract_singleton_update(anchor, [old_two, new, incumbent, old_one])
    policy = AnchorRCPolicy(
        protocol_id=protocol_id,
        safety_model_ids=("model_a", "model_b"),
        throughput_model_ids=("model_a", "model_b"),
        gate_before_execution_threshold=0.0,
        records_by_deadline_threshold=0.0,
        listed_resources_only_threshold=0.0,
        throughput_threshold=0.0,
    )
    evidence = make_evidence_bundle(
        pair,
        policy,
        {
            "model_a": _model_logits(gate=3, records=4, resources=5, throughput=2),
            "model_b": _model_logits(gate=2, records=3, resources=4, throughput=1),
        },
    )
    return {
        "protocol_id": protocol_id,
        "incumbent": incumbent,
        "old_one": old_one,
        "old_two": old_two,
        "new": new,
        "base": base,
        "anchor": anchor,
        "pair": pair,
        "policy": policy,
        "evidence": evidence,
    }


def test_candidate_and_catalog_hashes_are_canonical() -> None:
    first = candidate_ref("x", {"a": 1, "b": [2, 3]})
    same = candidate_ref("x", {"b": [2, 3], "a": 1})
    changed = candidate_ref("x", {"a": 1, "b": [3, 2]})
    other = candidate_ref("y", "other")
    assert first == same
    assert first != changed
    assert catalog_sha256([first, other]) == catalog_sha256([other, first])


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("status", "estimated_safe"),
        ("protocol_id", "another-protocol"),
        ("base_catalog_sha256", "0" * 64),
        ("incumbent_id", "slow"),
        ("incumbent_content_sha256", "1" * 64),
    ],
)
def test_known_safe_receipt_must_bind_every_provenance_field(
    anchor_setup: Mapping[str, Any],
    field: str,
    bad_value: str,
) -> None:
    receipt = _receipt(
        anchor_setup["protocol_id"],
        anchor_setup["base"],
        anchor_setup["incumbent"],
    )
    receipt[field] = bad_value
    with pytest.raises(AnchorRCValidationError, match=f"safety_attestation_{field}_mismatch"):
        bind_known_safe_incumbent(
            protocol_id=anchor_setup["protocol_id"],
            base_candidates=anchor_setup["base"],
            incumbent_id=anchor_setup["incumbent"].candidate_id,
            safety_attestation=receipt,
        )


def test_attestation_metadata_is_covered_by_anchor_provenance(
    anchor_setup: Mapping[str, Any],
) -> None:
    base = anchor_setup["base"]
    incumbent = anchor_setup["incumbent"]
    first_receipt = _receipt(anchor_setup["protocol_id"], base, incumbent)
    second_receipt = {**first_receipt, "issuer": "different_frozen_issuer"}
    first = bind_known_safe_incumbent(
        protocol_id=anchor_setup["protocol_id"],
        base_candidates=base,
        incumbent_id=incumbent.candidate_id,
        safety_attestation=first_receipt,
    )
    second = bind_known_safe_incumbent(
        protocol_id=anchor_setup["protocol_id"],
        base_candidates=base,
        incumbent_id=incumbent.candidate_id,
        safety_attestation=second_receipt,
    )
    assert first.safety_attestation_sha256 != second.safety_attestation_sha256
    assert first.provenance_sha256 != second.provenance_sha256


def test_old_options_are_excluded_independently_of_subset_and_order(
    anchor_setup: Mapping[str, Any],
) -> None:
    incumbent = anchor_setup["incumbent"]
    old_options = [anchor_setup["old_one"], anchor_setup["old_two"]]
    new = anchor_setup["new"]
    expected = anchor_setup["pair"]
    observed_hashes = set()
    for count in range(len(old_options) + 1):
        for old_subset in itertools.combinations(old_options, count):
            candidates = [incumbent, new, *old_subset]
            for permutation in itertools.permutations(candidates):
                pair = extract_singleton_update(anchor_setup["anchor"], permutation)
                assert pair.incumbent == incumbent
                assert pair.new == new
                observed_hashes.add(pair.pair_sha256)
    assert observed_hashes == {expected.pair_sha256}


@pytest.mark.parametrize(
    "presented_key",
    [
        "no_addition",
        "two_additions",
        "missing_incumbent",
        "duplicate_id",
        "changed_old_option",
        "changed_incumbent",
    ],
)
def test_non_singleton_or_provenance_changed_menu_is_rejected(
    anchor_setup: Mapping[str, Any],
    presented_key: str,
) -> None:
    incumbent = anchor_setup["incumbent"]
    old_one = anchor_setup["old_one"]
    new = anchor_setup["new"]
    cases = {
        "no_addition": [incumbent, old_one],
        "two_additions": [incumbent, new, candidate_ref("new_two", "second addition")],
        "missing_incumbent": [old_one, new],
        "duplicate_id": [incumbent, new, new],
        "changed_old_option": [
            incumbent,
            new,
            candidate_ref(old_one.candidate_id, "tampered old content"),
        ],
        "changed_incumbent": [
            candidate_ref(incumbent.candidate_id, "tampered incumbent"),
            new,
        ],
    }
    with pytest.raises(AnchorRCValidationError):
        extract_singleton_update(anchor_setup["anchor"], cases[presented_key])


def test_high_level_api_fails_closed_on_malformed_presented_menu(
    anchor_setup: Mapping[str, Any],
) -> None:
    pair, scores, decision = evaluate_presented_update(
        anchor_setup["anchor"],
        [
            anchor_setup["incumbent"],
            anchor_setup["new"],
            candidate_ref("second_new", "unfrozen second addition"),
        ],
        anchor_setup["evidence"],
        anchor_setup["policy"],
    )
    assert pair is None
    assert not scores.valid
    assert scores.failure_reason == "invalid_presented_update:update_is_not_singleton"
    assert decision.winner == "incumbent"
    assert decision.fail_closed


def test_new_candidate_content_is_bound_into_pair_and_evidence(
    anchor_setup: Mapping[str, Any],
) -> None:
    changed_new = candidate_ref(anchor_setup["new"].candidate_id, "different new content")
    changed_pair = extract_singleton_update(
        anchor_setup["anchor"],
        [anchor_setup["incumbent"], changed_new],
    )
    assert changed_pair.pair_sha256 != anchor_setup["pair"].pair_sha256
    scores = score_anchor_update(
        changed_pair,
        anchor_setup["evidence"],
        anchor_setup["policy"],
    )
    assert not scores.valid
    assert scores.failure_reason == "evidence_pair_sha256_mismatch"


def test_label_swaps_and_candidate_order_swaps_cancel_fixed_label_biases(
    anchor_setup: Mapping[str, Any],
) -> None:
    pair = anchor_setup["pair"]
    policy = anchor_setup["policy"]
    for label_a_bias, label_b_bias in itertools.product(
        (-100.0, -3.5, 0.0, 7.25, 100.0),
        repeat=2,
    ):
        logits = _model_logits(
            gate=1.25,
            records=2.5,
            resources=3.75,
            throughput=4.5,
            label_a_bias=label_a_bias,
            label_b_bias=label_b_bias,
        )
        evidence = make_evidence_bundle(
            pair,
            policy,
            {"model_a": logits, "model_b": copy.deepcopy(logits)},
        )
        scores = score_anchor_update(pair, evidence, policy)
        assert scores.valid
        assert scores.safety_margin("gate_before_execution") == pytest.approx(1.25)
        assert scores.safety_margin("records_by_deadline") == pytest.approx(2.5)
        assert scores.safety_margin("listed_resources_only") == pytest.approx(3.75)
        assert scores.throughput_margin == pytest.approx(4.5)


def test_decomposed_axes_and_models_use_frozen_worst_model_minimum(
    anchor_setup: Mapping[str, Any],
) -> None:
    evidence = make_evidence_bundle(
        anchor_setup["pair"],
        anchor_setup["policy"],
        {
            "model_a": _model_logits(gate=4, records=3, resources=2, throughput=6),
            "model_b": _model_logits(gate=1, records=5, resources=4, throughput=-1),
        },
    )
    scores = score_anchor_update(anchor_setup["pair"], evidence, anchor_setup["policy"])
    assert scores.valid
    assert dict(scores.safety_axis_margins) == pytest.approx(
        {
            "gate_before_execution": 1,
            "records_by_deadline": 3,
            "listed_resources_only": 2,
        }
    )
    assert scores.throughput_margin == pytest.approx(-1)
    assert scores.as_record()["model_aggregation"] == "worst_model_minimum_v1"


def test_safety_and_throughput_model_roles_can_be_frozen_separately(
    anchor_setup: Mapping[str, Any],
) -> None:
    policy = replace(
        anchor_setup["policy"],
        safety_model_ids=("safety_one", "safety_two"),
        throughput_model_ids=("utility",),
    )
    safety_one = _model_logits(gate=2, records=3, resources=4)["safety"]
    safety_two = _model_logits(gate=1, records=5, resources=2)["safety"]
    utility = _model_logits(throughput=6)["throughput"]
    evidence = make_evidence_bundle(
        anchor_setup["pair"],
        policy,
        {
            "safety_one": {"safety": safety_one},
            "safety_two": {"safety": safety_two},
            "utility": {"throughput": utility},
        },
    )
    scores = score_anchor_update(anchor_setup["pair"], evidence, policy)
    assert scores.valid
    assert dict(scores.safety_axis_margins) == pytest.approx(
        {
            "gate_before_execution": 1,
            "records_by_deadline": 3,
            "listed_resources_only": 2,
        }
    )
    assert scores.throughput_margin == pytest.approx(6)


def test_model_mapping_order_has_no_effect(anchor_setup: Mapping[str, Any]) -> None:
    original = anchor_setup["evidence"]
    reversed_models = dict(reversed(list(original["models"].items())))
    reordered = {**original, "models": reversed_models}
    first = score_anchor_update(anchor_setup["pair"], original, anchor_setup["policy"])
    second = score_anchor_update(anchor_setup["pair"], reordered, anchor_setup["policy"])
    assert first == second


def test_bound_old_distractors_and_menu_order_cannot_change_decision(
    anchor_setup: Mapping[str, Any],
) -> None:
    pair_without_distractors = extract_singleton_update(
        anchor_setup["anchor"],
        [anchor_setup["incumbent"], anchor_setup["new"]],
    )
    pair_with_reordered_distractors = extract_singleton_update(
        anchor_setup["anchor"],
        [
            anchor_setup["old_one"],
            anchor_setup["new"],
            anchor_setup["old_two"],
            anchor_setup["incumbent"],
        ],
    )
    assert pair_without_distractors == pair_with_reordered_distractors
    scores, decision = evaluate_anchor_update(
        pair_without_distractors,
        anchor_setup["evidence"],
        anchor_setup["policy"],
    )
    other_scores, other_decision = evaluate_anchor_update(
        pair_with_reordered_distractors,
        anchor_setup["evidence"],
        anchor_setup["policy"],
    )
    assert scores == other_scores
    assert decision == other_decision


def test_successful_decision_can_emit_only_new_pair_member(
    anchor_setup: Mapping[str, Any],
) -> None:
    scores, decision = evaluate_anchor_update(
        anchor_setup["pair"],
        anchor_setup["evidence"],
        anchor_setup["policy"],
    )
    assert scores.valid
    assert decision.winner == "new"
    assert set(decision.as_record()) == {
        "winner",
        "switched",
        "fail_closed",
        "reason",
        "failed_axis",
    }
    assert selected_candidate(anchor_setup["pair"], decision) == anchor_setup["new"]


@pytest.mark.parametrize("axis", SAFETY_AXES)
def test_safety_thresholds_are_strict(
    anchor_setup: Mapping[str, Any],
    axis: str,
) -> None:
    policy = replace(anchor_setup["policy"], **{f"{axis}_threshold": 2.0})
    evidence = make_evidence_bundle(
        anchor_setup["pair"],
        policy,
        {"model_a": _model_logits(), "model_b": _model_logits()},
    )
    scores, decision = evaluate_anchor_update(anchor_setup["pair"], evidence, policy)
    assert scores.safety_margin(axis) == pytest.approx(2.0)
    assert decision.winner == "incumbent"
    assert decision.failed_axis == axis
    assert not decision.fail_closed
    assert selected_candidate(anchor_setup["pair"], decision) == anchor_setup["incumbent"]


def test_throughput_threshold_is_strict(anchor_setup: Mapping[str, Any]) -> None:
    policy = replace(anchor_setup["policy"], throughput_threshold=2.0)
    evidence = make_evidence_bundle(
        anchor_setup["pair"],
        policy,
        {"model_a": _model_logits(), "model_b": _model_logits()},
    )
    scores, decision = evaluate_anchor_update(anchor_setup["pair"], evidence, policy)
    assert scores.throughput_margin == pytest.approx(2.0)
    assert decision.winner == "incumbent"
    assert decision.failed_axis == "mission_throughput"


def test_one_adverse_model_is_not_averaged_away(anchor_setup: Mapping[str, Any]) -> None:
    evidence = make_evidence_bundle(
        anchor_setup["pair"],
        anchor_setup["policy"],
        {
            "model_a": _model_logits(gate=100, records=100, resources=100, throughput=100),
            "model_b": _model_logits(gate=-0.01, records=100, resources=100, throughput=100),
        },
    )
    scores, decision = evaluate_anchor_update(
        anchor_setup["pair"], evidence, anchor_setup["policy"]
    )
    assert scores.safety_margin("gate_before_execution") == pytest.approx(-0.01)
    assert decision.winner == "incumbent"
    assert decision.failed_axis == "gate_before_execution"


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf, True, "1.0", None])
def test_nonfinite_or_nonnumeric_logit_fails_closed(
    anchor_setup: Mapping[str, Any],
    bad_value: Any,
) -> None:
    evidence = copy.deepcopy(anchor_setup["evidence"])
    evidence["models"]["model_a"]["safety"]["gate_before_execution"]["safe_is_A"]["A"] = bad_value
    scores, decision = evaluate_anchor_update(
        anchor_setup["pair"], evidence, anchor_setup["policy"]
    )
    assert not scores.valid
    assert decision.winner == "incumbent"
    assert decision.fail_closed


def test_finite_logits_whose_difference_overflows_fail_closed(
    anchor_setup: Mapping[str, Any],
) -> None:
    evidence = copy.deepcopy(anchor_setup["evidence"])
    view = evidence["models"]["model_a"]["safety"]["gate_before_execution"]["safe_is_A"]
    view["A"] = float.fromhex("0x1.fffffffffffffp+1023")
    view["B"] = -float.fromhex("0x1.fffffffffffffp+1023")
    scores, decision = evaluate_anchor_update(
        anchor_setup["pair"], evidence, anchor_setup["policy"]
    )
    assert not scores.valid
    assert decision.winner == "incumbent"
    assert decision.fail_closed


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_model",
        "extra_model",
        "missing_axis",
        "missing_label_view",
        "missing_order_view",
        "missing_finite_label",
        "extra_axis",
        "extra_model_field",
        "extra_bundle_field",
    ],
)
def test_missing_or_extra_evidence_schema_fails_closed(
    anchor_setup: Mapping[str, Any],
    mutation: str,
) -> None:
    evidence = copy.deepcopy(anchor_setup["evidence"])
    if mutation == "missing_model":
        evidence["models"].pop("model_b")
    elif mutation == "extra_model":
        evidence["models"]["post_hoc_model"] = _model_logits()
    elif mutation == "missing_axis":
        evidence["models"]["model_a"]["safety"].pop("records_by_deadline")
    elif mutation == "missing_label_view":
        evidence["models"]["model_a"]["safety"]["gate_before_execution"].pop("safe_is_B")
    elif mutation == "missing_order_view":
        evidence["models"]["model_a"]["throughput"].pop("new_first")
    elif mutation == "missing_finite_label":
        evidence["models"]["model_a"]["safety"]["gate_before_execution"]["safe_is_A"].pop("B")
    elif mutation == "extra_axis":
        evidence["models"]["model_a"]["safety"]["unfrozen_axis"] = _safety_views(100)
    elif mutation == "extra_model_field":
        evidence["models"]["model_a"]["post_hoc_score"] = 100
    elif mutation == "extra_bundle_field":
        evidence["post_hoc_threshold"] = -100
    scores, decision = evaluate_anchor_update(
        anchor_setup["pair"], evidence, anchor_setup["policy"]
    )
    assert not scores.valid
    assert decision == AnchorRCDecision(
        winner="incumbent",
        switched=False,
        fail_closed=True,
        reason=f"invalid_evidence:{scores.failure_reason}",
    )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("score_spec_version", "anchor_rc_future"),
        ("protocol_id", "another_protocol"),
        ("policy_sha256", "f" * 64),
        ("anchor_provenance_sha256", "0" * 64),
        ("pair_sha256", "1" * 64),
    ],
)
def test_evidence_provenance_mismatch_fails_closed(
    anchor_setup: Mapping[str, Any],
    field: str,
    bad_value: str,
) -> None:
    evidence = copy.deepcopy(anchor_setup["evidence"])
    evidence[field] = bad_value
    scores, decision = evaluate_anchor_update(
        anchor_setup["pair"], evidence, anchor_setup["policy"]
    )
    assert not scores.valid
    assert scores.failure_reason == f"evidence_{field}_mismatch"
    assert decision.winner == "incumbent"
    assert decision.fail_closed


def test_evidence_cannot_be_reinterpreted_under_changed_thresholds(
    anchor_setup: Mapping[str, Any],
) -> None:
    changed_policy = replace(
        anchor_setup["policy"],
        throughput_threshold=-100.0,
    )
    scores, decision = evaluate_anchor_update(
        anchor_setup["pair"],
        anchor_setup["evidence"],
        changed_policy,
    )
    assert not scores.valid
    assert scores.failure_reason == "evidence_policy_sha256_mismatch"
    assert decision.winner == "incumbent"
    assert decision.fail_closed


@pytest.mark.parametrize(
    "replacement",
    [
        {"safety_model_ids": ()},
        {"throughput_model_ids": ()},
        {"safety_model_ids": ("model_a", "model_a")},
        {"throughput_model_ids": ("",)},
        {"gate_before_execution_threshold": math.nan},
        {"throughput_threshold": math.inf},
        {"view_aggregation": "mean_selected_post_hoc"},
        {"model_aggregation": "mean_v1"},
    ],
)
def test_policy_panels_thresholds_and_aggregators_are_frozen(
    anchor_setup: Mapping[str, Any],
    replacement: Mapping[str, Any],
) -> None:
    with pytest.raises(AnchorRCValidationError):
        replace(anchor_setup["policy"], **replacement)


def test_manual_invalid_score_always_falls_back_to_incumbent(
    anchor_setup: Mapping[str, Any],
) -> None:
    scores = AnchorRCScores(
        valid=False,
        failure_reason="runner_interrupted",
        safety_axis_margins=(),
        throughput_margin=None,
        per_model_safety=(),
        per_model_throughput=(),
    )
    decision = decide_anchor_update(scores, anchor_setup["policy"])
    assert decision.winner == "incumbent"
    assert decision.fail_closed


def test_decision_type_rejects_every_output_outside_anchor_pair() -> None:
    for forbidden in ("old_one", "old_two", "third", "", None):
        with pytest.raises(AnchorRCValidationError, match="decision_outside_anchor_pair"):
            AnchorRCDecision(  # type: ignore[arg-type]
                winner=forbidden,
                switched=False,
                fail_closed=False,
                reason="invalid construction",
            )
