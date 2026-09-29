"""Stateful risk-limiting catalog updates for Sequential AnchorRC v1.

The module contains no model inference.  A runner supplies fixed safety scores
and a utility margin bound to the current state and singleton candidate.  The
controller computes exact rank p-values against fresh negative calibration
scores, applies a precommitted sequence risk budget, and records a hash-chained
transition.  Its output is always either the incumbent or the new candidate.

The statistical guarantee is deliberately scoped.  Rank p-values are valid
when a violating test candidate is exchangeable with the corresponding
requirement-specific calibration negatives.  Equal alpha spending and a union
bound then control the probability of any unsafe admission in a precommitted
sequence.  The code verifies the accounting and structural invariants; it does
not manufacture exchangeability from provenance metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real
from typing import Any, Mapping, Optional, Sequence

from nontransitive_safety.provenance import canonical_json_bytes, sha256_bytes
from option_set_instability.anchor_rc_v1 import CandidateRef, KnownSafeAnchor

SCHEMA_VERSION = "1.0.0"
PROTOCOL_FAMILY = "sequential_anchorrc_rank_fwer_v1"


class SequentialAnchorRCValidationError(ValueError):
    """Malformed state, calibration, or evidence."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise SequentialAnchorRCValidationError(code)


def _nonempty(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(code)
    return value


def _sha256(value: Any, code: str) -> str:
    text = _nonempty(value, code)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        _fail(code)
    return text


def _finite(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _fail(code)
    number = float(value)
    if not math.isfinite(number):
        _fail(code)
    return number


def _positive_integer(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(code)
    return value


def _canonical_fraction(value: Fraction) -> Mapping[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


@dataclass(frozen=True)
class SequenceRiskPolicy:
    """A fixed-horizon family-wise unsafe-admission budget."""

    protocol_id: str
    horizon: int
    alpha_numerator: int
    alpha_denominator: int
    allocation: str = "equal_bonferroni"
    protocol_family: str = PROTOCOL_FAMILY

    def __post_init__(self) -> None:
        _nonempty(self.protocol_id, "invalid_protocol_id")
        _positive_integer(self.horizon, "invalid_horizon")
        if (
            isinstance(self.alpha_numerator, bool)
            or not isinstance(self.alpha_numerator, int)
            or self.alpha_numerator <= 0
        ):
            _fail("invalid_alpha_numerator")
        _positive_integer(self.alpha_denominator, "invalid_alpha_denominator")
        if self.alpha_numerator >= self.alpha_denominator:
            _fail("alpha_not_in_open_unit_interval")
        if self.allocation != "equal_bonferroni":
            _fail("unsupported_risk_allocation")
        if self.protocol_family != PROTOCOL_FAMILY:
            _fail("unknown_protocol_family")

    @property
    def total_alpha(self) -> Fraction:
        return Fraction(self.alpha_numerator, self.alpha_denominator)

    @property
    def per_step_alpha(self) -> Fraction:
        return self.total_alpha / self.horizon

    def as_record(self) -> Mapping[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_family": self.protocol_family,
            "protocol_id": self.protocol_id,
            "horizon": self.horizon,
            "total_alpha": _canonical_fraction(self.total_alpha),
            "allocation": self.allocation,
            "per_step_alpha": _canonical_fraction(self.per_step_alpha),
        }

    @property
    def policy_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.as_record()))


@dataclass(frozen=True)
class AxisNegativeCalibration:
    """Fresh scores from candidates known to violate one safety requirement."""

    axis_id: str
    negative_scores: tuple[float, ...]
    scorer_sha256: str
    calibration_artifact_sha256: str

    def __post_init__(self) -> None:
        _nonempty(self.axis_id, "invalid_axis_id")
        _sha256(self.scorer_sha256, "invalid_scorer_sha256")
        _sha256(self.calibration_artifact_sha256, "invalid_calibration_artifact_sha256")
        values = tuple(
            _finite(value, "nonfinite_calibration_score")
            for value in self.negative_scores
        )
        if not values:
            _fail("empty_negative_calibration")
        if values != tuple(sorted(values)):
            _fail("negative_calibration_not_sorted")
        object.__setattr__(self, "negative_scores", values)

    @property
    def count(self) -> int:
        return len(self.negative_scores)

    @property
    def minimum_p_value(self) -> Fraction:
        return Fraction(1, self.count + 1)

    def p_value(self, candidate_score: Real) -> Fraction:
        """Conservative upper-tail rank p-value for the unsafe null.

        Larger scores mean stronger evidence that the requirement is met.
        Calibration ties count against admission.
        """

        score = _finite(candidate_score, "nonfinite_candidate_score")
        at_least = sum(value >= score for value in self.negative_scores)
        return Fraction(1 + at_least, self.count + 1)

    def as_record(self) -> Mapping[str, Any]:
        return {
            "axis_id": self.axis_id,
            "negative_count": self.count,
            "negative_scores_hex": [value.hex() for value in self.negative_scores],
            "minimum_p_value": _canonical_fraction(self.minimum_p_value),
            "scorer_sha256": self.scorer_sha256,
            "calibration_artifact_sha256": self.calibration_artifact_sha256,
            "tie_rule": "greater_than_or_equal_counts_against_admission",
        }

    @property
    def calibration_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.as_record()))


def validate_calibration_panel(
    calibrations: Sequence[AxisNegativeCalibration],
    policy: SequenceRiskPolicy,
    *,
    require_attainable_rejection: bool = True,
) -> tuple[AxisNegativeCalibration, ...]:
    """Canonicalize a calibration panel and verify its scorer and risk fit."""

    values = tuple(calibrations)
    if not values or any(not isinstance(value, AxisNegativeCalibration) for value in values):
        _fail("invalid_calibration_panel")
    axis_ids = [value.axis_id for value in values]
    if len(axis_ids) != len(set(axis_ids)):
        _fail("duplicate_calibration_axis")
    if len({value.scorer_sha256 for value in values}) != 1:
        _fail("calibration_scorer_mismatch")
    ordered = tuple(sorted(values, key=lambda value: value.axis_id))
    if require_attainable_rejection and any(
        value.minimum_p_value > policy.per_step_alpha for value in ordered
    ):
        _fail("calibration_too_small_for_risk_budget")
    return ordered


@dataclass(frozen=True)
class SequentialAnchorState:
    """Minimal state required to process the next singleton catalog update."""

    protocol_id: str
    risk_policy_sha256: str
    step: int
    horizon: int
    incumbent: CandidateRef
    excluded_content_sha256: tuple[str, ...]
    seen_candidate_ids: tuple[str, ...]
    transition_chain_sha256: str
    initial_anchor_provenance_sha256: str

    def __post_init__(self) -> None:
        _nonempty(self.protocol_id, "invalid_state_protocol_id")
        _sha256(self.risk_policy_sha256, "invalid_state_policy_sha256")
        _positive_integer(self.horizon, "invalid_state_horizon")
        if (
            isinstance(self.step, bool)
            or not isinstance(self.step, int)
            or not 0 <= self.step <= self.horizon
        ):
            _fail("invalid_state_step")
        if not isinstance(self.incumbent, CandidateRef):
            _fail("invalid_state_incumbent")
        excluded = tuple(self.excluded_content_sha256)
        if excluded != tuple(sorted(set(excluded))):
            _fail("noncanonical_exclusion_ledger")
        for value in excluded:
            _sha256(value, "invalid_excluded_content_sha256")
        if self.incumbent.content_sha256 in excluded:
            _fail("incumbent_in_exclusion_ledger")
        seen = tuple(self.seen_candidate_ids)
        if seen != tuple(sorted(set(seen))) or not seen:
            _fail("noncanonical_seen_candidate_ids")
        if self.incumbent.candidate_id not in seen:
            _fail("incumbent_id_not_seen")
        _sha256(self.transition_chain_sha256, "invalid_transition_chain_sha256")
        _sha256(self.initial_anchor_provenance_sha256, "invalid_initial_anchor_sha256")

    def as_record(self) -> Mapping[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": self.protocol_id,
            "risk_policy_sha256": self.risk_policy_sha256,
            "step": self.step,
            "horizon": self.horizon,
            "incumbent": self.incumbent.as_record(),
            "excluded_content_sha256": list(self.excluded_content_sha256),
            "seen_candidate_ids": list(self.seen_candidate_ids),
            "transition_chain_sha256": self.transition_chain_sha256,
            "initial_anchor_provenance_sha256": self.initial_anchor_provenance_sha256,
        }

    @property
    def state_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.as_record()))


def open_sequence(anchor: KnownSafeAnchor, policy: SequenceRiskPolicy) -> SequentialAnchorState:
    """Bind a one-step known-safe anchor to a new sequential controller state."""

    if not isinstance(anchor, KnownSafeAnchor):
        _fail("invalid_initial_anchor")
    if anchor.protocol_id != policy.protocol_id:
        _fail("initial_anchor_policy_protocol_mismatch")
    excluded = tuple(
        sorted(
            candidate.content_sha256
            for candidate in anchor.base_catalog
            if candidate != anchor.incumbent
        )
    )
    seed = {
        "event": "open_sequence",
        "protocol_id": policy.protocol_id,
        "policy_sha256": policy.policy_sha256,
        "anchor_provenance_sha256": anchor.provenance_sha256,
        "incumbent": anchor.incumbent.as_record(),
        "excluded_content_sha256": list(excluded),
    }
    return SequentialAnchorState(
        protocol_id=policy.protocol_id,
        risk_policy_sha256=policy.policy_sha256,
        step=0,
        horizon=policy.horizon,
        incumbent=anchor.incumbent,
        excluded_content_sha256=excluded,
        seen_candidate_ids=tuple(
            sorted(candidate.candidate_id for candidate in anchor.base_catalog)
        ),
        transition_chain_sha256=sha256_bytes(canonical_json_bytes(seed)),
        initial_anchor_provenance_sha256=anchor.provenance_sha256,
    )


@dataclass(frozen=True)
class SequentialUpdateEvidence:
    """Runner evidence bound to one state and one singleton candidate."""

    protocol_id: str
    risk_policy_sha256: str
    state_sha256: str
    candidate: CandidateRef
    scorer_sha256: str
    calibration_panel_sha256: str
    safety_scores: tuple[tuple[str, float], ...]
    utility_margin: float
    utility_threshold: float

    def __post_init__(self) -> None:
        _nonempty(self.protocol_id, "invalid_evidence_protocol_id")
        _sha256(self.risk_policy_sha256, "invalid_evidence_policy_sha256")
        _sha256(self.state_sha256, "invalid_evidence_state_sha256")
        if not isinstance(self.candidate, CandidateRef):
            _fail("invalid_evidence_candidate")
        _sha256(self.scorer_sha256, "invalid_evidence_scorer_sha256")
        _sha256(self.calibration_panel_sha256, "invalid_evidence_calibration_sha256")
        values = tuple(
            (str(axis), _finite(score, "nonfinite_safety_score"))
            for axis, score in self.safety_scores
        )
        if (
            not values
            or values != tuple(sorted(values))
            or len({axis for axis, _ in values}) != len(values)
        ):
            _fail("noncanonical_safety_scores")
        object.__setattr__(self, "safety_scores", values)
        _finite(self.utility_margin, "nonfinite_utility_margin")
        _finite(self.utility_threshold, "nonfinite_utility_threshold")

    @property
    def utility_improves(self) -> bool:
        return self.utility_margin > self.utility_threshold

    def as_record(self) -> Mapping[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": self.protocol_id,
            "risk_policy_sha256": self.risk_policy_sha256,
            "state_sha256": self.state_sha256,
            "candidate": self.candidate.as_record(),
            "scorer_sha256": self.scorer_sha256,
            "calibration_panel_sha256": self.calibration_panel_sha256,
            "safety_scores_hex": {axis: score.hex() for axis, score in self.safety_scores},
            "utility_margin_hex": self.utility_margin.hex(),
            "utility_threshold_hex": self.utility_threshold.hex(),
            "utility_comparison": "strictly_greater_than",
        }

    @property
    def evidence_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.as_record()))


def calibration_panel_sha256(calibrations: Sequence[AxisNegativeCalibration]) -> str:
    values = tuple(calibrations)
    if not values:
        _fail("empty_calibration_panel")
    return sha256_bytes(canonical_json_bytes([value.as_record() for value in values]))


def make_update_evidence(
    *,
    state: SequentialAnchorState,
    candidate: CandidateRef,
    policy: SequenceRiskPolicy,
    calibrations: Sequence[AxisNegativeCalibration],
    safety_scores: Mapping[str, Real],
    utility_margin: Real,
    utility_threshold: Real,
) -> SequentialUpdateEvidence:
    """Create a provenance-bound evidence bundle from runner-produced scores."""

    panel = validate_calibration_panel(calibrations, policy)
    if state.protocol_id != policy.protocol_id or state.risk_policy_sha256 != policy.policy_sha256:
        _fail("state_policy_mismatch")
    if set(safety_scores) != {value.axis_id for value in panel}:
        _fail("safety_score_axes_mismatch")
    return SequentialUpdateEvidence(
        protocol_id=policy.protocol_id,
        risk_policy_sha256=policy.policy_sha256,
        state_sha256=state.state_sha256,
        candidate=candidate,
        scorer_sha256=panel[0].scorer_sha256,
        calibration_panel_sha256=calibration_panel_sha256(panel),
        safety_scores=tuple(sorted((axis, float(score)) for axis, score in safety_scores.items())),
        utility_margin=float(utility_margin),
        utility_threshold=float(utility_threshold),
    )


@dataclass(frozen=True)
class SequentialAnchorDecision:
    """One state transition and its exact risk-test evidence."""

    winner: str
    selected: CandidateRef
    accepted: bool
    fail_closed: bool
    reason: str
    p_values: tuple[tuple[str, Fraction], ...]
    allocated_alpha: Optional[Fraction]
    evidence_sha256: Optional[str]

    def __post_init__(self) -> None:
        if self.winner not in {"incumbent", "new"}:
            _fail("decision_outside_pair")
        if self.accepted != (self.winner == "new"):
            _fail("incoherent_acceptance")
        if not isinstance(self.selected, CandidateRef):
            _fail("invalid_selected_candidate")
        _nonempty(self.reason, "missing_decision_reason")
        axes = [axis for axis, value in self.p_values]
        if axes != sorted(set(axes)) or any(
            not isinstance(value, Fraction) for _, value in self.p_values
        ):
            _fail("noncanonical_decision_p_values")
        if self.evidence_sha256 is not None:
            _sha256(self.evidence_sha256, "invalid_decision_evidence_sha256")

    def as_record(self) -> Mapping[str, Any]:
        return {
            "winner": self.winner,
            "selected": self.selected.as_record(),
            "accepted": self.accepted,
            "fail_closed": self.fail_closed,
            "reason": self.reason,
            "p_values": {axis: _canonical_fraction(value) for axis, value in self.p_values},
            "allocated_alpha": (
                None if self.allocated_alpha is None else _canonical_fraction(self.allocated_alpha)
            ),
            "evidence_sha256": self.evidence_sha256,
        }


def _decision(
    *,
    incumbent: CandidateRef,
    candidate: CandidateRef,
    accepted: bool,
    fail_closed: bool,
    reason: str,
    p_values: tuple[tuple[str, Fraction], ...] = (),
    allocated_alpha: Optional[Fraction] = None,
    evidence_sha256: Optional[str] = None,
) -> SequentialAnchorDecision:
    return SequentialAnchorDecision(
        winner="new" if accepted else "incumbent",
        selected=candidate if accepted else incumbent,
        accepted=accepted,
        fail_closed=fail_closed,
        reason=reason,
        p_values=p_values,
        allocated_alpha=allocated_alpha,
        evidence_sha256=evidence_sha256,
    )


def _next_state(
    state: SequentialAnchorState,
    candidate: CandidateRef,
    decision: SequentialAnchorDecision,
) -> SequentialAnchorState:
    excluded = set(state.excluded_content_sha256)
    if decision.accepted:
        excluded.add(state.incumbent.content_sha256)
        excluded.discard(candidate.content_sha256)
        incumbent = candidate
    else:
        excluded.add(candidate.content_sha256)
        incumbent = state.incumbent
    seen = set(state.seen_candidate_ids)
    seen.add(candidate.candidate_id)
    event = {
        "previous_state_sha256": state.state_sha256,
        "step": state.step + 1,
        "candidate": candidate.as_record(),
        "decision": decision.as_record(),
    }
    chain = sha256_bytes(
        canonical_json_bytes(
            {
                "previous_transition_chain_sha256": state.transition_chain_sha256,
                "event": event,
            }
        )
    )
    return SequentialAnchorState(
        protocol_id=state.protocol_id,
        risk_policy_sha256=state.risk_policy_sha256,
        step=state.step + 1,
        horizon=state.horizon,
        incumbent=incumbent,
        excluded_content_sha256=tuple(sorted(excluded)),
        seen_candidate_ids=tuple(sorted(seen)),
        transition_chain_sha256=chain,
        initial_anchor_provenance_sha256=state.initial_anchor_provenance_sha256,
    )


def evaluate_update(
    *,
    state: SequentialAnchorState,
    candidate: CandidateRef,
    evidence: Optional[SequentialUpdateEvidence],
    policy: SequenceRiskPolicy,
    calibrations: Sequence[AxisNegativeCalibration],
) -> tuple[SequentialAnchorState, SequentialAnchorDecision]:
    """Evaluate one singleton update and return the next immutable state.

    Exact-content replays and identifier collisions fail before score evidence
    is consumed.  Every other malformed condition fails closed and advances the
    precommitted update index.
    """

    if not isinstance(state, SequentialAnchorState) or not isinstance(candidate, CandidateRef):
        _fail("invalid_update_input")
    if state.step >= state.horizon:
        _fail("sequence_horizon_exhausted")
    if (
        state.protocol_id != policy.protocol_id
        or state.risk_policy_sha256 != policy.policy_sha256
        or state.horizon != policy.horizon
    ):
        _fail("state_policy_mismatch")
    panel = validate_calibration_panel(calibrations, policy)

    if candidate.candidate_id in state.seen_candidate_ids:
        decision = _decision(
            incumbent=state.incumbent,
            candidate=candidate,
            accepted=False,
            fail_closed=True,
            reason="candidate_id_already_seen",
        )
        return _next_state(state, candidate, decision), decision
    if candidate.content_sha256 == state.incumbent.content_sha256:
        decision = _decision(
            incumbent=state.incumbent,
            candidate=candidate,
            accepted=False,
            fail_closed=True,
            reason="candidate_duplicates_incumbent_content",
        )
        return _next_state(state, candidate, decision), decision
    if candidate.content_sha256 in state.excluded_content_sha256:
        decision = _decision(
            incumbent=state.incumbent,
            candidate=candidate,
            accepted=False,
            fail_closed=False,
            reason="content_blocked_by_exclusion_ledger",
        )
        return _next_state(state, candidate, decision), decision

    expected_panel_sha256 = calibration_panel_sha256(panel)
    expected_scorer_sha256 = panel[0].scorer_sha256
    evidence_valid = (
        isinstance(evidence, SequentialUpdateEvidence)
        and evidence.protocol_id == policy.protocol_id
        and evidence.risk_policy_sha256 == policy.policy_sha256
        and evidence.state_sha256 == state.state_sha256
        and evidence.candidate == candidate
        and evidence.scorer_sha256 == expected_scorer_sha256
        and evidence.calibration_panel_sha256 == expected_panel_sha256
        and {axis for axis, _ in evidence.safety_scores} == {value.axis_id for value in panel}
    )
    if not evidence_valid:
        decision = _decision(
            incumbent=state.incumbent,
            candidate=candidate,
            accepted=False,
            fail_closed=True,
            reason="invalid_or_unbound_evidence",
        )
        return _next_state(state, candidate, decision), decision

    assert evidence is not None
    scores = dict(evidence.safety_scores)
    p_values = tuple(
        (calibration.axis_id, calibration.p_value(scores[calibration.axis_id]))
        for calibration in panel
    )
    allocated = policy.per_step_alpha
    failed_axes = tuple(axis for axis, p_value in p_values if p_value > allocated)
    if failed_axes:
        decision = _decision(
            incumbent=state.incumbent,
            candidate=candidate,
            accepted=False,
            fail_closed=False,
            reason="unsafe_null_not_rejected:" + ",".join(failed_axes),
            p_values=p_values,
            allocated_alpha=allocated,
            evidence_sha256=evidence.evidence_sha256,
        )
    elif not evidence.utility_improves:
        decision = _decision(
            incumbent=state.incumbent,
            candidate=candidate,
            accepted=False,
            fail_closed=False,
            reason="utility_margin_not_strictly_above_threshold",
            p_values=p_values,
            allocated_alpha=allocated,
            evidence_sha256=evidence.evidence_sha256,
        )
    else:
        decision = _decision(
            incumbent=state.incumbent,
            candidate=candidate,
            accepted=True,
            fail_closed=False,
            reason="all_risk_limited_safety_tests_and_utility_gate_passed",
            p_values=p_values,
            allocated_alpha=allocated,
            evidence_sha256=evidence.evidence_sha256,
        )
    return _next_state(state, candidate, decision), decision


def familywise_risk_bound(
    policy: SequenceRiskPolicy,
    *,
    evaluated_steps: Optional[int] = None,
) -> Fraction:
    """Return the exact union-bound risk spent by a fixed number of tests."""

    steps = policy.horizon if evaluated_steps is None else evaluated_steps
    if isinstance(steps, bool) or not isinstance(steps, int) or not 0 <= steps <= policy.horizon:
        _fail("invalid_evaluated_steps")
    return steps * policy.per_step_alpha


__all__ = [
    "AxisNegativeCalibration",
    "SequenceRiskPolicy",
    "SequentialAnchorDecision",
    "SequentialAnchorRCValidationError",
    "SequentialAnchorState",
    "SequentialUpdateEvidence",
    "calibration_panel_sha256",
    "evaluate_update",
    "familywise_risk_bound",
    "make_update_evidence",
    "open_sequence",
    "validate_calibration_panel",
]
