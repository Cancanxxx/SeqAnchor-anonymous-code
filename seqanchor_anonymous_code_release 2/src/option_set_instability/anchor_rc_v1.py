"""AnchorRC v1: anchored, symmetrized logit scoring for singleton updates.

The module is deliberately independent of model inference. A runner supplies
finite logits from two label-swapped safety views and the four-element
candidate-order-by-label utility group. This module verifies their provenance, symmetrizes the
views, applies a frozen worst-model aggregation, and returns only one of the
two admissible actions: keep the known-safe incumbent or select the singleton
new option.

The safety of the incumbent is an external premise represented by a receipt.
``bind_known_safe_incumbent`` verifies that the receipt is bound to the exact
protocol, base catalog, incumbent identity, and incumbent content; it does not
attempt to infer safety from the receipt itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, Mapping, Optional, Sequence

from nontransitive_safety.provenance import canonical_json_bytes, sha256_bytes

SCORE_SPEC_VERSION = "anchor_rc_finite_label_logits_v1"
VIEW_AGGREGATION = "finite_nuisance_group_mean_v1"
MODEL_AGGREGATION = "worst_model_minimum_v1"

SAFETY_AXES = (
    "gate_before_execution",
    "records_by_deadline",
    "listed_resources_only",
)
LABEL_SWAPPED_VIEWS = ("safe_is_A", "safe_is_B")
CANDIDATE_ORDER_VIEWS = ("incumbent_first", "new_first")
UTILITY_LABEL_VIEWS = ("new_is_A", "new_is_B")
FINITE_LABELS = ("A", "B")
WINNERS = ("incumbent", "new")

__all__ = [
    "SAFETY_AXES",
    "AnchorPair",
    "AnchorRCDecision",
    "AnchorRCPolicy",
    "AnchorRCScores",
    "AnchorRCValidationError",
    "CandidateRef",
    "KnownSafeAnchor",
    "bind_known_safe_incumbent",
    "candidate_ref",
    "catalog_sha256",
    "decide_anchor_update",
    "evaluate_anchor_update",
    "evaluate_presented_update",
    "evidence_bindings",
    "extract_singleton_update",
    "make_evidence_bundle",
    "score_anchor_update",
    "selected_candidate",
]


class AnchorRCValidationError(ValueError):
    """A malformed or provenance-inconsistent AnchorRC input."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise AnchorRCValidationError(code)


def _require_nonempty_string(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(code)
    return value


def _require_sha256(value: Any, code: str) -> str:
    text = _require_nonempty_string(value, code)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        _fail(code)
    return text


def _finite_number(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _fail(code)
    number = float(value)
    if not math.isfinite(number):
        _fail(code)
    return number


def _exact_mapping(value: Any, keys: Sequence[str], code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        _fail(code)
    return value


def _finite_difference(left: float, right: float, code: str) -> float:
    value = left - right
    if not math.isfinite(value):
        _fail(code)
    return value


def _symmetric_mean(first: float, second: float, code: str) -> float:
    # Scaling before addition avoids unnecessary overflow for large,
    # same-signed finite margins. A non-finite result fails closed.
    value = 0.5 * first + 0.5 * second
    if not math.isfinite(value):
        _fail(code)
    return value


@dataclass(frozen=True)
class CandidateRef:
    """Content-addressed public identity for one candidate."""

    candidate_id: str
    content_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.candidate_id, "invalid_candidate_id")
        _require_sha256(self.content_sha256, "invalid_candidate_content_sha256")

    def as_record(self) -> Mapping[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "content_sha256": self.content_sha256,
        }


def candidate_ref(candidate_id: str, content: Any) -> CandidateRef:
    """Bind a candidate ID to the canonical JSON representation of its content."""

    _require_nonempty_string(candidate_id, "invalid_candidate_id")
    try:
        digest = sha256_bytes(canonical_json_bytes(content))
    except (TypeError, ValueError):
        _fail("candidate_content_not_canonical_json")
    return CandidateRef(candidate_id=candidate_id, content_sha256=digest)


def _canonical_catalog(candidates: Sequence[CandidateRef]) -> tuple[CandidateRef, ...]:
    values = tuple(candidates)
    if not values or any(not isinstance(candidate, CandidateRef) for candidate in values):
        _fail("invalid_base_catalog")
    ids = [candidate.candidate_id for candidate in values]
    if len(ids) != len(set(ids)):
        _fail("duplicate_base_candidate_id")
    return tuple(sorted(values, key=lambda candidate: candidate.candidate_id))


def catalog_sha256(candidates: Sequence[CandidateRef]) -> str:
    """Hash a base catalog independently of input ordering."""

    catalog = _canonical_catalog(candidates)
    payload = {"base_candidates": [candidate.as_record() for candidate in catalog]}
    return sha256_bytes(canonical_json_bytes(payload))


@dataclass(frozen=True)
class KnownSafeAnchor:
    """A base incumbent bound to an external known-safe attestation."""

    protocol_id: str
    base_catalog: tuple[CandidateRef, ...]
    base_catalog_sha256: str
    incumbent: CandidateRef
    safety_attestation_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.protocol_id, "invalid_anchor_protocol_id")
        catalog = _canonical_catalog(self.base_catalog)
        if catalog != self.base_catalog:
            _fail("noncanonical_anchor_catalog")
        if catalog_sha256(catalog) != self.base_catalog_sha256:
            _fail("anchor_catalog_hash_mismatch")
        _require_sha256(self.safety_attestation_sha256, "invalid_safety_attestation_sha256")
        by_id = {candidate.candidate_id: candidate for candidate in catalog}
        if by_id.get(self.incumbent.candidate_id) != self.incumbent:
            _fail("incumbent_not_bound_to_base_catalog")

    def as_record(self) -> Mapping[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "base_catalog": [candidate.as_record() for candidate in self.base_catalog],
            "base_catalog_sha256": self.base_catalog_sha256,
            "incumbent": self.incumbent.as_record(),
            "safety_attestation_sha256": self.safety_attestation_sha256,
        }

    @property
    def provenance_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.as_record()))


def bind_known_safe_incumbent(
    *,
    protocol_id: str,
    base_candidates: Sequence[CandidateRef],
    incumbent_id: str,
    safety_attestation: Mapping[str, Any],
) -> KnownSafeAnchor:
    """Verify and bind an external ``known_safe`` receipt to the base incumbent.

    The receipt may carry additional issuer or signature metadata, but it must
    contain the five binding fields checked below.
    """

    protocol = _require_nonempty_string(protocol_id, "invalid_anchor_protocol_id")
    incumbent_name = _require_nonempty_string(incumbent_id, "invalid_incumbent_id")
    catalog = _canonical_catalog(base_candidates)
    by_id = {candidate.candidate_id: candidate for candidate in catalog}
    if incumbent_name not in by_id:
        _fail("incumbent_not_in_base_catalog")
    if not isinstance(safety_attestation, Mapping):
        _fail("invalid_safety_attestation")
    incumbent = by_id[incumbent_name]
    digest = catalog_sha256(catalog)
    expected = {
        "status": "known_safe",
        "protocol_id": protocol,
        "base_catalog_sha256": digest,
        "incumbent_id": incumbent.candidate_id,
        "incumbent_content_sha256": incumbent.content_sha256,
    }
    for key, value in expected.items():
        if safety_attestation.get(key) != value:
            _fail(f"safety_attestation_{key}_mismatch")
    try:
        receipt_digest = sha256_bytes(canonical_json_bytes(safety_attestation))
    except (TypeError, ValueError):
        _fail("safety_attestation_not_canonical_json")
    return KnownSafeAnchor(
        protocol_id=protocol,
        base_catalog=catalog,
        base_catalog_sha256=digest,
        incumbent=incumbent,
        safety_attestation_sha256=receipt_digest,
    )


@dataclass(frozen=True)
class AnchorPair:
    """The only two candidates visible to AnchorRC scoring."""

    protocol_id: str
    anchor_provenance_sha256: str
    incumbent: CandidateRef
    new: CandidateRef

    def __post_init__(self) -> None:
        _require_nonempty_string(self.protocol_id, "invalid_pair_protocol_id")
        _require_sha256(self.anchor_provenance_sha256, "invalid_anchor_provenance_sha256")
        if self.incumbent.candidate_id == self.new.candidate_id:
            _fail("new_candidate_equals_incumbent")

    def as_record(self) -> Mapping[str, Any]:
        return {
            "score_spec_version": SCORE_SPEC_VERSION,
            "protocol_id": self.protocol_id,
            "anchor_provenance_sha256": self.anchor_provenance_sha256,
            "incumbent": self.incumbent.as_record(),
            "new": self.new.as_record(),
        }

    @property
    def pair_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.as_record()))


def extract_singleton_update(
    anchor: KnownSafeAnchor,
    presented_candidates: Sequence[CandidateRef],
) -> AnchorPair:
    """Exclude bound old options and extract exactly one unbound new option.

    Any subset of the base catalog may accompany the incumbent. Present old
    options are content-verified and then ignored. The incumbent must be
    present, and exactly one candidate ID may fall outside the bound catalog.
    Thus old-option order and the presence of bound distractors cannot affect
    the scored pair, while a second unbound addition fails closed.
    """

    if not isinstance(anchor, KnownSafeAnchor):
        _fail("invalid_known_safe_anchor")
    values = tuple(presented_candidates)
    if not values or any(not isinstance(candidate, CandidateRef) for candidate in values):
        _fail("invalid_presented_candidates")
    ids = [candidate.candidate_id for candidate in values]
    if len(ids) != len(set(ids)):
        _fail("duplicate_presented_candidate_id")
    presented = {candidate.candidate_id: candidate for candidate in values}
    if presented.get(anchor.incumbent.candidate_id) != anchor.incumbent:
        _fail("presented_incumbent_missing_or_changed")
    base = {candidate.candidate_id: candidate for candidate in anchor.base_catalog}
    additions = []
    for candidate in values:
        expected = base.get(candidate.candidate_id)
        if expected is None:
            additions.append(candidate)
        elif candidate != expected:
            _fail("presented_base_candidate_content_changed")
    if len(additions) != 1:
        _fail("update_is_not_singleton")
    return AnchorPair(
        protocol_id=anchor.protocol_id,
        anchor_provenance_sha256=anchor.provenance_sha256,
        incumbent=anchor.incumbent,
        new=additions[0],
    )


@dataclass(frozen=True)
class AnchorRCPolicy:
    """Frozen score panels and strict decision thresholds."""

    protocol_id: str
    safety_model_ids: tuple[str, ...]
    throughput_model_ids: tuple[str, ...]
    gate_before_execution_threshold: float
    records_by_deadline_threshold: float
    listed_resources_only_threshold: float
    throughput_threshold: float
    score_spec_version: str = SCORE_SPEC_VERSION
    view_aggregation: str = VIEW_AGGREGATION
    model_aggregation: str = MODEL_AGGREGATION

    def __post_init__(self) -> None:
        _require_nonempty_string(self.protocol_id, "invalid_policy_protocol_id")
        object.__setattr__(self, "safety_model_ids", tuple(self.safety_model_ids))
        object.__setattr__(self, "throughput_model_ids", tuple(self.throughput_model_ids))
        for name, panel in (
            ("safety", self.safety_model_ids),
            ("throughput", self.throughput_model_ids),
        ):
            if (
                not panel
                or len(panel) != len(set(panel))
                or any(not isinstance(model_id, str) or not model_id for model_id in panel)
            ):
                _fail(f"invalid_{name}_model_panel")
        if self.score_spec_version != SCORE_SPEC_VERSION:
            _fail("unknown_score_spec_version")
        if self.view_aggregation != VIEW_AGGREGATION:
            _fail("unfrozen_view_aggregation")
        if self.model_aggregation != MODEL_AGGREGATION:
            _fail("unfrozen_model_aggregation")
        for field_name in (
            "gate_before_execution_threshold",
            "records_by_deadline_threshold",
            "listed_resources_only_threshold",
            "throughput_threshold",
        ):
            _finite_number(getattr(self, field_name), f"nonfinite_{field_name}")

    def safety_threshold(self, axis: str) -> float:
        if axis not in SAFETY_AXES:
            _fail("unknown_safety_axis")
        return float(getattr(self, f"{axis}_threshold"))

    @property
    def expected_model_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.safety_model_ids) | set(self.throughput_model_ids)))

    def as_record(self) -> Mapping[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "safety_model_ids": list(self.safety_model_ids),
            "throughput_model_ids": list(self.throughput_model_ids),
            "safety_thresholds": {axis: self.safety_threshold(axis) for axis in SAFETY_AXES},
            "throughput_threshold": float(self.throughput_threshold),
            "score_spec_version": self.score_spec_version,
            "view_aggregation": self.view_aggregation,
            "model_aggregation": self.model_aggregation,
            "comparison": "strictly_greater_than",
        }

    @property
    def policy_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.as_record()))


def evidence_bindings(pair: AnchorPair, policy: AnchorRCPolicy) -> Mapping[str, str]:
    """Return the exact provenance header a runner must attach to raw logits."""

    if pair.protocol_id != policy.protocol_id:
        _fail("pair_policy_protocol_mismatch")
    return {
        "score_spec_version": SCORE_SPEC_VERSION,
        "protocol_id": policy.protocol_id,
        "policy_sha256": policy.policy_sha256,
        "anchor_provenance_sha256": pair.anchor_provenance_sha256,
        "pair_sha256": pair.pair_sha256,
    }


def make_evidence_bundle(
    pair: AnchorPair,
    policy: AnchorRCPolicy,
    model_logits: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Attach immutable pair/policy bindings to runner-produced model logits."""

    return {**evidence_bindings(pair, policy), "models": model_logits}


@dataclass(frozen=True)
class AnchorRCScores:
    """Immutable analyzer result before thresholds are applied."""

    valid: bool
    failure_reason: Optional[str]
    safety_axis_margins: tuple[tuple[str, float], ...]
    throughput_margin: Optional[float]
    per_model_safety: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    per_model_throughput: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if self.valid:
            if self.failure_reason is not None:
                _fail("valid_scores_have_failure_reason")
            if tuple(axis for axis, _ in self.safety_axis_margins) != SAFETY_AXES:
                _fail("valid_scores_missing_safety_axis")
            if self.throughput_margin is None:
                _fail("valid_scores_missing_throughput_margin")
            for _, value in self.safety_axis_margins:
                _finite_number(value, "nonfinite_aggregated_safety_margin")
            _finite_number(self.throughput_margin, "nonfinite_aggregated_throughput_margin")
        elif not self.failure_reason:
            _fail("invalid_scores_missing_failure_reason")

    def safety_margin(self, axis: str) -> float:
        values = dict(self.safety_axis_margins)
        if axis not in values:
            _fail("score_missing_safety_axis")
        return values[axis]

    def as_record(self) -> Mapping[str, Any]:
        return {
            "valid": self.valid,
            "failure_reason": self.failure_reason,
            "safety_axis_margins": dict(self.safety_axis_margins),
            "throughput_margin": self.throughput_margin,
            "per_model_safety": {
                model_id: dict(axis_values) for model_id, axis_values in self.per_model_safety
            },
            "per_model_throughput": dict(self.per_model_throughput),
            "view_aggregation": VIEW_AGGREGATION,
            "model_aggregation": MODEL_AGGREGATION,
        }


def _invalid_scores(code: str) -> AnchorRCScores:
    return AnchorRCScores(
        valid=False,
        failure_reason=code,
        safety_axis_margins=(),
        throughput_margin=None,
        per_model_safety=(),
        per_model_throughput=(),
    )


def _label_logits(value: Any, code: str) -> Mapping[str, float]:
    logits = _exact_mapping(value, FINITE_LABELS, code)
    return {
        label: _finite_number(logits[label], f"{code}_nonfinite_{label}") for label in FINITE_LABELS
    }


def _safety_axis_margin(value: Any, code: str) -> float:
    views = _exact_mapping(value, LABEL_SWAPPED_VIEWS, code)
    safe_a = _label_logits(views["safe_is_A"], f"{code}_safe_is_A")
    safe_b = _label_logits(views["safe_is_B"], f"{code}_safe_is_B")
    first = _finite_difference(safe_a["A"], safe_a["B"], f"{code}_margin_safe_is_A")
    second = _finite_difference(safe_b["B"], safe_b["A"], f"{code}_margin_safe_is_B")
    return _symmetric_mean(first, second, f"{code}_symmetrized_margin")


def _throughput_margin(value: Any, code: str) -> float:
    orders = _exact_mapping(value, CANDIDATE_ORDER_VIEWS, code)
    margins = []
    for order in CANDIDATE_ORDER_VIEWS:
        views = _exact_mapping(
            orders[order],
            UTILITY_LABEL_VIEWS,
            f"{code}_{order}",
        )
        new_a = _label_logits(views["new_is_A"], f"{code}_{order}_new_is_A")
        new_b = _label_logits(views["new_is_B"], f"{code}_{order}_new_is_B")
        margins.append(
            _finite_difference(
                new_a["A"],
                new_a["B"],
                f"{code}_{order}_margin_new_is_A",
            )
        )
        margins.append(
            _finite_difference(
                new_b["B"],
                new_b["A"],
                f"{code}_{order}_margin_new_is_B",
            )
        )
    first_pair = _symmetric_mean(margins[0], margins[1], f"{code}_incumbent_first_mean")
    second_pair = _symmetric_mean(margins[2], margins[3], f"{code}_new_first_mean")
    return _symmetric_mean(first_pair, second_pair, f"{code}_group_symmetrized_margin")


def _score_anchor_update_or_raise(
    pair: AnchorPair,
    evidence: Mapping[str, Any],
    policy: AnchorRCPolicy,
) -> AnchorRCScores:
    header = evidence_bindings(pair, policy)
    bundle = _exact_mapping(
        evidence,
        (*header.keys(), "models"),
        "malformed_evidence_bundle",
    )
    for key, expected in header.items():
        if bundle[key] != expected:
            _fail(f"evidence_{key}_mismatch")
    models = bundle["models"]
    if not isinstance(models, Mapping) or set(models) != set(policy.expected_model_ids):
        _fail("evidence_model_panel_mismatch")

    per_model_safety = []
    for model_id in policy.safety_model_ids:
        expected_keys = ["safety"]
        if model_id in policy.throughput_model_ids:
            expected_keys.append("throughput")
        model = _exact_mapping(models[model_id], expected_keys, "malformed_model_evidence")
        axes = _exact_mapping(model["safety"], SAFETY_AXES, "malformed_safety_evidence")
        axis_values = tuple(
            (
                axis,
                _safety_axis_margin(
                    axes[axis],
                    f"safety_{model_id}_{axis}",
                ),
            )
            for axis in SAFETY_AXES
        )
        per_model_safety.append((model_id, axis_values))

    per_model_throughput = []
    for model_id in policy.throughput_model_ids:
        expected_keys = ["throughput"]
        if model_id in policy.safety_model_ids:
            expected_keys.insert(0, "safety")
        model = _exact_mapping(models[model_id], expected_keys, "malformed_model_evidence")
        per_model_throughput.append(
            (
                model_id,
                _throughput_margin(model["throughput"], f"throughput_{model_id}"),
            )
        )

    # The panel aggregation is a frozen conjunction: the least favorable
    # model margin is the deployed margin. No model is selected post hoc.
    safety_axis_margins = tuple(
        (
            axis,
            min(dict(axis_values)[axis] for _, axis_values in per_model_safety),
        )
        for axis in SAFETY_AXES
    )
    throughput = min(value for _, value in per_model_throughput)
    return AnchorRCScores(
        valid=True,
        failure_reason=None,
        safety_axis_margins=safety_axis_margins,
        throughput_margin=throughput,
        per_model_safety=tuple(per_model_safety),
        per_model_throughput=tuple(per_model_throughput),
    )


def score_anchor_update(
    pair: AnchorPair,
    evidence: Mapping[str, Any],
    policy: AnchorRCPolicy,
) -> AnchorRCScores:
    """Validate and aggregate raw logits; malformed evidence becomes invalid.

    This is the fail-closed boundary for runner output. Missing views, missing
    models, extra schema fields, non-numeric values, NaNs, infinities, or any
    provenance mismatch yield ``valid=False`` instead of a partial score.
    """

    try:
        return _score_anchor_update_or_raise(pair, evidence, policy)
    except AnchorRCValidationError as error:
        return _invalid_scores(error.code)


@dataclass(frozen=True)
class AnchorRCDecision:
    """A two-action decision; no old option can be emitted."""

    winner: Literal["incumbent", "new"]
    switched: bool
    fail_closed: bool
    reason: str
    failed_axis: Optional[str] = None

    def __post_init__(self) -> None:
        if self.winner not in WINNERS:
            _fail("decision_outside_anchor_pair")
        if self.switched != (self.winner == "new"):
            _fail("incoherent_switch_indicator")
        if not isinstance(self.reason, str) or not self.reason:
            _fail("decision_missing_reason")

    def as_record(self) -> Mapping[str, Any]:
        return {
            "winner": self.winner,
            "switched": self.switched,
            "fail_closed": self.fail_closed,
            "reason": self.reason,
            "failed_axis": self.failed_axis,
        }


def decide_anchor_update(scores: AnchorRCScores, policy: AnchorRCPolicy) -> AnchorRCDecision:
    """Apply all strict thresholds, retaining the incumbent on every failure."""

    if not scores.valid:
        return AnchorRCDecision(
            winner="incumbent",
            switched=False,
            fail_closed=True,
            reason=f"invalid_evidence:{scores.failure_reason}",
        )
    for axis in SAFETY_AXES:
        if not scores.safety_margin(axis) > policy.safety_threshold(axis):
            return AnchorRCDecision(
                winner="incumbent",
                switched=False,
                fail_closed=False,
                reason="safety_margin_not_strictly_above_threshold",
                failed_axis=axis,
            )
    if scores.throughput_margin is None or not scores.throughput_margin > float(
        policy.throughput_threshold
    ):
        return AnchorRCDecision(
            winner="incumbent",
            switched=False,
            fail_closed=False,
            reason="throughput_margin_not_strictly_above_threshold",
            failed_axis="mission_throughput",
        )
    return AnchorRCDecision(
        winner="new",
        switched=True,
        fail_closed=False,
        reason="all_strict_anchor_rc_gates_passed",
    )


def evaluate_anchor_update(
    pair: AnchorPair,
    evidence: Mapping[str, Any],
    policy: AnchorRCPolicy,
) -> tuple[AnchorRCScores, AnchorRCDecision]:
    """Convenience API for an analyzer: score once, then make the decision."""

    scores = score_anchor_update(pair, evidence, policy)
    return scores, decide_anchor_update(scores, policy)


def evaluate_presented_update(
    anchor: KnownSafeAnchor,
    presented_candidates: Sequence[CandidateRef],
    evidence: Mapping[str, Any],
    policy: AnchorRCPolicy,
) -> tuple[Optional[AnchorPair], AnchorRCScores, AnchorRCDecision]:
    """Extract, score, and decide while failing closed on a malformed menu."""

    try:
        pair = extract_singleton_update(anchor, presented_candidates)
    except AnchorRCValidationError as error:
        scores = _invalid_scores(f"invalid_presented_update:{error.code}")
        return None, scores, decide_anchor_update(scores, policy)
    scores, decision = evaluate_anchor_update(pair, evidence, policy)
    return pair, scores, decision


def selected_candidate(pair: AnchorPair, decision: AnchorRCDecision) -> CandidateRef:
    """Resolve the semantic two-action output to one of the exact pair members."""

    return pair.incumbent if decision.winner == "incumbent" else pair.new
