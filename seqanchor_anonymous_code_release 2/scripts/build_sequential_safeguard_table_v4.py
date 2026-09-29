#!/usr/bin/env python3
"""Build a publication-scoped sequential safeguard comparison table."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any, Mapping

from nontransitive_safety.provenance import canonical_json_bytes, sha256_file

SAFETY_PROTECTION = (
    "<=5% probability of any unsafe admission over the fixed six-update horizon, "
    "under family exchangeability, calibration coverage, a fixed scorer, and a "
    "precommitted sequence or conditionally super-uniform rank values"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def _fraction(events: int, total: int) -> str:
    return f"{events}/{total} ({100 * events / total:.1f}%)"


def _learned_utility_protection(learned: Mapping[str, Any]) -> str:
    calibration = learned["balanced_per_update_utility_budget"]["calibration"]
    bound = calibration["implied_finite_sample_bound"]
    numerator = int(bound["numerator"])
    denominator = int(bound["denominator"])
    _require((numerator, denominator) == (6, 121), "learned utility bound differs")
    return (
        f"{SAFETY_PROTECTION}; the balanced learned-utility gate separately bounds "
        f"each non-improving replacement at {numerator}/{denominator} under its own "
        "family-exchangeability and coverage assumptions"
    )


def build_rows(
    *,
    exact_document: Mapping[str, Any],
    full: Mapping[str, Any],
    learned_document: Mapping[str, Any],
    efficiency: Mapping[str, Any],
) -> list[Mapping[str, str]]:
    exact = exact_document["methods"]
    learned = learned_document["balanced_per_update_utility_budget"]["sequence_metrics"]

    def exact_row(
        method: str,
        label: str,
        latency: str,
        guarantee: str,
        *,
        old_action: str = "N/A (previous actions not offered)",
        provenance: str = (
            "controlled-study oracle recomputed from frozen simulator-executed "
            "mission-item counts after auditing the earlier derived field; no model "
            "inference was rerun, and this is not a deployable learned utility score"
        ),
    ) -> Mapping[str, str]:
        result = exact[method]
        sequence = result["sequence_metrics"]
        updates = result["update_metrics"]
        return {
            "method": label,
            "evaluation interface": (
                "singleton workflow update + frozen simulator-executed mission throughput"
            ),
            "result provenance": provenance,
            "unsafe sequence": _fraction(sequence["any_unsafe_admission"], 60),
            "safe useful update adopted": _fraction(
                sequence["useful_safe_improvement_adopted"], 60
            ),
            "unsafe candidates passed by safety gate": _fraction(
                updates["unsafe_safety_passes"], 180
            ),
            "old rejected action resurrection": old_action,
            "warm latency median / p95 (scope)": latency,
            "formal protection": guarantee,
        }

    a_lat = efficiency["anchorrc"]["latency_seconds"]
    q_lat = efficiency["qwen3guard"]["latency_seconds"]
    t_lat = efficiency["tsguard"]["latency_seconds"]
    s_lat = efficiency["stepguard"]["latency_seconds"]
    exact_provenance = (
        "controlled-study oracle recomputed from frozen simulator-executed mission-item "
        "counts after auditing the earlier derived field; no model inference was rerun, "
        "and this is not a deployable learned utility score"
    )
    rows = [
        {
            "method": "Always retain incumbent",
            "evaluation interface": "singleton workflow update",
            "result provenance": "frozen six-update comparison panel",
            "unsafe sequence": _fraction(0, 60),
            "safe useful update adopted": _fraction(0, 60),
            "unsafe candidates passed by safety gate": "not evaluated",
            "old rejected action resurrection": "0 (structural)",
            "warm latency median / p95 (scope)": "controller only",
            "formal protection": "safe only if the initial incumbent remains suitable",
        },
        {
            "method": "Growing full-catalog reselection (Qwen-2.5-14B)",
            "evaluation interface": "all accumulated workflows reconsidered",
            "result provenance": "frozen six-update full-catalog evaluation",
            "unsafe sequence": _fraction(full["safe_to_unsafe_sequences"], full["base_safe"]),
            "safe useful update adopted": _fraction(full["useful_S1_selected_after_available"], 60),
            "unsafe candidates passed by safety gate": "not separable",
            "old rejected action resurrection": _fraction(
                full["old_U0_resurrections"], full["base_safe"]
            ),
            "warm latency median / p95 (scope)": ("not measured on the singleton latency panel"),
            "formal protection": "none",
        },
        exact_row(
            "Exact throughput only",
            "Exact throughput only",
            "controller only",
            "none",
        ),
        exact_row(
            "Qwen3Guard + exact throughput",
            "Qwen3Guard-Gen-4B + exact throughput",
            (
                f"{q_lat['median']:.2f}s / {q_lat['p95']:.2f}s "
                "(guard inference only; exact controller excluded)"
            ),
            "no workflow-update or sequence-level guarantee",
        ),
        exact_row(
            "TS-Guard + exact throughput",
            "TS-Guard + exact throughput",
            (
                f"{t_lat['median']:.2f}s / {t_lat['p95']:.2f}s "
                "(guard inference only; exact controller excluded)"
            ),
            "no workflow-update or sequence-level guarantee",
        ),
        exact_row(
            "StepGuard + exact throughput",
            "StepGuard action filter + exact throughput",
            (
                f"{s_lat['median']:.2f}s / {s_lat['p95']:.2f}s "
                "(guard inference only; exact controller excluded)"
            ),
            "no persistent ledger or sequence-level risk guarantee",
            provenance=(
                f"{exact_provenance}; StepGuard was added post hoc under a transfer "
                "protocol frozen before its predictions"
            ),
        ),
        exact_row(
            "SeqAnchor + exact throughput",
            "SeqAnchor/Qwen2.5 + exact throughput",
            (
                f"{a_lat['median']:.2f}s / {a_lat['p95']:.2f}s learned-utility-interface "
                "proxy; exact-throughput latency was not measured and this proxy is not "
                "an upper bound"
            ),
            SAFETY_PROTECTION,
            old_action="0 (structural ledger)",
        ),
        {
            "method": "SeqAnchor/Qwen2.5 + learned balanced utility",
            "evaluation interface": "singleton safety and learned utility scoring",
            "result provenance": (
                "prospectively frozen post-primary Qwen replication with a separately "
                "frozen complete-coverage extension after the initial output was opened; "
                "the extension changed no prompt, threshold, calibration example, or "
                "safety decision"
            ),
            "unsafe sequence": _fraction(learned["any_unsafe_admission"], 60),
            "safe useful update adopted": _fraction(learned["useful_improvement_adopted"], 60),
            "unsafe candidates passed by safety gate": _fraction(0, 180),
            "old rejected action resurrection": "0 (structural ledger)",
            "warm latency median / p95 (scope)": (
                f"{a_lat['median']:.2f}s / {a_lat['p95']:.2f}s (measured complete "
                "safety + learned-utility interface)"
            ),
            "formal protection": _learned_utility_protection(learned_document),
        },
    ]
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-isolation", required=True, type=Path)
    parser.add_argument("--full-catalog", required=True, type=Path)
    parser.add_argument("--learned-utility-seqanchor", required=True, type=Path)
    parser.add_argument("--efficiency", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "exact_isolation": args.exact_isolation.resolve(),
        "full_catalog": args.full_catalog.resolve(),
        "learned_utility_seqanchor": args.learned_utility_seqanchor.resolve(),
        "efficiency": args.efficiency.resolve(),
    }
    documents = {name: _load(path) for name, path in paths.items()}
    exact_document = documents["exact_isolation"]
    learned_document = documents["learned_utility_seqanchor"]
    efficiency = documents["efficiency"]
    _require(
        exact_document.get("artifact_type") == "sequential_exact_throughput_safety_isolation_v2",
        "exact-throughput artifact differs",
    )
    _require(
        exact_document.get("utility_definition")
        == "strictly_more_simulator-executed_mission_items_than_current_incumbent",
        "exact-throughput provenance differs",
    )
    _require(
        learned_document.get("status") == "post_frozen_complete_coverage_extension",
        "learned-utility extension status differs",
    )
    _require(
        learned_document.get("utility_extension", {}).get("thresholds_changed") is False,
        "learned-utility extension changed thresholds",
    )
    _require(
        efficiency.get("artifact_type") == "safeguard_efficiency_comparison_v4",
        "efficiency artifact is not v4",
    )
    full = documents["full_catalog"]["sequence_metrics"]
    rows = build_rows(
        exact_document=exact_document,
        full=full,
        learned_document=learned_document,
        efficiency=efficiency,
    )

    fieldnames = list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_payload = buffer.getvalue().encode("utf-8")
    csv_target = args.output_csv.resolve()
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    if csv_target.exists() and csv_target.read_bytes() != csv_payload:
        raise RuntimeError(f"existing table differs: {csv_target}")
    if not csv_target.exists():
        csv_target.write_bytes(csv_payload)

    learned_bound = learned_document["balanced_per_update_utility_budget"]["calibration"][
        "implied_finite_sample_bound"
    ]
    output = {
        "schema_version": "4.0.0",
        "artifact_type": "sequential_safeguard_main_table_v4",
        "rows": rows,
        "source_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "result_scope": {
            "exact_throughput": {
                "utility_definition": exact_document["utility_definition"],
                "provenance": (
                    "Frozen simulator-executed mission-item counts from the controlled "
                    "workflow specifications. This is a controlled-study oracle, not a "
                    "deployable learned utility score."
                ),
                "artifact_status": exact_document.get("status"),
                "stepguard_comparator_timing": exact_document["audit"][
                    "stepguard_comparator_timing"
                ],
            },
            "learned_utility": {
                "artifact_status": learned_document["status"],
                "post_primary_replication": True,
                "complete_coverage_extension_frozen_after_initial_output_opened": True,
                "extension_purpose": learned_document["utility_extension"]["purpose"],
                "extension_changed_thresholds": learned_document["utility_extension"][
                    "thresholds_changed"
                ],
                "balanced_per_update_finite_sample_bound": learned_bound,
                "utility_bound_is_sequence_wide": False,
                "utility_calibration_strengthens_safety_bound": False,
            },
            "seqanchor_exact_latency": {
                "measurement_available": False,
                "displayed_value": "learned-utility-interface proxy",
                "proxy_is_upper_bound": False,
            },
        },
        "comparison_notes": [
            (
                "Exact-throughput rows share the same 60 sequences and frozen executable "
                "utility rule based on simulator-executed mission-item counts."
            ),
            (
                "Exact throughput is a controlled-study oracle and is not available as a "
                "deployable learned utility score."
            ),
            (
                "The displayed SeqAnchor exact-throughput latency is the measured complete "
                "learned-utility interface used only as a proxy; exact-throughput latency "
                "was not measured, and the proxy is not an upper bound."
            ),
            (
                "The final 53/60 learned-utility result is a prospectively frozen "
                "post-primary Qwen replication with a separately frozen complete-coverage "
                "extension after the initial output was opened."
            ),
            (
                "The extension changed no prompt, threshold, calibration example, or "
                "safety decision."
            ),
            (
                "StepGuard is a post-hoc workflow-admission transfer whose protocol was "
                "frozen before its predictions."
            ),
            (
                "Singleton filters cannot be credited with zero old-action resurrection "
                "because previous actions are not offered to them."
            ),
            (
                "Latency is a descriptive A100 80GB warm-model comparison with method-"
                "specific batching, synchronization, and clocked regions documented in "
                "the v4 efficiency artifact."
            ),
            "Observed zero failures does not prove zero deployment risk.",
        ],
        "human_annotations_collected": False,
    }
    payload = canonical_json_bytes(output) + b"\n"
    json_target = args.output_json.resolve()
    if json_target.exists() and json_target.read_bytes() != payload:
        raise RuntimeError(f"existing table differs: {json_target}")
    if not json_target.exists():
        json_target.write_bytes(payload)
    print(buffer.getvalue(), end="")


if __name__ == "__main__":
    main()
