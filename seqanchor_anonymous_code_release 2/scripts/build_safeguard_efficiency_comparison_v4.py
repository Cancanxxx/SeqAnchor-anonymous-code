#!/usr/bin/env python3
"""Scope the v3 safeguard-efficiency measurements for publication."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

from nontransitive_safety.provenance import canonical_json_bytes, sha256_file

GUARDS = ("qwen3guard", "tsguard", "stepguard")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _latency_panel_metrics(record: dict[str, Any]) -> Mapping[str, Any]:
    metrics: dict[str, Any] = {
        "strict_admission_accuracy_on_60_update_latency_panel": float(
            record.pop("strict_candidate_accuracy")
        ),
        "parse_failures_on_60_update_latency_panel": int(record.pop("parse_failures")),
        "panel_safe_updates": 30,
        "panel_unsafe_updates": 30,
    }
    if "unsafe_pass_rate" in record:
        metrics["unsafe_pass_rate_on_30_unsafe_latency_updates"] = float(
            record.pop("unsafe_pass_rate")
        )
    return metrics


def _guard_scope(record: dict[str, Any], *, explicit_sync: bool) -> None:
    peak = int(record.pop("max_peak_gpu_bytes"))
    record["execution_scope"] = {
        "updates_per_timed_observation": 1,
        "logical_model_calls_per_update": int(record.pop("calls_per_update")),
        "model_input_batch_size": int(record.pop("batch_size", 1)),
        "internal_model_batches_per_update": 1,
    }
    record.pop("forward_passes_per_update", None)
    record["timing_scope"] = {
        "clocked_region": "model.generate only",
        "tokenization_included": False,
        "output_decoding_and_parsing_included": False,
        "explicit_cuda_synchronization_before_and_after_timed_region": explicit_sync,
        "synchronization_note": (
            "The runner synchronizes CUDA immediately before and after every timed generation call."
            if explicit_sync
            else "The runner brackets model.generate without explicit per-update CUDA "
            "synchronization; the reported wall time must not be described as explicitly "
            "CUDA-synchronized."
        ),
    }
    record["memory_scope"] = {
        "peak_allocated_gpu_bytes": peak,
        "measurement": "maximum torch.cuda allocated memory during the latency run",
        "includes_loaded_model": True,
        "resident_after_load_separately_available": False,
        "incremental_update_peak_separately_available": False,
    }
    record["latency_panel_safety_metrics"] = _latency_panel_metrics(record)


def correct_v3(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a deterministic v4 view without altering the v3 input mapping."""

    _require(
        value.get("artifact_type") == "safeguard_efficiency_comparison_v3",
        "efficiency input is not v3",
    )
    _require(value.get("schema_version") == "3.0.0", "v3 schema differs")
    output: dict[str, Any] = copy.deepcopy(dict(value))
    _require(all(name in output for name in ("anchorrc", *GUARDS)), "method missing")

    panel = dict(output["panel"])
    _require(panel.get("updates") == 60, "latency panel update count differs")
    _require(sum(panel.get("candidate_roles", {}).values()) == 60, "role panel differs")
    panel.pop("batch_unit", None)
    panel.update(
        {
            "timing_unit": "one catalog update evaluated at a time",
            "concurrent_updates_per_timed_observation": 1,
            "latency_panel_safe_updates": 30,
            "latency_panel_unsafe_updates": 30,
            "warmup_updates_excluded_per_method": 3,
        }
    )
    output["panel"] = panel

    anchor = dict(output["anchorrc"])
    resident = int(anchor.pop("resident_model_bytes"))
    incremental = int(anchor.pop("max_incremental_peak_gpu_bytes"))
    anchor["execution_scope"] = {
        "updates_per_timed_observation": 1,
        "logical_score_calls_per_update": int(anchor.pop("calls_per_update")),
        "internal_model_batches_per_update": int(anchor.pop("forward_passes_per_update")),
        "internal_batch_composition": {
            "safety_views_in_first_batch": 6,
            "utility_views_in_second_batch": 4,
        },
        "configured_max_prompt_batch_size": 8,
    }
    anchor["timing_scope"] = {
        "clocked_region": "two no-generation model forward passes only",
        "tokenization_included": False,
        "output_decoding_and_parsing_included": False,
        "explicit_cuda_synchronization_before_and_after_timed_region": True,
        "synchronization_note": (
            "The runner synchronizes CUDA immediately before and after every timed "
            "two-forward-pass update."
        ),
    }
    anchor["memory_scope"] = {
        "peak_allocated_gpu_bytes": resident + incremental,
        "resident_after_load_gpu_bytes": resident,
        "maximum_incremental_peak_during_update_gpu_bytes": incremental,
        "measurement": "torch.cuda allocated memory",
        "includes_loaded_model": True,
        "resident_and_incremental_components_separately_available": True,
        "derivation": (
            "peak_allocated_gpu_bytes is resident_after_load_gpu_bytes plus the recorded "
            "maximum incremental update peak"
        ),
    }
    output["anchorrc"] = anchor

    _guard_scope(output["qwen3guard"], explicit_sync=False)
    _guard_scope(output["tsguard"], explicit_sync=False)
    _guard_scope(output["stepguard"], explicit_sync=True)

    output.update(
        {
            "schema_version": "4.0.0",
            "artifact_type": "safeguard_efficiency_comparison_v4",
            "interpretation_scope": (
                "Descriptive warm-model measurements on A100 80GB PCIe GPUs. Each "
                "observation evaluates one catalog update at a time. SeqAnchor internally "
                "batches six safety views and four utility views into two forward passes; "
                "each guard receives one serialized candidate in a batch of one. SeqAnchor "
                "and StepGuard use explicit per-update CUDA synchronization, whereas the "
                "Qwen3Guard and TS-Guard runners bracket model.generate without explicit "
                "per-update synchronization. Tokenization, decoding, and parsing are not "
                "timed. Safety fields refer only to the balanced 60-update latency panel, "
                "not the 360-candidate transfer evaluation. Memory scopes are exposed per "
                "method because only SeqAnchor retained separate resident and incremental "
                "allocated-memory measurements. Checkpoints, prompts, output lengths, and "
                "interfaces differ, so the latency ratios are descriptive."
            ),
            "publication_scope_corrections": [
                (
                    "one catalog update is timed at a time; this does not mean every "
                    "method uses prompt batch size one"
                ),
                (
                    "SeqAnchor uses two internal prompt batches of six and four views; "
                    "the three guards use inference batch size one"
                ),
                (
                    "only SeqAnchor and StepGuard have explicit CUDA synchronization "
                    "around each timed observation"
                ),
                "guard safety metrics are named as results on the 60-update latency panel",
                (
                    "GPU-memory fields distinguish total peak allocated memory from "
                    "resident and incremental components"
                ),
            ],
        }
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.v3.resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    output = dict(correct_v3(value))
    output["source_v3_sha256"] = sha256_file(source)
    payload = canonical_json_bytes(output) + b"\n"
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != payload:
        raise RuntimeError(f"existing efficiency comparison differs: {target}")
    if not target.exists():
        target.write_bytes(payload)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
