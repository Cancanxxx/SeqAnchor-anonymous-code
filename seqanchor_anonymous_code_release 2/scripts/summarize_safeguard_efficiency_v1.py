#!/usr/bin/env python3
"""Create the compact frozen AnchorRC versus TS-Guard efficiency result."""

from __future__ import annotations

import argparse
from pathlib import Path

from nontransitive_safety.provenance import (
    canonical_json_bytes,
    loads_json_strict,
    sha256_file,
)
from option_set_instability.anchorrc_v1 import load_jsonl_strict
from option_set_instability.sequential_efficiency_v1 import latency_summary


def _json(path: Path):
    value = loads_json_strict(path.read_text(encoding="utf-8"), source=str(path))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchorrc-result", required=True, type=Path)
    parser.add_argument("--tsguard-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    anchor_path = args.anchorrc_result.resolve()
    tsguard_run = args.tsguard_run.resolve()
    anchor = _json(anchor_path)
    tsguard_completion_path = tsguard_run / "completion.json"
    tsguard_predictions_path = tsguard_run / "predictions.jsonl"
    tsguard = _json(tsguard_completion_path)
    predictions = load_jsonl_strict(tsguard_predictions_path)
    if anchor.get("artifact_type") != "anchorrc_update_latency_result_v1":
        raise RuntimeError("unexpected AnchorRC latency artifact")
    if len(anchor.get("observations", ())) != 60 or len(predictions) != 60:
        raise RuntimeError("efficiency comparison requires 60 updates per method")
    if any(int(row["batch_size_realized"]) != 1 for row in predictions):
        raise RuntimeError("TS-Guard latency rows were not executed at batch size one")
    if tsguard.get("warmup", {}).get("rows") != 3:
        raise RuntimeError("TS-Guard latency run lacks the frozen three-row warmup")
    tsguard_latencies = [float(row["batch_seconds"]) for row in predictions]
    tsguard_strict_correct = sum(
        bool(row["parse_valid"])
        and (
            (float(row["truth_label"]) == 0.0 and float(row["prediction"]) == 0.0)
            or (float(row["truth_label"]) == 1.0 and float(row["prediction"]) != 0.0)
        )
        for row in predictions
    )
    anchor_latency = dict(anchor["latency_seconds"])
    tsguard_latency = dict(latency_summary(tsguard_latencies))
    output = {
        "schema_version": "1.0.0",
        "artifact_type": "safeguard_efficiency_comparison_v1",
        "panel": {
            "updates": 60,
            "candidate_roles": {
                "S1": 10,
                "E": 10,
                "N": 10,
                "C_GR": 10,
                "C_GA": 10,
                "C_RA": 10,
            },
            "hardware": "NVIDIA A100 80GB PCIe",
            "warm_model": True,
            "batch_unit": "one_catalog_update",
        },
        "anchorrc": {
            "model": "Qwen2.5-14B-Instruct",
            "calls_per_update": 10,
            "forward_passes_per_update": 2,
            "autoregressive_generation": False,
            "latency_seconds": anchor_latency,
            "input_tokens_per_update": anchor["input_tokens_per_update"],
            "resident_model_bytes": anchor["resident_model_bytes"],
            "max_incremental_peak_gpu_bytes": anchor[
                "max_incremental_peak_gpu_bytes"
            ],
            "model_load_seconds": anchor["model_load_seconds"],
            "evidence_sha256": sha256_file(anchor_path),
        },
        "tsguard": {
            "model": "MurrayTom/TS-Guard (7B)",
            "calls_per_update": 1,
            "forward_passes_per_update": None,
            "autoregressive_generation": True,
            "max_new_tokens": 384,
            "latency_seconds": tsguard_latency,
            "input_tokens_per_update": {
                "mean": sum(int(row["input_tokens"]) for row in predictions)
                / len(predictions),
                "minimum": min(int(row["input_tokens"]) for row in predictions),
                "maximum": max(int(row["input_tokens"]) for row in predictions),
            },
            "generated_tokens_per_update": {
                "mean": sum(int(row["generated_tokens"]) for row in predictions)
                / len(predictions),
                "minimum": min(int(row["generated_tokens"]) for row in predictions),
                "maximum": max(int(row["generated_tokens"]) for row in predictions),
            },
            "max_peak_gpu_bytes": tsguard["max_peak_gpu_bytes"],
            "model_load_seconds": tsguard["load_seconds"],
            "parse_failures": tsguard["metrics"]["parse_failures"],
            "strict_candidate_accuracy": tsguard_strict_correct / len(predictions),
            "evidence_sha256": sha256_file(tsguard_completion_path),
            "predictions_sha256": sha256_file(tsguard_predictions_path),
        },
        "descriptive_latency_ratio": {
            "tsguard_over_anchorrc_median": (
                tsguard_latency["median"] / anchor_latency["median"]
            ),
            "tsguard_over_anchorrc_p95": (
                tsguard_latency["p95"] / anchor_latency["p95"]
            ),
        },
        "interpretation_scope": (
            "Descriptive same-hardware warm-model comparison. The methods use different "
            "model sizes and inference interfaces, so latency is not attributed to "
            "controller architecture alone."
        ),
        "human_annotations_collected": False,
    }
    payload = canonical_json_bytes(output) + b"\n"
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != payload:
        raise RuntimeError(f"existing efficiency summary differs: {target}")
    target.write_bytes(payload)
    print(output["anchorrc"]["latency_seconds"])
    print(output["tsguard"]["latency_seconds"])
    print(output["descriptive_latency_ratio"])


if __name__ == "__main__":
    main()
