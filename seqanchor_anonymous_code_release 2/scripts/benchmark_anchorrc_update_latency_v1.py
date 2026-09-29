#!/usr/bin/env python3
"""Measure Sequential AnchorRC latency on a frozen balanced update panel."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from nontransitive_safety.local_hf_registry import (
    load_model,
    load_model_registry,
    load_tokenizer,
    registry_chat_template_kwargs,
    validate_runtime_versions,
)
from nontransitive_safety.provenance import (
    canonical_json_bytes,
    loads_json_strict,
    sha256_bytes,
    sha256_file,
)
from option_set_instability.anchorrc_v1 import (
    configure_deterministic_torch,
    load_jsonl_strict,
    load_protocol,
    prepare_prompts,
    score_allowed_label_logits_batch,
    validate_deterministic_process_environment,
)
from option_set_instability.sequential_efficiency_v1 import (
    build_update_call_groups,
    first_arrival_panel,
    latency_summary,
)


REPO = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = loads_json_strict(path.read_text(encoding="utf-8"), source=str(path))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--safety-call-plan", required=True, type=Path)
    parser.add_argument("--utility-call-plan", required=True, type=Path)
    parser.add_argument("--safety-protocol", required=True, type=Path)
    parser.add_argument("--utility-protocol", required=True, type=Path)
    parser.add_argument("--model-registry", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--warmup-updates", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup_updates < 0:
        raise ValueError("--warmup-updates cannot be negative")
    status = _git("status", "--porcelain", "--untracked-files=normal")
    if status:
        raise RuntimeError("AnchorRC latency measurement requires a clean source tree")
    environment = dict(validate_deterministic_process_environment())

    paths = {
        "sequences": args.sequences.resolve(),
        "safety_call_plan": args.safety_call_plan.resolve(),
        "utility_call_plan": args.utility_call_plan.resolve(),
        "safety_protocol": args.safety_protocol.resolve(),
        "utility_protocol": args.utility_protocol.resolve(),
        "model_registry": args.model_registry.resolve(),
    }
    for path in paths.values():
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"latency input must be a regular file: {path}")
    output_root = args.output_root.resolve()
    try:
        output_root.relative_to(REPO.resolve())
    except ValueError:
        pass
    else:
        raise RuntimeError("latency output root must be outside the source repository")

    sequences = load_jsonl_strict(paths["sequences"])
    safety_calls = load_jsonl_strict(paths["safety_call_plan"])
    utility_calls = load_jsonl_strict(paths["utility_call_plan"])
    panel = first_arrival_panel(sequences)
    groups = build_update_call_groups(panel, safety_calls, utility_calls)
    snapshot_ids = {str(group["snapshot_id"]) for group in groups}
    if len(snapshot_ids) != 1:
        raise RuntimeError("latency panel spans multiple snapshots")
    snapshot_id = snapshot_ids.pop()

    safety_protocol = load_protocol(paths["safety_protocol"])
    utility_protocol = load_protocol(paths["utility_protocol"])
    if safety_protocol["fixed_chat_date"] != utility_protocol["fixed_chat_date"]:
        raise RuntimeError("safety and utility protocols use different chat dates")
    if safety_protocol["inference"] != utility_protocol["inference"]:
        raise RuntimeError("safety and utility protocols use different inference settings")
    inference = safety_protocol["inference"]
    if inference["batch_size"] < 6:
        raise RuntimeError("frozen batch size cannot hold one six-view safety panel")

    registry = load_model_registry(
        paths["model_registry"], verify_files=True, require_frozen=True
    )
    if registry["candidate_snapshot_id"] != snapshot_id:
        raise RuntimeError("model registry differs from the selected call plans")
    validate_runtime_versions(registry)

    identity = {
        "schema_version": "1.0.0",
        "artifact_type": "anchorrc_update_latency_identity_v1",
        "git_commit": _git("rev-parse", "HEAD"),
        "snapshot_id": snapshot_id,
        "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "panel": {
            "selection": "all_60_test_sequences_first_arrival",
            "updates": len(groups),
            "candidate_roles": {role: 10 for role in ("S1", "E", "N", "C_GR", "C_GA", "C_RA")},
            "safety_calls_per_update": 6,
            "utility_calls_per_update": 4,
            "forward_passes_per_update": 2,
        },
        "warmup_updates": args.warmup_updates,
        "inference": dict(inference),
    }
    run_id = sha256_bytes(canonical_json_bytes(identity))[:12]
    run_dir = output_root / run_id
    result_path = run_dir / "result.json"
    if result_path.is_file():
        print(result_path.read_text(encoding="utf-8"), end="")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "identity.json", identity)

    import torch

    configure_deterministic_torch(torch)
    tokenizer = load_tokenizer(registry)
    flattened_calls = []
    spans = []
    for group in groups:
        start = len(flattened_calls)
        flattened_calls.extend(group["safety_calls"])
        safety_stop = len(flattened_calls)
        flattened_calls.extend(group["utility_calls"])
        spans.append((start, safety_stop, len(flattened_calls)))
    prepared, answer_tokens = prepare_prompts(
        tokenizer=tokenizer,
        calls=flattened_calls,
        fixed_chat_date=safety_protocol["fixed_chat_date"],
        chat_template_kwargs=registry_chat_template_kwargs(registry),
        max_input_tokens=inference["max_input_tokens"],
    )

    load_started = time.perf_counter()
    model = load_model(registry)
    load_seconds = time.perf_counter() - load_started
    resident_model_bytes = int(torch.cuda.memory_allocated())

    def score_group(index: int) -> None:
        start, safety_stop, stop = spans[index]
        score_allowed_label_logits_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prepared[start:safety_stop],
            answer_tokens=answer_tokens,
            max_input_tokens=inference["max_input_tokens"],
        )
        score_allowed_label_logits_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prepared[safety_stop:stop],
            answer_tokens=answer_tokens,
            max_input_tokens=inference["max_input_tokens"],
        )

    for index in range(min(args.warmup_updates, len(groups))):
        score_group(index)
    torch.cuda.synchronize()

    observations = []
    for index, group in enumerate(groups):
        start, _, stop = spans[index]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        score_group(index)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        observations.append(
            {
                "test_index": group["test_index"],
                "family_id": group["family_id"],
                "sector": group["sector"],
                "candidate_role": group["candidate_role"],
                "latency_seconds": elapsed,
                "input_tokens": sum(
                    len(prompt.input_token_ids) for prompt in prepared[start:stop]
                ),
                "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
                "incremental_peak_gpu_bytes": max(
                    0, int(torch.cuda.max_memory_allocated()) - resident_model_bytes
                ),
            }
        )
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "completed_updates": index + 1,
                    "total_updates": len(groups),
                    "latency_seconds": round(elapsed, 4),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    latencies = [float(row["latency_seconds"]) for row in observations]
    input_tokens = [int(row["input_tokens"]) for row in observations]
    result = {
        "schema_version": "1.0.0",
        "artifact_type": "anchorrc_update_latency_result_v1",
        "run_id": run_id,
        "identity_sha256": sha256_file(run_dir / "identity.json"),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "environment": environment,
        },
        "model_load_seconds": load_seconds,
        "resident_model_bytes": resident_model_bytes,
        "latency_seconds": dict(latency_summary(latencies)),
        "input_tokens_per_update": {
            "mean": sum(input_tokens) / len(input_tokens),
            "minimum": min(input_tokens),
            "maximum": max(input_tokens),
        },
        "max_peak_gpu_bytes": max(int(row["peak_gpu_bytes"]) for row in observations),
        "max_incremental_peak_gpu_bytes": max(
            int(row["incremental_peak_gpu_bytes"]) for row in observations
        ),
        "observations": observations,
        "claim_scope": (
            "Warm-model latency for one catalog update using six safety views and four "
            "utility views in two no-generation forward passes; model loading is reported "
            "separately and no network or tool-execution latency is included."
        ),
    }
    _write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
