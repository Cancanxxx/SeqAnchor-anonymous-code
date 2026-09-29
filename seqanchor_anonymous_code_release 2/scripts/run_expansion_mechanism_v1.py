#!/usr/bin/env python3
"""Run resumable, deterministic StateExpansion mechanism diagnostics."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from nontransitive_safety.chat_rendering import render_chat_prompt
from nontransitive_safety.provenance import canonical_json_bytes, sha256_bytes, sha256_file
from option_set_instability.expansion_mechanism_v1 import LABELS, build_call_plan, build_families
from option_set_instability.state_expansion_180 import load_workflow_specs

REPO = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return value


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    raw = path.read_bytes() if path.exists() else b""
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError(f"partial JSONL record: {path}")
    return [json.loads(line) for line in raw.splitlines()]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"existing artifact differs: {path}")
    path.write_bytes(payload)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"existing artifact differs: {path}")
    path.write_bytes(payload)


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("ab") as handle:
        handle.write(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
        handle.flush()
        os.fsync(handle.fileno())


def _model(protocol: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    matches = [row for row in protocol["models"] if row["key"] == key]
    if len(matches) != 1:
        raise ValueError(f"model key is not uniquely declared: {key}")
    return matches[0]


def _verify_protocol(protocol_path: Path, protocol: Mapping[str, Any], model: Mapping[str, Any]) -> None:
    if protocol.get("status") != "frozen_before_new_model_inference":
        raise RuntimeError("mechanism protocol is not frozen")
    benchmark = protocol["benchmark"]
    for path_key, hash_key in (
        ("public_path", "public_sha256"),
        ("manifest_path", "manifest_sha256"),
    ):
        path = REPO / benchmark[path_key]
        if sha256_file(path) != benchmark[hash_key]:
            raise RuntimeError(f"frozen benchmark artifact differs: {path}")
    for entry in [*benchmark["private_truth_files"], *benchmark["workflow_spec_files"]]:
        path = REPO / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"frozen mechanism input differs: {path}")
    for entry in protocol["implementation"]:
        path = REPO / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"frozen mechanism implementation differs: {path}")
    snapshot = Path(model["local_snapshot"])
    if not snapshot.is_dir() or snapshot.name != model["revision"]:
        raise RuntimeError("declared local model revision is unavailable")
    if sha256_file(snapshot / "config.json") != model["config_sha256"]:
        raise RuntimeError("model config hash differs")


def _prompt_ids(tokenizer: Any, prompt: str) -> tuple[int, ...]:
    values = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not values or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise RuntimeError("tokenizer returned invalid prompt IDs")
    return tuple(values)


def _prepare(
    calls: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    fixed_date: str,
    chat_template_kwargs: Mapping[str, Any],
    max_input_tokens: int,
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, int]]:
    prepared = []
    token_contracts = set()
    for call in calls:
        prompt = render_chat_prompt(
            tokenizer,
            call["messages"],
            fixed_date=fixed_date,
            chat_template_kwargs=chat_template_kwargs,
        )
        prefix = _prompt_ids(tokenizer, prompt)
        if len(prefix) > max_input_tokens:
            raise RuntimeError(f"mechanism prompt exceeds token limit: {call['call_id']}")
        answer_tokens = {}
        for label in call["allowed_labels"]:
            combined = _prompt_ids(tokenizer, prompt + label)
            if len(combined) != len(prefix) + 1 or combined[:-1] != prefix:
                raise RuntimeError(f"answer label is not a one-token continuation: {label}")
            answer_tokens[label] = combined[-1]
        token_contracts.add(tuple(sorted(answer_tokens.items())))
        prepared.append(
            {
                **{key: value for key, value in call.items() if key != "messages"},
                "rendered_prompt": prompt,
                "input_tokens": len(prefix),
                "rendered_prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            }
        )
    answer_token_ids: dict[str, int] = {}
    for contract in token_contracts:
        for label, token_id in contract:
            if label in answer_token_ids and answer_token_ids[label] != token_id:
                raise RuntimeError("answer token ID changes across prompts")
            answer_token_ids[label] = token_id
    if set(answer_token_ids) != set(LABELS):
        raise RuntimeError("complete A/B/C answer-token contract is absent")
    return (
        tuple(sorted(prepared, key=lambda row: (row["input_tokens"], row["call_id"]))),
        answer_token_ids,
    )


def _last_logit_kwargs(model: Any, kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    parameters = inspect.signature(model.forward).parameters
    if "logits_to_keep" in parameters:
        return {**kwargs, "logits_to_keep": 1}
    if "num_logits_to_keep" in parameters:
        return {**kwargs, "num_logits_to_keep": 1}
    return dict(kwargs)


def _score_batch(
    model: Any,
    tokenizer: Any,
    batch: Sequence[Mapping[str, Any]],
    answer_token_ids: Mapping[str, int],
) -> tuple[list[Mapping[str, float]], float, int]:
    encoded = tokenizer(
        [row["rendered_prompt"] for row in batch],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in encoded.items()}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(
            **_last_logit_kwargs(
                model,
                {**inputs, "use_cache": False, "return_dict": True},
            )
        )
    elapsed = time.perf_counter() - started
    token_ids = [answer_token_ids[label] for label in LABELS]
    values = output.logits[:, -1, token_ids].detach().float().cpu().tolist()
    result = []
    for vector in values:
        row = {label: float(value) for label, value in zip(LABELS, vector)}
        if any(not math.isfinite(value) for value in row.values()):
            raise RuntimeError("model returned a non-finite allowed-label logit")
        result.append(row)
    return result, elapsed, int(torch.cuda.max_memory_allocated())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=REPO / "configs/expansion_mechanism_diagnostics_v1.json"
    )
    parser.add_argument("--model-key", required=True)
    parser.add_argument(
        "--model-snapshot",
        type=Path,
        help="Optional path to the exact local model revision declared by the protocol.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit-families", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = _load_json(protocol_path)
    model_spec = dict(_model(protocol, args.model_key))
    if args.model_snapshot is not None:
        model_spec["local_snapshot"] = str(args.model_snapshot.resolve())
    _verify_protocol(protocol_path, protocol, model_spec)
    frozen_batch_size = int(model_spec["batch_size"])
    if args.batch_size is not None and args.batch_size != frozen_batch_size:
        raise RuntimeError("requested batch size differs from the frozen model setting")
    benchmark = protocol["benchmark"]
    public = _load_jsonl(REPO / benchmark["public_path"])
    private = [
        row
        for entry in benchmark["private_truth_files"]
        for row in _load_jsonl(REPO / entry["path"])
    ]
    specs = load_workflow_specs(
        [REPO / entry["path"] for entry in benchmark["workflow_spec_files"]]
    )
    families = list(build_families(public, private, specs))
    if args.limit_families is not None:
        if args.limit_families <= 0:
            raise ValueError("--limit-families must be positive")
        families = families[: args.limit_families]

    snapshot = Path(model_spec["local_snapshot"])
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, use_fast=True, trust_remote_code=False
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    calls = build_call_plan(families, model_spec["key"])
    plan, answer_token_ids = _prepare(
        calls,
        tokenizer,
        fixed_date=protocol["diagnostic_design"]["fixed_chat_date"],
        chat_template_kwargs=model_spec["chat_template_kwargs"],
        max_input_tokens=int(protocol["diagnostic_design"]["max_input_tokens"]),
    )
    identity = {
        "schema_version": "1.0.0",
        "artifact_type": "expansion_mechanism_run_identity_v1",
        "protocol_sha256": sha256_file(protocol_path),
        "model_key": model_spec["key"],
        "model_id": model_spec["model_id"],
        "model_revision": model_spec["revision"],
        "panel_role": model_spec["panel_role"],
        "answer_token_ids": answer_token_ids,
        "settings": {
            "batch_size": frozen_batch_size,
            "limit_families": args.limit_families,
            "max_input_tokens": protocol["diagnostic_design"]["max_input_tokens"],
            "sampling": False,
            "forward_mode": protocol["diagnostic_design"]["forward_mode"],
        },
        "planned_call_ids_sha256": sha256_bytes(
            canonical_json_bytes([row["call_id"] for row in plan])
        ),
    }
    run_id = sha256_bytes(canonical_json_bytes(identity))[:12]
    run_dir = args.output_root.resolve() / run_id / str(model_spec["key"])
    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "model_key": model_spec["key"],
                    "families": len(families),
                    "calls": len(plan),
                    "input_tokens": {
                        "min": min(row["input_tokens"] for row in plan),
                        "median": sorted(row["input_tokens"] for row in plan)[len(plan) // 2],
                        "max": max(row["input_tokens"] for row in plan),
                        "total": sum(row["input_tokens"] for row in plan),
                    },
                    "run_dir": str(run_dir),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    identity_path = run_dir / "identity.json"
    plan_path = run_dir / "plan.jsonl"
    metadata_path = run_dir / "metadata.json"
    output_path = run_dir / "logits.jsonl"
    completion_path = run_dir / "completion.json"
    _write_json(identity_path, identity)
    _write_jsonl(
        plan_path,
        [
            {key: value for key, value in row.items() if key != "rendered_prompt"}
            for row in plan
        ],
    )
    _write_json(
        metadata_path,
        {
            "schema_version": "1.0.0",
            "artifact_type": "expansion_mechanism_run_metadata_v1",
            "run_id": run_id,
            "identity_sha256": sha256_file(identity_path),
            "plan_sha256": sha256_file(plan_path),
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "environment": {
                name: os.getenv(name)
                for name in (
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "TOKENIZERS_PARALLELISM",
                    "CUBLAS_WORKSPACE_CONFIG",
                    "SLURM_JOB_ID",
                )
            },
        },
    )
    existing = _load_jsonl(output_path)
    if [row.get("call_id") for row in existing] != [
        row["call_id"] for row in plan[: len(existing)]
    ]:
        raise RuntimeError("existing output is not the exact planned prefix")
    if len(existing) == len(plan) and completion_path.exists():
        print(completion_path.read_text(encoding="utf-8"), end="")
        return

    torch.use_deterministic_algorithms(True)
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started
    completed_prefix = len(existing)
    start_index = completed_prefix - (completed_prefix % frozen_batch_size)
    for batch_index, begin in enumerate(
        range(start_index, len(plan), frozen_batch_size),
        start=start_index // frozen_batch_size,
    ):
        batch = plan[begin : begin + frozen_batch_size]
        scores, elapsed, peak = _score_batch(model, tokenizer, batch, answer_token_ids)
        realized = []
        for position, (planned, all_scores) in enumerate(zip(batch, scores)):
            allowed = {label: all_scores[label] for label in planned["allowed_labels"]}
            realized.append(
                {
                    "schema_version": "1.0.0",
                    "artifact_type": "expansion_mechanism_logits_v1",
                    "ordinal": begin + position,
                    **{
                        key: value
                        for key, value in planned.items()
                        if key not in {"rendered_prompt", "input_tokens"}
                    },
                    "input_tokens": planned["input_tokens"],
                    "allowed_label_logits": allowed,
                    "argmax_label": max(sorted(allowed), key=lambda label: allowed[label]),
                    "batch_index": batch_index,
                    "batch_position": position,
                    "batch_size_realized": len(batch),
                    "batch_seconds": elapsed,
                    "batch_peak_gpu_bytes": peak,
                }
            )
        overlap = max(0, completed_prefix - begin)
        stable_keys = set(realized[0]) - {"batch_seconds", "batch_peak_gpu_bytes"}
        for position in range(overlap):
            if any(
                existing[begin + position].get(key) != realized[position].get(key)
                for key in stable_keys
            ):
                raise RuntimeError("replayed partial batch differs")
        _append_jsonl(output_path, realized[overlap:])
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "model_key": model_spec["key"],
                    "completed": begin + len(batch),
                    "total": len(plan),
                    "batch_seconds": round(elapsed, 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    completed = _load_jsonl(output_path)
    if [row["call_id"] for row in completed] != [row["call_id"] for row in plan]:
        raise RuntimeError("completed output order differs")
    batches = sorted({int(row["batch_index"]) for row in completed})
    completion = {
        "schema_version": "1.0.0",
        "artifact_type": "expansion_mechanism_completion_v1",
        "run_id": run_id,
        "model_key": model_spec["key"],
        "model_id": model_spec["model_id"],
        "model_revision": model_spec["revision"],
        "identity_sha256": sha256_file(identity_path),
        "plan_sha256": sha256_file(plan_path),
        "metadata_sha256": sha256_file(metadata_path),
        "logits_sha256": sha256_file(output_path),
        "rows": len(completed),
        "families": len(families),
        "model_load_seconds": load_seconds,
        "input_tokens": sum(int(row["input_tokens"]) for row in completed),
        "batch_seconds_sum": sum(
            max(
                float(row["batch_seconds"])
                for row in completed
                if int(row["batch_index"]) == index
            )
            for index in batches
        ),
        "max_peak_gpu_bytes": max(int(row["batch_peak_gpu_bytes"]) for row in completed),
        "human_annotations_collected": False,
    }
    _write_json(completion_path, completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
