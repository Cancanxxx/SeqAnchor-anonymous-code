#!/usr/bin/env python3
"""Run resumable growing-catalog reselection on one pinned local model."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from nontransitive_safety.chat_rendering import render_chat_prompt
from nontransitive_safety.local_hf_registry import (
    load_model,
    load_model_registry,
    load_tokenizer,
    registry_chat_template_kwargs,
    validate_runtime_versions,
)
from nontransitive_safety.provenance import canonical_json_bytes, sha256_bytes, sha256_file
from option_set_instability.sequential_full_catalog_v1 import LABELS, build_choice_calls

FIXED_DATE = "2026-09-01"
REPO = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    raw = path.read_bytes()
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


def _prompt_ids(tokenizer: Any, prompt: str) -> tuple[int, ...]:
    values = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not values or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise RuntimeError("tokenizer returned invalid prompt IDs")
    return tuple(values)


def _prepare(
    calls: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    chat_kwargs: Mapping[str, Any],
    max_input_tokens: int,
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, int]]:
    prepared = []
    contracts = set()
    for call in calls:
        prompt = render_chat_prompt(
            tokenizer,
            call["messages"],
            fixed_date=FIXED_DATE,
            chat_template_kwargs=chat_kwargs,
        )
        prefix = _prompt_ids(tokenizer, prompt)
        if len(prefix) > max_input_tokens:
            raise RuntimeError(f"full-catalog prompt exceeds limit: {call['call_id']}")
        token_ids = {}
        for label in call["allowed_labels"]:
            combined = _prompt_ids(tokenizer, prompt + label)
            if len(combined) != len(prefix) + 1 or combined[:-1] != prefix:
                raise RuntimeError(f"answer label is not one-token continuation: {label}")
            token_ids[label] = combined[-1]
        contracts.add(tuple(sorted(token_ids.items())))
        prepared.append(
            {
                **{key: value for key, value in call.items() if key != "messages"},
                "rendered_prompt": prompt,
                "input_token_ids": prefix,
                "rendered_prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                "input_token_ids_sha256": sha256_bytes(
                    canonical_json_bytes(list(prefix))
                ),
                "input_tokens": len(prefix),
            }
        )
    # The first n labels must retain identical token IDs across all menu sizes.
    answer_tokens: dict[str, int] = {}
    for contract in contracts:
        for label, token_id in contract:
            if label in answer_tokens and answer_tokens[label] != token_id:
                raise RuntimeError("answer label token ID drifts across prompts")
            answer_tokens[label] = token_id
    if set(answer_tokens) != set(LABELS):
        raise RuntimeError("full A-H answer-token contract is absent")
    return tuple(
        sorted(prepared, key=lambda row: (row["input_tokens"], row["call_id"]))
    ), answer_tokens


def _last_logit_kwargs(model: Any, kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    parameters = inspect.signature(model.forward).parameters
    if "logits_to_keep" in parameters:
        return {**kwargs, "logits_to_keep": 1}
    if "num_logits_to_keep" in parameters:
        return {**kwargs, "num_logits_to_keep": 1}
    return dict(kwargs)


def _score(
    model: Any,
    tokenizer: Any,
    batch: Sequence[Mapping[str, Any]],
    answer_tokens: Mapping[str, int],
) -> tuple[list[Mapping[str, float]], float, int]:
    encoded = tokenizer(
        [row["rendered_prompt"] for row in batch],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(
            **_last_logit_kwargs(
                model,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "use_cache": False,
                    "return_dict": True,
                },
            )
        )
    elapsed = time.perf_counter() - started
    token_ids = [answer_tokens[label] for label in LABELS]
    values = output.logits[:, -1, token_ids].detach().float().cpu().tolist()
    rows = []
    for vector in values:
        row = {label: float(value) for label, value in zip(LABELS, vector)}
        if any(not math.isfinite(value) for value in row.values()):
            raise RuntimeError("nonfinite full-catalog logit")
        rows.append(row)
    return rows, elapsed, torch.cuda.max_memory_allocated()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-registry", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--limit-streams", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_input_tokens <= 0:
        raise ValueError("inference limits must be positive")
    if _git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("full-catalog inference requires a clean source tree")
    streams_path = args.streams.resolve()
    manifest_path = args.manifest.resolve()
    registry_path = args.model_registry.resolve()
    streams = _load_jsonl(streams_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_bytes(streams_path.read_bytes()) != manifest["public_sha256"]:
        raise RuntimeError("full-catalog public stream hash differs")
    if args.limit_streams is not None:
        streams = streams[: args.limit_streams]
    registry = load_model_registry(registry_path, verify_files=True, require_frozen=True)
    validate_runtime_versions(registry)
    tokenizer = load_tokenizer(registry)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    calls = build_choice_calls(streams, registry["candidate_snapshot_id"])
    plan, answer_tokens = _prepare(
        calls,
        tokenizer,
        registry_chat_template_kwargs(registry),
        args.max_input_tokens,
    )
    identity = {
        "schema_version": "1.0.0",
        "artifact_type": "sequential_full_catalog_run_identity_v1",
        "streams_sha256": sha256_file(streams_path),
        "manifest_sha256": sha256_file(manifest_path),
        "model_registry_sha256": sha256_file(registry_path),
        "snapshot_id": registry["candidate_snapshot_id"],
        "model_revision": registry["revision"],
        "code_commit": _git("rev-parse", "HEAD"),
        "answer_token_ids": answer_tokens,
        "fixed_date": FIXED_DATE,
        "settings": {
            "batch_size": args.batch_size,
            "max_input_tokens": args.max_input_tokens,
            "limit_streams": args.limit_streams,
            "forward_mode": "no_cache_single_next_token_A_to_H_logits",
        },
        "planned_call_ids_sha256": sha256_bytes(
            canonical_json_bytes([row["call_id"] for row in plan])
        ),
    }
    run_id = sha256_bytes(canonical_json_bytes(identity))[:12]
    run_dir = args.output_root.resolve() / run_id / registry["candidate_snapshot_id"]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "streams": len(streams),
                    "calls": len(plan),
                    "input_tokens": {
                        "min": min(row["input_tokens"] for row in plan),
                        "max": max(row["input_tokens"] for row in plan),
                    },
                    "run_dir": str(run_dir),
                },
                indent=2,
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
            {
                key: value
                for key, value in row.items()
                if key not in {"rendered_prompt", "input_token_ids"}
            }
            for row in plan
        ],
    )
    _write_json(
        metadata_path,
        {
            "schema_version": "1.0.0",
            "artifact_type": "sequential_full_catalog_run_metadata_v1",
            "run_id": run_id,
            "identity_sha256": sha256_file(identity_path),
            "plan_sha256": sha256_file(plan_path),
            "model_id": registry["model_id"],
            "snapshot_id": registry["candidate_snapshot_id"],
            "runtime": registry["runtime"],
            "environment": {
                name: os.getenv(name)
                for name in (
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "TOKENIZERS_PARALLELISM",
                    "CUBLAS_WORKSPACE_CONFIG",
                )
            },
        },
    )
    existing = _load_jsonl(output_path) if output_path.exists() else []
    if [row.get("call_id") for row in existing] != [
        row["call_id"] for row in plan[: len(existing)]
    ]:
        raise RuntimeError("existing full-catalog output is not the planned prefix")
    if len(existing) == len(plan) and completion_path.exists():
        print(completion_path.read_text(encoding="utf-8"), end="")
        return

    torch.use_deterministic_algorithms(True)
    load_started = time.perf_counter()
    model = load_model(registry)
    model.eval()
    load_seconds = time.perf_counter() - load_started
    completed_prefix = len(existing)
    start_index = completed_prefix - (completed_prefix % args.batch_size)
    for batch_index, begin in enumerate(
        range(start_index, len(plan), args.batch_size), start=start_index // args.batch_size
    ):
        batch = plan[begin : begin + args.batch_size]
        scores, elapsed, peak = _score(model, tokenizer, batch, answer_tokens)
        realized = []
        for position, (planned, all_scores) in enumerate(zip(batch, scores)):
            allowed = {
                label: all_scores[label] for label in planned["allowed_labels"]
            }
            realized.append(
                {
                    "schema_version": "1.0.0",
                    "artifact_type": "sequential_full_catalog_logits_v1",
                    "ordinal": begin + position,
                    "call_id": planned["call_id"],
                    "stream_id": planned["stream_id"],
                    "family_id": planned["family_id"],
                    "test_index": planned["test_index"],
                    "sector": planned["sector"],
                    "step": planned["step"],
                    "view_index": planned["view_index"],
                    "candidate_id_by_label": planned["candidate_id_by_label"],
                    "allowed_label_logits": allowed,
                    "argmax_label": max(sorted(allowed), key=lambda label: allowed[label]),
                    "input_tokens": planned["input_tokens"],
                    "rendered_prompt_sha256": planned["rendered_prompt_sha256"],
                    "batch_index": batch_index,
                    "batch_position": position,
                    "batch_size_realized": len(batch),
                    "batch_seconds": elapsed,
                    "batch_peak_gpu_bytes": peak,
                }
            )
        overlap = max(0, completed_prefix - begin)
        stable = set(realized[0]) - {"batch_seconds", "batch_peak_gpu_bytes"}
        for position in range(overlap):
            if any(
                existing[begin + position].get(key) != realized[position].get(key)
                for key in stable
            ):
                raise RuntimeError("replayed full-catalog partial batch differs")
        _append_jsonl(output_path, realized[overlap:])
        print(
            json.dumps(
                {
                    "run_id": run_id,
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
        raise RuntimeError("completed full-catalog output order differs")
    batches = sorted({int(row["batch_index"]) for row in completed})
    completion = {
        "schema_version": "1.0.0",
        "artifact_type": "sequential_full_catalog_completion_v1",
        "run_id": run_id,
        "snapshot_id": registry["candidate_snapshot_id"],
        "model_id": registry["model_id"],
        "identity_sha256": sha256_file(identity_path),
        "plan_sha256": sha256_file(plan_path),
        "metadata_sha256": sha256_file(metadata_path),
        "logits_sha256": sha256_file(output_path),
        "rows": len(completed),
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
