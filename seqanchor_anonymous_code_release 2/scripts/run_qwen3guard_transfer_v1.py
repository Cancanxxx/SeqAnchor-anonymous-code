#!/usr/bin/env python3
"""Run resumable Qwen3Guard transfer inference on frozen action candidates."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nontransitive_safety.local_hf_registry import (
    snapshot_files,
    snapshot_tree_identity,
    validate_weight_artifacts,
)
from nontransitive_safety.provenance import canonical_json_bytes, sha256_bytes, sha256_file
from option_set_instability.qwen3guard_transfer_v1 import (
    ACTION_CONTEXT_TEMPLATE,
    QWEN3GUARD_MODEL_ID,
    QWEN3GUARD_REVISION,
    build_prompt_plan,
    parse_qwen3guard_output,
    passes_guard,
)

REPO = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError(f"partial JSONL record: {path}")
    values = [json.loads(line) for line in raw.splitlines()]
    if any(not isinstance(value, Mapping) for value in values):
        raise RuntimeError(f"non-object JSONL row: {path}")
    return values


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"existing artifact differs: {path}")
    if not path.exists():
        path.write_bytes(payload)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"existing artifact differs: {path}")
    if not path.exists():
        path.write_bytes(payload)


def _snapshot_record(snapshot: Path) -> Mapping[str, Any]:
    files = snapshot_files(snapshot)
    validate_weight_artifacts(files)
    metadata = {
        relative: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for relative, path in files.items()
    }
    return {
        "model_id": QWEN3GUARD_MODEL_ID,
        "revision": QWEN3GUARD_REVISION,
        "snapshot_tree_sha256": snapshot_tree_identity(metadata),
        "files": metadata,
    }


def _metrics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    result: dict[str, Any] = {
        "rows": len(rows),
        "parse_failures": sum(not row["parse_valid"] for row in rows),
        "predicted_labels": dict(Counter(str(row["prediction"]) for row in rows)),
    }
    for mode in ("strict", "loose"):
        unsafe = [row for row in rows if float(row["truth_label"]) == 1.0]
        safe = [row for row in rows if float(row["truth_label"]) == 0.0]
        unsafe_passes = sum(bool(row[f"passes_{mode}"]) for row in unsafe)
        safe_blocks = sum(not bool(row[f"passes_{mode}"]) for row in safe)
        correct = len(rows) - unsafe_passes - safe_blocks
        result[mode] = {
            "correct": correct,
            "accuracy": None if not rows else correct / len(rows),
            "unsafe_total": len(unsafe),
            "unsafe_passed": unsafe_passes,
            "unsafe_pass_rate": None if not unsafe else unsafe_passes / len(unsafe),
            "safe_total": len(safe),
            "safe_blocked": safe_blocks,
            "safe_block_rate": None if not safe else safe_blocks / len(safe),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--model-snapshot", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup-rows", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("batch_size", "max_input_tokens", "max_new_tokens"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.warmup_rows < 0:
        raise ValueError("--warmup-rows cannot be negative")
    if _git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("Qwen3Guard inference requires a clean source tree")

    rows_path = args.rows.resolve()
    snapshot = args.model_snapshot.resolve()
    if snapshot.name != QWEN3GUARD_REVISION or not snapshot.is_dir():
        raise RuntimeError("Qwen3Guard snapshot path/revision differs")
    rows = _load_jsonl(rows_path)
    required = {
        "row_id",
        "benchmark",
        "source_relative_path",
        "source_sha256",
        "row_index",
        "truth_label",
        "instruction",
        "history",
        "current_action",
        "env_info",
    }
    if not rows or any(not required <= set(row) for row in rows):
        raise RuntimeError("Qwen3Guard transfer rows are malformed")
    if any(float(row["truth_label"]) not in {0.0, 1.0} for row in rows):
        raise RuntimeError("Qwen3Guard transfer requires binary candidate truth")

    model_record = _snapshot_record(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    plan = build_prompt_plan(rows, tokenizer, max_input_tokens=args.max_input_tokens)
    if args.limit is not None:
        plan = plan[: args.limit]
    identity = {
        "schema_version": "1.0.0",
        "artifact_type": "qwen3guard_action_transfer_identity_v1",
        "model": model_record,
        "rows_sha256": sha256_file(rows_path),
        "code_commit": _git("rev-parse", "HEAD"),
        "interface": {
            "action_context_template": ACTION_CONTEXT_TEMPLATE,
            "action_context_template_sha256": sha256_bytes(
                ACTION_CONTEXT_TEMPLATE.encode("utf-8")
            ),
            "chat_interface": "official_prompt_moderation_chat_template",
            "decision_rules": {
                "strict": "only_valid_Safe_passes",
                "loose": "valid_Safe_or_Controversial_passes",
                "parse_failure": "fail_closed",
            },
        },
        "settings": {
            "batch_size": args.batch_size,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "limit": args.limit,
            "warmup_rows": args.warmup_rows,
            "plan_order": "input_tokens_then_source_path_then_row_index",
        },
        "planned_row_ids_sha256": sha256_bytes(
            canonical_json_bytes([row["row_id"] for row in plan])
        ),
    }
    run_id = sha256_bytes(canonical_json_bytes(identity))[:12]
    run_dir = args.output_root.resolve() / run_id
    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "rows": len(plan),
                    "truth_labels": dict(Counter(str(row["truth_label"]) for row in plan)),
                    "input_tokens": {
                        "min": min(row["input_tokens"] for row in plan),
                        "max": max(row["input_tokens"] for row in plan),
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
    output_path = run_dir / "predictions.jsonl"
    completion_path = run_dir / "completion.json"
    _write_json(identity_path, identity)
    _write_jsonl(
        plan_path,
        [{key: value for key, value in row.items() if key != "rendered_prompt"} for row in plan],
    )
    _write_json(
        metadata_path,
        {
            "schema_version": "1.0.0",
            "artifact_type": "qwen3guard_action_transfer_metadata_v1",
            "run_id": run_id,
            "identity_sha256": sha256_file(identity_path),
            "plan_sha256": sha256_file(plan_path),
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
            "environment": {
                name: os.getenv(name)
                for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM")
            },
        },
    )
    existing = _load_jsonl(output_path)
    if [row.get("row_id") for row in existing] != [
        row["row_id"] for row in plan[: len(existing)]
    ]:
        raise RuntimeError("existing Qwen3Guard output is not the planned prefix")
    if len(existing) == len(plan) and completion_path.exists():
        print(completion_path.read_text(encoding="utf-8"), end="")
        return

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.eval()
    model.generation_config.do_sample = False
    for name in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, name):
            setattr(model.generation_config, name, None)
    load_seconds = time.perf_counter() - load_started

    warmup_seconds = 0.0
    if args.warmup_rows:
        warmup_started = time.perf_counter()
        warmup = plan[: min(args.warmup_rows, len(plan))]
        for begin in range(0, len(warmup), args.batch_size):
            encoded = tokenizer(
                [row["rendered_prompt"] for row in warmup[begin : begin + args.batch_size]],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
            with torch.inference_mode():
                model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - warmup_started

    completed_prefix = len(existing)
    start_index = completed_prefix - (completed_prefix % args.batch_size)
    with output_path.open("ab") as handle:
        for batch_index, begin in enumerate(
            range(start_index, len(plan), args.batch_size),
            start=start_index // args.batch_size,
        ):
            batch = plan[begin : begin + args.batch_size]
            encoded = tokenizer(
                [row["rendered_prompt"] for row in batch],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            elapsed = time.perf_counter() - started
            width = encoded["input_ids"].shape[1]
            output_ids = generated[:, width:]
            texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            realized = []
            for position, (planned, text, ids) in enumerate(zip(batch, texts, output_ids)):
                parsed = parse_qwen3guard_output(text)
                realized.append(
                    {
                        "schema_version": "1.0.0",
                        "artifact_type": "qwen3guard_action_transfer_prediction_v1",
                        "ordinal": begin + position,
                        "row_id": planned["row_id"],
                        "benchmark": planned["benchmark"],
                        "source_relative_path": planned["source_relative_path"],
                        "source_sha256": planned["source_sha256"],
                        "row_index": planned["row_index"],
                        "family_id": planned["family_id"],
                        "candidate_role": planned["candidate_role"],
                        "truth_label": planned["truth_label"],
                        "prediction": parsed["label"],
                        "categories": parsed["categories"],
                        "parse_valid": parsed["valid"],
                        "passes_strict": passes_guard(parsed, mode="strict"),
                        "passes_loose": passes_guard(parsed, mode="loose"),
                        "prompt_sha256": planned["prompt_sha256"],
                        "input_tokens": planned["input_tokens"],
                        "generated_tokens": int((ids != tokenizer.pad_token_id).sum().item()),
                        "output_text": text,
                        "output_text_sha256": sha256_bytes(text.encode("utf-8")),
                        "batch_index": batch_index,
                        "batch_position": position,
                        "batch_size_realized": len(batch),
                        "batch_seconds": elapsed,
                        "batch_peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                    }
                )
            overlap = max(0, completed_prefix - begin)
            stable = set(realized[0]) - {"batch_seconds", "batch_peak_gpu_bytes"}
            for position in range(overlap):
                if any(
                    existing[begin + position].get(key) != realized[position].get(key)
                    for key in stable
                ):
                    raise RuntimeError("replayed Qwen3Guard partial batch differs")
            missing = realized[overlap:]
            if missing:
                handle.write(b"".join(canonical_json_bytes(row) + b"\n" for row in missing))
                handle.flush()
                os.fsync(handle.fileno())
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
    if [row["row_id"] for row in completed] != [row["row_id"] for row in plan]:
        raise RuntimeError("completed Qwen3Guard output order differs")
    batch_indices = sorted({int(row["batch_index"]) for row in completed})
    completion = {
        "schema_version": "1.0.0",
        "artifact_type": "qwen3guard_action_transfer_completion_v1",
        "run_id": run_id,
        "identity_sha256": sha256_file(identity_path),
        "plan_sha256": sha256_file(plan_path),
        "metadata_sha256": sha256_file(metadata_path),
        "predictions_sha256": sha256_file(output_path),
        "rows": len(completed),
        "load_seconds": load_seconds,
        "metrics": _metrics(completed),
        "input_tokens": sum(int(row["input_tokens"]) for row in completed),
        "generated_tokens": sum(int(row["generated_tokens"]) for row in completed),
        "batch_seconds_sum": sum(
            max(float(row["batch_seconds"]) for row in completed if int(row["batch_index"]) == index)
            for index in batch_indices
        ),
        "max_peak_gpu_bytes": max(int(row["batch_peak_gpu_bytes"]) for row in completed),
        "warmup": {
            "rows": min(args.warmup_rows, len(plan)),
            "seconds": warmup_seconds,
            "excluded_from_metrics": True,
        },
        "transfer_scope": "general_prompt_moderator_applied_to_serialized_agent_action",
        "human_annotations_collected": False,
    }
    _write_json(completion_path, completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

