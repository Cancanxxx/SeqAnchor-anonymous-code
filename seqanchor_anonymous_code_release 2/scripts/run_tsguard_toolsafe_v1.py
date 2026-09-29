#!/usr/bin/env python3
"""Run a resumable, pinned TS-Guard reproduction on TS-Bench trajectories."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nontransitive_safety.provenance import canonical_json_bytes, sha256_bytes, sha256_file
from option_set_instability.toolsafe_tsguard_v1 import (
    SOURCE_RELATIVE_PATHS,
    TSGUARD_REVISION,
    build_prompt_plan,
    load_source_rows,
    load_toolsafe_interface,
    parse_tsguard_output,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"existing artifact differs: {path}")
        return
    path.write_bytes(payload)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"existing artifact differs: {path}")
        return
    path.write_bytes(payload)


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError("partial final JSONL record; refusing silent truncation")
    rows = []
    for index, line in enumerate(raw.splitlines()):
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise RuntimeError(f"non-object output row {index}")
        rows.append(value)
    return rows


def _source_selection(benchmark: str) -> tuple[str, ...]:
    if benchmark == "all":
        return SOURCE_RELATIVE_PATHS
    prefix = {
        "agentharm": "agentharm-traj/",
        "agentdojo": "agentdojo-traj/",
        "agent_safety_bench": "asb-traj/",
    }[benchmark]
    return tuple(value for value in SOURCE_RELATIVE_PATHS if value.startswith(prefix))


def _public_plan_rows(plan: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "rendered_prompt"}
        for row in plan
    ]


def _confusion(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    valid = [row for row in rows if row["parse_valid"]]
    exact = Counter((str(row["truth_label"]), str(row["prediction"])) for row in valid)

    def mapped(value: float, mode: str) -> int:
        if mode == "strict":
            return 0 if value == 0.0 else 1
        if mode == "loose":
            return 1 if value == 1.0 else 0
        raise ValueError(mode)

    binary = {}
    for mode in ("strict", "loose"):
        correct = sum(
            mapped(float(row["truth_label"]), mode)
            == mapped(float(row["prediction"]), mode)
            for row in valid
        )
        binary[mode] = {
            "correct": correct,
            "total": len(valid),
            "accuracy": None if not valid else correct / len(valid),
        }
    exact_correct = sum(row["truth_label"] == row["prediction"] for row in valid)
    return {
        "rows": len(rows),
        "valid_predictions": len(valid),
        "parse_failures": len(rows) - len(valid),
        "exact": {
            "correct": exact_correct,
            "total": len(valid),
            "accuracy": None if not valid else exact_correct / len(valid),
            "confusion": {
                f"truth_{truth}__pred_{prediction}": count
                for (truth, prediction), count in sorted(exact.items())
            },
        },
        "binary": binary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolsafe-repo", required=True, type=Path)
    parser.add_argument("--model-snapshot", required=True, type=Path)
    parser.add_argument("--model-revision", default=TSGUARD_REVISION)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--benchmark",
        choices=(
            "all",
            "agentharm",
            "agentdojo",
            "agent_safety_bench",
            "sequential_anchorrc",
        ),
        default="all",
    )
    parser.add_argument(
        "--custom-rows",
        type=Path,
        help="Frozen adapter rows used only with --benchmark sequential_anchorrc.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--warmup-rows",
        type=int,
        default=0,
        help="Run and discard this many leading rows after model load; omitted from identity when zero.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_revision != TSGUARD_REVISION:
        raise RuntimeError("TS-Guard revision differs from the frozen protocol")
    for name in ("batch_size", "max_input_tokens", "max_new_tokens"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.warmup_rows < 0:
        raise ValueError("--warmup-rows cannot be negative")

    toolsafe_repo = args.toolsafe_repo.resolve()
    model_snapshot = args.model_snapshot.resolve()
    output_root = args.output_root.resolve()
    if _git(toolsafe_repo, "rev-parse", "HEAD") != "46358fa424a927a895c6c8322f99032c4eb5155e":
        raise RuntimeError("ToolSafe checkout differs from the frozen commit")
    if _git(toolsafe_repo, "status", "--porcelain"):
        raise RuntimeError("ToolSafe checkout is not clean")
    if model_snapshot.name != args.model_revision or not model_snapshot.is_dir():
        raise RuntimeError("TS-Guard snapshot path/revision mismatch")

    interface, upstream_parser = load_toolsafe_interface(toolsafe_repo)
    if args.benchmark == "sequential_anchorrc":
        if args.custom_rows is None:
            raise RuntimeError("sequential_anchorrc requires --custom-rows")
        custom_path = args.custom_rows.resolve()
        rows = tuple(_load_jsonl(custom_path))
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
            raise RuntimeError("custom TS-Guard rows are malformed")
        if any(row["benchmark"] != "sequential_anchorrc" for row in rows):
            raise RuntimeError("custom TS-Guard benchmark binding differs")
        source_paths = (str(custom_path),)
        source_hashes = {str(custom_path): sha256_file(custom_path)}
    else:
        if args.custom_rows is not None:
            raise RuntimeError("--custom-rows is only valid for sequential_anchorrc")
        source_paths = _source_selection(args.benchmark)
        rows = load_source_rows(
            toolsafe_repo / "TS-Bench",
            source_relative_paths=source_paths,
        )
        source_hashes = {
            value: sha256_file(toolsafe_repo / "TS-Bench" / value)
            for value in source_paths
        }
    tokenizer = AutoTokenizer.from_pretrained(model_snapshot, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    plan = build_prompt_plan(
        rows,
        interface,
        tokenizer,
        max_input_tokens=args.max_input_tokens,
    )
    if args.limit is not None:
        plan = plan[: args.limit]
    public_plan = _public_plan_rows(plan)
    identity = {
        "schema_version": "1.0.0",
        "artifact_type": "tsguard_toolsafe_run_identity_v1",
        "interface": interface.as_record(),
        "model_revision": args.model_revision,
        "benchmark": args.benchmark,
        "source_paths": list(source_paths),
        "source_hashes": source_hashes,
        "settings": {
            "batch_size": args.batch_size,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "limit": args.limit,
            "plan_order": "input_tokens_then_source_path_then_row_index",
        },
        "planned_row_ids_sha256": sha256_bytes(
            canonical_json_bytes([row["row_id"] for row in plan])
        ),
    }
    if args.warmup_rows:
        identity["settings"]["warmup_rows"] = args.warmup_rows
    run_id = sha256_bytes(canonical_json_bytes(identity))[:12]
    run_dir = output_root / run_id
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
    _write_jsonl(plan_path, public_plan)
    metadata = {
        "schema_version": "1.0.0",
        "artifact_type": "tsguard_toolsafe_run_metadata_v1",
        "run_id": run_id,
        "identity_sha256": sha256_file(identity_path),
        "plan_sha256": sha256_file(plan_path),
        "planned_rows": len(plan),
        "model_snapshot": str(model_snapshot),
        "model_revision": args.model_revision,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "environment": {
            "HF_HUB_OFFLINE": os.getenv("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.getenv("TRANSFORMERS_OFFLINE"),
            "TOKENIZERS_PARALLELISM": os.getenv("TOKENIZERS_PARALLELISM"),
        },
    }
    _write_json(metadata_path, metadata)
    existing = _load_jsonl(output_path)
    if [row.get("row_id") for row in existing] != [
        row["row_id"] for row in plan[: len(existing)]
    ]:
        raise RuntimeError("existing TS-Guard output is not the exact planned prefix")
    if len(existing) > len(plan):
        raise RuntimeError("existing output is longer than the plan")
    if len(existing) == len(plan) and completion_path.exists():
        print(completion_path.read_text(encoding="utf-8"), end="")
        return

    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_snapshot,
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
    load_seconds = time.perf_counter() - load_start

    warmup_seconds = 0.0
    if args.warmup_rows:
        warmup_plan = plan[: min(args.warmup_rows, len(plan))]
        warmup_started = time.perf_counter()
        for begin in range(0, len(warmup_plan), args.batch_size):
            batch = warmup_plan[begin : begin + args.batch_size]
            inputs = tokenizer(
                [row["rendered_prompt"] for row in batch],
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
            with torch.inference_mode():
                model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - warmup_started

    completed_prefix = len(existing)
    start_index = (completed_prefix // args.batch_size) * args.batch_size
    with output_path.open("ab") as handle:
        for batch_index, begin in enumerate(
            range(start_index, len(plan), args.batch_size),
            start=start_index // args.batch_size,
        ):
            batch = plan[begin : begin + args.batch_size]
            prompts = [row["rendered_prompt"] for row in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            batch_seconds = time.perf_counter() - started
            input_width = inputs["input_ids"].shape[1]
            output_ids = generated[:, input_width:]
            texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            realized_rows = []
            for position, (planned, text, ids) in enumerate(zip(batch, texts, output_ids)):
                parsed = parse_tsguard_output(text, upstream_parser)
                generated_tokens = int((ids != tokenizer.pad_token_id).sum().item())
                row = {
                    "schema_version": "1.0.0",
                    "artifact_type": "tsguard_toolsafe_prediction_v1",
                    "ordinal": begin + position,
                    "row_id": planned["row_id"],
                    "benchmark": planned["benchmark"],
                    "source_relative_path": planned["source_relative_path"],
                    "source_sha256": planned["source_sha256"],
                    "row_index": planned["row_index"],
                    "truth_label": planned["truth_label"],
                    "prediction": parsed["prediction"],
                    "parse_valid": parsed["valid"],
                    "parser_details": parsed["details"],
                    "parser_error": parsed["raw_parser"],
                    "prompt_sha256": planned["prompt_sha256"],
                    "input_tokens": planned["input_tokens"],
                    "generated_tokens": generated_tokens,
                    "output_text": text,
                    "output_text_sha256": sha256_bytes(text.encode("utf-8")),
                    "batch_index": batch_index,
                    "batch_position": position,
                    "batch_size_realized": len(batch),
                    "batch_seconds": batch_seconds,
                    "batch_peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                }
                realized_rows.append(row)
            overlap = max(0, completed_prefix - begin)
            comparison_keys = {
                "ordinal",
                "row_id",
                "truth_label",
                "prediction",
                "parse_valid",
                "parser_details",
                "parser_error",
                "prompt_sha256",
                "input_tokens",
                "generated_tokens",
                "output_text",
                "output_text_sha256",
                "batch_index",
                "batch_position",
                "batch_size_realized",
            }
            for position in range(overlap):
                observed = existing[begin + position]
                replayed = realized_rows[position]
                if any(observed.get(key) != replayed.get(key) for key in comparison_keys):
                    raise RuntimeError("replayed partial batch differs from existing prefix")
            missing = realized_rows[overlap:]
            if missing:
                handle.write(b"".join(canonical_json_bytes(row) + b"\n" for row in missing))
                handle.flush()
                os.fsync(handle.fileno())
            print(
                json.dumps(
                    {
                        "run_id": run_id,
                        "completed": max(completed_prefix, begin + len(batch)),
                        "total": len(plan),
                        "batch_seconds": round(batch_seconds, 3),
                        "max_input_tokens": max(row["input_tokens"] for row in batch),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    completed = _load_jsonl(output_path)
    if [row["row_id"] for row in completed] != [row["row_id"] for row in plan]:
        raise RuntimeError("completed output order differs from the plan")
    by_benchmark = defaultdict(list)
    for row in completed:
        by_benchmark[row["benchmark"]].append(row)
    completion = {
        "schema_version": "1.0.0",
        "artifact_type": "tsguard_toolsafe_completion_v1",
        "run_id": run_id,
        "identity_sha256": sha256_file(identity_path),
        "plan_sha256": sha256_file(plan_path),
        "metadata_sha256": sha256_file(metadata_path),
        "predictions_sha256": sha256_file(output_path),
        "load_seconds": load_seconds,
        "metrics": _confusion(completed),
        "metrics_by_benchmark": {
            key: _confusion(value) for key, value in sorted(by_benchmark.items())
        },
        "input_tokens": sum(int(row["input_tokens"]) for row in completed),
        "generated_tokens": sum(int(row["generated_tokens"]) for row in completed),
        "batch_seconds_sum": sum(
            max(float(row["batch_seconds"]) for row in completed if row["batch_index"] == index)
            for index in sorted({int(row["batch_index"]) for row in completed})
        ),
        "max_peak_gpu_bytes": max(int(row["batch_peak_gpu_bytes"]) for row in completed),
    }
    if args.warmup_rows:
        completion["warmup"] = {
            "rows": min(args.warmup_rows, len(plan)),
            "seconds": warmup_seconds,
            "excluded_from_metrics": True,
        }
    _write_json(completion_path, completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
