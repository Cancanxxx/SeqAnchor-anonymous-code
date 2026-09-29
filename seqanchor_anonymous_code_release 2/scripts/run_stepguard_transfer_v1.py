#!/usr/bin/env python3
"""Run resumable, source-bound StepGuard inference on frozen workflow updates."""

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
    snapshot_tree_identity,
    validate_weight_artifacts,
)
from nontransitive_safety.provenance import canonical_json_bytes, sha256_bytes, sha256_file
from option_set_instability.stepguard_transfer_v1 import (
    STEPGUARD_MODEL_ID,
    STEPGUARD_MODEL_REVISION,
    build_prompt_plan,
    parse_stepguard_output,
    passes_guard,
    verify_upstream,
)

REPO = Path(__file__).resolve().parents[1]
RUNNER_RELATIVE_PATH = "scripts/run_stepguard_transfer_v1.py"
ADAPTER_RELATIVE_PATH = "src/option_set_instability/stepguard_transfer_v1.py"
PROTOCOL_RELATIVE_PATH = "docs/stepguard_transfer_protocol_v1.md"

# Frozen before any StepGuard transfer prediction was inspected.  This digest
# covers every regular file in the local snapshot except Hugging Face's mutable
# local-dir bookkeeping beneath .cache/; .gitattributes remains included.
STEPGUARD_SNAPSHOT_TREE_SHA256 = (
    "d715185d1413d6d83522a7178b115dd8aaabed2acdb22989d8e73db6954d4c4c"
)
SNAPSHOT_EXCLUSION = "exclude_regular_files_beneath_.cache/_only"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError(f"partial JSONL record: {path}")
    values = []
    for index, line in enumerate(raw.splitlines()):
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise RuntimeError(f"non-object JSONL row {index}: {path}")
        values.append(value)
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


def _snapshot_files(snapshot: Path) -> dict[str, Path]:
    """Return immutable snapshot payload files, excluding local HF cache state."""

    if not snapshot.is_dir() or snapshot.is_symlink():
        raise ValueError(f"missing or symlinked StepGuard snapshot: {snapshot}")
    files: dict[str, Path] = {}
    for path in snapshot.rglob("*"):
        relative = path.relative_to(snapshot)
        if relative.parts and relative.parts[0] == ".cache":
            continue
        if path.is_symlink():
            raise ValueError(f"symlinked StepGuard snapshot payload: {relative.as_posix()}")
        if path.is_file():
            files[relative.as_posix()] = path
    if not files:
        raise ValueError("StepGuard snapshot has no payload files")
    return dict(sorted(files.items()))


def _revision_metadata(snapshot: Path, payload_files: Sequence[str]) -> Mapping[str, Any]:
    """Validate the revision lines written by ``snapshot_download(local_dir=...)``."""

    metadata_root = snapshot / ".cache" / "huggingface" / "download"
    records: dict[str, str] = {}
    if metadata_root.is_dir():
        for path in sorted(metadata_root.rglob("*.metadata")):
            lines = path.read_text(encoding="utf-8").splitlines()
            if not lines or not lines[0].strip():
                raise RuntimeError(f"empty Hugging Face revision metadata: {path}")
            target = path.relative_to(metadata_root).as_posix()
            target = target[: -len(".metadata")]
            records[target] = lines[0].strip()
    expected_targets = set(payload_files)
    if set(records) != expected_targets:
        raise RuntimeError(
            "StepGuard Hugging Face revision metadata targets differ from payload; "
            f"missing={sorted(expected_targets - set(records))}, "
            f"unexpected={sorted(set(records) - expected_targets)}"
        )
    revisions = set(records.values())
    if revisions != {STEPGUARD_MODEL_REVISION}:
        raise RuntimeError(f"StepGuard snapshot revision metadata differs: {sorted(revisions)}")
    return {
        "source": "huggingface_local_dir_metadata_first_line",
        "validated_revision": STEPGUARD_MODEL_REVISION,
        "targets": sorted(records),
    }


def _snapshot_record(snapshot: Path) -> Mapping[str, Any]:
    files = _snapshot_files(snapshot)
    validate_weight_artifacts(files)
    metadata = {
        relative: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for relative, path in files.items()
    }
    tree_sha256 = snapshot_tree_identity(metadata)
    if tree_sha256 != STEPGUARD_SNAPSHOT_TREE_SHA256:
        raise RuntimeError(
            "StepGuard snapshot payload differs from the frozen tree: "
            f"{tree_sha256}"
        )
    return {
        "model_id": STEPGUARD_MODEL_ID,
        "revision": STEPGUARD_MODEL_REVISION,
        "snapshot_tree_sha256": tree_sha256,
        "snapshot_file_selection": {
            "rule": SNAPSHOT_EXCLUSION,
            "excluded_prefix": ".cache/",
        },
        "revision_metadata": _revision_metadata(snapshot, tuple(files)),
        "files": metadata,
    }


def _code_provenance() -> Mapping[str, Any]:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    source_paths = (
        RUNNER_RELATIVE_PATH,
        ADAPTER_RELATIVE_PATH,
        PROTOCOL_RELATIVE_PATH,
    )
    missing = [relative for relative in source_paths if not (REPO / relative).is_file()]
    if missing:
        raise RuntimeError(f"StepGuard provenance files are missing: {missing}")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_porcelain_sha256": sha256_bytes(status.encode("utf-8")),
        "status_entry_count": 0 if not status else len(status.splitlines()),
        "source_files": {
            relative: sha256_file(REPO / relative) for relative in source_paths
        },
    }


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    required = {
        "row_id",
        "benchmark",
        "source_relative_path",
        "source_sha256",
        "row_index",
        "family_id",
        "candidate_role",
        "truth_label",
        "instruction",
        "history",
        "current_action",
        "env_info",
    }
    if not rows:
        raise RuntimeError("StepGuard transfer rows are empty")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise RuntimeError(f"StepGuard transfer row {index} is missing {sorted(missing)}")
        row_id = row["row_id"]
        if not isinstance(row_id, str) or not row_id:
            raise RuntimeError(f"StepGuard transfer row {index} has an invalid row_id")
        if row_id in seen:
            raise RuntimeError(f"duplicate StepGuard row_id: {row_id}")
        seen.add(row_id)
        if row["benchmark"] != "sequential_anchorrc":
            raise RuntimeError(f"StepGuard benchmark binding differs: {row_id}")
        if float(row["truth_label"]) not in {0.0, 1.0}:
            raise RuntimeError(f"StepGuard transfer requires binary truth: {row_id}")
        if isinstance(row["row_index"], bool) or not isinstance(row["row_index"], int):
            raise RuntimeError(f"StepGuard row_index is invalid: {row_id}")
        for field in (
            "source_relative_path",
            "source_sha256",
            "family_id",
            "candidate_role",
            "instruction",
            "history",
            "current_action",
            "env_info",
        ):
            if not isinstance(row[field], str):
                raise RuntimeError(f"StepGuard {field} is not text: {row_id}")


def _public_plan_rows(
    plan: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "rendered_prompt"}
        for row in plan
    ]


def _metrics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    unsafe = [row for row in rows if float(row["truth_label"]) == 1.0]
    safe = [row for row in rows if float(row["truth_label"]) == 0.0]
    unsafe_passes = sum(bool(row["passes_strict"]) for row in unsafe)
    safe_blocks = sum(not bool(row["passes_strict"]) for row in safe)
    correct = len(rows) - unsafe_passes - safe_blocks
    return {
        "rows": len(rows),
        "parse_failures": sum(not bool(row["parse_valid"]) for row in rows),
        "predicted_labels": dict(
            Counter(str(row.get("prediction")) for row in rows)
        ),
        "strict": {
            "correct": correct,
            "accuracy": None if not rows else correct / len(rows),
            "unsafe_total": len(unsafe),
            "unsafe_passed": unsafe_passes,
            "unsafe_pass_rate": None if not unsafe else unsafe_passes / len(unsafe),
            "safe_total": len(safe),
            "safe_blocked": safe_blocks,
            "safe_block_rate": None if not safe else safe_blocks / len(safe),
        },
    }


def _verify_completed_run(
    *,
    completion_path: Path,
    identity_path: Path,
    plan_path: Path,
    metadata_path: Path,
    predictions_path: Path,
    run_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    expected = {
        "artifact_type": "stepguard_action_transfer_completion_v1",
        "run_id": run_id,
        "identity_sha256": sha256_file(identity_path),
        "plan_sha256": sha256_file(plan_path),
        "metadata_sha256": sha256_file(metadata_path),
        "predictions_sha256": sha256_file(predictions_path),
        "rows": len(rows),
        "metrics": _metrics(rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "generated_tokens": sum(int(row["generated_tokens"]) for row in rows),
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            raise RuntimeError(f"completed StepGuard receipt differs: {key}")
    return completion


def _dry_run_summary(
    *,
    identity: Mapping[str, Any],
    plan: Sequence[Mapping[str, Any]],
    run_dir: Path,
) -> Mapping[str, Any]:
    token_counts = [int(row["input_tokens"]) for row in plan]
    return {
        "run_id": sha256_bytes(canonical_json_bytes(identity))[:12],
        "rows": len(plan),
        "truth_labels": dict(Counter(str(row["truth_label"]) for row in plan)),
        "input_tokens": {
            "min": min(token_counts),
            "mean": sum(token_counts) / len(token_counts),
            "max": max(token_counts),
            "sum": sum(token_counts),
        },
        "model_snapshot_tree_sha256": identity["model"]["snapshot_tree_sha256"],
        "code_commit": identity["code"]["commit"],
        "code_dirty": identity["code"]["dirty"],
        "run_dir": str(run_dir),
    }


def _timed_generate(
    model: Any,
    encoded: Mapping[str, Any],
    *,
    max_new_tokens: int,
    pad_token_id: int,
) -> tuple[Any, float, int]:
    """Generate once with CUDA synchronization immediately around the timer."""

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return generated, elapsed, int(torch.cuda.max_memory_allocated())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--stepguard-repo", required=True, type=Path)
    parser.add_argument("--model-snapshot", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
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

    rows_path = args.rows.resolve()
    stepguard_repo = args.stepguard_repo.resolve()
    snapshot = args.model_snapshot.resolve()
    upstream = verify_upstream(stepguard_repo)
    rows = _load_jsonl(rows_path)
    _validate_rows(rows)
    model_record = _snapshot_record(snapshot)
    code_record = _code_provenance()

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise RuntimeError("StepGuard tokenizer has no padding token")
    plan = build_prompt_plan(rows, tokenizer, max_input_tokens=args.max_input_tokens)
    if args.limit is not None:
        plan = plan[: args.limit]

    identity = {
        "schema_version": "1.0.0",
        "artifact_type": "stepguard_action_transfer_identity_v1",
        "model": model_record,
        "rows_sha256": sha256_file(rows_path),
        "code": code_record,
        "interface": {
            "upstream": dict(upstream),
            "prompt_name": "stepguard",
            "chat_interface": "one_user_message_official_tokenizer_chat_template",
            "adapter": {
                "instruction": "user_request_verbatim",
                "history": "one_environment_observation_verbatim",
                "current_action": "raw_action_at_step_1_verbatim",
                "env_info": "hash_bound_but_not_rendered_not_a_tool_schema",
                "available_tools": "empty",
            },
            "decision_rules": {
                "strict": "only_valid_safe_passes",
                "parse_failure": "fail_closed",
            },
        },
        "settings": {
            "batch_size": args.batch_size,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "local_files_only": True,
            "trust_remote_code": False,
            "limit": args.limit,
            "warmup_rows": args.warmup_rows,
            "plan_order": "input_tokens_then_source_path_then_row_index",
            "timing": "cuda_synchronized_model_generate_only",
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
                _dry_run_summary(identity=identity, plan=plan, run_dir=run_dir),
                indent=2,
                sort_keys=True,
            )
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("StepGuard inference requires CUDA")
    run_dir.mkdir(parents=True, exist_ok=True)
    identity_path = run_dir / "identity.json"
    plan_path = run_dir / "plan.jsonl"
    metadata_path = run_dir / "metadata.json"
    output_path = run_dir / "predictions.jsonl"
    completion_path = run_dir / "completion.json"
    _write_json(identity_path, identity)
    _write_jsonl(plan_path, _public_plan_rows(plan))
    _write_json(
        metadata_path,
        {
            "schema_version": "1.0.0",
            "artifact_type": "stepguard_action_transfer_metadata_v1",
            "run_id": run_id,
            "identity_sha256": sha256_file(identity_path),
            "plan_sha256": sha256_file(plan_path),
            "planned_rows": len(plan),
            "model_snapshot": str(snapshot),
            "stepguard_repo": str(stepguard_repo),
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
            "environment": {
                name: os.getenv(name)
                for name in (
                    "CUDA_VISIBLE_DEVICES",
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "TOKENIZERS_PARALLELISM",
                )
            },
        },
    )
    existing = _load_jsonl(output_path)
    if len(existing) > len(plan):
        raise RuntimeError("existing StepGuard output is longer than the plan")
    if [row.get("row_id") for row in existing] != [
        row["row_id"] for row in plan[: len(existing)]
    ]:
        raise RuntimeError("existing StepGuard output is not the planned prefix")
    if completion_path.exists() and len(existing) != len(plan):
        raise RuntimeError("StepGuard completion exists for incomplete predictions")
    if len(existing) == len(plan) and completion_path.exists():
        _verify_completed_run(
            completion_path=completion_path,
            identity_path=identity_path,
            plan_path=plan_path,
            metadata_path=metadata_path,
            predictions_path=output_path,
            run_id=run_id,
            rows=existing,
        )
        print(completion_path.read_text(encoding="utf-8"), end="")
        return

    torch.cuda.synchronize()
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.eval()
    model.config.use_cache = True
    model.generation_config.use_cache = True
    model.generation_config.do_sample = False
    for name in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, name):
            setattr(model.generation_config, name, None)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    warmup_seconds = 0.0
    if args.warmup_rows:
        warmup = plan[: min(args.warmup_rows, len(plan))]
        torch.cuda.synchronize()
        warmup_started = time.perf_counter()
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
                    use_cache=True,
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
            generated, elapsed, peak_gpu_bytes = _timed_generate(
                model,
                encoded,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            width = encoded["input_ids"].shape[1]
            output_ids = generated[:, width:]
            texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            realized = []
            for position, (planned, text, ids) in enumerate(
                zip(batch, texts, output_ids)
            ):
                parsed = parse_stepguard_output(text)
                realized.append(
                    {
                        "schema_version": "1.0.0",
                        "artifact_type": "stepguard_action_transfer_prediction_v1",
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
                        "confidence": parsed["confidence"],
                        "parse_valid": parsed["valid"],
                        "passes_strict": passes_guard(parsed),
                        "prompt_sha256": planned["prompt_sha256"],
                        "env_info_sha256": planned["env_info_sha256"],
                        "input_tokens": planned["input_tokens"],
                        "generated_tokens": int(
                            (ids != tokenizer.pad_token_id).sum().item()
                        ),
                        "output_text": text,
                        "output_text_sha256": sha256_bytes(text.encode("utf-8")),
                        "batch_index": batch_index,
                        "batch_position": position,
                        "batch_size_realized": len(batch),
                        "batch_seconds": elapsed,
                        "batch_peak_gpu_bytes": peak_gpu_bytes,
                    }
                )
            overlap = max(0, completed_prefix - begin)
            stable = set(realized[0]) - {"batch_seconds", "batch_peak_gpu_bytes"}
            for position in range(overlap):
                if any(
                    existing[begin + position].get(key) != realized[position].get(key)
                    for key in stable
                ):
                    raise RuntimeError("replayed StepGuard partial batch differs")
            missing = realized[overlap:]
            if missing:
                handle.write(
                    b"".join(canonical_json_bytes(row) + b"\n" for row in missing)
                )
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
        raise RuntimeError("completed StepGuard output order differs")
    batch_indices = sorted({int(row["batch_index"]) for row in completed})
    completion = {
        "schema_version": "1.0.0",
        "artifact_type": "stepguard_action_transfer_completion_v1",
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
            max(
                float(row["batch_seconds"])
                for row in completed
                if int(row["batch_index"]) == index
            )
            for index in batch_indices
        ),
        "max_peak_gpu_bytes": max(
            int(row["batch_peak_gpu_bytes"]) for row in completed
        ),
        "warmup": {
            "rows": min(args.warmup_rows, len(plan)),
            "seconds": warmup_seconds,
            "excluded_from_metrics": True,
        },
        "transfer_scope": "stepguard_action_guard_applied_to_raw_workflow_update",
        "old_action_resurrection": "not_applicable_singleton_filter",
        "human_annotations_collected": False,
    }
    _write_json(completion_path, completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
