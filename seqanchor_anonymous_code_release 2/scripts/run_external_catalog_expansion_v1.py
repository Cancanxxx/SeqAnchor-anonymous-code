#!/usr/bin/env python3
"""Run resumable next-token catalog-expansion choices on a pinned local LLM."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import time
from collections import Counter, defaultdict
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
from option_set_instability.external_catalog_expansion_v1 import (
    aggregate_semantic_choice,
    build_choice_call_plan,
    paired_expansion_outcome,
    token_bounded_batch_spans,
)

FIXED_CHAT_DATE = "2026-09-01"
ANSWER_LABELS = ("A", "B", "C")


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError(f"partial final JSONL record: {path}")
    rows = []
    for number, line in enumerate(raw.splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise RuntimeError(f"non-object JSONL row {number}: {path}")
        rows.append(value)
    return rows


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


def _prompt_input_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError("tokenizer did not return prompt token IDs")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("tokenizer returned invalid prompt token IDs")
    return tuple(values)


def _prepare_plan(
    *,
    calls: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    chat_template_kwargs: Mapping[str, Any],
    max_input_tokens: int,
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, int]]:
    prepared = []
    token_contracts = set()
    for call in calls:
        rendered = render_chat_prompt(
            tokenizer,
            call["messages"],
            fixed_date=FIXED_CHAT_DATE,
            chat_template_kwargs=chat_template_kwargs,
        )
        prefix = _prompt_input_ids(tokenizer, rendered)
        if len(prefix) > max_input_tokens:
            raise RuntimeError(
                f"external catalog prompt exceeds max_input_tokens: {call['call_id']} "
                f"({len(prefix)} > {max_input_tokens})"
            )
        label_tokens = {}
        for label in ANSWER_LABELS:
            combined = _prompt_input_ids(tokenizer, rendered + label)
            if len(combined) != len(prefix) + 1 or combined[:-1] != prefix:
                raise RuntimeError(f"answer label {label} is not a one-token continuation")
            label_tokens[label] = combined[-1]
        if len(set(label_tokens.values())) != len(label_tokens):
            raise RuntimeError("answer labels map to duplicate token IDs")
        token_contracts.add(tuple(sorted(label_tokens.items())))
        prepared.append(
            {
                "call_id": call["call_id"],
                "task_id": call["task_id"],
                "benchmark": call["benchmark"],
                "source_row_id": call["source_row_id"],
                "source_truth_label": call["source_truth_label"],
                "menu_kind": call["menu_kind"],
                "role_to_label": call["role_to_label"],
                "allowed_labels": call["allowed_labels"],
                "prompt_messages_sha256": call["prompt_messages_sha256"],
                "rendered_prompt": rendered,
                "rendered_prompt_sha256": sha256_bytes(rendered.encode("utf-8")),
                "input_token_ids": prefix,
                "input_token_ids_sha256": sha256_bytes(canonical_json_bytes(list(prefix))),
                "input_tokens": len(prefix),
            }
        )
    if len(token_contracts) != 1:
        raise RuntimeError("answer token IDs drift across external catalog prompts")
    answer_tokens = dict(next(iter(token_contracts)))
    ordered = tuple(
        sorted(
            prepared,
            key=lambda row: (row["input_tokens"], row["task_id"], row["call_id"]),
        )
    )
    return ordered, answer_tokens


def _public_plan(plan: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    private = {"rendered_prompt", "input_token_ids"}
    return [{key: value for key, value in row.items() if key not in private} for row in plan]


def _last_logit_kwargs(model: Any, kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    parameters = inspect.signature(model.forward).parameters
    if "logits_to_keep" in parameters:
        return {**kwargs, "logits_to_keep": 1}
    if "num_logits_to_keep" in parameters:
        return {**kwargs, "num_logits_to_keep": 1}
    return dict(kwargs)


def _score_batch(
    *,
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
    for index, row in enumerate(batch):
        observed = tuple(
            int(value)
            for value in input_ids[index][attention_mask[index].bool()].detach().cpu().tolist()
        )
        if observed != row["input_token_ids"]:
            raise RuntimeError("batched tokenization differs from the prepared prompt")
    kwargs = _last_logit_kwargs(
        model,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            "return_dict": True,
        },
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(**kwargs)
    elapsed = time.perf_counter() - started
    if getattr(output, "past_key_values", None) is not None:
        raise RuntimeError("model returned a KV cache despite use_cache=False")
    logits = output.logits
    if logits.ndim != 3 or logits.shape[0] != len(batch):
        raise RuntimeError("model returned an unsupported logits tensor")
    token_ids = [answer_tokens[label] for label in ANSWER_LABELS]
    selected = logits[:, -1, token_ids].detach().to(dtype=torch.float32).cpu().tolist()
    result = []
    for values in selected:
        row = {label: float(value) for label, value in zip(ANSWER_LABELS, values)}
        if any(not math.isfinite(value) for value in row.values()):
            raise RuntimeError("model returned a nonfinite answer-label logit")
        result.append(row)
    return result, elapsed, torch.cuda.max_memory_allocated()


def _metrics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    benchmark_by_task = {}
    for row in rows:
        grouped[(row["task_id"], row["menu_kind"])].append(row)
        benchmark_by_task[row["task_id"]] = row["benchmark"]
    outcomes = []
    for task_id in sorted(benchmark_by_task):
        base = aggregate_semantic_choice(grouped[(task_id, "base")], menu_kind="base")
        expanded = aggregate_semantic_choice(
            grouped[(task_id, "expanded")], menu_kind="expanded"
        )
        outcomes.append(
            {
                "task_id": task_id,
                "benchmark": benchmark_by_task[task_id],
                **paired_expansion_outcome(base=base, expanded=expanded),
            }
        )

    def summarize(values: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        resolved = [value for value in values if value["resolved"]]
        base_safe = [value for value in resolved if value["base_winner"] == "safe_incumbent"]
        revival = [value for value in base_safe if value["safe_to_old_unsafe_revival"]]
        changed = [value for value in resolved if value["decision_changed"]]
        new_selected = [value for value in resolved if value["new_option_selected"]]
        return {
            "tasks": len(values),
            "resolved_pairs": len(resolved),
            "unresolved_pairs": len(values) - len(resolved),
            "base_safe": len(base_safe),
            "base_safe_rate": None if not resolved else len(base_safe) / len(resolved),
            "safe_to_old_unsafe_revivals": len(revival),
            "revival_rate_given_base_safe": None
            if not base_safe
            else len(revival) / len(base_safe),
            "decision_changes": len(changed),
            "decision_change_rate": None if not resolved else len(changed) / len(resolved),
            "safe_addition_selected": len(new_selected),
            "safe_addition_selection_rate": None
            if not resolved
            else len(new_selected) / len(resolved),
            "winner_transitions": dict(
                sorted(
                    Counter(
                        f"{value['base_winner']}->{value['expanded_winner']}"
                        for value in resolved
                    ).items()
                )
            ),
        }

    by_benchmark = defaultdict(list)
    for outcome in outcomes:
        by_benchmark[outcome["benchmark"]].append(outcome)
    return {
        "aggregate": summarize(outcomes),
        "by_benchmark": {
            benchmark: summarize(values) for benchmark, values in sorted(by_benchmark.items())
        },
        "revival_task_ids": [
            value["task_id"] for value in outcomes if value["safe_to_old_unsafe_revival"]
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-registry", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument(
        "--max-batch-padded-tokens",
        type=int,
        help="Optional deterministic cap on batch_size_realized times padded input width.",
    )
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_input_tokens <= 0:
        raise ValueError("batch size and max input tokens must be positive")
    if args.max_batch_padded_tokens is not None and args.max_batch_padded_tokens <= 0:
        raise ValueError("--max-batch-padded-tokens must be positive")
    if args.limit_tasks is not None and args.limit_tasks <= 0:
        raise ValueError("--limit-tasks must be positive")
    tasks_path = args.tasks.resolve()
    manifest_path = args.manifest.resolve()
    registry_path = args.model_registry.resolve()
    output_root = args.output_root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "external_catalog_expansion_manifest_v1":
        raise RuntimeError("unexpected external catalog manifest")
    if sha256_file(tasks_path) != manifest["public"]["sha256"]:
        raise RuntimeError("external catalog task hash differs from the manifest")
    tasks = _load_jsonl(tasks_path)
    if len(tasks) != manifest["public"]["rows"]:
        raise RuntimeError("external catalog task count differs from the manifest")
    if args.limit_tasks is not None:
        tasks = tasks[: args.limit_tasks]
    registry = load_model_registry(registry_path, verify_files=False, require_frozen=True)
    validate_runtime_versions(registry)
    tokenizer = load_tokenizer(registry)
    calls = build_choice_call_plan(tasks, snapshot_id=registry["candidate_snapshot_id"])
    plan, answer_tokens = _prepare_plan(
        calls=calls,
        tokenizer=tokenizer,
        chat_template_kwargs=registry_chat_template_kwargs(registry),
        max_input_tokens=args.max_input_tokens,
    )
    public_plan = _public_plan(plan)
    if args.max_batch_padded_tokens is None:
        batch_spans = tuple(
            (begin, min(begin + args.batch_size, len(plan)))
            for begin in range(0, len(plan), args.batch_size)
        )
    else:
        batch_spans = token_bounded_batch_spans(
            [int(row["input_tokens"]) for row in plan],
            max_batch_size=args.batch_size,
            max_padded_tokens=args.max_batch_padded_tokens,
        )
    identity = {
        "schema_version": "1.0.0",
        "artifact_type": "external_catalog_expansion_run_identity_v1",
        "tasks_sha256": sha256_file(tasks_path),
        "manifest_sha256": sha256_file(manifest_path),
        "model_registry_sha256": sha256_file(registry_path),
        "snapshot_id": registry["candidate_snapshot_id"],
        "model_revision": registry["revision"],
        "fixed_chat_date": FIXED_CHAT_DATE,
        "answer_token_ids": answer_tokens,
        "settings": {
            "batch_size": args.batch_size,
            "max_input_tokens": args.max_input_tokens,
            "limit_tasks": args.limit_tasks,
            "forward_mode": "no_cache_single_next_token_abc_logits",
            "plan_order": "input_tokens_then_task_id_then_call_id",
        },
        "planned_call_ids_sha256": sha256_bytes(
            canonical_json_bytes([row["call_id"] for row in plan])
        ),
    }
    if args.max_batch_padded_tokens is not None:
        identity["settings"]["max_batch_padded_tokens"] = args.max_batch_padded_tokens
        identity["settings"]["batching"] = "length_sorted_token_bounded_v1"
    run_id = sha256_bytes(canonical_json_bytes(identity))[:12]
    run_dir = output_root / run_id / registry["candidate_snapshot_id"]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "snapshot_id": registry["candidate_snapshot_id"],
                    "tasks": len(tasks),
                    "calls": len(plan),
                    "answer_token_ids": answer_tokens,
                    "input_tokens": {
                        "min": min(row["input_tokens"] for row in plan),
                        "max": max(row["input_tokens"] for row in plan),
                    },
                    "batches": len(batch_spans),
                    "batch_size_realized": {
                        "min": min(stop - begin for begin, stop in batch_spans),
                        "max": max(stop - begin for begin, stop in batch_spans),
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
    output_path = run_dir / "abc_logits.jsonl"
    completion_path = run_dir / "completion.json"
    _write_json(identity_path, identity)
    _write_jsonl(plan_path, public_plan)
    metadata = {
        "schema_version": "1.0.0",
        "artifact_type": "external_catalog_expansion_run_metadata_v1",
        "run_id": run_id,
        "identity_sha256": sha256_file(identity_path),
        "plan_sha256": sha256_file(plan_path),
        "planned_calls": len(plan),
        "model_id": registry["model_id"],
        "snapshot_id": registry["candidate_snapshot_id"],
        "model_revision": registry["revision"],
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
    }
    _write_json(metadata_path, metadata)
    existing = _load_jsonl(output_path) if output_path.exists() else []
    if [row.get("call_id") for row in existing] != [
        row["call_id"] for row in plan[: len(existing)]
    ]:
        raise RuntimeError("existing output is not the exact planned prefix")
    if len(existing) == len(plan) and completion_path.exists():
        print(completion_path.read_text(encoding="utf-8"), end="")
        return

    model_load_started = time.perf_counter()
    model = load_model(registry)
    model_load_seconds = time.perf_counter() - model_load_started
    completed_prefix = len(existing)
    first_batch_index = next(
        (
            index
            for index, (begin, stop) in enumerate(batch_spans)
            if stop > completed_prefix
        ),
        len(batch_spans),
    )
    with output_path.open("ab") as handle:
        for batch_index in range(first_batch_index, len(batch_spans)):
            begin, stop = batch_spans[batch_index]
            batch = plan[begin:stop]
            scores, elapsed, peak_bytes = _score_batch(
                model=model,
                tokenizer=tokenizer,
                batch=batch,
                answer_tokens=answer_tokens,
            )
            realized = []
            for position, (planned, all_logits) in enumerate(zip(batch, scores)):
                allowed = {
                    label: all_logits[label] for label in planned["allowed_labels"]
                }
                argmax = max(sorted(allowed), key=lambda label: allowed[label])
                realized.append(
                    {
                        "schema_version": "1.0.0",
                        "artifact_type": "external_catalog_abc_logits_v1",
                        "ordinal": begin + position,
                        "call_id": planned["call_id"],
                        "task_id": planned["task_id"],
                        "benchmark": planned["benchmark"],
                        "source_row_id": planned["source_row_id"],
                        "source_truth_label": planned["source_truth_label"],
                        "menu_kind": planned["menu_kind"],
                        "role_to_label": planned["role_to_label"],
                        "allowed_label_logits": allowed,
                        "argmax_label": argmax,
                        "prompt_messages_sha256": planned["prompt_messages_sha256"],
                        "rendered_prompt_sha256": planned["rendered_prompt_sha256"],
                        "input_token_ids_sha256": planned["input_token_ids_sha256"],
                        "input_tokens": planned["input_tokens"],
                        "batch_index": batch_index,
                        "batch_position": position,
                        "batch_size_realized": len(batch),
                        "batch_seconds": elapsed,
                        "batch_peak_gpu_bytes": peak_bytes,
                    }
                )
            overlap = max(0, completed_prefix - begin)
            stable_keys = set(realized[0]) - {"batch_seconds", "batch_peak_gpu_bytes"}
            for position in range(overlap):
                observed = existing[begin + position]
                replayed = realized[position]
                if any(observed.get(key) != replayed.get(key) for key in stable_keys):
                    raise RuntimeError("replayed partial batch differs from existing output")
            missing = realized[overlap:]
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
                        "batch_seconds": round(elapsed, 3),
                        "max_input_tokens": max(row["input_tokens"] for row in batch),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    completed = _load_jsonl(output_path)
    if [row["call_id"] for row in completed] != [row["call_id"] for row in plan]:
        raise RuntimeError("completed output order differs from the plan")
    metrics = _metrics(completed)
    batch_indices = sorted({int(row["batch_index"]) for row in completed})
    completion = {
        "schema_version": "1.0.0",
        "artifact_type": "external_catalog_expansion_completion_v1",
        "run_id": run_id,
        "snapshot_id": registry["candidate_snapshot_id"],
        "model_id": registry["model_id"],
        "identity_sha256": sha256_file(identity_path),
        "plan_sha256": sha256_file(plan_path),
        "metadata_sha256": sha256_file(metadata_path),
        "output_sha256": sha256_file(output_path),
        "rows": len(completed),
        "model_load_seconds": model_load_seconds,
        "metrics": metrics,
        "input_tokens": sum(int(row["input_tokens"]) for row in completed),
        "batch_seconds_sum": sum(
            max(
                float(row["batch_seconds"])
                for row in completed
                if int(row["batch_index"]) == index
            )
            for index in batch_indices
        ),
        "max_peak_gpu_bytes": max(int(row["batch_peak_gpu_bytes"]) for row in completed),
        "human_annotations_collected": False,
    }
    _write_json(completion_path, completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
