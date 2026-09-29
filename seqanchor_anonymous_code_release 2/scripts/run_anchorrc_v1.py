#!/usr/bin/env python3
"""Run resumable deterministic AnchorRC v1 allowed-label logit acquisition."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    MARGIN_SERIALIZATION,
    append_jsonl_rows,
    batching_metadata,
    build_call_plan,
    build_completion_receipt,
    build_output_row,
    call_plan_sha256,
    configure_deterministic_torch,
    load_jsonl_strict,
    load_protocol,
    load_public_tasks,
    prepare_prompts,
    score_allowed_label_logits_batch,
    validate_deterministic_process_environment,
    verify_output_rows,
    write_or_verify_canonical_json,
    write_or_verify_canonical_jsonl,
)

REPO = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _execution_state(repo: Path) -> Mapping[str, Any]:
    status = _git(repo, "status", "--porcelain", "--untracked-files=normal")
    if status:
        raise RuntimeError("AnchorRC inference requires a clean tracked and untracked tree")
    return {
        "git_commit": _git(repo, "rev-parse", "HEAD"),
        "tracked_and_untracked_tree_clean": True,
        "python_executable": str(Path(sys.executable).resolve()),
        "process_environment": dict(validate_deterministic_process_environment()),
    }


def _require_external_output_root(repo: Path, output_root: Path) -> None:
    resolved_repo = repo.resolve()
    resolved_output = output_root.resolve()
    try:
        resolved_output.relative_to(resolved_repo)
    except ValueError:
        return
    raise RuntimeError("AnchorRC output root must be outside the clean source repository")


def _registry_reference(
    *, repo: Path, protocol: Mapping[str, Any], snapshot_id: str
) -> tuple[Path, str]:
    try:
        reference = protocol["model_registries"][snapshot_id]
    except KeyError as error:
        raise ValueError("snapshot ID is outside the AnchorRC protocol") from error
    path = repo / reference["path"]
    if not path.is_file() or sha256_file(path) != reference["sha256"]:
        raise RuntimeError("AnchorRC model registry differs from its protocol binding")
    return path, reference["sha256"]


def _metadata(
    *,
    run_id: str,
    snapshot_id: str,
    protocol_path: Path,
    tasks_path: Path,
    registry_path: Path,
    call_plan_path: Path,
    protocol: Mapping[str, Any],
    registry: Mapping[str, Any],
    execution: Mapping[str, Any],
    calls: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    batch_size = protocol["inference"]["batch_size"]
    return {
        "schema_version": "1.0.0",
        "artifact_type": "anchorrc_logit_run_v1",
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "model_id": registry["model_id"],
        "model_revision": registry["revision"],
        "tokenizer_revision": registry["tokenizer_revision"],
        "snapshot_tree_sha256": registry["snapshot_tree_sha256"],
        "model_registry_path": str(registry_path.resolve()),
        "model_registry_sha256": sha256_file(registry_path),
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "tasks_path": str(tasks_path.resolve()),
        "tasks_sha256": sha256_file(tasks_path),
        "execution": dict(execution),
        "call_plan_sha256": call_plan_sha256(calls),
        "call_plan_artifact": {
            "path": str(call_plan_path.resolve()),
            "sha256": sha256_file(call_plan_path),
            "bytes": call_plan_path.stat().st_size,
        },
        "planned_call_count": len(calls),
        "executed_call_kinds": sorted({call["call_kind"] for call in calls}),
        "planned_call_ids_sha256": sha256_bytes(
            canonical_json_bytes([call["call_id"] for call in calls])
        ),
        "batching": batching_metadata(call_count=len(calls), batch_size=batch_size),
        "inference": dict(protocol["inference"]),
        "answer_labels": ["A", "B"],
        "margin_serialization": MARGIN_SERIALIZATION,
        "raw_rows_store_messages": False,
        "uses_private_task_fields": False,
        "human_annotations_collected": False,
    }


def _read_completion(path: Path) -> Mapping[str, Any]:
    value = loads_json_strict(path.read_text(encoding="utf-8"), source=str(path))
    if not isinstance(value, Mapping):
        raise ValueError("AnchorRC completion receipt must contain an object")
    return value


def _finalize(
    *,
    run_id: str,
    snapshot_id: str,
    protocol_path: Path,
    tasks_path: Path,
    registry_path: Path,
    metadata_path: Path,
    output_path: Path,
    completion_path: Path,
    calls: Sequence[Mapping[str, Any]],
    prepared: Sequence[Any],
    answer_tokens: Any,
    batch_size: int,
) -> Mapping[str, Any]:
    if not output_path.is_file() or output_path.is_symlink():
        raise RuntimeError("cannot finalize an absent or symlinked AnchorRC output")
    rows = load_jsonl_strict(output_path)
    metadata_bytes = metadata_path.read_bytes()
    output_bytes = output_path.read_bytes()
    expected = build_completion_receipt(
        run_id=run_id,
        snapshot_id=snapshot_id,
        protocol_sha256=sha256_file(protocol_path),
        tasks_sha256=sha256_file(tasks_path),
        model_registry_sha256=sha256_file(registry_path),
        metadata_bytes=metadata_bytes,
        output_bytes=output_bytes,
        rows=rows,
        calls=calls,
        prepared=prepared,
        answer_tokens=answer_tokens,
        frozen_batch_size=batch_size,
    )
    if completion_path.exists():
        observed = _read_completion(completion_path)
        if observed != expected or completion_path.read_bytes() != (
            canonical_json_bytes(expected) + b"\n"
        ):
            raise RuntimeError("existing AnchorRC completion receipt differs")
        return observed
    write_or_verify_canonical_json(completion_path, expected)
    return expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--model-registry",
        type=Path,
        help=(
            "Optional local registry override. This makes the release portable while "
            "leaving the experiment protocol unchanged."
        ),
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--call-kind",
        choices=("safety_axis", "utility_pair"),
        help="Run only one predeclared scoring family; the filtered call plan remains fully bound.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    tasks_path = args.tasks.resolve()
    output_root = args.output_root.resolve()
    _require_external_output_root(REPO, output_root)
    protocol = load_protocol(protocol_path)
    tasks = load_public_tasks(tasks_path)
    frozen_batch_size = protocol["inference"]["batch_size"]
    if args.batch_size is not None and args.batch_size != frozen_batch_size:
        raise RuntimeError("--batch-size differs from the frozen AnchorRC protocol")

    if args.model_registry is None:
        registry_path, registry_sha256 = _registry_reference(
            repo=REPO, protocol=protocol, snapshot_id=args.snapshot_id
        )
    else:
        registry_path = args.model_registry.resolve()
        if not registry_path.is_file() or registry_path.is_symlink():
            raise RuntimeError("local model registry is absent or symbolic")
        registry_sha256 = sha256_file(registry_path)
    execution = _execution_state(REPO)
    registry = load_model_registry(registry_path, verify_files=True, require_frozen=True)
    if registry["candidate_snapshot_id"] != args.snapshot_id:
        raise RuntimeError("AnchorRC snapshot ID differs from the bound model registry")
    if sha256_file(registry_path) != registry_sha256:
        raise RuntimeError("AnchorRC model registry changed during validation")
    validate_runtime_versions(registry)

    calls = build_call_plan(tasks, protocol, snapshot_id=args.snapshot_id)
    if args.call_kind is not None:
        calls = tuple(call for call in calls if call["call_kind"] == args.call_kind)
        if not calls:
            raise RuntimeError("the requested AnchorRC call-kind filter produced no calls")
    identity = {
        "protocol_sha256": sha256_file(protocol_path),
        "tasks_sha256": sha256_file(tasks_path),
        "model_registry_sha256": registry_sha256,
        "snapshot_id": args.snapshot_id,
        "call_plan_sha256": call_plan_sha256(calls),
    }
    run_id = sha256_bytes(canonical_json_bytes(identity))[:12]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "snapshot_id": args.snapshot_id,
                    "planned_call_count": len(calls),
                    "safety_axis_calls": sum(
                        call["call_kind"] == "safety_axis" for call in calls
                    ),
                    "utility_pair_calls": sum(
                        call["call_kind"] == "utility_pair" for call in calls
                    ),
                    "frozen_batch_size": frozen_batch_size,
                    "sampling": False,
                    "temperature_applied": False,
                    "top_p_applied": False,
                    "git_commit": execution["git_commit"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    import torch

    configure_deterministic_torch(torch)
    tokenizer = load_tokenizer(registry)
    prepared, answer_tokens = prepare_prompts(
        tokenizer=tokenizer,
        calls=calls,
        fixed_chat_date=protocol["fixed_chat_date"],
        chat_template_kwargs=registry_chat_template_kwargs(registry),
        max_input_tokens=protocol["inference"]["max_input_tokens"],
    )

    output_dir = output_root / run_id / args.snapshot_id
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "anchorrc_v1.metadata.json"
    call_plan_path = output_dir / "anchorrc_v1.call_plan.jsonl"
    output_path = output_dir / "anchorrc_v1.jsonl"
    completion_path = output_dir / "anchorrc_v1.completion.json"
    write_or_verify_canonical_jsonl(call_plan_path, calls)
    metadata = _metadata(
        run_id=run_id,
        snapshot_id=args.snapshot_id,
        protocol_path=protocol_path,
        tasks_path=tasks_path,
        registry_path=registry_path,
        call_plan_path=call_plan_path,
        protocol=protocol,
        registry=registry,
        execution=execution,
        calls=calls,
    )
    write_or_verify_canonical_json(metadata_path, metadata)

    if completion_path.exists():
        completion = _finalize(
            run_id=run_id,
            snapshot_id=args.snapshot_id,
            protocol_path=protocol_path,
            tasks_path=tasks_path,
            registry_path=registry_path,
            metadata_path=metadata_path,
            output_path=output_path,
            completion_path=completion_path,
            calls=calls,
            prepared=prepared,
            answer_tokens=answer_tokens,
            batch_size=frozen_batch_size,
        )
        print(
            f"{args.snapshot_id}: verified completion for {completion['output']['rows']} calls",
            flush=True,
        )
        return

    rows = load_jsonl_strict(output_path) if output_path.exists() else []
    verify_output_rows(
        rows,
        calls,
        prepared,
        answer_tokens,
        frozen_batch_size=frozen_batch_size,
        require_resume_boundary=False,
    )
    completed = len(rows)
    if completed < len(calls):
        model = load_model(registry)
        # If interruption happened after one or more complete row appends but
        # before a frozen batch finished, replay that entire batch.  Existing
        # rows must match bit-for-bit; only its missing suffix is appended.
        resume_batch_start = completed - (completed % frozen_batch_size)
        for start in range(resume_batch_start, len(calls), frozen_batch_size):
            stop = min(start + frozen_batch_size, len(calls))
            batch_logits = score_allowed_label_logits_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prepared[start:stop],
                answer_tokens=answer_tokens,
                max_input_tokens=protocol["inference"]["max_input_tokens"],
            )
            replayed_batch_rows = [
                build_output_row(
                    ordinal=ordinal,
                    call=calls[ordinal],
                    prepared=prepared[ordinal],
                    logits=batch_logits[ordinal - start],
                    answer_tokens=answer_tokens,
                    frozen_batch_size=frozen_batch_size,
                    frozen_batch_index=ordinal // frozen_batch_size,
                    frozen_batch_position=ordinal % frozen_batch_size,
                    batch_size_realized=stop - start,
                )
                for ordinal in range(start, stop)
            ]
            existing_in_batch = max(0, min(completed, stop) - start)
            if rows[start : start + existing_in_batch] != replayed_batch_rows[
                :existing_in_batch
            ]:
                raise RuntimeError(
                    "partial AnchorRC batch differs under deterministic replay"
                )
            missing_batch_rows = replayed_batch_rows[existing_in_batch:]
            append_jsonl_rows(output_path, missing_batch_rows)
            rows.extend(missing_batch_rows)
            verify_output_rows(
                rows,
                calls,
                prepared,
                answer_tokens,
                frozen_batch_size=frozen_batch_size,
                require_resume_boundary=False,
            )
            print(f"{args.snapshot_id}: {stop}/{len(calls)} calls complete", flush=True)

    completion = _finalize(
        run_id=run_id,
        snapshot_id=args.snapshot_id,
        protocol_path=protocol_path,
        tasks_path=tasks_path,
        registry_path=registry_path,
        metadata_path=metadata_path,
        output_path=output_path,
        completion_path=completion_path,
        calls=calls,
        prepared=prepared,
        answer_tokens=answer_tokens,
        batch_size=frozen_batch_size,
    )
    print(f"{args.snapshot_id}: finalized all {completion['output']['rows']} calls", flush=True)


if __name__ == "__main__":
    main()
