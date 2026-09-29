#!/usr/bin/env python3
"""Authorize replication inference after deterministic registry/tokenizer preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nontransitive_safety.local_hf_registry import load_tokenizer, registry_chat_template_kwargs
from nontransitive_safety.provenance import canonical_json_bytes, loads_json_strict, sha256_file
from option_set_instability.anchorrc_v1 import (
    call_plan_sha256,
    load_protocol,
    load_public_tasks,
    prepare_prompts,
)
from option_set_instability.seqanchor_multisafety_replication_v1 import (
    PREFLIGHT_ARTIFACT_TYPE,
    _expected_safety_calls,
    load_replication_model_registry,
    prepared_prompt_contract,
    reject_duplicate_cli_options,
    require_clean_committed_tree,
    verify_bound_sources,
    write_once_bytes,
)


REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "configs" / "seqanchor_multisafety_replication_analysis_v1.json"
PROTOCOL = REPO / "configs" / "seqanchor_multisafety_replication_inference_v1.json"


def _external(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO.resolve())
    except ValueError:
        return resolved
    raise RuntimeError("authorization receipt must be outside the clean repository")


def _write_once(path: Path, payload: bytes) -> None:
    write_once_bytes(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    reject_duplicate_cli_options(sys.argv[1:])
    args = parse_args()
    output = _external(args.output)
    commit = require_clean_committed_tree(REPO)
    spec = loads_json_strict(SPEC.read_text(encoding="utf-8"), source=str(SPEC))
    paths = verify_bound_sources(spec, repo=REPO)
    protocol = load_protocol(PROTOCOL)
    tasks = load_public_tasks(paths["safety_tasks_public"])
    models = {}
    for row in spec["predeclared_models"]:
        snapshot_id = str(row["snapshot_id"])
        registry_path = paths[str(row["registry_binding"])]
        registry = load_replication_model_registry(
            registry_path, verify_files=True, require_frozen=True
        )
        if registry["candidate_snapshot_id"] != snapshot_id:
            raise RuntimeError("preflight registry/model binding differs")
        reference = protocol["model_registries"].get(snapshot_id)
        if reference is None or reference["sha256"] != sha256_file(registry_path):
            raise RuntimeError("preflight protocol/registry binding differs")
        calls = _expected_safety_calls(
            protocol=protocol, tasks=tasks, snapshot_id=snapshot_id
        )
        tokenizer = load_tokenizer(registry)
        prepared, answer_tokens = prepare_prompts(
            tokenizer=tokenizer,
            calls=calls,
            fixed_chat_date=protocol["fixed_chat_date"],
            chat_template_kwargs=registry_chat_template_kwargs(registry),
            max_input_tokens=protocol["inference"]["max_input_tokens"],
        )
        contract = prepared_prompt_contract(prepared, answer_tokens)
        if contract["prompts"] != 5_040 or contract["answer_prompt_count"] != 5_040:
            raise RuntimeError("preflight prompt coverage differs")
        models[snapshot_id] = {
            "model_id": registry["model_id"],
            "revision": registry["revision"],
            "registry_sha256": sha256_file(registry_path),
            "snapshot_tree_sha256": registry["snapshot_tree_sha256"],
            "local_snapshot": registry["local_snapshot"],
            "snapshot_files_fully_hash_verified": True,
            "call_plan_sha256": call_plan_sha256(calls),
            "calls": len(calls),
            "prompt_contract": contract,
        }
    receipt = {
        "schema_version": "1.0.0",
        "artifact_type": PREFLIGHT_ARTIFACT_TYPE,
        "status": "authorized_for_inference",
        "authorization_commit": commit,
        "tree_clean": True,
        "analysis_spec_sha256": sha256_file(SPEC),
        "protocol_sha256": sha256_file(PROTOCOL),
        "tasks_sha256": sha256_file(paths["safety_tasks_public"]),
        "models": models,
        "model_inference_performed": False,
        "human_annotations_collected": False,
    }
    _write_once(output, canonical_json_bytes(receipt) + b"\n")
    print(output)


if __name__ == "__main__":
    main()
