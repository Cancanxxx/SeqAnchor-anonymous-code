#!/usr/bin/env python3
"""Run authorized SeqAnchor safety-only acquisition for the replication panel."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import run_anchorrc_v1 as implementation

from nontransitive_safety.provenance import loads_json_strict, sha256_file
from option_set_instability.seqanchor_multisafety_replication_v1 import (
    load_replication_model_registry,
    validate_preflight_authorization,
)


REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "configs" / "seqanchor_multisafety_replication_analysis_v1.json"
PROTOCOL = REPO / "configs" / "seqanchor_multisafety_replication_inference_v1.json"
TASKS = REPO / "data" / "sequential_anchorrc_v1" / "safety_tasks_public.jsonl"


def _reject_duplicate_options() -> None:
    names = [value.split("=", 1)[0] for value in sys.argv[1:] if value.startswith("--")]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SystemExit(f"duplicate command-line options are forbidden: {duplicates}")


def _take_argument(name: str) -> str:
    positions = [index for index, value in enumerate(sys.argv) if value == name]
    if len(positions) != 1 or any(value.startswith(name + "=") for value in sys.argv):
        raise SystemExit(f"argument must occur exactly once in split form: {name}")
    index = positions[0]
    if index + 1 >= len(sys.argv) or sys.argv[index + 1].startswith("--"):
        raise SystemExit(f"required argument value is absent: {name}")
    value = sys.argv[index + 1]
    del sys.argv[index : index + 2]
    return value


def _peek_argument(name: str) -> str:
    positions = [index for index, value in enumerate(sys.argv) if value == name]
    if len(positions) != 1 or any(value.startswith(name + "=") for value in sys.argv):
        raise SystemExit(f"argument must occur exactly once in split form: {name}")
    index = positions[0]
    if index + 1 >= len(sys.argv) or sys.argv[index + 1].startswith("--"):
        raise SystemExit(f"required argument value is absent: {name}")
    return sys.argv[index + 1]


def main() -> None:
    _reject_duplicate_options()
    authorization_path = Path(_take_argument("--preflight-authorization")).resolve()
    snapshot_id = _peek_argument("--snapshot-id")
    protocol_path = Path(_peek_argument("--protocol")).resolve()
    tasks_path = Path(_peek_argument("--tasks")).resolve()
    if protocol_path != PROTOCOL.resolve() or tasks_path != TASKS.resolve():
        raise RuntimeError("runner protocol or task path differs from the authorized study")
    if _peek_argument("--call-kind") != "safety_axis":
        raise RuntimeError("replication runner requires --call-kind safety_axis")
    receipt = loads_json_strict(
        authorization_path.read_text(encoding="utf-8"), source=str(authorization_path)
    )
    validate_preflight_authorization(
        receipt,
        receipt_path=authorization_path,
        repo=REPO,
        spec_path=SPEC,
        protocol_path=PROTOCOL,
        require_current_commit=True,
    )
    if snapshot_id not in receipt["models"]:
        raise RuntimeError("snapshot is outside the authorized replication panel")

    implementation.load_model_registry = load_replication_model_registry
    original_metadata = implementation._metadata

    def authorized_metadata(**kwargs):
        value = dict(original_metadata(**kwargs))
        value["preflight_authorization_sha256"] = sha256_file(authorization_path)
        value["preflight_authorization_commit"] = receipt["authorization_commit"]
        return value

    implementation._metadata = authorized_metadata
    implementation.main()


if __name__ == "__main__":
    main()
