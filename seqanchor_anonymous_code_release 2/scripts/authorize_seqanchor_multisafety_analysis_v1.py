#!/usr/bin/env python3
"""Seal both public model runs before private SeqAnchor truth is accessed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nontransitive_safety.provenance import canonical_json_bytes, loads_json_strict, sha256_file
from option_set_instability.seqanchor_multisafety_replication_v1 import (
    ANALYSIS_AUTHORIZATION_ARTIFACT_TYPE,
    audit_safety_run,
    require_clean_committed_tree,
    reject_duplicate_cli_options,
    validate_preflight_authorization,
    validate_spec,
    write_once_bytes,
)
from option_set_instability.sequential_analysis_v1 import run_evidence_sha256


REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "configs" / "seqanchor_multisafety_replication_analysis_v1.json"
PROTOCOL = REPO / "configs" / "seqanchor_multisafety_replication_inference_v1.json"


def _parse_run(value: str) -> tuple[str, Path]:
    snapshot_id, separator, raw_path = value.partition("=")
    if not separator or not snapshot_id or not Path(raw_path).is_absolute():
        raise argparse.ArgumentTypeError("--run must be SNAPSHOT_ID=/absolute/run/directory")
    return snapshot_id, Path(raw_path).resolve()


def _external(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO.resolve())
    except ValueError:
        return resolved
    raise RuntimeError("analysis authorization must be outside the clean repository")


def _write_once(path: Path, payload: bytes) -> None:
    write_once_bytes(path, payload)


def _bound_path(spec: dict, group: str, name: str) -> Path:
    binding = spec[group][name]
    path = (REPO / binding["path"]).resolve()
    if sha256_file(path) != binding["sha256"]:
        raise RuntimeError(f"bound public source differs: {name}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-authorization", type=Path, required=True)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    reject_duplicate_cli_options(sys.argv[1:], repeatable=("--run",))
    args = parse_args()
    output = _external(args.output)
    run_dirs = dict(args.run)
    if len(run_dirs) != len(args.run):
        raise RuntimeError("duplicate run model")
    analysis_commit = require_clean_committed_tree(REPO)
    spec = loads_json_strict(SPEC.read_text(encoding="utf-8"), source=str(SPEC))
    validate_spec(spec)
    declared = [row["snapshot_id"] for row in spec["predeclared_models"]]
    if set(run_dirs) != set(declared):
        raise RuntimeError("both and only the predeclared models are required")
    preflight_path = args.preflight_authorization.resolve()
    preflight = loads_json_strict(
        preflight_path.read_text(encoding="utf-8"), source=str(preflight_path)
    )
    validate_preflight_authorization(
        preflight,
        receipt_path=preflight_path,
        repo=REPO,
        spec_path=SPEC,
        protocol_path=PROTOCOL,
        require_current_commit=False,
    )
    inference_commit = str(preflight["authorization_commit"])
    if analysis_commit != inference_commit:
        raise RuntimeError("analysis must use the same clean source commit as inference")
    tasks_path = _bound_path(spec, "inputs", "safety_tasks_public")
    records = {}
    for model in spec["predeclared_models"]:
        snapshot_id = str(model["snapshot_id"])
        registry_path = _bound_path(spec, "source_bindings", model["registry_binding"])
        _scores, audit = audit_safety_run(
            run_dirs[snapshot_id],
            repo=REPO,
            protocol_path=PROTOCOL,
            tasks_path=tasks_path,
            registry_path=registry_path,
            snapshot_id=snapshot_id,
            preflight_model=preflight["models"][snapshot_id],
            authorization_commit=inference_commit,
            preflight_sha256=sha256_file(preflight_path),
        )
        records[snapshot_id] = {
            "run_id": audit["run_id"],
            "completion_sha256": sha256_file(
                run_dirs[snapshot_id] / "anchorrc_v1.completion.json"
            ),
            "run_evidence_sha256": run_evidence_sha256(run_dirs[snapshot_id]),
            "output_sha256": audit["output_sha256"],
            "strict_public_audit_passed": True,
        }
    receipt = {
        "schema_version": "1.0.0",
        "artifact_type": ANALYSIS_AUTHORIZATION_ARTIFACT_TYPE,
        "status": "authorized_for_private_analysis",
        "authorization_commit": analysis_commit,
        "tree_clean": True,
        "preflight_receipt_sha256": sha256_file(preflight_path),
        "analysis_spec_sha256": sha256_file(SPEC),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runs": records,
        "private_truth_accessed": False,
        "model_inference_performed": False,
    }
    _write_once(output, canonical_json_bytes(receipt) + b"\n")
    print(output)


if __name__ == "__main__":
    main()
