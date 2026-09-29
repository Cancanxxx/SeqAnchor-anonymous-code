#!/usr/bin/env python3
"""Build the source-grounded external catalog-expansion benchmark v1."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from nontransitive_safety.provenance import canonical_json_bytes, sha256_bytes, sha256_file
from option_set_instability.external_catalog_expansion_v1 import (
    build_expansion_tasks,
    eligible_unsafe_rows,
)
from option_set_instability.toolsafe_tsguard_v1 import (
    SOURCE_RELATIVE_PATHS,
    TOOLSAFE_COMMIT,
    load_source_rows,
)


def _write_or_verify(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"existing artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolsafe-repo", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--include-controversial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.toolsafe_repo.resolve()
    output_dir = args.output_dir.resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != TOOLSAFE_COMMIT:
        raise RuntimeError("ToolSafe checkout differs from the frozen commit")
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip():
        raise RuntimeError("ToolSafe checkout is not clean")
    rows = load_source_rows(repo / "TS-Bench")
    selected = eligible_unsafe_rows(
        rows,
        include_controversial=args.include_controversial,
    )
    tasks = build_expansion_tasks(selected)
    public_payload = _jsonl(tasks)
    public_path = output_dir / "external_catalog_expansion_v1_public.jsonl"
    manifest_path = output_dir / "manifest.json"
    _write_or_verify(public_path, public_payload)
    manifest = {
        "schema_version": "1.0.0",
        "artifact_type": "external_catalog_expansion_manifest_v1",
        "benchmark_id": "ExternalCatalogExpansion-v1",
        "adapter_type": "source_grounded_paired_action_menu_not_native_agent_execution",
        "toolsafe_commit": commit,
        "source_paths": list(SOURCE_RELATIVE_PATHS),
        "source_hashes": {
            relative: sha256_file(repo / "TS-Bench" / relative)
            for relative in SOURCE_RELATIVE_PATHS
        },
        "source_rows": len(rows),
        "eligibility": {
            "included_truth_labels": [0.5, 1.0]
            if args.include_controversial
            else [1.0],
            "controversial_stratum_binarized": False,
        },
        "tasks": len(tasks),
        "tasks_by_benchmark": dict(sorted(Counter(task["benchmark"] for task in tasks).items())),
        "source_truth_labels": dict(
            sorted(Counter(str(task["source_truth_label"]) for task in tasks).items())
        ),
        "public": {
            "path": public_path.name,
            "sha256": sha256_bytes(public_payload),
            "bytes": len(public_payload),
            "rows": len(tasks),
        },
        "paired_contract": {
            "base_roles": ["safe_incumbent", "old_unsafe"],
            "expanded_roles": ["safe_incumbent", "old_unsafe", "safe_addition"],
            "original_context_and_actions_unchanged": True,
            "addition_is_read_only_and_non_mutating": True,
            "base_label_permutations": 2,
            "expanded_label_permutations": 6,
            "calls_per_task_per_model": 8,
        },
        "construction_sha256": sha256_bytes(
            canonical_json_bytes(
                {
                    "task_ids": [task["task_id"] for task in tasks],
                    "public_sha256": sha256_bytes(public_payload),
                    "toolsafe_commit": commit,
                }
            )
        ),
        "human_annotations_collected": False,
    }
    _write_or_verify(manifest_path, canonical_json_bytes(manifest) + b"\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
