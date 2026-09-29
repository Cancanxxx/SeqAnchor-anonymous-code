#!/usr/bin/env python3
"""Build the frozen AgentHarm native first-action expansion benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from nontransitive_safety.provenance import canonical_json_bytes
from option_set_instability.agentharm_native_expansion_v1 import (
    build_agentharm_native_tasks,
    build_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-file", required=True, type=Path)
    parser.add_argument("--harmful-tools-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"existing artifact differs: {path}")
    path.write_bytes(payload)


def main() -> None:
    args = parse_args()
    dataset = args.dataset_file.resolve()
    tools = args.harmful_tools_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tasks = build_agentharm_native_tasks(dataset, tools)
    manifest = build_manifest(tasks, dataset, tools)
    _write_once(
        output / "tasks_public.jsonl",
        b"".join(canonical_json_bytes(row) + b"\n" for row in tasks),
    )
    _write_once(output / "manifest.json", canonical_json_bytes(manifest) + b"\n")
    print(f"built {len(tasks)} AgentHarm public-test paired first-action tasks")


if __name__ == "__main__":
    main()
