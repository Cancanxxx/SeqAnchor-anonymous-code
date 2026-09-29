#!/usr/bin/env python3
"""Build complete E/N-incumbent utility coverage for Sequential AnchorRC."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from nontransitive_safety.provenance import canonical_json_bytes
from option_set_instability.sequential_benchmark_v1 import (
    build_complete_test_utility_extension,
)
from option_set_instability.state_expansion_180 import load_workflow_specs


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SPECS = tuple(
    REPO / "data" / "workflow_specs" / name
    for name in ("part_a.json", "part_b.json", "part_c.json")
)


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"existing extension artifact differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "data" / "sequential_anchorrc_utility_extension_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bank = build_complete_test_utility_extension(load_workflow_specs(DEFAULT_SPECS))
    output = args.output_dir.resolve()
    _write_once(output / "utility_tasks_public.jsonl", _jsonl(bank["utility_tasks"]))
    _write_once(output / "extension_truth.jsonl", _jsonl(bank["extension_truth"]))
    _write_once(
        output / "manifest.json", canonical_json_bytes(bank["manifest"]) + b"\n"
    )
    print(f"built {len(bank['utility_tasks'])} utility extension tasks in {output}")


if __name__ == "__main__":
    main()
