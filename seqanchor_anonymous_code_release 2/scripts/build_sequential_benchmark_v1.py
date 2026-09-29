#!/usr/bin/env python3
"""Build the fresh calibration and six-update Sequential AnchorRC benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from nontransitive_safety.provenance import canonical_json_bytes
from option_set_instability.sequential_benchmark_v1 import build_sequential_benchmark
from option_set_instability.state_expansion_180 import load_workflow_specs

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SPECS = tuple(
    REPO / "data" / "workflow_specs" / name
    for name in ("part_a.json", "part_b.json", "part_c.json")
)


def _write_once(path: Path, payload: bytes, *, overwrite: bool) -> None:
    if path.exists():
        if path.read_bytes() == payload:
            return
        if not overwrite:
            raise RuntimeError(f"existing artifact differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "data" / "sequential_anchorrc_v1",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace differing generated artifacts in the explicitly named output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    bank = build_sequential_benchmark(load_workflow_specs(DEFAULT_SPECS))
    artifacts = {
        "safety_tasks_public.jsonl": _jsonl(bank["safety_tasks"]),
        "utility_tasks_public.jsonl": _jsonl(bank["utility_tasks"]),
        "calibration_truth.jsonl": _jsonl(bank["calibration_truth"]),
        "sequences_truth.jsonl": _jsonl(bank["sequences"]),
        "tsguard_adaptation.jsonl": _jsonl(bank["tsguard_rows"]),
        "manifest.json": canonical_json_bytes(bank["manifest"]) + b"\n",
    }
    for name, payload in artifacts.items():
        _write_once(output / name, payload, overwrite=args.overwrite)
    print(
        f"built {len(bank['safety_tasks'])} safety tasks, "
        f"{len(bank['utility_tasks'])} utility tasks, and "
        f"{len(bank['sequences'])} sequences in {output}"
    )


if __name__ == "__main__":
    main()
