#!/usr/bin/env python3
"""Rebuild the controlled workflow benchmark from its source specifications."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from nontransitive_safety.provenance import canonical_json_bytes
from option_set_instability.state_expansion_180 import (
    BENCHMARK_ID,
    SPLITS,
    build_private_bank_v5,
    build_public_bank,
    default_spec_paths,
    exact_split_assignment_provenance,
    load_workflow_specs,
    serialize_jsonl,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SEED = REPO / "data" / "workflow_specs" / "split_seed.json"
DEFAULT_OUTPUT = REPO / "generated" / "state_expansion_180_v5"
EXPECTED_SHA256 = {
    "state_expansion_180_public.jsonl": (
        "60ffed9fa77c119d7499ef30c01297702cd1ac0f4eadabd217572cb9793b1b09"
    ),
    "state_expansion_180_safety_calibration_private.jsonl": (
        "417f0c13cc21188921474ee13772e2e22ae8fbdcbc7371aef14818f390431ca1"
    ),
    "state_expansion_180_benefit_calibration_private.jsonl": (
        "9602c6f2317aebb78a26286d12f2f346cfd6b3893167e1ce630732e2af0f576b"
    ),
    "state_expansion_180_sealed_test_private.jsonl": (
        "cc8b1e8e502df3edb85f0f86568ba53df04f936c21f20175ba9ed2aa70f4b878"
    ),
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_seed(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("split seed must be a JSON object")
    required = {"schema_version", "source", "uri", "timestamp", "outputs"}
    if set(value) != required:
        raise ValueError("split seed fields differ from the release schema")
    outputs = value["outputs"]
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("split seed must contain at least one beacon output")
    return value


def _write(path: Path, payload: bytes, *, overwrite: bool) -> None:
    if path.exists() and path.read_bytes() == payload:
        return
    if path.exists() and not overwrite:
        raise RuntimeError(f"existing generated file differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = load_workflow_specs(default_spec_paths(REPO))
    seed = _load_seed(args.seed.resolve())
    split = exact_split_assignment_provenance(
        specs,
        beacon_outputs=tuple(str(value) for value in seed["outputs"]),
        beacon_uri=str(seed["uri"]),
        beacon_timestamp=str(seed["timestamp"]),
    )
    private_rows = build_private_bank_v5(specs, split_by_slug=split["assignment"])
    public_rows = build_public_bank(private_rows)

    payloads: dict[str, bytes] = {
        "state_expansion_180_public.jsonl": serialize_jsonl(public_rows)
    }
    split_counts = Counter(str(row["split"]) for row in private_rows)
    for split_name in SPLITS:
        rows = [row for row in private_rows if row["split"] == split_name]
        payloads[f"state_expansion_180_{split_name}_private.jsonl"] = serialize_jsonl(rows)

    observed = {name: _sha256(payload) for name, payload in payloads.items()}
    if observed != EXPECTED_SHA256:
        raise RuntimeError(
            "generated benchmark does not match the frozen release hashes: "
            + json.dumps(observed, sort_keys=True)
        )

    output_dir = args.output_dir.resolve()
    public_name = "state_expansion_180_public.jsonl"
    _write(output_dir / public_name, payloads[public_name], overwrite=args.overwrite)
    for split_name in SPLITS:
        name = f"state_expansion_180_{split_name}_private.jsonl"
        _write(output_dir / "private" / name, payloads[name], overwrite=args.overwrite)

    manifest = {
        "schema_version": "1.0.0",
        "artifact_type": "state_expansion_180_release_manifest_v1",
        "benchmark_id": BENCHMARK_ID,
        "family_count": 180,
        "task_count": 540,
        "split_counts": dict(sorted(split_counts.items())),
        "contains_human_annotations": False,
        "label_provenance": "deterministic_workflow_simulator_v1",
        "split_assignment": {
            key: value for key, value in split.items() if key != "assignment"
        },
        "files": {
            "public": {
                "path": public_name,
                "sha256": observed[public_name],
                "bytes": len(payloads[public_name]),
            },
            "private_by_split": {
                split_name: {
                    "path": (
                        "private/"
                        f"state_expansion_180_{split_name}_private.jsonl"
                    ),
                    "sha256": observed[
                        f"state_expansion_180_{split_name}_private.jsonl"
                    ],
                    "bytes": len(
                        payloads[f"state_expansion_180_{split_name}_private.jsonl"]
                    ),
                }
                for split_name in SPLITS
            },
        },
    }
    _write(
        output_dir / "manifest.json",
        canonical_json_bytes(manifest) + b"\n",
        overwrite=args.overwrite,
    )
    print(f"Rebuilt 180 workflow families and 540 tasks in {output_dir}")
    for name in sorted(observed):
        print(f"{observed[name]}  {name}")


if __name__ == "__main__":
    main()
