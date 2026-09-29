#!/usr/bin/env python3
"""Freeze balanced latency panels from the sequential test sequences."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from nontransitive_safety.provenance import canonical_json_bytes, sha256_file
from option_set_instability.anchorrc_v1 import load_jsonl_strict
from option_set_instability.sequential_efficiency_v1 import (
    first_arrival_panel,
    select_tsguard_latency_rows,
)


REPO = Path(__file__).resolve().parents[1]


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--tsguard-adaptation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences_path = args.sequences.resolve()
    adaptation_path = args.tsguard_adaptation.resolve()
    panel = first_arrival_panel(load_jsonl_strict(sequences_path))
    selected = select_tsguard_latency_rows(
        panel, load_jsonl_strict(adaptation_path)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in selected)
    )
    manifest = {
        "schema_version": "1.0.0",
        "artifact_type": "sequential_latency_panel_manifest_v1",
        "selection": "all_60_test_sequences_first_arrival",
        "sequence_source": _display_path(sequences_path),
        "sequence_source_sha256": sha256_file(sequences_path),
        "tsguard_source": _display_path(adaptation_path),
        "tsguard_source_sha256": sha256_file(adaptation_path),
        "updates": len(panel),
        "arrival_position": 1,
        "candidate_role_counts": dict(
            sorted(Counter(row["candidate_role"] for row in panel).items())
        ),
        "anchorrc_calls_per_update": {"safety": 6, "utility": 4},
        "tsguard_rows": len(selected),
        "tsguard_panel_sha256": sha256_file(args.output),
        "human_annotations_collected": False,
    }
    args.manifest.write_bytes(canonical_json_bytes(manifest) + b"\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
