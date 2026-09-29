#!/usr/bin/env python3
"""Analyze a completed growing-catalog reselection run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nontransitive_safety.provenance import canonical_json_bytes, sha256_file
from option_set_instability.sequential_analysis_v1 import load_jsonl
from option_set_instability.sequential_full_catalog_analysis_v1 import (
    aggregate_full_catalog_winners,
    analyze_full_catalog_sequences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    completion = run / "completion.json"
    if not completion.is_file():
        raise RuntimeError("full-catalog evidence is incomplete")
    winners = aggregate_full_catalog_winners(load_jsonl(run / "logits.jsonl"))
    result = analyze_full_catalog_sequences(winners, load_jsonl(args.truth.resolve()))
    completed = json.loads(completion.read_text(encoding="utf-8"))
    output = {
        "schema_version": "1.0.0",
        "artifact_type": "sequential_full_catalog_analysis_v1",
        "snapshot_id": completed["snapshot_id"],
        "model_id": completed["model_id"],
        "completion_sha256": sha256_file(completion),
        **result,
        "human_annotations_collected": False,
    }
    payload = canonical_json_bytes(output) + b"\n"
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != payload:
        raise RuntimeError(f"existing analysis differs: {target}")
    target.write_bytes(payload)
    print(json.dumps(output["sequence_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
