#!/usr/bin/env python3
"""Build the frozen growing-catalog reselection baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from nontransitive_safety.provenance import canonical_json_bytes, sha256_bytes
from option_set_instability.sequential_analysis_v1 import load_jsonl
from option_set_instability.sequential_full_catalog_v1 import build_full_catalog_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequential-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"existing artifact differs: {path}")
    path.write_bytes(payload)


def main() -> None:
    args = parse_args()
    source = args.sequential_data.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    public, truth = build_full_catalog_tasks(
        load_jsonl(source / "sequences_truth.jsonl"),
        load_jsonl(source / "safety_tasks_public.jsonl"),
    )
    public_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in public)
    truth_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in truth)
    manifest = {
        "schema_version": "1.0.0",
        "artifact_type": "sequential_full_catalog_manifest_v1",
        "construction_status": "frozen_before_full_catalog_inference",
        "streams": len(public),
        "steps_per_stream_including_base": 7,
        "calls_per_model": 60 * 26,
        "public_sha256": sha256_bytes(public_payload),
        "truth_sha256": sha256_bytes(truth_payload),
        "source_manifest_sha256": sha256_bytes((source / "manifest.json").read_bytes()),
        "human_annotations_collected": False,
    }
    _write_once(output / "streams_public.jsonl", public_payload)
    _write_once(output / "streams_truth.jsonl", truth_payload)
    _write_once(output / "manifest.json", canonical_json_bytes(manifest) + b"\n")
    print(f"built {len(public)} growing-catalog streams and {60 * 26} calls per model")


if __name__ == "__main__":
    main()
