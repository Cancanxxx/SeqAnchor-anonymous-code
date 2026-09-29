#!/usr/bin/env python3
"""Correct inherited guard labels in a StepGuard sequential comparison artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nontransitive_safety.provenance import canonical_json_bytes, sha256_file
from option_set_instability.stepguard_schema_v2 import (
    migrate_stepguard_comparison_v1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.v1.resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    output = migrate_stepguard_comparison_v1(
        value,
        source_v1_sha256=sha256_file(source),
    )
    payload = canonical_json_bytes(output) + b"\n"
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != payload:
        raise RuntimeError(f"existing StepGuard v2 comparison differs: {target}")
    if not target.exists():
        target.write_bytes(payload)
    print(target)


if __name__ == "__main__":
    main()
