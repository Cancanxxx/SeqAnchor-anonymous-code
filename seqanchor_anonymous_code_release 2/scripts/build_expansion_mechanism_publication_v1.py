#!/usr/bin/env python3
"""Build publication artifacts for the complete mechanism diagnostic panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from option_set_instability.expansion_mechanism_publication_v1 import (
    generate_artifacts,
    load_bundle,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO / "results/expansion_mechanism_v1",
        help="directory containing the official multimodel JSON and CSV artifacts",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO / "configs/expansion_mechanism_diagnostics_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/expansion_mechanism_v1/publication_artifacts",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate an interim subset without emitting publication artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_bundle(
        args.input_dir,
        args.protocol,
        require_complete=not args.validate_only,
    )
    if args.validate_only:
        expected = len(bundle.protocol["models"])
        result = {
            "status": "valid_complete_bundle"
            if len(bundle.model_order) == expected
            else "valid_interim_bundle",
            "models_found": list(bundle.model_order),
            "model_count": len(bundle.model_order),
            "expected_model_count": expected,
            "artifacts_written": False,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    outputs = generate_artifacts(bundle, args.output_dir)
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
