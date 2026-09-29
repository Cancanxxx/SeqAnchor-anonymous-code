#!/usr/bin/env python3
"""Analyze the authorized Phi-4 and Qwen3-8B SeqAnchor safety replication."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from nontransitive_safety.provenance import canonical_json_bytes, loads_json_strict
from option_set_instability.seqanchor_multisafety_replication_v1 import (
    DIAGNOSTIC_VARIANTS,
    PRIMARY_VARIANT,
    analyze_replication,
    reject_duplicate_cli_options,
    write_once_bytes,
)


REPO = Path(__file__).resolve().parents[1]


def _parse_run(value: str) -> tuple[str, Path]:
    snapshot_id, separator, raw_path = value.partition("=")
    if not separator or not snapshot_id or not raw_path:
        raise argparse.ArgumentTypeError("--run must be SNAPSHOT_ID=/absolute/run/directory")
    path = Path(raw_path)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("--run directory must be absolute")
    return snapshot_id, path


def _write_once(path: Path, payload: bytes) -> None:
    write_once_bytes(path, payload)


def _fraction(metric: Mapping[str, Any]) -> str:
    return f"{metric['events']}/{metric['total']} ({100.0 * metric['rate']:.1f}%)"


def _summary_csv(result: Mapping[str, Any]) -> bytes:
    rows = []
    for snapshot_id in result["model_order"]:
        model = result["models"][snapshot_id]
        variants = {PRIMARY_VARIANT: model["primary"], **model["diagnostic_ablations"]}
        for variant_id in (PRIMARY_VARIANT, *DIAGNOSTIC_VARIANTS):
            candidate = variants[variant_id]["candidate_metrics"]
            sequence = variants[variant_id]["sequence_metrics"]
            rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "model_id": model["model_id"],
                    "analysis": "primary" if variant_id == PRIMARY_VARIANT else "diagnostic_ablation",
                    "variant_id": variant_id,
                    "unsafe_safety_passes": _fraction(candidate["unsafe_safety_passes"]),
                    "unsafe_admissions": _fraction(candidate["unsafe_admissions"]),
                    "any_unsafe_sequence": _fraction(sequence["any_unsafe_admission"]),
                    "useful_safe_adoption": _fraction(
                        sequence["useful_safe_improvement_adopted"]
                    ),
                    "exact_sequences": _fraction(sequence["exact_sequence_correct"]),
                }
            )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=REPO / "configs" / "seqanchor_multisafety_replication_analysis_v1.json",
    )
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run,
        required=True,
        metavar="SNAPSHOT_ID=/ABSOLUTE/RUN/DIRECTORY",
        help="Repeat exactly once for each predeclared model.",
    )
    parser.add_argument("--preflight-authorization", type=Path, required=True)
    parser.add_argument("--analysis-authorization", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    reject_duplicate_cli_options(sys.argv[1:], repeatable=("--run",))
    args = parse_args()
    run_dirs = dict(args.run)
    if len(run_dirs) != len(args.run):
        raise ValueError("--run contains a duplicate snapshot ID")
    spec_path = args.spec.resolve()
    spec = loads_json_strict(spec_path.read_text(encoding="utf-8"), source=str(spec_path))
    result = analyze_replication(
        spec,
        spec_path=spec_path,
        repo=REPO,
        run_dirs=run_dirs,
        preflight_path=args.preflight_authorization.resolve(),
        analysis_authorization_path=args.analysis_authorization.resolve(),
    )
    _write_once(args.output_json.resolve(), canonical_json_bytes(result) + b"\n")
    _write_once(args.output_csv.resolve(), _summary_csv(result))
    for snapshot_id in result["model_order"]:
        primary = result["models"][snapshot_id]["primary"]
        print(snapshot_id, primary["candidate_metrics"], primary["sequence_metrics"])


if __name__ == "__main__":
    main()
