#!/usr/bin/env python3
"""Replay the frozen SeqAnchor calibration-component ablation."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any, Mapping

from nontransitive_safety.provenance import canonical_json_bytes
from option_set_instability.sequential_calibration_ablation_v1 import analyze_from_spec

REPO = Path(__file__).resolve().parents[1]


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"existing ablation artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _fraction(value: Mapping[str, Any]) -> str:
    return f"{value['events']}/{value['total']} ({100.0 * value['rate']:.1f}%)"


def _summary_csv(result: Mapping[str, Any], labels: Mapping[str, str]) -> bytes:
    rows = []
    for variant_id in result["variant_order"]:
        value = result["variants"][variant_id]
        candidate = value["candidate_metrics"]
        sequence = value["sequence_metrics"]
        rows.append(
            {
                "variant_id": variant_id,
                "variant": labels[variant_id],
                "unsafe_safety_passes": _fraction(candidate["unsafe_safety_passes"]),
                "unsafe_admissions": _fraction(candidate["unsafe_admissions"]),
                "unsafe_sequences": _fraction(sequence["any_unsafe_admission"]),
                "safe_useful_safety_passes": _fraction(
                    candidate["safe_useful_safety_passes"]
                ),
                "safe_useful_admissions": _fraction(candidate["safe_useful_admissions"]),
                "useful_sequences": _fraction(
                    sequence["useful_safe_improvement_adopted"]
                ),
                "safe_nonimproving_safety_passes": _fraction(
                    candidate["safe_nonimproving_safety_passes"]
                ),
                "safe_nonimproving_admissions": _fraction(
                    candidate["safe_nonimproving_admissions"]
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
        default=REPO / "configs/sequential_anchorrc_calibration_ablation_v1.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO / "results/sequential_anchorrc_calibration_ablation_v1.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO / "results/sequential_anchorrc_calibration_ablation_v1.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    result = analyze_from_spec(spec, spec_path=spec_path, repo=REPO)
    labels = {str(value["id"]): str(value["label"]) for value in spec["variants"]}
    _write_once(args.output_json.resolve(), canonical_json_bytes(result) + b"\n")
    _write_once(args.output_csv.resolve(), _summary_csv(result, labels))
    for variant_id in result["variant_order"]:
        value = result["variants"][variant_id]
        print(variant_id, value["candidate_metrics"], value["sequence_metrics"])


if __name__ == "__main__":
    main()
