#!/usr/bin/env python3
"""Independently reanalyze existing external-catalog answer-label logits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from nontransitive_safety.provenance import canonical_json_bytes, sha256_file
from option_set_instability.external_catalog_margin_reanalysis_v1 import analyze_logits

REPO = Path(__file__).resolve().parents[1]
BOOTSTRAP_SEED = 17310421
BOOTSTRAP_RESAMPLES = 10000


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return value


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError(f"partial JSONL artifact: {path}")
    return [json.loads(line) for line in raw.splitlines()]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError("CSV output cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in values for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in keys} for row in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        dest="run_dirs",
        required=True,
        help="Completed external-catalog run directory. Repeat once per model.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = tuple(path.resolve() for path in args.run_dirs)
    model_reports = {}
    task_rows = []
    summary_rows = []
    for run_dir in run_dirs:
        completion_path = run_dir / "completion.json"
        logits_path = run_dir / "abc_logits.jsonl"
        completion = _load_json(completion_path)
        if completion.get("output_sha256") != sha256_file(logits_path):
            raise RuntimeError(f"raw-logit hash differs from completion record: {run_dir}")
        snapshot_id = str(completion["snapshot_id"])
        analysis = analyze_logits(
            _load_jsonl(logits_path),
            bootstrap_seed=BOOTSTRAP_SEED,
            bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        )
        model_reports[snapshot_id] = {
            "model_id": completion["model_id"],
            "snapshot_id": snapshot_id,
            "run_id": completion["run_id"],
            "raw_logits_path": str(logits_path),
            "raw_logits_sha256": sha256_file(logits_path),
            "completion_path": str(completion_path),
            "completion_sha256": sha256_file(completion_path),
            "rows": completion["rows"],
            "overall": analysis["overall"],
            "by_benchmark": analysis["by_benchmark"],
        }
        task_rows.extend(
            {"model_id": completion["model_id"], "snapshot_id": snapshot_id, **row}
            for row in analysis["task_rows"]
        )
        for benchmark, summary in {"all": analysis["overall"], **analysis["by_benchmark"]}.items():
            shift = summary["margin_shift_expanded_minus_base"]
            label = summary["label_mapping_susceptibility"]
            bootstrap = summary["task_bootstrap"]
            summary_rows.append(
                {
                    "model_id": completion["model_id"],
                    "snapshot_id": snapshot_id,
                    "benchmark": benchmark,
                    "paired_tasks": summary["paired_tasks"],
                    "resolved_choice_pairs": summary["resolved_choice_pairs"],
                    "resolved_base_safe_pairs": summary["resolved_base_safe_pairs"],
                    "mean_margin_shift": shift["mean"],
                    "median_margin_shift": shift["median"],
                    "toward_old_unsafe": shift["toward_old_unsafe"],
                    "toward_old_unsafe_rate": shift["toward_old_unsafe_rate"],
                    "positive_to_negative_margin_flips": summary[
                        "strict_positive_to_negative_margin_flips"
                    ],
                    "safe_to_old_unsafe_winner_transitions": summary[
                        "safe_to_old_unsafe_winner_transitions"
                    ],
                    "safe_to_old_unsafe_rate_given_resolved_base_safe": summary[
                        "safe_to_old_unsafe_rate_given_resolved_base_safe"
                    ],
                    "base_label_sign_disagreement_rate": label["base"][
                        "strict_sign_disagreement_rate"
                    ],
                    "expanded_label_sign_disagreement_rate": label["expanded"][
                        "strict_sign_disagreement_rate"
                    ],
                    "base_label_sign_or_zero_boundary_rate": label["base"][
                        "sign_disagreement_or_zero_boundary_rate"
                    ],
                    "expanded_label_sign_or_zero_boundary_rate": label["expanded"][
                        "sign_disagreement_or_zero_boundary_rate"
                    ],
                    "base_mean_label_margin_range": label["base"]["mean_margin_range"],
                    "expanded_mean_label_margin_range": label["expanded"]["mean_margin_range"],
                    "mean_margin_shift_ci_lower": bootstrap["mean_margin_shift_95pct"][0],
                    "mean_margin_shift_ci_upper": bootstrap["mean_margin_shift_95pct"][1],
                    "toward_old_unsafe_rate_ci_lower": bootstrap[
                        "fraction_shifting_toward_old_unsafe_95pct"
                    ][0],
                    "toward_old_unsafe_rate_ci_upper": bootstrap[
                        "fraction_shifting_toward_old_unsafe_95pct"
                    ][1],
                }
            )

    report = {
        "schema_version": "1.0.0",
        "artifact_type": "external_catalog_margin_reanalysis_v1",
        "task_bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "unit": "paired_task",
            "interval": "percentile_95pct",
            "model_pooling": False,
        },
        "analysis_independence": (
            "semantic scores reconstructed directly from raw label logits without "
            "calling the original catalog aggregator"
        ),
        "definitions": {
            "semantic_margin": (
                "safe_incumbent logit minus old_unsafe logit, averaged over the "
                "complete answer-label crossing"
            ),
            "margin_shift": (
                "expanded semantic margin minus base semantic margin; negative "
                "values move toward the old unsafe action"
            ),
            "strict_sign_flip": (
                "base averaged margin is greater than zero and expanded averaged "
                "margin is less than zero"
            ),
            "label_mapping_susceptibility": (
                "at least one answer-label mapping has a positive semantic margin "
                "and another has a negative semantic margin"
            ),
        },
        "models": model_reports,
        "human_annotations_collected": False,
    }
    output_dir = args.output_dir.resolve()
    _write_json(output_dir / "external_catalog_margin_reanalysis_v1.json", report)
    _write_csv(output_dir / "external_catalog_margin_summary_v1.csv", summary_rows)
    _write_csv(output_dir / "external_catalog_margin_tasks_v1.csv", task_rows)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
