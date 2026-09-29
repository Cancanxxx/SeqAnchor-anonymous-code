#!/usr/bin/env python3
"""Analyze every complete model run in the mechanism diagnostic panel."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from nontransitive_safety.provenance import canonical_json_bytes, sha256_file
from option_set_instability.expansion_mechanism_analysis_v1 import analyze_model

REPO = Path(__file__).resolve().parents[1]


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


def _complete_run(input_root: Path, model_key: str) -> tuple[Path, Mapping[str, Any]]:
    matches = []
    for completion_path in input_root.glob(f"*/{model_key}/completion.json"):
        completion = _load_json(completion_path)
        if completion.get("families") == 180 and completion.get("rows") == 28080:
            matches.append((completion_path.parent, completion))
    if len(matches) != 1:
        raise RuntimeError(f"expected one complete full run for {model_key}, found {len(matches)}")
    run_dir, completion = matches[0]
    logits_path = run_dir / "logits.jsonl"
    if sha256_file(logits_path) != completion["logits_sha256"]:
        raise RuntimeError(f"logits hash differs for {model_key}")
    return run_dir, completion


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = list(rows)
    if not values:
        raise ValueError("CSV output cannot be empty")
    keys = sorted({key for row in values for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in values:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=REPO / "configs/expansion_mechanism_diagnostics_v1.json"
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-key", action="append", dest="model_keys")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = _load_json(protocol_path)
    if protocol.get("status") != "frozen_before_new_model_inference":
        raise RuntimeError("mechanism protocol is not frozen")
    seed = int(protocol["analysis"]["bootstrap_seed"])
    reports = {}
    summary_rows = []
    all_updates = []
    all_families = []
    run_artifacts = {}
    selected_models = (
        {str(value) for value in args.model_keys}
        if args.model_keys
        else {str(spec["key"]) for spec in protocol["models"]}
    )
    known_models = {str(spec["key"]) for spec in protocol["models"]}
    if not selected_models.issubset(known_models):
        raise ValueError("requested analysis model is outside the frozen panel")
    for model_index, model_spec in enumerate(protocol["models"]):
        model_key = str(model_spec["key"])
        if model_key not in selected_models:
            continue
        run_dir, completion = _complete_run(args.input_root.resolve(), model_key)
        logits = _load_jsonl(run_dir / "logits.jsonl")
        result = analyze_model(
            logits,
            bootstrap_seed=seed + 1000 * model_index,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        reports[model_key] = result["summary"]
        run_artifacts[model_key] = {
            "panel_role": model_spec["panel_role"],
            "model_id": model_spec["model_id"],
            "revision": model_spec["revision"],
            "run_dir": str(run_dir),
            "completion_sha256": sha256_file(run_dir / "completion.json"),
            "logits_sha256": completion["logits_sha256"],
        }
        for addition_role, role_result in result["summary"]["by_addition_role"].items():
            summary_rows.append(
                {
                    "model_key": model_key,
                    "panel_role": model_spec["panel_role"],
                    "addition_role": addition_role,
                    "eligible_safe_base": role_result["eligible_safe_base"],
                    "direct_reversals": role_result["direct_reversal"]["events"],
                    "direct_reversal_rate": role_result["direct_reversal"]["rate"],
                    "margin_flips": role_result["pairwise_margin_flip"]["events"],
                    "margin_flip_rate": role_result["pairwise_margin_flip"]["rate"],
                    "mean_margin_shift": role_result["margin_shift"]["mean"],
                    "median_margin_shift": role_result["margin_shift"]["median"],
                    "fraction_shift_toward_U0": role_result["margin_shift"]["fraction_toward_U0"],
                    "any_view_selects_U0_rate": role_result["presentation"]["any_view_selects_U0"][
                        "rate"
                    ],
                    "view_sign_change_rate": role_result["presentation"][
                        "margin_sign_changes_across_views"
                    ]["rate"],
                }
            )
        all_updates.extend({"model_key": model_key, **row} for row in result["update_rows"])
        all_families.extend({"model_key": model_key, **row} for row in result["family_rows"])

    report = {
        "schema_version": "1.0.0",
        "artifact_type": "expansion_mechanism_multimodel_analysis_v1",
        "protocol_sha256": sha256_file(protocol_path),
        "bootstrap_resamples": args.bootstrap_resamples,
        "model_count": len(reports),
        "model_pooling_for_inference": False,
        "run_artifacts": run_artifacts,
        "models": reports,
        "human_annotations_collected": False,
    }
    output_dir = args.output_dir.resolve()
    _write_json(output_dir / "expansion_mechanism_multimodel_analysis_v1.json", report)
    _write_csv(output_dir / "expansion_mechanism_summary_v1.csv", summary_rows)
    _write_csv(output_dir / "expansion_mechanism_updates_v1.csv", all_updates)
    _write_csv(output_dir / "expansion_mechanism_families_v1.csv", all_families)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
