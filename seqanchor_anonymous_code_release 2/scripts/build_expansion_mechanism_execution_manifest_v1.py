#!/usr/bin/env python3
"""Audit and bind the complete expansion-mechanism execution bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
REPORT_NAME = "expansion_mechanism_multimodel_analysis_v1.json"
PUBLICATION_MANIFEST_NAME = "expansion_publication_artifacts_manifest_v1.json"
ANALYSIS_FILES = (
    REPORT_NAME,
    "expansion_mechanism_summary_v1.csv",
    "expansion_mechanism_updates_v1.csv",
    "expansion_mechanism_families_v1.csv",
)
SOURCE_FILES = (
    "configs/expansion_mechanism_diagnostics_v1.json",
    "scripts/run_expansion_mechanism_v1.py",
    "scripts/analyze_expansion_mechanism_v1.py",
    "scripts/build_expansion_mechanism_publication_v1.py",
    "scripts/build_expansion_mechanism_execution_manifest_v1.py",
    "src/option_set_instability/expansion_mechanism_v1.py",
    "src/option_set_instability/expansion_mechanism_analysis_v1.py",
    "src/option_set_instability/expansion_mechanism_publication_v1.py",
    "src/option_set_instability/state_expansion_180.py",
    "src/option_set_instability/workflow_simulator_v1.py",
    "src/nontransitive_safety/chat_rendering.py",
)
MODEL_SPECIFIC_PLAN_FIELDS = {
    "call_id",
    "input_tokens",
    "rendered_prompt_sha256",
    "snapshot_key",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_line(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def audit_run(run_dir: Path, protocol_sha256: str) -> tuple[Mapping[str, Any], str]:
    paths = {name: run_dir / name for name in (
        "identity.json", "metadata.json", "plan.jsonl", "logits.jsonl", "completion.json"
    )}
    require(all(path.is_file() for path in paths.values()), f"incomplete run directory: {run_dir}")
    identity = load_json(paths["identity.json"])
    metadata = load_json(paths["metadata.json"])
    completion = load_json(paths["completion.json"])
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    require(identity["protocol_sha256"] == protocol_sha256, "run uses a different protocol")
    require(completion["families"] == 180 and completion["rows"] == 28080, "wrong run size")
    for name in ("identity", "metadata", "plan", "logits"):
        suffix = ".json" if name in {"identity", "metadata"} else ".jsonl"
        require(
            completion[f"{name}_sha256"] == hashes[f"{name}{suffix}"],
            f"{name} hash differs",
        )
    require(
        metadata["identity_sha256"] == hashes["identity.json"],
        "metadata identity hash differs",
    )
    require(metadata["plan_sha256"] == hashes["plan.jsonl"], "metadata plan hash differs")

    semantic_cells: list[str] = []
    rows = 0
    with (
        paths["plan.jsonl"].open(encoding="utf-8") as planned,
        paths["logits.jsonl"].open(encoding="utf-8") as observed,
    ):
        while True:
            plan_line = planned.readline()
            logit_line = observed.readline()
            require(bool(plan_line) == bool(logit_line), "plan/logit row counts differ")
            if not plan_line:
                break
            plan_row = json.loads(plan_line)
            logit_row = json.loads(logit_line)
            require(
                plan_row["call_id"] == logit_row["call_id"],
                f"call sequence differs at row {rows}",
            )
            require(logit_row["ordinal"] == rows, f"non-contiguous ordinal at row {rows}")
            for key, value in plan_row.items():
                require(
                    logit_row.get(key) == value,
                    f"logged plan field differs at row {rows}: {key}",
                )
            semantic_cells.append(
                canonical_line(
                    {
                        key: value
                        for key, value in plan_row.items()
                        if key not in MODEL_SPECIFIC_PLAN_FIELDS
                    }
                )
            )
            rows += 1
    require(rows == 28080, f"wrong verified row count: {rows}")
    payload = ("\n".join(sorted(semantic_cells)) + "\n").encode("utf-8")
    semantic_plan_sha256 = hashlib.sha256(payload).hexdigest()
    return {
        "run_dir": str(run_dir),
        "model_id": completion["model_id"],
        "model_revision": completion["model_revision"],
        "run_id": completion["run_id"],
        "families": completion["families"],
        "rows": completion["rows"],
        "input_tokens": completion["input_tokens"],
        "max_peak_gpu_bytes": completion["max_peak_gpu_bytes"],
        "runtime": metadata["runtime"],
        "artifact_sha256": hashes,
        "exact_call_sequence_verified": True,
    }, semantic_plan_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=REPO / SOURCE_FILES[0])
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--publication-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    input_root = args.input_root.resolve()
    analysis_dir = args.analysis_dir.resolve()
    publication_dir = args.publication_dir.resolve()
    output = (
        args.output
        or analysis_dir / "expansion_mechanism_execution_manifest_v1.json"
    ).resolve()
    protocol = load_json(protocol_path)
    protocol_sha256 = sha256_file(protocol_path)
    report = load_json(analysis_dir / REPORT_NAME)
    require(report["protocol_sha256"] == protocol_sha256, "analysis protocol hash differs")
    require(
        report["model_count"] == 6
        and report["model_pooling_for_inference"] is False,
        "analysis is not the complete no-pooling panel",
    )

    runs: dict[str, Mapping[str, Any]] = {}
    semantic_hashes: dict[str, str] = {}
    for spec in protocol["models"]:
        model_key = str(spec["key"])
        matches = list(input_root.glob(f"*/{model_key}/completion.json"))
        require(len(matches) == 1, f"expected one completion for {model_key}, found {len(matches)}")
        run, semantic_hash = audit_run(matches[0].parent, protocol_sha256)
        require(run["model_id"] == spec["model_id"], f"model ID differs: {model_key}")
        require(run["model_revision"] == spec["revision"], f"revision differs: {model_key}")
        reported = report["run_artifacts"][model_key]
        require(
            reported["completion_sha256"]
            == run["artifact_sha256"]["completion.json"],
            "analysis completion binding differs",
        )
        require(
            reported["logits_sha256"] == run["artifact_sha256"]["logits.jsonl"],
            "analysis logits binding differs",
        )
        runs[model_key] = {"panel_role": spec["panel_role"], **run}
        semantic_hashes[model_key] = semantic_hash
    require(
        len(set(semantic_hashes.values())) == 1,
        "semantic call-cell multisets differ across models",
    )

    publication_manifest_path = publication_dir / PUBLICATION_MANIFEST_NAME
    publication_manifest = load_json(publication_manifest_path)
    for name, expected in publication_manifest["artifact_sha256"].items():
        filename = {
            "forest_pdf": "expansion_margin_shift_forest_v1.pdf",
            "direct_table": "expansion_direct_reversals_table_v1.tex",
            "dissociation_table": "expansion_joint_independent_table_v1.tex",
            "presentation_table": "expansion_presentation_effects_table_v1.tex",
            "semantic_table": "expansion_semantic_contrasts_table_v1.tex",
            "local_accuracy_table": "expansion_candidate_local_accuracy_table_v1.tex",
            "macros": "expansion_mechanism_macros_v1.tex",
        }[name]
        require(
            sha256_file(publication_dir / filename) == expected,
            f"publication artifact differs: {name}",
        )

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "status", "--short"], cwd=REPO, text=True
    ).splitlines()
    manifest = {
        "schema_version": "1.0.0",
        "artifact_type": "expansion_mechanism_execution_manifest_v1",
        "protocol_sha256": protocol_sha256,
        "common_semantic_plan_sha256": next(iter(semantic_hashes.values())),
        "semantic_plan_definition": (
            "SHA256 of sorted canonical plan rows after removing call_id, "
            "input_tokens, rendered_prompt_sha256, and snapshot_key"
        ),
        "exact_call_sequence_verified_for_all_models": True,
        "model_pooling_for_inference": False,
        "human_annotations_collected": False,
        "models": runs,
        "analysis_artifact_sha256": {
            name: sha256_file(analysis_dir / name) for name in ANALYSIS_FILES
        },
        "publication_manifest_sha256": sha256_file(publication_manifest_path),
        "publication_artifact_sha256": publication_manifest["artifact_sha256"],
        "source_sha256": {name: sha256_file(REPO / name) for name in SOURCE_FILES},
        "git": {
            "commit": git_commit,
            "working_tree_clean": not git_status,
            "status_short": git_status,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_line(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output)}, indent=2))


if __name__ == "__main__":
    main()
