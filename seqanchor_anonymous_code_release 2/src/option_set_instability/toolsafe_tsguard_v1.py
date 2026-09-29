"""Pinned, source-preserving adapter for ToolSafe's TS-Bench and TS-Guard.

The adapter never edits the upstream ToolSafe repository.  It loads the
official prompt and parser modules from a caller-supplied checkout, binds
their file hashes, and constructs a deterministic inference plan over the
published trajectory JSON files.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

from nontransitive_safety.provenance import canonical_json_bytes, sha256_bytes, sha256_file

SCHEMA_VERSION = "1.0.0"
TOOLSAFE_COMMIT = "46358fa424a927a895c6c8322f99032c4eb5155e"
TSGUARD_REVISION = "ad2f82df3ae3e0dcdaed7b7c71e8b7f0b16ead7a"

SOURCE_RELATIVE_PATHS = (
    "agentdojo-traj/banking.json",
    "agentdojo-traj/slack.json",
    "agentdojo-traj/travel.json",
    "agentdojo-traj/workspace.json",
    "agentharm-traj/benign_steps.json",
    "agentharm-traj/harmful_steps.json",
    "asb-traj/test/DPI_attack_success.json",
    "asb-traj/test/OPI_attack_success.json",
    "asb-traj/test/atttack_failure.json",
)
REQUIRED_ROW_KEYS = {
    "id-interaction",
    "id-segment",
    "instruction",
    "history",
    "current_action",
    "env_info",
    "score",
}
VALID_LABELS = {0.0, 0.5, 1.0}


def _load_module(path: Path, name: str) -> ModuleType:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or symlinked upstream module: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot construct module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


@dataclass(frozen=True)
class ToolSafeInterface:
    """Exact upstream prompt and output parser with source-file bindings."""

    prompt_template: str
    prompt_source_sha256: str
    parser_source_sha256: str
    toolsafe_commit: str = TOOLSAFE_COMMIT

    def render(self, row: Mapping[str, Any]) -> str:
        return self.prompt_template.format(
            user_request=row["instruction"],
            agent_action={
                "interaction_history": row["history"],
                "current_action": row["current_action"],
            },
            env_info=row["env_info"],
        )

    def as_record(self) -> Mapping[str, str]:
        return {
            "toolsafe_commit": self.toolsafe_commit,
            "prompt_source_sha256": self.prompt_source_sha256,
            "parser_source_sha256": self.parser_source_sha256,
            "prompt_template_sha256": hashlib.sha256(
                self.prompt_template.encode("utf-8")
            ).hexdigest(),
        }


def load_toolsafe_interface(toolsafe_repo: Path) -> tuple[ToolSafeInterface, Any]:
    """Load ToolSafe's official TS-Guard prompt and parser from a pinned tree."""

    repo = toolsafe_repo.resolve()
    prompt_path = repo / "src" / "agent" / "agent_prompts.py"
    parser_path = repo / "src" / "utils" / "guardian_parser.py"
    prompt_module = _load_module(prompt_path, "_toolsafe_agent_prompts_v1")
    parser_module = _load_module(parser_path, "_toolsafe_guardian_parser_v1")
    template = getattr(prompt_module, "instruction_ashellguardian_template_v2", None)
    parser = getattr(parser_module, "ashellguardian_parser_v2", None)
    if not isinstance(template, str) or not template:
        raise RuntimeError("ToolSafe TS-Guard prompt template is missing")
    if not callable(parser):
        raise RuntimeError("ToolSafe TS-Guard parser is missing")
    return (
        ToolSafeInterface(
            prompt_template=template,
            prompt_source_sha256=sha256_file(prompt_path),
            parser_source_sha256=sha256_file(parser_path),
        ),
        parser,
    )


def normalize_label(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean TS-Bench score")
    try:
        label = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("non-numeric TS-Bench score") from error
    if label not in VALID_LABELS:
        raise ValueError(f"unsupported TS-Bench score {label}")
    return label


def _row_id(source_relative_path: str, row_index: int, row: Mapping[str, Any]) -> str:
    identity = {
        "source_relative_path": source_relative_path,
        "row_index": row_index,
        "id_interaction": row["id-interaction"],
        "id_segment": row["id-segment"],
        "row_sha256": sha256_bytes(canonical_json_bytes(row)),
    }
    return "tsb_" + sha256_bytes(canonical_json_bytes(identity))[:20]


def benchmark_name(source_relative_path: str) -> str:
    if source_relative_path.startswith("agentharm-traj/"):
        return "agentharm"
    if source_relative_path.startswith("agentdojo-traj/"):
        return "agentdojo"
    if source_relative_path.startswith("asb-traj/"):
        return "agent_safety_bench"
    raise ValueError(f"unrecognized TS-Bench source {source_relative_path!r}")


def load_source_rows(
    ts_bench_root: Path,
    *,
    source_relative_paths: Sequence[str] = SOURCE_RELATIVE_PATHS,
) -> tuple[Mapping[str, Any], ...]:
    """Load strict public trajectory rows and attach deterministic identities."""

    root = ts_bench_root.resolve()
    selected = tuple(str(value) for value in source_relative_paths)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("source path list is empty or contains duplicates")
    if any(value not in SOURCE_RELATIVE_PATHS for value in selected):
        raise ValueError("source path outside the frozen TS-Bench list")
    output = []
    for relative in selected:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or symlinked TS-Bench source: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"TS-Bench source is not an array: {relative}")
        source_sha256 = sha256_file(path)
        for index, row in enumerate(value):
            if not isinstance(row, Mapping) or not REQUIRED_ROW_KEYS <= set(row):
                raise ValueError(f"malformed TS-Bench row: {relative}:{index}")
            for key in ("instruction", "history", "current_action", "env_info"):
                if not isinstance(row[key], str):
                    raise ValueError(f"non-string {key}: {relative}:{index}")
            output.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "row_id": _row_id(relative, index, row),
                    "benchmark": benchmark_name(relative),
                    "source_relative_path": relative,
                    "source_sha256": source_sha256,
                    "row_index": index,
                    "id_interaction": row["id-interaction"],
                    "id_segment": row["id-segment"],
                    "truth_label": normalize_label(row["score"]),
                    "instruction": row["instruction"],
                    "history": row["history"],
                    "current_action": row["current_action"],
                    "env_info": row["env_info"],
                }
            )
    return tuple(output)


def build_prompt_plan(
    rows: Sequence[Mapping[str, Any]],
    interface: ToolSafeInterface,
    tokenizer: Any,
    *,
    max_input_tokens: int,
) -> tuple[Mapping[str, Any], ...]:
    """Render and tokenize a length-bucketed deterministic inference plan."""

    if isinstance(max_input_tokens, bool) or not isinstance(max_input_tokens, int):
        raise ValueError("max_input_tokens must be an integer")
    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive")
    plan = []
    for row in rows:
        prompt_body = interface.render(row)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_body}],
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        if len(token_ids) > max_input_tokens:
            raise RuntimeError(
                f"TS-Guard prompt exceeds max_input_tokens: {row['row_id']} "
                f"({len(token_ids)} > {max_input_tokens})"
            )
        plan.append(
            {
                "row_id": row["row_id"],
                "benchmark": row["benchmark"],
                "source_relative_path": row["source_relative_path"],
                "source_sha256": row["source_sha256"],
                "row_index": row["row_index"],
                "truth_label": row["truth_label"],
                "input_tokens": len(token_ids),
                "prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                "rendered_prompt": rendered,
            }
        )
    return tuple(
        sorted(
            plan,
            key=lambda value: (
                value["input_tokens"],
                value["source_relative_path"],
                value["row_index"],
            ),
        )
    )


def parse_tsguard_output(text: str, upstream_parser: Any) -> Mapping[str, Any]:
    """Normalize ToolSafe's official parser result without changing its labels."""

    result = upstream_parser(text)
    if not isinstance(result, tuple) or len(result) != 2:
        return {"valid": False, "prediction": None, "details": {}, "raw_parser": repr(result)}
    prediction, details = result
    if prediction not in VALID_LABELS:
        return {
            "valid": False,
            "prediction": None,
            "details": details if isinstance(details, Mapping) else {},
            "raw_parser": repr(result),
        }
    return {
        "valid": True,
        "prediction": float(prediction),
        "details": dict(details) if isinstance(details, Mapping) else {},
        "raw_parser": None,
    }


__all__ = [
    "SOURCE_RELATIVE_PATHS",
    "TOOLSAFE_COMMIT",
    "TSGUARD_REVISION",
    "ToolSafeInterface",
    "benchmark_name",
    "build_prompt_plan",
    "load_source_rows",
    "load_toolsafe_interface",
    "normalize_label",
    "parse_tsguard_output",
]
