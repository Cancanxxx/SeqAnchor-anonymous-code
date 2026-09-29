#!/usr/bin/env python3
"""Hash a complete local Hugging Face snapshot into a reviewable model registry."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict

from nontransitive_safety.chat_rendering import (
    CHAT_RENDERER_ID,
    chat_template_kwargs_sha256,
    validate_chat_template_kwargs,
)
from nontransitive_safety.local_hf_registry import (
    CHAT_RENDERER_PATH,
    REGISTRY_ARTIFACT_TYPE,
    REGISTRY_SCHEMA_VERSION,
    load_model_registry,
    snapshot_files,
    snapshot_tree_identity,
    validate_weight_artifacts,
)
from nontransitive_safety.provenance import loads_json_strict, sha256_file, sha256_text

REPO = Path(__file__).resolve().parents[1]


def write_json_exclusive(path: Path, value: Dict[str, Any]) -> None:
    """Create a model registry without overwriting an existing artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--candidate-snapshot-id", required=True)
    parser.add_argument(
        "--pretraining-family",
        choices=("Gemma", "Meta", "Mistral", "Phi", "Qwen"),
        required=True,
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), required=True)
    parser.add_argument("--attn-implementation", required=True)
    parser.add_argument(
        "--chat-template-kwargs-json",
        required=True,
        help='Exact canonical JSON object, for example {} or {"enable_thinking":false}',
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="mechanically freeze the exact registry identity and hashes",
    )
    return parser.parse_args()


def file_role(relative: str) -> str:
    name = Path(relative).name.casefold()
    if name.endswith((".safetensors", ".bin", ".pt", ".pth")) or name.endswith(
        (".safetensors.index.json", ".bin.index.json")
    ):
        return "model_shard"
    if name == "generation_config.json":
        return "generation"
    if name in {"config.json", "adapter_config.json"}:
        return "config"
    tokenizer_markers = (
        "tokenizer",
        "special_tokens",
        "added_tokens",
        "vocab",
        "merges",
        "sentencepiece",
        "spiece",
    )
    if any(marker in name for marker in tokenizer_markers) or name.endswith(".model"):
        return "tokenizer"
    return "other"


def parse_device(value: str) -> int | str:
    try:
        parsed = int(value)
    except ValueError:
        return value
    if parsed < 0:
        raise ValueError("device index must be non-negative")
    return parsed


def runtime_identity() -> Dict[str, str]:
    import accelerate
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "interpreter": str(Path(sys.executable).resolve()),
    }


def main() -> None:
    args = parse_args()
    chat_template_kwargs = validate_chat_template_kwargs(
        loads_json_strict(args.chat_template_kwargs_json)
    )
    snapshot = args.snapshot.resolve()
    if snapshot.name != args.revision:
        raise ValueError("snapshot directory name must equal the exact frozen revision")
    snapshot_content = snapshot_files(snapshot)
    validate_weight_artifacts(snapshot_content)
    files = {
        relative: {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "role": file_role(relative),
        }
        for relative, path in snapshot_content.items()
    }
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )
    registry: Dict[str, Any] = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "artifact_type": REGISTRY_ARTIFACT_TYPE,
        "status": "frozen_before_inference" if args.freeze else "draft_unfrozen",
        "candidate_snapshot_id": args.candidate_snapshot_id,
        "pretraining_family": args.pretraining_family,
        "model_id": args.model_id,
        "revision": args.revision,
        "tokenizer_revision": args.tokenizer_revision,
        "local_snapshot": str(snapshot),
        "offline_required": True,
        "chat_template_sha256": sha256_text(tokenizer.chat_template or ""),
        "chat_template_kwargs": chat_template_kwargs,
        "chat_template_kwargs_sha256": chat_template_kwargs_sha256(
            chat_template_kwargs
        ),
        "chat_renderer": {
            "id": CHAT_RENDERER_ID,
            "path": CHAT_RENDERER_PATH,
            "sha256": sha256_file(REPO / CHAT_RENDERER_PATH),
        },
        "snapshot_tree_sha256": snapshot_tree_identity(files),
        "files": files,
        "runtime": runtime_identity(),
        "loading": {
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "device_map": {"": parse_device(args.device)},
            "trust_remote_code": False,
        },
    }
    write_json_exclusive(args.output, registry)
    load_model_registry(
        args.output,
        verify_files=True,
        require_frozen=args.freeze,
    )
    print(json.dumps(registry, indent=2), flush=True)


if __name__ == "__main__":
    main()
