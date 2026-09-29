"""Fail-closed registry for arbitrary local Hugging Face causal language models."""

from __future__ import annotations

import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set

from .chat_rendering import (
    CHAT_RENDERER_ID,
    validate_template_declaration,
)
from .provenance import (
    canonical_json_bytes,
    load_json_strict,
    sha256_bytes,
    sha256_file,
    sha256_text,
)

REGISTRY_SCHEMA_VERSION = "2.0.0"
REGISTRY_ARTIFACT_TYPE = "frozen_local_hf_causal_lm_registry_v2"
CHAT_RENDERER_PATH = "src/nontransitive_safety/chat_rendering.py"
REGISTRY_KEYS = {
    "schema_version",
    "artifact_type",
    "status",
    "candidate_snapshot_id",
    "pretraining_family",
    "model_id",
    "revision",
    "tokenizer_revision",
    "local_snapshot",
    "offline_required",
    "chat_template_sha256",
    "chat_template_kwargs",
    "chat_template_kwargs_sha256",
    "chat_renderer",
    "snapshot_tree_sha256",
    "files",
    "runtime",
    "loading",
}
CHAT_RENDERER_KEYS = {"id", "path", "sha256"}
FILE_KEYS = {"sha256", "bytes", "role"}
RUNTIME_KEYS = {"python", "torch", "transformers", "accelerate", "interpreter"}
LOADING_KEYS = {
    "dtype",
    "attn_implementation",
    "device_map",
    "trust_remote_code",
}
FILE_ROLES = {"model_shard", "tokenizer", "config", "generation", "other"}
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
WEIGHT_INDEX_SUFFIXES = (".safetensors.index.json", ".bin.index.json")
SHARDED_WEIGHT_PATTERN = re.compile(r"-[0-9]{5}-of-[0-9]{5}(?:\.safetensors|\.bin)$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

CANONICAL_PANEL_CANDIDATES: Mapping[str, Mapping[str, Any]] = {
    "llama31-8b-instruct-0e9e39f": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "tokenizer_revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "pretraining_family": "Meta",
        "hidden_size": 4096,
        "chat_template_kwargs": {},
    },
    "phi-4-2db69c1": {
        "model_id": "microsoft/phi-4",
        "revision": "2db69c1c3e91a05d2c64a3185acfbaf36f744e25",
        "tokenizer_revision": "2db69c1c3e91a05d2c64a3185acfbaf36f744e25",
        "pretraining_family": "Phi",
        "hidden_size": 5120,
        "chat_template_kwargs": {},
    },
    "qwen25-14b-instruct-cf98f3b": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        "tokenizer_revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        "pretraining_family": "Qwen",
        "hidden_size": 5120,
        "chat_template_kwargs": {},
    },
    "qwen3-14b-40c0698": {
        "model_id": "Qwen/Qwen3-14B",
        "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
        "tokenizer_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
        "pretraining_family": "Qwen",
        "hidden_size": 5120,
        "chat_template_kwargs": {"enable_thinking": False},
    },
}
CANONICAL_PANEL_ORDER = (
    "llama31-8b-instruct-0e9e39f",
    "phi-4-2db69c1",
    "qwen25-14b-instruct-cf98f3b",
    "qwen3-14b-40c0698",
)

# Additional exact checkpoints admitted only to the portable-interface v9 study.
# Keeping these identities separate preserves the already-frozen four-candidate
# v2 predictor panel and all of its historical decisions.
PORTABLE_INTERFACE_CANDIDATES: Mapping[str, Mapping[str, Any]] = {
    "gemma11-7b-it-065a528": {
        "model_id": "google/gemma-1.1-7b-it",
        "revision": "065a528791af6f57f013e8e42b7276992b45ef71",
        "tokenizer_revision": "065a528791af6f57f013e8e42b7276992b45ef71",
        "pretraining_family": "Gemma",
        "hidden_size": 3072,
        "chat_template_kwargs": {},
    },
    "mistral7b-instruct-v02-3ad372f": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "revision": "3ad372fc79158a2148299e3318516c786aeded6c",
        "tokenizer_revision": "3ad372fc79158a2148299e3318516c786aeded6c",
        "pretraining_family": "Mistral",
        "hidden_size": 4096,
        "chat_template_kwargs": {},
    },
}
REGISTRY_CANDIDATES: Mapping[str, Mapping[str, Any]] = {
    **CANONICAL_PANEL_CANDIDATES,
    **PORTABLE_INTERFACE_CANDIDATES,
}
MODEL_PANEL_ID = "predictor-panel-v2-four-candidate"
MODEL_PANEL_SCHEMA_VERSION = "2.0.0"
MODEL_PANEL_ARTIFACT_TYPE = "prospective_four_candidate_model_panel_v2"
MODEL_PANEL_KEYS = {
    "schema_version",
    "artifact_type",
    "panel_id",
    "status",
    "candidate_count",
    "pretraining_family_units",
    "candidate_substitution_allowed",
    "engineering_smoke_authorized",
    "scientific_model_calls_authorized",
    "shared_chat_renderer",
    "candidates",
    "registry_set_sha256",
}
MODEL_PANEL_CANDIDATE_KEYS = {
    "candidate_snapshot_id",
    "pretraining_family",
    "hidden_size",
    "model_id",
    "revision",
    "tokenizer_revision",
    "model_registry_path",
    "model_registry_sha256",
    "snapshot_tree_sha256",
    "chat_template_sha256",
    "chat_template_kwargs",
    "chat_template_kwargs_sha256",
}


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], name: str) -> None:
    expected_set = set(expected)
    if set(value) != expected_set:
        raise ValueError(
            f"{name} keys differ; missing={sorted(expected_set - set(value))}, "
            f"unknown={sorted(set(value) - expected_set)}"
        )


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def validate_panel_candidate_identity(value: Mapping[str, Any]) -> None:
    """Require one exact prospectively frozen member of the four-model panel."""

    snapshot_id = _nonempty(
        value["candidate_snapshot_id"], "model registry candidate_snapshot_id"
    )
    expected = REGISTRY_CANDIDATES.get(snapshot_id)
    if expected is None:
        raise ValueError("model registry candidate is outside the frozen panel")
    for key in ("model_id", "revision", "tokenizer_revision", "pretraining_family"):
        if value[key] != expected[key]:
            raise ValueError(f"model registry {key} differs from the frozen panel")
    if value["chat_template_kwargs"] != expected["chat_template_kwargs"]:
        raise ValueError(
            "model registry chat_template_kwargs differ from the frozen panel"
        )


def snapshot_files(snapshot: Path) -> Dict[str, Path]:
    if not snapshot.is_dir():
        raise FileNotFoundError(f"local model snapshot is absent: {snapshot}")
    files = {
        path.relative_to(snapshot).as_posix(): path
        for path in snapshot.rglob("*")
        if path.is_file()
    }
    if not files:
        raise ValueError("local model snapshot contains no regular files")
    return dict(sorted(files.items()))


def snapshot_tree_identity(files: Mapping[str, Mapping[str, Any]]) -> str:
    projection = {
        path: {"sha256": item["sha256"], "bytes": item["bytes"]}
        for path, item in sorted(files.items())
    }
    return sha256_bytes(canonical_json_bytes(projection))


def _weight_file_names(relative_names: Iterable[str]) -> tuple[set[str], set[str]]:
    names = set(relative_names)
    indexes = {
        name for name in names if name.casefold().endswith(WEIGHT_INDEX_SUFFIXES)
    }
    weights = {
        name
        for name in names
        if name.casefold().endswith(WEIGHT_SUFFIXES) and name not in indexes
    }
    if not weights:
        raise ValueError("model snapshot contains no weight tensor file")
    if any(SHARDED_WEIGHT_PATTERN.search(Path(name).name) for name in weights) and not indexes:
        raise ValueError("sharded model snapshot has no weight index")
    return weights, indexes


def validate_weight_artifacts(files: Mapping[str, Path]) -> None:
    """Require real weights and complete index-to-shard references."""

    weights, indexes = _weight_file_names(files)
    for relative in weights:
        if files[relative].stat().st_size <= 0:
            raise ValueError(f"model weight tensor file is empty: {relative}")
    referenced: set[str] = set()
    for relative in indexes:
        try:
            value = load_json_strict(files[relative])
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"model weight index is unreadable: {relative}") from error
        if not isinstance(value, Mapping):
            raise ValueError(f"model weight index must contain an object: {relative}")
        weight_map = value.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ValueError(f"model weight index has no nonempty weight_map: {relative}")
        parent = Path(relative).parent
        for shard in weight_map.values():
            if not isinstance(shard, str) or not shard:
                raise ValueError(f"model weight index has an invalid shard name: {relative}")
            resolved = (parent / shard).as_posix()
            if Path(resolved).is_absolute() or ".." in Path(resolved).parts:
                raise ValueError(f"model weight index shard path escapes snapshot: {shard}")
            if resolved not in weights:
                raise ValueError(f"model weight index references an absent shard: {resolved}")
            referenced.add(resolved)
    sharded = {
        name for name in weights if SHARDED_WEIGHT_PATTERN.search(Path(name).name)
    }
    if sharded and referenced != sharded:
        raise ValueError(
            "weight index/shard set differs; "
            f"unreferenced={sorted(sharded - referenced)}, "
            f"unknown={sorted(referenced - sharded)}"
        )


def load_model_registry(
    path: Path, *, verify_files: bool = True, require_frozen: bool = True
) -> Mapping[str, Any]:
    value = load_json_strict(path)
    if not isinstance(value, Mapping):
        raise ValueError("model registry must contain an object")
    _exact_keys(value, REGISTRY_KEYS, "model registry")
    if value["schema_version"] != REGISTRY_SCHEMA_VERSION:
        raise ValueError("unsupported model registry schema")
    if value["artifact_type"] != REGISTRY_ARTIFACT_TYPE:
        raise ValueError("unexpected model registry artifact type")
    if value["status"] not in {"draft_unfrozen", "frozen_before_inference"}:
        raise ValueError("unknown model registry status")
    if require_frozen and value["status"] != "frozen_before_inference":
        raise RuntimeError("model registry is not frozen before inference")
    for key in ("model_id", "revision", "tokenizer_revision", "local_snapshot"):
        _nonempty(value[key], f"model registry {key}")
    _nonempty(value["pretraining_family"], "model registry pretraining_family")
    validate_template_declaration(
        chat_template_kwargs=value["chat_template_kwargs"],
        declared_sha256=value["chat_template_kwargs_sha256"],
    )
    validate_panel_candidate_identity(value)
    renderer = value["chat_renderer"]
    if not isinstance(renderer, Mapping):
        raise ValueError("model registry chat_renderer must be an object")
    _exact_keys(renderer, CHAT_RENDERER_KEYS, "model registry chat_renderer")
    if renderer["id"] != CHAT_RENDERER_ID or renderer["path"] != CHAT_RENDERER_PATH:
        raise ValueError("model registry chat renderer identity differs")
    if not isinstance(renderer["sha256"], str) or SHA256_PATTERN.fullmatch(
        renderer["sha256"]
    ) is None:
        raise ValueError("model registry chat renderer hash is invalid")
    renderer_path = Path(__file__).resolve().parents[2] / renderer["path"]
    if not renderer_path.is_file() or sha256_file(renderer_path) != renderer["sha256"]:
        raise RuntimeError("shared chat renderer source differs from the registry")
    if value["offline_required"] is not True:
        raise RuntimeError("confirmation inference requires an offline model registry")
    if not isinstance(value["chat_template_sha256"], str) or SHA256_PATTERN.fullmatch(
        value["chat_template_sha256"]
    ) is None:
        raise ValueError("chat_template_sha256 must be a SHA-256 digest")
    if not isinstance(value["snapshot_tree_sha256"], str) or SHA256_PATTERN.fullmatch(
        value["snapshot_tree_sha256"]
    ) is None:
        raise ValueError("snapshot_tree_sha256 must be a SHA-256 digest")
    snapshot = Path(value["local_snapshot"])
    if snapshot.name != value["revision"]:
        raise ValueError("snapshot directory name must equal the frozen model revision")
    files_value = value["files"]
    if not isinstance(files_value, Mapping) or not files_value:
        raise ValueError("model registry files must be a non-empty object")
    roles = set()
    for relative, item in files_value.items():
        _nonempty(relative, "model registry relative file path")
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"model registry file path escapes snapshot: {relative}")
        if not isinstance(item, Mapping):
            raise ValueError(f"model registry file {relative} must be an object")
        _exact_keys(item, FILE_KEYS, f"model registry file {relative}")
        if not isinstance(item["sha256"], str) or SHA256_PATTERN.fullmatch(
            item["sha256"]
        ) is None:
            raise ValueError(f"model registry file {relative} has invalid SHA-256")
        if isinstance(item["bytes"], bool) or not isinstance(item["bytes"], int):
            raise ValueError(f"model registry file {relative} has invalid byte count")
        if item["bytes"] < 0 or item["role"] not in FILE_ROLES:
            raise ValueError(f"model registry file {relative} has invalid metadata")
        roles.add(item["role"])
    if "model_shard" not in roles or "tokenizer" not in roles or "config" not in roles:
        raise ValueError("registry must identify model_shard, tokenizer, and config files")
    _weight_file_names(files_value)
    if snapshot_tree_identity(files_value) != value["snapshot_tree_sha256"]:
        raise ValueError("model registry snapshot-tree hash is internally inconsistent")
    runtime = value["runtime"]
    loading = value["loading"]
    if not isinstance(runtime, Mapping) or not isinstance(loading, Mapping):
        raise ValueError("model runtime/loading declarations must be objects")
    _exact_keys(runtime, RUNTIME_KEYS, "model runtime")
    _exact_keys(loading, LOADING_KEYS, "model loading")
    for key in RUNTIME_KEYS:
        _nonempty(runtime[key], f"model runtime {key}")
    if loading["dtype"] not in {"bfloat16", "float16", "float32"}:
        raise ValueError("unsupported frozen model dtype")
    if not isinstance(loading["device_map"], Mapping) or set(loading["device_map"]) != {""}:
        raise ValueError("device_map must contain exactly the root model mapping")
    device = loading["device_map"][""]
    if isinstance(device, bool) or not isinstance(device, (int, str)):
        raise ValueError("root device_map value must be a device index or device string")
    if not isinstance(loading["attn_implementation"], str) or not loading[
        "attn_implementation"
    ]:
        raise ValueError("attn_implementation must be a frozen non-empty string")
    if loading["trust_remote_code"] is not False:
        raise ValueError("confirmation runner forbids trust_remote_code")

    if verify_files:
        actual = snapshot_files(snapshot)
        if set(actual) != set(files_value):
            raise RuntimeError(
                "snapshot file set differs from registry; "
                f"missing={sorted(set(files_value)-set(actual))}, "
                f"unregistered={sorted(set(actual)-set(files_value))}"
            )
        for relative, file_path in actual.items():
            metadata = files_value[relative]
            if file_path.stat().st_size != metadata["bytes"]:
                raise RuntimeError(f"model file byte count differs: {relative}")
            if sha256_file(file_path) != metadata["sha256"]:
                raise RuntimeError(f"model file hash differs: {relative}")
        validate_weight_artifacts(actual)
    return value


def registry_chat_template_kwargs(registry: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the already validated model-specific renderer kwargs."""

    return validate_template_declaration(
        chat_template_kwargs=registry["chat_template_kwargs"],
        declared_sha256=registry["chat_template_kwargs_sha256"],
    )


def canonical_model_registry_path(candidate_snapshot_id: str) -> str:
    if candidate_snapshot_id not in CANONICAL_PANEL_CANDIDATES:
        raise ValueError("candidate snapshot is outside the frozen panel")
    return f"manifests/{candidate_snapshot_id}_confirmation_model_registry_v2.json"


def model_panel_registry_payload(repo: Path) -> Mapping[str, Any]:
    """Build the blocked whole-panel closure from the four exact registries."""

    candidates = []
    renderer: Optional[Mapping[str, Any]] = None
    for snapshot_id in CANONICAL_PANEL_ORDER:
        relative = canonical_model_registry_path(snapshot_id)
        path = repo / relative
        registry = load_model_registry(path, verify_files=False, require_frozen=True)
        expected = CANONICAL_PANEL_CANDIDATES[snapshot_id]
        current_renderer = dict(registry["chat_renderer"])
        if renderer is None:
            renderer = current_renderer
        elif current_renderer != renderer:
            raise ValueError("candidate registries do not share one renderer identity")
        candidates.append(
            {
                "candidate_snapshot_id": snapshot_id,
                "pretraining_family": expected["pretraining_family"],
                "hidden_size": expected["hidden_size"],
                "model_id": registry["model_id"],
                "revision": registry["revision"],
                "tokenizer_revision": registry["tokenizer_revision"],
                "model_registry_path": relative,
                "model_registry_sha256": sha256_file(path),
                "snapshot_tree_sha256": registry["snapshot_tree_sha256"],
                "chat_template_sha256": registry["chat_template_sha256"],
                "chat_template_kwargs": registry["chat_template_kwargs"],
                "chat_template_kwargs_sha256": registry[
                    "chat_template_kwargs_sha256"
                ],
            }
        )
    assert renderer is not None
    projection = {"shared_chat_renderer": renderer, "candidates": candidates}
    return {
        "schema_version": MODEL_PANEL_SCHEMA_VERSION,
        "artifact_type": MODEL_PANEL_ARTIFACT_TYPE,
        "panel_id": MODEL_PANEL_ID,
        "status": "blocked_pending_audits_smoke_and_predictor_handoff",
        "candidate_count": len(candidates),
        "pretraining_family_units": ["Meta", "Phi", "Qwen"],
        "candidate_substitution_allowed": False,
        "engineering_smoke_authorized": False,
        "scientific_model_calls_authorized": False,
        "shared_chat_renderer": renderer,
        "candidates": candidates,
        "registry_set_sha256": sha256_bytes(canonical_json_bytes(projection)),
    }


def load_model_panel_registry(
    path: Path, *, repo: Path, verify_snapshot_files: bool = False
) -> Mapping[str, Any]:
    """Validate the exact four-candidate panel and every linked registry."""

    value = load_json_strict(path)
    if not isinstance(value, Mapping):
        raise ValueError("model panel registry must contain an object")
    _exact_keys(value, MODEL_PANEL_KEYS, "model panel registry")
    if (
        value["schema_version"] != MODEL_PANEL_SCHEMA_VERSION
        or value["artifact_type"] != MODEL_PANEL_ARTIFACT_TYPE
        or value["panel_id"] != MODEL_PANEL_ID
    ):
        raise ValueError("model panel registry identity differs")
    if value["status"] != "blocked_pending_audits_smoke_and_predictor_handoff":
        raise ValueError("model panel registry has an unsupported status")
    if value["candidate_count"] != 4 or value["pretraining_family_units"] != [
        "Meta",
        "Phi",
        "Qwen",
    ]:
        raise ValueError("model panel size or family units differ")
    for key in (
        "candidate_substitution_allowed",
        "engineering_smoke_authorized",
        "scientific_model_calls_authorized",
    ):
        if value[key] is not False:
            raise RuntimeError("blocked model panel may not authorize calls or substitution")
    rows = value["candidates"]
    if not isinstance(rows, list) or len(rows) != len(CANONICAL_PANEL_ORDER):
        raise ValueError("model panel candidates must be the exact four-member list")
    renderer = value["shared_chat_renderer"]
    projection_rows = []
    for snapshot_id, row in zip(CANONICAL_PANEL_ORDER, rows):
        if not isinstance(row, Mapping):
            raise ValueError("model panel candidate must be an object")
        _exact_keys(row, MODEL_PANEL_CANDIDATE_KEYS, "model panel candidate")
        expected = CANONICAL_PANEL_CANDIDATES[snapshot_id]
        for key in (
            "candidate_snapshot_id",
            "pretraining_family",
            "hidden_size",
            "model_id",
            "revision",
            "tokenizer_revision",
            "chat_template_kwargs",
        ):
            expected_value = snapshot_id if key == "candidate_snapshot_id" else expected[key]
            if row[key] != expected_value:
                raise ValueError(f"model panel candidate {key} differs")
        relative = canonical_model_registry_path(snapshot_id)
        if row["model_registry_path"] != relative:
            raise ValueError("model panel registry path differs")
        registry_path = repo / relative
        if sha256_file(registry_path) != row["model_registry_sha256"]:
            raise RuntimeError("model panel linked registry hash differs")
        registry = load_model_registry(
            registry_path,
            verify_files=verify_snapshot_files,
            require_frozen=True,
        )
        for panel_key, registry_key in (
            ("snapshot_tree_sha256", "snapshot_tree_sha256"),
            ("chat_template_sha256", "chat_template_sha256"),
            ("chat_template_kwargs_sha256", "chat_template_kwargs_sha256"),
        ):
            if row[panel_key] != registry[registry_key]:
                raise RuntimeError(f"model panel {panel_key} differs from registry")
        if dict(registry["chat_renderer"]) != renderer:
            raise RuntimeError("model panel renderer differs from a candidate registry")
        projection_rows.append(dict(row))
    observed = sha256_bytes(
        canonical_json_bytes(
            {"shared_chat_renderer": renderer, "candidates": projection_rows}
        )
    )
    if observed != value["registry_set_sha256"]:
        raise ValueError("model panel registry-set digest differs")
    return value


def validate_runtime_versions(registry: Mapping[str, Any]) -> None:
    import accelerate
    import torch
    import transformers

    observed = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "interpreter": str(Path(sys.executable).resolve()),
    }
    expected = dict(registry["runtime"])
    expected["interpreter"] = str(Path(expected["interpreter"]).resolve())
    if observed != expected:
        raise RuntimeError(f"runtime identity differs: observed={observed}, expected={expected}")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) not in {"1", "true", "TRUE"}:
            raise RuntimeError(f"{name} must require offline inference")


def load_tokenizer(registry: Mapping[str, Any]):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        registry["local_snapshot"],
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    observed = sha256_text(tokenizer.chat_template or "")
    if observed != registry["chat_template_sha256"]:
        raise RuntimeError("loaded tokenizer chat-template hash differs from registry")
    return tokenizer


def load_model(registry: Mapping[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[registry["loading"]["dtype"]]
    model = AutoModelForCausalLM.from_pretrained(
        registry["local_snapshot"],
        local_files_only=True,
        dtype=dtype,
        attn_implementation=registry["loading"]["attn_implementation"],
        device_map=dict(registry["loading"]["device_map"]),
        trust_remote_code=False,
    )
    model.eval()
    return model


def resolve_declared_stop_token_ids(registry: Mapping[str, Any], tokenizer) -> Set[int]:
    from transformers import AutoConfig, GenerationConfig

    model_config = AutoConfig.from_pretrained(
        registry["local_snapshot"], local_files_only=True, trust_remote_code=False
    )
    try:
        generation_config = GenerationConfig.from_pretrained(
            registry["local_snapshot"], local_files_only=True
        )
    except OSError:
        generation_config = GenerationConfig.from_model_config(model_config)
    result: Set[int] = set()
    for declared in (
        tokenizer.eos_token_id,
        model_config.eos_token_id,
        generation_config.eos_token_id,
    ):
        if isinstance(declared, int):
            result.add(declared)
        elif isinstance(declared, Sequence):
            if any(isinstance(item, bool) or not isinstance(item, int) for item in declared):
                raise ValueError("EOS token list must contain only integers")
            result.update(declared)
        elif declared is not None:
            raise ValueError(f"unsupported EOS declaration {declared!r}")
    if not result:
        raise RuntimeError("no stop token can be resolved from the frozen local model")
    return result
