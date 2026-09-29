"""Deterministic hashing and manifest helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


def _reject_duplicate_json_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def loads_json_strict(text: str, *, source: str = "JSON") -> Any:
    """Parse UTF-8 JSON while rejecting duplicate keys at every object depth."""

    if not isinstance(text, str):
        raise TypeError(f"{source} input must be text")
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{source} is not strict duplicate-free JSON: {error}") from error


def load_json_strict(path: Path) -> Any:
    """Read one UTF-8 JSON artifact and reject duplicate keys recursively."""

    return loads_json_strict(path.read_text(encoding="utf-8"), source=str(path))


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> Optional[str]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def runtime_identity(repo: Path) -> Dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": git_commit(repo),
    }


def build_manifest(
    *,
    run_id: str,
    repo: Path,
    config: Mapping[str, Any],
    inputs: Mapping[str, Path],
    outputs: Mapping[str, Path],
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "runtime": runtime_identity(repo),
        "config": dict(config),
        "config_sha256": sha256_bytes(canonical_json_bytes(config)),
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in outputs.items()
        },
    }
