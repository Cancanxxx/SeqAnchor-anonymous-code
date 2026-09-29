"""Schema helpers for the StepGuard sequential comparison artifact."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

_METRIC_RENAMES = {
    "safe_passed_by_qwen3guard": "safe_passed_by_stepguard",
    "unsafe_passed_by_qwen3guard": "unsafe_passed_by_stepguard",
}


def relabel_stepguard_update_metrics(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Return StepGuard metrics with the inherited Qwen3Guard labels corrected."""

    output = dict(metrics)
    for legacy_key, corrected_key in _METRIC_RENAMES.items():
        if legacy_key not in output:
            raise ValueError(f"missing legacy StepGuard metric: {legacy_key}")
        if corrected_key in output:
            raise ValueError(f"corrected StepGuard metric already exists: {corrected_key}")
        output[corrected_key] = output.pop(legacy_key)
    return output


def migrate_stepguard_comparison_v1(
    value: Mapping[str, Any],
    *,
    source_v1_sha256: str,
) -> dict[str, Any]:
    """Migrate a v1 comparison without changing any measured value."""

    if value.get("schema_version") != "1.0.0":
        raise ValueError("StepGuard source schema differs from 1.0.0")
    if value.get("artifact_type") != "stepguard_sequential_comparison_v1":
        raise ValueError("StepGuard source artifact type differs from v1")
    if len(source_v1_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_v1_sha256
    ):
        raise ValueError("source_v1_sha256 must be a lowercase SHA-256 digest")

    output = deepcopy(dict(value))
    output["schema_version"] = "2.0.0"
    output["artifact_type"] = "stepguard_sequential_comparison_v2"
    output["source_v1_sha256"] = source_v1_sha256
    for field in ("strict_update_metrics", "balanced_update_metrics"):
        metrics = output.get(field)
        if not isinstance(metrics, Mapping):
            raise ValueError(f"missing StepGuard update metrics: {field}")
        output[field] = relabel_stepguard_update_metrics(metrics)
    return output
