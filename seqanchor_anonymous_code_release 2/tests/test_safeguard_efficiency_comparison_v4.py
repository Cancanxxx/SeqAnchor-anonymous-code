from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "build_safeguard_efficiency_comparison_v4.py"
    spec = importlib.util.spec_from_file_location("build_safeguard_efficiency_comparison_v4", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _guard(*, accuracy: float, peak: int, unsafe_pass: float | None = None):
    value = {
        "model": "guard",
        "calls_per_update": 1,
        "batch_size": 1,
        "max_peak_gpu_bytes": peak,
        "strict_candidate_accuracy": accuracy,
        "parse_failures": 0,
        "latency_seconds": {"median": 1.0, "p95": 2.0},
    }
    if unsafe_pass is not None:
        value["unsafe_pass_rate"] = unsafe_pass
    return value


def _v3():
    return {
        "schema_version": "3.0.0",
        "artifact_type": "safeguard_efficiency_comparison_v3",
        "panel": {
            "updates": 60,
            "batch_unit": "one_catalog_update",
            "hardware": "NVIDIA A100 80GB PCIe",
            "warm_model": True,
            "candidate_roles": {
                "S1": 10,
                "E": 10,
                "N": 10,
                "C_GR": 10,
                "C_GA": 10,
                "C_RA": 10,
            },
        },
        "anchorrc": {
            "model": "anchor",
            "calls_per_update": 10,
            "forward_passes_per_update": 2,
            "resident_model_bytes": 1000,
            "max_incremental_peak_gpu_bytes": 200,
            "latency_seconds": {"median": 1.0, "p95": 2.0},
        },
        "qwen3guard": _guard(accuracy=0.5, peak=300, unsafe_pass=0.8),
        "tsguard": _guard(accuracy=0.6, peak=400),
        "stepguard": {
            **_guard(accuracy=0.7, peak=500, unsafe_pass=0.0),
            "timing_scope": "cuda_synchronized_model_generate_only",
        },
    }


def test_v4_scopes_update_batching_sync_safety_and_memory() -> None:
    module = _module()
    source = _v3()
    untouched = copy.deepcopy(source)

    value = module.correct_v3(source)

    assert source == untouched
    assert value["panel"]["concurrent_updates_per_timed_observation"] == 1
    anchor = value["anchorrc"]
    assert anchor["execution_scope"]["internal_batch_composition"] == {
        "safety_views_in_first_batch": 6,
        "utility_views_in_second_batch": 4,
    }
    assert anchor["execution_scope"]["internal_model_batches_per_update"] == 2
    assert anchor["timing_scope"]["explicit_cuda_synchronization_before_and_after_timed_region"]
    assert anchor["memory_scope"]["peak_allocated_gpu_bytes"] == 1200

    assert not value["qwen3guard"]["timing_scope"][
        "explicit_cuda_synchronization_before_and_after_timed_region"
    ]
    assert not value["tsguard"]["timing_scope"][
        "explicit_cuda_synchronization_before_and_after_timed_region"
    ]
    assert value["stepguard"]["timing_scope"][
        "explicit_cuda_synchronization_before_and_after_timed_region"
    ]
    qwen = value["qwen3guard"]
    assert "strict_candidate_accuracy" not in qwen
    assert "unsafe_pass_rate" not in qwen
    assert (
        qwen["latency_panel_safety_metrics"]["strict_admission_accuracy_on_60_update_latency_panel"]
        == 0.5
    )
    assert (
        qwen["latency_panel_safety_metrics"]["unsafe_pass_rate_on_30_unsafe_latency_updates"] == 0.8
    )
    assert value["stepguard"]["memory_scope"]["peak_allocated_gpu_bytes"] == 500


def test_v4_transform_is_deterministic_and_rejects_wrong_schema() -> None:
    module = _module()
    source = _v3()
    assert module.correct_v3(source) == module.correct_v3(source)
    source["artifact_type"] = "wrong"
    try:
        module.correct_v3(source)
    except RuntimeError as exc:
        assert "not v3" in str(exc)
    else:
        raise AssertionError("wrong artifact type was accepted")
