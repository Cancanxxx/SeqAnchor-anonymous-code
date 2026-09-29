from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "build_sequential_safeguard_table_v4.py"
    spec = importlib.util.spec_from_file_location("build_sequential_safeguard_table_v4", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _method(unsafe: int, useful: int, passed: int):
    return {
        "sequence_metrics": {
            "any_unsafe_admission": unsafe,
            "useful_safe_improvement_adopted": useful,
        },
        "update_metrics": {"unsafe_safety_passes": passed},
    }


def _exact():
    return {
        "methods": {
            "Exact throughput only": _method(50, 10, 180),
            "Qwen3Guard + exact throughput": _method(46, 14, 154),
            "TS-Guard + exact throughput": _method(7, 36, 10),
            "StepGuard + exact throughput": _method(0, 0, 0),
            "SeqAnchor + exact throughput": _method(0, 60, 0),
        }
    }


def _learned():
    return {
        "balanced_per_update_utility_budget": {
            "calibration": {"implied_finite_sample_bound": {"numerator": 6, "denominator": 121}},
            "sequence_metrics": {
                "any_unsafe_admission": 0,
                "useful_improvement_adopted": 53,
            },
        }
    }


def test_v4_labels_exact_latency_as_non_upper_bound_proxy() -> None:
    module = _module()
    latency = {"median": 1.0, "p95": 2.0}
    rows = module.build_rows(
        exact_document=_exact(),
        full={
            "safe_to_unsafe_sequences": 17,
            "base_safe": 60,
            "useful_S1_selected_after_available": 58,
            "old_U0_resurrections": 12,
        },
        learned_document=_learned(),
        efficiency={
            name: {"latency_seconds": latency}
            for name in ("anchorrc", "qwen3guard", "tsguard", "stepguard")
        },
    )
    exact_anchor = next(
        row for row in rows if row["method"] == "SeqAnchor/Qwen2.5 + exact throughput"
    )
    learned_anchor = next(
        row for row in rows if row["method"] == "SeqAnchor/Qwen2.5 + learned balanced utility"
    )
    stepguard = next(row for row in rows if row["method"].startswith("StepGuard"))

    latency_label = exact_anchor["warm latency median / p95 (scope)"]
    assert "learned-utility-interface proxy" in latency_label
    assert "not an upper bound" in latency_label
    assert "simulator-executed" in exact_anchor["result provenance"]
    assert "not a deployable learned utility" in exact_anchor["result provenance"]
    assert "post hoc" in stepguard["result provenance"]

    protection = learned_anchor["formal protection"]
    assert "fixed six-update horizon" in protection
    assert "family exchangeability" in protection
    assert "calibration coverage" in protection
    assert "6/121" in protection
    assert "separately bounds each non-improving replacement" in protection
    assert "post-primary" in learned_anchor["result provenance"]
    assert "after the initial output was opened" in learned_anchor["result provenance"]
