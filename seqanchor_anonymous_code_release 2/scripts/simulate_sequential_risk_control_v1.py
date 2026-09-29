#!/usr/bin/env python3
"""Run frozen Monte Carlo checks for Sequential AnchorRC risk accounting."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from nontransitive_safety.provenance import canonical_json_bytes

SEED = 20260901
TRIALS = 100_000
CALIBRATION_FAMILIES = 120
HORIZON = 6
ALPHA = 0.05


def _two_sided_wilson(events: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    probability = events / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total**2)
    ) / denominator
    return center - half, center + half


def _one_sided_wilson_upper(
    events: int, total: int, z: float = 1.6448536269514722
) -> float:
    probability = events / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total**2)
    ) / denominator
    return center + half


def _scenario(
    rng: np.random.Generator,
    *,
    trials: int,
    calibration_prototypes: int,
    correlation: float,
    test_shift: float,
    adaptive_candidates_per_step: int,
    batch_size: int = 1000,
) -> dict:
    events = 0
    realized = 0
    while realized < trials:
        size = min(batch_size, trials - realized)
        calibration_latent = rng.standard_normal((size, CALIBRATION_FAMILIES, 1))
        calibration_noise = rng.standard_normal(
            (size, CALIBRATION_FAMILIES, calibration_prototypes)
        )
        calibration = (
            math.sqrt(correlation) * calibration_latent
            + math.sqrt(1.0 - correlation) * calibration_noise
        )
        threshold = calibration.max(axis=(1, 2))
        test = (
            rng.standard_normal((size, HORIZON, adaptive_candidates_per_step))
            + test_shift
        )
        selected = test.max(axis=2)
        events += int(np.any(selected > threshold[:, None], axis=1).sum())
        realized += size
    lower, upper = _two_sided_wilson(events, trials)
    return {
        "events": events,
        "trials": trials,
        "rate": events / trials,
        "wilson_95pct": [lower, upper],
        "one_sided_wilson_95pct_upper": _one_sided_wilson_upper(events, trials),
        "calibration_prototypes_per_family": calibration_prototypes,
        "within_family_gaussian_correlation": correlation,
        "test_mean_shift": test_shift,
        "adaptive_candidates_selected_per_step": adaptive_candidates_per_step,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=TRIALS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    rng = np.random.default_rng(SEED)
    scenarios = {
        "exchangeable_single_prototype": _scenario(
            rng,
            trials=args.trials,
            calibration_prototypes=1,
            correlation=0.0,
            test_shift=0.0,
            adaptive_candidates_per_step=1,
        ),
        "exchangeable_family_max_two_correlated": _scenario(
            rng,
            trials=args.trials,
            calibration_prototypes=2,
            correlation=0.75,
            test_shift=0.0,
            adaptive_candidates_per_step=1,
        ),
        "distribution_shift_stress": _scenario(
            rng,
            trials=args.trials,
            calibration_prototypes=1,
            correlation=0.0,
            test_shift=0.5,
            adaptive_candidates_per_step=1,
        ),
        "adaptive_best_of_four_stress": _scenario(
            rng,
            trials=args.trials,
            calibration_prototypes=1,
            correlation=0.0,
            test_shift=0.0,
            adaptive_candidates_per_step=4,
        ),
        "adaptive_best_of_four_matched_family_max": _scenario(
            rng,
            trials=args.trials,
            calibration_prototypes=4,
            correlation=0.0,
            test_shift=0.0,
            adaptive_candidates_per_step=4,
        ),
    }
    result = {
        "schema_version": "1.0.0",
        "artifact_type": "sequential_risk_control_simulation_v1",
        "seed": SEED,
        "trials_per_scenario": args.trials,
        "calibration_families": CALIBRATION_FAMILIES,
        "horizon": HORIZON,
        "total_alpha": ALPHA,
        "per_step_alpha": 1 / 120,
        "minimum_rank_p_value": 1 / 121,
        "exact_exchangeable_single_prototype_sequence_probability": HORIZON
        / (CALIBRATION_FAMILIES + HORIZON),
        "scenarios": scenarios,
        "interpretation": {
            "confirmatory": [
                "exchangeable_single_prototype",
                "exchangeable_family_max_two_correlated",
                "adaptive_best_of_four_matched_family_max",
            ],
            "scope_stress_only": [
                "distribution_shift_stress",
                "adaptive_best_of_four_stress",
            ],
        },
        "human_annotations_collected": False,
    }
    payload = canonical_json_bytes(result) + b"\n"
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != payload:
        raise RuntimeError(f"existing simulation differs: {target}")
    target.write_bytes(payload)
    for name, row in scenarios.items():
        print(name, row["events"], row["rate"], row["wilson_95pct"])


if __name__ == "__main__":
    main()
