from option_set_instability.sequential_comparators_v1 import (
    analyze_tsguard_sequential,
    fresh_utility_threshold,
)


def _calibration():
    truth = []
    scores = {}
    for index in range(120):
        by_role = {}
        for role in ("E", "N"):
            by_role[role] = {}
            for incumbent in ("S0", "S1"):
                task_id = f"cal-{index}-{role}-{incumbent}"
                scores[task_id] = float(index) / 1000
                by_role[role][incumbent] = task_id
        truth.append({"utility_task_id_by_role_and_incumbent": by_role})
    return truth, scores


def test_fresh_utility_threshold_uses_family_maxima():
    truth, scores = _calibration()
    assert fresh_utility_threshold(truth, scores) == 0.119


def test_tsguard_composition_replays_sequences():
    sequences = []
    adapters = []
    predictions = []
    utility_scores = {}
    roles = ("S1", "E", "N", "C_GR", "C_GA", "C_RA")
    for index in range(60):
        family = f"f-{index}"
        task_map = {}
        for role in roles:
            row_id = f"p-{index}-{role}"
            adapters.append(
                {
                    "row_id": row_id,
                    "family_id": family,
                    "candidate_role": role,
                    "truth_label": 0.0 if role in {"S1", "E", "N"} else 1.0,
                }
            )
            predictions.append(
                {
                    "row_id": row_id,
                    "truth_label": adapters[-1]["truth_label"],
                    "parse_valid": True,
                    "prediction": 0.0 if role in {"S1", "E", "N"} else 1.0,
                }
            )
            task_map[role] = {}
            for incumbent in ("S0", "S1"):
                task_id = f"u-{index}-{role}-{incumbent}"
                utility_scores[task_id] = 1.0 if role == "S1" else -1.0
                task_map[role][incumbent] = task_id
        sequences.append(
            {
                "test_index": index,
                "family_id": family,
                "arrival_roles": list(roles),
                "policy_compliant_by_role": {
                    role: role in {"S1", "E", "N"} for role in roles
                },
                "utility_task_id_by_role_and_incumbent": task_map,
            }
        )
    result = analyze_tsguard_sequential(
        sequences, adapters, predictions, utility_scores, 0.0
    )
    assert result["tsguard_safety_only"]["any_unsafe_admission"] == 0
    assert result["tsguard_plus_fresh_utility"]["useful_improvement_adopted"] == 60
    assert result["tsguard_plus_fresh_utility"]["exact_sequence_correct"] == 60
