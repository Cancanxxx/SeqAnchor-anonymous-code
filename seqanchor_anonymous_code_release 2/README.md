# SeqAnchor anonymous code artifact

This repository contains the code and source workflow specifications needed to reproduce the experiments for *Safe Additions, Unsafe Actions: Safety Reversals in LLM Agents*. It is intentionally code only. It contains no model outputs, run logs, result tables, cached files, manuscript source, credentials, or model weights.

The three JSON files in `data/workflow_specs/` are the source definitions for the 180 controlled workflow families. They are included because the benchmark cannot be regenerated without them. All task banks, private simulator labels, model outputs, and analyses are generated locally and are excluded from the release.

## Repository layout

- `data/workflow_specs/` contains 180 workflow specifications across 12 sectors and the public randomness used for the fixed 60/60/60 split.
- `scripts/build_controlled_workflows.py` deterministically renders the controlled safety reversal benchmark and verifies its frozen hashes.
- `scripts/build_sequential_benchmark_v1.py` creates the disjoint calibration and six update SeqAnchor benchmark.
- `src/option_set_instability/` contains the workflow simulator, safety reversal analyses, SeqAnchor controller, calibration code, comparator adapters, and statistical analyses.
- `scripts/` contains the benchmark builders, model runners, analyses, guard comparisons, latency measurements, and publication table builders.
- `configs/` contains the frozen experiment settings. Model registry paths are machine specific and must be bound locally before inference.
- `tests/` contains tests that do not require archived experimental outputs.

## Environment

The recorded inference environment used Python 3.9, PyTorch 2.8.0, Transformers 4.57.6, Accelerate 1.10.1, and CUDA 12.8. The analysis environment is CPU compatible.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis-lock.txt
python -m pip install -e . --no-deps
```

Install the inference environment only on a machine with the required GPU software.

```bash
python -m pip install -r requirements-inference-lock.txt
python -m pip install -e . --no-deps
```

## Rebuild the controlled workflows

The following command rebuilds all 540 paired tasks, the simulator derived labels, and the fixed split manifest. The builder stops if any generated task bank differs from the frozen SHA-256 hashes.

```bash
python scripts/build_controlled_workflows.py
```

Generated files appear under `generated/state_expansion_180_v5/`. They are not part of this archive.

To rebuild the repeated update benchmark, run:

```bash
python scripts/build_sequential_benchmark_v1.py \
  --output-dir data/sequential_anchorrc_v1
python scripts/build_sequential_utility_extension_v1.py \
  --output-dir data/sequential_anchorrc_utility_complete_v1
```

The first command produces 840 safety scoring tasks, 1,140 utility scoring tasks, and 60 six update sequences. The second command adds complete utility coverage for later incumbent states.

## Local model registries

Model files are not redistributed. Download each checkpoint from its official source at the exact revision declared in the corresponding config. Build a local hash registry before running SeqAnchor inference. For example:

```bash
python scripts/build_local_hf_registry.py \
  --snapshot /absolute/path/to/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 \
  --candidate-snapshot-id qwen25-14b-instruct-cf98f3b \
  --pretraining-family Qwen \
  --model-id Qwen/Qwen2.5-14B-Instruct \
  --revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 \
  --tokenizer-revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --chat-template-kwargs-json '{}' \
  --output manifests/qwen25-14b-instruct-cf98f3b_anchorrc_model_registry_v1.json \
  --freeze
```

`run_anchorrc_v1.py` accepts the generated registry directly through `--model-registry`. The optional `bind_model_registries.py` helper can also place one or more registry hashes into a protocol while keeping only repository relative paths.

## Experiment map

| Paper result | Build or inference entry point | Analysis entry point |
|---|---|---|
| Controlled safety reversals and presentation tests | `build_controlled_workflows.py`, `run_expansion_mechanism_v1.py` | `analyze_expansion_mechanism_v1.py`, `build_expansion_mechanism_publication_v1.py` |
| Agent SafetyBench, AgentDojo, and AgentHarm paired adaptations | `build_external_catalog_expansion_v1.py`, `run_external_catalog_expansion_v1.py` | `analyze_external_catalog_uncertainty_v1.py`, `analyze_external_catalog_margins_v1.py` |
| Native AgentHarm first action evaluation | `build_agentharm_native_expansion_v1.py`, `run_agentharm_native_expansion_v1.py` | `analyze_agentharm_native_uncertainty_v1.py` |
| Full action list reselection | `build_sequential_full_catalog_v1.py`, `run_sequential_full_catalog_v1.py` | `analyze_sequential_full_catalog_v1.py` |
| SeqAnchor safety and learned utility | `build_sequential_benchmark_v1.py`, `run_anchorrc_v1.py` | `analyze_sequential_anchorrc_v1.py`, `analyze_sequential_oracle_utility_v2.py` |
| Calibration ablation and risk simulation | `run_anchorrc_v1.py` | `analyze_sequential_calibration_ablation_v1.py`, `simulate_sequential_risk_control_v1.py` |
| Phi-4 and Qwen3 safety scorer replications | `preflight_seqanchor_multisafety_replication_v1.py`, `run_seqanchor_multisafety_replication_v1.py` | `authorize_seqanchor_multisafety_analysis_v1.py`, `analyze_seqanchor_multisafety_replication_v1.py` |
| Qwen3Guard comparison | `run_qwen3guard_transfer_v1.py` | `analyze_qwen3guard_sequential_v1.py` |
| TS-Guard comparison | `run_tsguard_toolsafe_v1.py` | `analyze_tsguard_sequential_v1.py` |
| StepGuard comparison | `run_stepguard_transfer_v1.py` | `analyze_stepguard_sequential_v1.py`, `migrate_stepguard_sequential_comparison_v2.py` |
| Inference latency and final comparison | `build_sequential_latency_panels_v1.py`, `benchmark_anchorrc_update_latency_v1.py` | `build_safeguard_efficiency_comparison_v4.py`, `analyze_guard_comparator_statistics_v2.py`, `build_sequential_safeguard_table_v4.py` |

Every entry point documents its required paths through `--help`. Generated benchmark files belong under `generated/` or `data/sequential_anchorrc_v1/`. Inference outputs should be written to a separate directory outside the source repository, as required by the SeqAnchor runner. None of these generated artifacts should be committed to the anonymous code repository.

## External benchmark and guard dependencies

The external benchmark adapters require separately obtained upstream sources. `build_external_catalog_expansion_v1.py` checks the ToolSafe checkout against the commit frozen in `toolsafe_tsguard_v1.py`. `build_agentharm_native_expansion_v1.py` accepts the official AgentHarm dataset and harmful tool directory as explicit arguments. The TS-Guard and StepGuard runners likewise require local upstream checkouts or model snapshots at the exact revisions declared in the source.

Upstream benchmark data, repositories, and model weights remain under their original licenses and are not redistributed here.

## Validation

Run the focused code-only test suite with:

```bash
pytest
```

The release was checked by rebuilding both controlled benchmarks, importing every command line entry point, and running the complete included test suite.

