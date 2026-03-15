# MoE Model Evaluation Scripts (Distributed)

## Overview

These scripts provide comprehensive evaluation of MoE (Mixture of Experts) models trained with TorchTitan. The evaluation runs distributed across multiple GPUs using the same parallelism configuration as training.

## Single Evaluation Usage

```bash
# Quick 30-second routing check
python scripts/eval/run_eval_moe.py \
    --checkpoint_dir ./outputs/checkpoint/step-1000 \
    --config_file ./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b_nvidia_4x_a100_80GBmem.toml \
    --preset 30s

# 5-minute evaluation for ablation studies
python scripts/eval/run_eval_moe.py \
    --checkpoint_dir ./outputs/checkpoint/step-1000 \
    --config_file ./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b_nvidia_4x_a100_80GBmem.toml \
    --preset 5min

# Full evaluation (default preset)
python scripts/eval/run_eval_moe.py \
    --checkpoint_dir ./outputs/checkpoint/step-1000 \
    --config_file ./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b_nvidia_4x_a100_80GBmem.toml
```

### Evaluation Presets

| Preset | Duration | What it runs |
|--------|----------|--------------|
| `30s` | ~30 seconds | Routing only (10 samples), no inference/lm_eval |
| `1min` | ~1 minute | Routing (10) + inference (1 iter) + lm_eval (5 samples) |
| `5min` | ~5 minutes | Routing (30) + inference (3 iters) + lm_eval (20 samples) |
| `15min` | ~15 minutes | Routing (50) + inference (5 iters) + lm_eval (100 samples) |
| `full` | ~10-15 minutes | Routing (100) + inference (10 iters) + lm_eval (hellaswag, arc_easy, mmlu_stem, limit=500) |

## Multi-Trial Evaluation

Run repeated trials to capture variance from GPU nondeterminism and triton autotuning:

```bash
# 50 trials of baseline, 5min preset
python scripts/eval/run_eval_trials.py \
    --checkpoint_dir ./outputs/checkpoint/step-1000 \
    --config_file ./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b_nvidia_4x_a100_80GBmem.toml \
    --experiment_name baseline

# Quick test with 3 trials, 30s preset
python scripts/eval/run_eval_trials.py \
    --checkpoint_dir ./outputs/checkpoint/step-1000 \
    --config_file ./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b_nvidia_4x_a100_80GBmem.toml \
    --num_trials 3 --preset 30s \
    --experiment_name baseline

# Full test with 50 trials, full preset
python scripts/eval/run_eval_trials.py \
    --checkpoint_dir ./outputs/checkpoint/step-1000 \
    --config_file ./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b_nvidia_4x_a100_80GBmem.toml \
    --num_trials 50 --preset full \
    --experiment_name baseline

# Override lm_eval tasks and limit
python scripts/eval/run_eval_trials.py \
    --checkpoint_dir ./outputs/checkpoint/step-1000 \
    --config_file ./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b_nvidia_4x_a100_80GBmem.toml \
    --num_trials 50 --preset full \
    --lm_eval_tasks mmlu hellaswag arc_easy \
    --lm_eval_limit 200 \
    --experiment_name baseline
```

Results are saved to `./results/<experiment_name>_<timestamp>/`:
- `trials/trial_001.json` ... `trial_050.json` - Individual trial results
- `summary.json` - Aggregated metrics with mean/stdev

Example summary output:
```
AGGREGATED RESULTS: baseline (50 trials)

[ROUTING EFFICIENCY]
  Avg CV:          1.5403 +/- 0.0204
  Avg Gini:        0.6586 +/- 0.0039
  Avg Utilization: 90.59% +/- 1.15%

[INFERENCE PERFORMANCE]
  Latency:         35563.60 +/- 120.50 ms
  Throughput:      3.60 +/- 0.02 tok/s

[COMPUTATIONAL COST]
  Active Params:   15.71B
  FLOPs/token:     1.51e+10

[WALLTIME]
  Per trial:       455.4 +/- 12.3s
  Total:           22770.0s (6.3h)
```

## Files

- `eval_moe_model.py` - Main evaluation script (distributed, launched via torchrun)
- `run_eval_moe.py` - Launcher that auto-detects GPUs and invokes torchrun
- `run_eval_trials.py` - Multi-trial runner for statistical comparisons
- `aggregate_trials.py` - Aggregates trial results into summary JSON with mean/stdev
- `compare_eval_results.py` - Compare two eval results (supports both summary.json and individual trial files)
- `torchtitan_lm_eval.py` - Custom lm_eval wrapper for torchtitan models
- `eval_convert_to_hf.py` - Standalone script to convert checkpoints to HuggingFace format

## Evaluation Metrics

### 1. Routing Efficiency
- **Gini Coefficient**: Measures load imbalance (0 = perfect equality, 1 = perfect inequality)
- **Coefficient of Variation**: Std dev / mean (lower = better balance)
- **Expert Utilization Rate**: Percentage of experts receiving tokens
- **Per-expert token distribution**: Full distribution across all experts

### 2. Inference Performance
- Latency (ms per batch)
- Throughput (tokens/second)
- Memory usage (allocated & reserved GB)

### 3. Computational Cost
- Total FLOPs per forward pass
- TFLOPs count
- Active parameters (billions)

### 4. Model Accuracy (Optional)
- Direct lm_eval integration (no HuggingFace conversion needed)
- Default tasks: HellaSwag, ARC-easy, MMLU-STEM (limit=500 per task)
- Override with `--lm_eval_tasks` and `--lm_eval_limit` for full MMLU or other benchmarks
- Requires separate lm-eval installation

## Direct torchrun Usage

```bash
torchrun --nproc_per_node=4 scripts/eval/eval_moe_model.py \
    --checkpoint_dir ./outputs/checkpoint/step-1000 \
    --config_file ./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b_nvidia_4x_a100_80GBmem.toml \
    --preset 5min \
    --output_dir ./eval_results
```

## CLI Reference

### run_eval_moe.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint_dir` | (required) | Path to checkpoint directory |
| `--config_file` | (required) | Path to training config TOML |
| `--output_dir` | auto | Where to save results |
| `--preset` | `full` | Evaluation preset: 30s, 1min, 5min, 15min, full |
| `--skip_lm_eval` | off | Skip lm_eval benchmark |
| `--lm_eval_only` | off | Only run lm_eval |
| `--lm_eval_tasks` | from preset | lm_eval tasks to run |
| `--lm_eval_limit` | from preset | Limit examples per task |
| `--ngpu` | auto | Number of GPUs (auto-detects from config) |

### run_eval_trials.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint_dir` | (required) | Path to checkpoint directory |
| `--config_file` | (required) | Path to training config TOML |
| `--num_trials` | 50 | Number of evaluation trials |
| `--experiment_name` | `baseline` | Name for this experiment |
| `--results_dir` | `./results` | Directory to save results |
| `--preset` | `5min` | Evaluation preset per trial |
| `--skip_lm_eval` | off | Skip lm_eval |
| `--lm_eval_tasks` | from preset | Override lm_eval tasks (e.g. hellaswag arc_easy mmlu) |
| `--lm_eval_limit` | from preset | Limit examples per lm_eval task |

### compare_eval_results.py

| Argument | Default | Description |
|----------|---------|-------------|
| `result_a` | (required) | Path to first result file (summary.json or trial file) |
| `result_b` | (required) | Path to second result file (summary.json or trial file) |
| `--json` | off | Output as JSON instead of formatted table |

## Requirements

### Minimum
- Python 3.10+
- PyTorch with CUDA support
- GPUs with enough VRAM to load the model (uses same parallelism as training)

### For lm_eval (Optional)

```bash
pip install lm-eval
```

The script runs lm_eval directly on the torchtitan model without HuggingFace conversion.

## Outputs

Single eval results are saved to `{dump_folder}/eval/eval_results_{timestamp}.json`.
Multi-trial results are saved to `./results/<experiment_name>_<timestamp>/`.

```json
{
  "routing_stats": {
    "layer_1": {
      "gini_coefficient": 0.12,
      "coefficient_of_variation": 0.15,
      "expert_utilization_rate": 1.0,
      "mean_tokens_per_expert": 512.5
    }
  },
  "inference_performance": {
    "latency_ms": 65067.35,
    "throughput_tokens_per_sec": 1.97,
    "memory_allocated_gb": 17.29,
    "memory_reserved_gb": 18.34
  },
  "computational_cost": {
    "total_flops": 7750000000000,
    "active_params_billions": 15.71,
    "num_flops_per_token": 1.51e+10
  }
}
```

## Comparing Results

### Two Multi-Trial Experiments (Recommended)

Compare `summary.json` files from multi-trial runs. This is the most robust comparison since it uses averaged metrics across many trials with standard deviations:

```bash
python scripts/eval/compare_eval_results.py \
    results/baseline_20260313/summary.json \
    results/mod_20260313/summary.json
```

The script auto-detects summary format and will:
- Show mean +/- stdev for all metrics
- Aggregate lm_eval scores from trial files
- Display config differences between the two models
- Determine an overall winner

### Two Single Trial Results

```bash
python scripts/eval/compare_eval_results.py \
    results/exp_a/trials/trial_001.json \
    results/exp_b/trials/trial_001.json
```

### JSON Output

```bash
# For programmatic use
python scripts/eval/compare_eval_results.py --json \
    results/exp_a/summary.json results/exp_b/summary.json
```

### Multiple Checkpoints

```bash
for step in 500 1000 1500 2000; do
    python scripts/eval/run_eval_moe.py \
        --checkpoint_dir ./outputs/checkpoint/step-${step} \
        --config_file ./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b_nvidia_4x_a100_80GBmem.toml \
        --output_dir ./outputs/eval/step-${step} \
        --skip_lm_eval
done
```

## Key Metrics

| Metric | Goal | Interpretation |
|--------|------|----------------|
| Gini Coefficient | Lower is better | <0.1 excellent, 0.1-0.2 good, >0.3 poor |
| Coefficient of Variation | Lower is better | <0.2 excellent, 0.2-0.4 good, >0.5 poor |
| Expert Utilization | Higher is better | >0.95 excellent, 0.8-0.95 good, <0.8 poor |
| Throughput | Higher is better | Depends on hardware |
| Memory | Lower is better | Should fit in available VRAM |

## Troubleshooting

### Checkpoint Loading Errors

1. **OOM Error**: Ensure you're using the same number of GPUs and parallelism config as training.
2. **Sharded Checkpoint**: DCP automatically handles checkpoints saved with expert/tensor parallelism.
3. **Missing Files**: Ensure the checkpoint directory contains `.distcp` files and `.metadata`.

### Pipeline Parallel Not Supported

Use a config with `pipeline_parallel_degree = 1` for evaluation.

### lm_eval Not Found

```bash
pip install lm-eval
```
