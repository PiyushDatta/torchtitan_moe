# MoE Model Evaluation Scripts (Distributed)

## Overview

These scripts provide comprehensive evaluation of MoE (Mixture of Experts) models trained with TorchTitan. The evaluation runs distributed across multiple GPUs using the same parallelism configuration as training.

## Files

- `scripts/eval/eval_moe_model.py` - Main evaluation script (distributed)
- `scripts/eval/run_eval_moe.sh` - Convenient bash wrapper with torchrun
- `scripts/eval/compare_eval_results.py` - Compare two eval result files

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
- Automatic HuggingFace checkpoint conversion
- lm_eval integration (requires separate installation)
- Supports MMLU, HellaSwag, ARC-easy, and other benchmarks

## Usage

### Quick Start (Skip lm_eval)

```bash
CHECKPOINT_DIR=./outputs/checkpoint/step-1000 \
CONFIG_FILE=./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_10b_nvidia_4x_a100_40GBmem.toml \
SKIP_LM_EVAL=1 \
./scripts/eval/run_eval_moe.sh
```

### Full Evaluation with lm_eval

```bash
CHECKPOINT_DIR=./outputs/checkpoint/step-1000 \
CONFIG_FILE=./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_10b_nvidia_4x_a100_40GBmem.toml \
./scripts/eval/run_eval_moe.sh
```

### Direct Python Usage (with torchrun)

```bash
torchrun --nproc_per_node=4 scripts/eval/eval_moe_model.py \
    --checkpoint_dir ./outputs/checkpoint/step-1000 \
    --config_file ./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_10b_nvidia_4x_a100_40GBmem.toml \
    --output_dir ./eval_results \
    --skip_lm_eval
```

### Custom Number of GPUs

```bash
NGPU=2 \
CHECKPOINT_DIR=./outputs/checkpoint/step-1000 \
CONFIG_FILE=./config.toml \
./scripts/eval/run_eval_moe.sh
```

## Requirements

### Minimum
- Python 3.10+
- PyTorch with CUDA support
- GPUs with enough VRAM to load the model (uses same parallelism as training)

### For lm_eval (Optional)
Install lm-eval:

```bash
pip install lm-eval
```

Note: The script uses the HuggingFace backend which is compatible with all Python versions. For faster inference with vllm backend, use Python 3.10/3.11 and install `pip install "lm-eval[vllm]"`.

## Outputs

Results are saved to `{dump_folder}/eval/eval_results_{timestamp}.json` by default (where `dump_folder` comes from the training config, typically `./outputs`). Each evaluation run creates a new timestamped file, so multiple runs don't overwrite each other.

Example filename: `eval_results_20260129_212638.json`

```json
{
  "routing_stats": {
    "layer_1": {
      "gini_coefficient": 0.12,
      "coefficient_of_variation": 0.15,
      "expert_utilization_rate": 1.0,
      "mean_tokens_per_expert": 512.5
    },
    "aggregate": {
      "avg_gini_coefficient": 0.11,
      "avg_coefficient_of_variation": 0.14
    }
  },
  "inference_performance": {
    "latency_ms": 150.5,
    "throughput_tokens_per_sec": 845.2,
    "memory_allocated_gb": 38.2,
    "memory_reserved_gb": 39.1
  },
  "computational_cost": {
    "tflops": 6.89,
    "active_params_billions": 9.7
  },
  "walltime_seconds": 532.45,
  "checkpoint_dir": "./outputs/checkpoint/step-1000",
  "config": {
    "job": {"description": "DeepSeek-V3 MoE ~10B training"},
    "model": {"name": "deepseek_v3", "flavor": "10B"},
    "training": {"seq_len": 4096, "local_batch_size": 2},
    "parallelism": {"expert_parallel_degree": 4, "tensor_parallel_degree": 1}
  },
  "model_args": {
    "dim": 1264,
    "n_layers": 27,
    "n_heads": 16,
    "moe_args": {"num_experts": 64, "top_k": 6}
  },
  "eval_timestamp": "20260129_212638",
  "output_file": "./torchtitan_moe/outputs/eval/eval_results_20260129_212638.json"
}
```

## Troubleshooting

### Checkpoint Loading Errors

If you get errors loading the checkpoint:

1. **OOM Error**: Ensure you're using the same number of GPUs and parallelism config as training. Use `NGPU=4` for the 4x A100 config.

2. **Sharded Checkpoint**: The script uses DCP (Distributed Checkpoint) which automatically handles checkpoints saved with expert/tensor parallelism.

3. **Missing Files**: Ensure the checkpoint directory contains `.distcp` files and `.metadata`.

### Pipeline Parallel Not Supported

If you get "Pipeline parallel evaluation is not yet supported", use a config without pipeline parallelism for evaluation, or set `pipeline_parallel_degree = 1` in the config.

### lm_eval Not Found

If lm_eval is not installed, the script will skip it automatically and provide installation instructions:

```bash
pip install lm-eval
```

For faster inference (requires Python 3.10/3.11):
```bash
pip install "lm-eval[vllm]"
```

## Comparing MoE Strategies

### Using the Comparison Script

Compare two evaluation results to see which performs better:

```bash
# Compare two timestamped eval results
python scripts/eval/compare_eval_results.py \
    ./outputs/eval/eval_results_20260129_212638.json \
    ./outputs/eval/eval_results_20260129_220145.json
```

The script compares:
- Routing efficiency (Gini coefficient, CV, expert utilization)
- Inference performance (latency, throughput, memory)
- Computational cost (TFLOPs, active parameters)
- Model accuracy (lm_eval scores, if available)

Output as JSON for programmatic use:
```bash
python scripts/eval/compare_eval_results.py --json \
    ./outputs/eval/eval_results_20260129_212638.json \
    ./outputs/eval/eval_results_20260129_220145.json
```

### Evaluating Multiple Checkpoints

Run eval on each checkpoint:
```bash
for step in 500 1000 1500 2000; do
    CHECKPOINT_DIR=./outputs/checkpoint/step-${step} \
    CONFIG_FILE=./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_10b_nvidia_4x_a100_40GBmem.toml \
    OUTPUT_DIR=./outputs/eval/step-${step} \
    SKIP_LM_EVAL=1 \
    ./scripts/eval/run_eval_moe.sh
done
```

Compare consecutive checkpoints:
```bash
# List eval results to find the timestamped filenames
ls ./outputs/eval/step-500/eval_results_*.json
ls ./outputs/eval/step-1000/eval_results_*.json

# Compare them (use actual filenames from above)
python scripts/eval/compare_eval_results.py \
    ./outputs/eval/step-500/eval_results_20260129_212638.json \
    ./outputs/eval/step-1000/eval_results_20260129_220145.json
```

### Manual Comparison (Python)

```python
import json
from pathlib import Path

# Find the latest eval results for each step
results = {}
for step in [500, 1000, 1500, 2000]:
    eval_dir = Path(f"./outputs/eval/step-{step}")
    # Get the most recent eval_results file
    result_files = sorted(eval_dir.glob("eval_results_*.json"))
    if result_files:
        with open(result_files[-1]) as f:
            results[step] = json.load(f)

# Compare routing efficiency
for step, data in results.items():
    gini = data["routing_stats"]["aggregate"]["avg_gini_coefficient"]
    cv = data["routing_stats"]["aggregate"]["avg_coefficient_of_variation"]
    print(f"Step {step}: Gini={gini:.3f}, CV={cv:.3f}")
```

## Key Metrics for MoE Comparison

| Metric | Goal | Interpretation |
|--------|------|----------------|
| Gini Coefficient | Lower is better | <0.1 = excellent, 0.1-0.2 = good, >0.3 = poor |
| Coefficient of Variation | Lower is better | <0.2 = excellent, 0.2-0.4 = good, >0.5 = poor |
| Expert Utilization | Higher is better | >0.95 = excellent, 0.8-0.95 = good, <0.8 = poor |
| Throughput | Higher is better | Depends on hardware |
| Memory | Lower is better | Should fit in available VRAM |

## Advanced Options

### Custom Evaluation Tasks

```bash
torchrun --nproc_per_node=4 scripts/eval/eval_moe_model.py \
    --checkpoint_dir ./outputs/checkpoint/step-1000 \
    --config_file ./config.toml \
    --lm_eval_tasks mmlu hellaswag winogrande arc_challenge \
    --output_dir ./eval_results
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| CHECKPOINT_DIR | (required) | Path to checkpoint directory |
| CONFIG_FILE | (required) | Path to training config TOML |
| OUTPUT_DIR | {dump_folder}/eval | Where to save results (uses dump_folder from config if not set) |
| SKIP_LM_EVAL | 0 | Set to 1 to skip lm_eval |
| SEED | 1337 | Random seed for reproducibility |
| NGPU | auto-detect | Number of GPUs to use (auto-detects from config parallelism, set to 0 for CPU) |
