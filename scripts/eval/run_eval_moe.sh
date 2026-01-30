#!/bin/bash
#
# MoE Model Evaluation Script Wrapper (Distributed)
#
# Usage:
#   ./scripts/eval/run_eval_moe.sh [options]
#
# Environment variables:
#   CHECKPOINT_DIR - Path to checkpoint directory (required)
#   CONFIG_FILE    - Path to training config TOML file (required)
#   OUTPUT_DIR     - Directory to save results (default: {dump_folder}/eval from config)
#   SKIP_LM_EVAL   - Set to 1 to skip lm_eval benchmark (default: 0)
#   SEED           - Random seed for reproducibility (default: 1337)
#   NGPU           - Number of GPUs to use (default: auto-detect from config)
#
# Examples:
#   CHECKPOINT_DIR=./outputs/checkpoint/step-1000 \
#   CONFIG_FILE=./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_10b_nvidia_4x_a100_40GBmem.toml \
#   ./scripts/eval/run_eval_moe.sh
#
#   # Skip lm_eval and only run inference/routing analysis:
#   CHECKPOINT_DIR=./outputs/checkpoint/step-1000 \
#   CONFIG_FILE=./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_10b_nvidia_4x_a100_40GBmem.toml \
#   SKIP_LM_EVAL=1 \
#   ./scripts/eval/run_eval_moe.sh
#
#   # With seed for reproducibility:
#   CHECKPOINT_DIR=./outputs/checkpoint/step-1000 \
#   CONFIG_FILE=./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_10b_nvidia_4x_a100_40GBmem.toml \
#   SEED=42 \
#   ./scripts/eval/run_eval_moe.sh
#
#   # Use specific number of GPUs (must match parallelism config):
#   NGPU=4 \
#   CHECKPOINT_DIR=./outputs/checkpoint/step-1000 \
#   CONFIG_FILE=./config.toml \
#   ./scripts/eval/run_eval_moe.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Validate required environment variables
if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "Error: CHECKPOINT_DIR environment variable is required"
    echo "Example: CHECKPOINT_DIR=./outputs/checkpoint/step-1000"
    exit 1
fi

if [ -z "${CONFIG_FILE}" ]; then
    echo "Error: CONFIG_FILE environment variable is required"
    echo "Example: CONFIG_FILE=./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_10b_nvidia_4x_a100_40GBmem.toml"
    exit 1
fi

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Extract parallelism degrees from config file
# These determine the minimum number of GPUs required
EP_DEGREE=$(grep -E "^expert_parallel_degree\s*=" "${CONFIG_FILE}" 2>/dev/null | sed 's/.*=\s*//' | tr -d ' ' || echo "1")
TP_DEGREE=$(grep -E "^tensor_parallel_degree\s*=" "${CONFIG_FILE}" 2>/dev/null | sed 's/.*=\s*//' | tr -d ' ' || echo "1")
PP_DEGREE=$(grep -E "^pipeline_parallel_degree\s*=" "${CONFIG_FILE}" 2>/dev/null | sed 's/.*=\s*//' | tr -d ' ' || echo "1")

# Calculate minimum required GPUs (EP * TP * PP for non-data-parallel dimensions)
# For evaluation, we primarily care about expert parallelism
MIN_GPUS=${EP_DEGREE}
if [ "${TP_DEGREE}" != "1" ]; then
    MIN_GPUS=$((MIN_GPUS * TP_DEGREE))
fi
if [ "${PP_DEGREE}" != "1" ]; then
    echo "Warning: Pipeline parallelism (PP=${PP_DEGREE}) is not yet supported for evaluation."
    echo "The script will fail if PP > 1. Please use a config with pipeline_parallel_degree = 1."
fi

# Get available GPUs
if command -v nvidia-smi &> /dev/null; then
    AVAILABLE_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1 | tr -d ' ')
else
    AVAILABLE_GPUS=0
fi

# Set NGPU: use user-specified value, or auto-detect based on config
if [ -z "${NGPU}" ]; then
    # Auto-detect: use the minimum required by config, or available GPUs, whichever is smaller
    if [ "${AVAILABLE_GPUS}" -gt 0 ]; then
        if [ "${AVAILABLE_GPUS}" -ge "${MIN_GPUS}" ]; then
            NGPU=${MIN_GPUS}
        else
            echo "Error: Config requires ${MIN_GPUS} GPUs (expert_parallel_degree=${EP_DEGREE}), but only ${AVAILABLE_GPUS} available."
            echo "Please use a config file with lower parallelism degree, or use more GPUs."
            exit 1
        fi
    else
        echo "Error: No GPUs detected. This evaluation requires ${MIN_GPUS} GPUs based on the config."
        echo "If you want to run on CPU (very slow), set NGPU=0"
        exit 1
    fi
else
    # User specified NGPU - validate it
    if [ "${NGPU}" != "0" ] && [ "${NGPU}" -lt "${MIN_GPUS}" ]; then
        echo "Error: NGPU=${NGPU} is less than required ${MIN_GPUS} GPUs."
        echo "Config requires expert_parallel_degree=${EP_DEGREE}"
        echo "Please set NGPU>=${MIN_GPUS} or use a config with lower parallelism."
        exit 1
    fi
fi

# Build command arguments
ARGS="--checkpoint_dir ${CHECKPOINT_DIR} \
    --config_file ${CONFIG_FILE}"

# Add output_dir if explicitly set (otherwise script uses dump_folder/eval from config)
if [ -n "${OUTPUT_DIR}" ]; then
    ARGS="${ARGS} --output_dir ${OUTPUT_DIR}"
fi

# Add skip_lm_eval flag if set
if [ "${SKIP_LM_EVAL}" = "1" ]; then
    ARGS="${ARGS} --skip_lm_eval"
fi

# Add seed if set
if [ -n "${SEED}" ]; then
    ARGS="${ARGS} --seed ${SEED}"
fi

# Append any additional arguments
ARGS="${ARGS} $@"

echo "============================================="
echo "MoE Model Evaluation (Distributed)"
echo "============================================="
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Config:     ${CONFIG_FILE}"
echo "Output:     ${OUTPUT_DIR:-(auto: {dump_folder}/eval)}"
echo "Skip lm_eval: ${SKIP_LM_EVAL:-0}"
echo "Seed:       ${SEED:-1337 (default)}"
echo "GPUs:       ${NGPU} (required: ${MIN_GPUS}, available: ${AVAILABLE_GPUS})"
echo "Parallelism: EP=${EP_DEGREE}, TP=${TP_DEGREE}, PP=${PP_DEGREE}"
echo "============================================="
echo ""

# Run evaluation
cd "${REPO_ROOT}"

if [ "${NGPU}" = "0" ]; then
    # CPU-only mode (very slow, for debugging only)
    echo "Warning: Running on CPU. This will be very slow!"
    echo "Set NGPU to a positive number for GPU execution."
    python ${SCRIPT_DIR}/eval_moe_model.py ${ARGS}
else
    # GPU mode with torchrun
    torchrun --nproc_per_node=${NGPU} ${SCRIPT_DIR}/eval_moe_model.py ${ARGS}
fi
