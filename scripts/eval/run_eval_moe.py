#!/usr/bin/env python3
"""
MoE Model Evaluation Launcher (Distributed)

Replaces run_eval_moe.sh - handles GPU detection, parallelism validation,
and launches eval_moe_model.py with torchrun.

Usage:
    python scripts/eval/run_eval_moe.py \
        --checkpoint_dir ./outputs/checkpoint/step-1000 \
        --config_file ./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b_nvidia_4x_a100_80GBmem.toml

    python scripts/eval/run_eval_moe.py \
        --checkpoint_dir ./outputs/checkpoint/step-1000 \
        --config_file ./config.toml \
        --preset 5min --skip_lm_eval
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def parse_toml_value(config_path: Path, key: str, default: str = "1") -> str:
    """Extract a value from a TOML file via simple regex (no toml dep needed)."""
    try:
        text = config_path.read_text()
        match = re.search(rf"^{key}\s*=\s*(\S+)", text, re.MULTILINE)
        if match:
            return match.group(1).strip().strip('"').strip("'")
    except OSError:
        pass
    return default


def get_available_gpus() -> int:
    """Detect available GPUs via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().splitlines()[0].strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 0


def main():
    parser = argparse.ArgumentParser(description="MoE Model Evaluation Launcher")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to checkpoint directory")
    parser.add_argument("--config_file", required=True, help="Path to training config TOML file")
    parser.add_argument("--output_dir", default=None, help="Directory to save results")
    parser.add_argument("--preset", default=None, help="Evaluation preset: 30s, 1min, 5min, 15min, full")
    parser.add_argument("--skip_lm_eval", action="store_true", help="Skip lm_eval benchmark")
    parser.add_argument("--lm_eval_only", action="store_true", help="Only run lm_eval")
    parser.add_argument("--lm_eval_tasks", nargs="*", default=None, help="lm_eval tasks to run")
    parser.add_argument("--lm_eval_limit", type=int, default=None, help="Limit examples per task")
    parser.add_argument("--ngpu", type=int, default=None, help="Number of GPUs (default: auto-detect)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    eval_script = script_dir / "eval_moe_model.py"

    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Extract parallelism from config
    ep = int(parse_toml_value(config_path, "expert_parallel_degree"))
    tp = int(parse_toml_value(config_path, "tensor_parallel_degree"))
    pp = int(parse_toml_value(config_path, "pipeline_parallel_degree"))
    min_gpus = ep * tp
    if pp > 1:
        print(f"Warning: Pipeline parallelism (PP={pp}) is not yet supported for evaluation.")

    # Determine GPU count
    available = get_available_gpus()
    if args.ngpu is not None:
        ngpu = args.ngpu
        if ngpu > 0 and ngpu < min_gpus:
            print(f"Error: --ngpu={ngpu} < required {min_gpus} GPUs (EP={ep}, TP={tp})", file=sys.stderr)
            sys.exit(1)
    elif available >= min_gpus:
        ngpu = min_gpus
    elif available > 0:
        print(f"Error: Config requires {min_gpus} GPUs but only {available} available.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Error: No GPUs detected. This evaluation requires {min_gpus} GPUs.", file=sys.stderr)
        sys.exit(1)

    # Build eval_moe_model.py arguments
    eval_args = ["--checkpoint_dir", args.checkpoint_dir, "--config_file", args.config_file]
    if args.output_dir:
        eval_args += ["--output_dir", args.output_dir]
    if args.preset:
        eval_args += ["--preset", args.preset]
    if args.skip_lm_eval:
        eval_args.append("--skip_lm_eval")
    if args.lm_eval_only:
        eval_args.append("--lm_eval_only")
    if args.lm_eval_tasks:
        eval_args += ["--lm_eval_tasks"] + args.lm_eval_tasks
    if args.lm_eval_limit is not None:
        eval_args += ["--lm_eval_limit", str(args.lm_eval_limit)]

    print("=============================================")
    print("MoE Model Evaluation (Distributed)")
    print("=============================================")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Config:     {args.config_file}")
    print(f"Output:     {args.output_dir or '(auto: {dump_folder}/eval)'}")
    print(f"Preset:     {args.preset or 'full (default)'}")
    print(f"GPUs:       {ngpu} (required: {min_gpus}, available: {available})")
    print(f"Parallelism: EP={ep}, TP={tp}, PP={pp}")
    print("=============================================")
    print()

    # Launch with torchrun or plain python
    if ngpu == 0:
        print("Warning: Running on CPU. This will be very slow!")
        cmd = [sys.executable, str(eval_script)] + eval_args
    else:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={ngpu}",
            str(eval_script),
        ] + eval_args

    result = subprocess.run(cmd, cwd=str(repo_root))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
