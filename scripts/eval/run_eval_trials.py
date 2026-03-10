#!/usr/bin/env python3
"""
Multi-Trial MoE Evaluation Runner

Runs eval_moe_model.py N times, collects all trial results into a single
directory, and computes averaged metrics for comparison across methods
(e.g., standard MoE vs MoD).

Variance across trials captures GPU nondeterminism and triton autotuning
effects rather than data randomness (the eval uses random inputs with a
fixed seed set by set_determinism in distributed mode).

Usage:
    python scripts/eval/run_eval_trials.py \
        --checkpoint_dir ./outputs/checkpoint/step-1000 \
        --config_file ./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b_nvidia_4x_a100_80GBmem.toml \
        --experiment_name baseline \
        --num_trials 50

    # Quick test with 3 trials
    python scripts/eval/run_eval_trials.py \
        --checkpoint_dir ./outputs/checkpoint/step-1000 \
        --config_file ./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b_nvidia_4x_a100_80GBmem.toml \
        --preset 30s --num_trials 3
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Ensure sibling modules are importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate_trials import build_summary, load_trials, print_summary


def find_result_json(directory: Path) -> Path | None:
    """Find the eval_results_*.json file in a directory."""
    results = list(directory.glob("**/eval_results_*.json"))
    return results[0] if results else None


def run_single_trial(
    trial_num: int,
    total: int,
    checkpoint_dir: str,
    config_file: str,
    preset: str,
    skip_lm_eval: bool,
    output_path: Path,
    script_dir: Path,
    repo_root: Path,
) -> bool:
    """Run a single eval trial. Returns True on success."""
    print(f"[Trial {trial_num}/{total}] ...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [
            sys.executable,
            str(script_dir / "run_eval_moe.py"),
            "--checkpoint_dir", checkpoint_dir,
            "--config_file", config_file,
            "--output_dir", tmp_dir,
            "--preset", preset,
        ]
        if skip_lm_eval:
            cmd.append("--skip_lm_eval")

        result = subprocess.run(cmd, cwd=str(repo_root))

        if result.returncode != 0:
            print(f"[Trial {trial_num}/{total}] FAILED (exit code {result.returncode})")
            return False

        result_json = find_result_json(Path(tmp_dir))
        if result_json is None:
            print(f"[Trial {trial_num}/{total}] WARNING: No result JSON found")
            return False

        shutil.copy2(result_json, output_path)
        print(f"[Trial {trial_num}/{total}] Saved to {output_path}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Multi-Trial MoE Evaluation Runner")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to checkpoint directory")
    parser.add_argument("--config_file", required=True, help="Path to training config TOML file")
    parser.add_argument("--num_trials", type=int, default=50, help="Number of evaluation trials (default: 50)")
    parser.add_argument("--experiment_name", default="baseline", help="Name for this experiment (default: baseline)")
    parser.add_argument("--results_dir", default=None, help="Directory to save results (default: ./results)")
    parser.add_argument("--preset", default="5min", help="Evaluation preset per trial (default: 5min)")
    parser.add_argument("--skip_lm_eval", action="store_true", help="Skip lm_eval")
    args = parser.parse_args()

    skip_lm_eval = args.skip_lm_eval

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    results_dir = Path(args.results_dir) if args.results_dir else repo_root / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = results_dir / f"{args.experiment_name}_{timestamp}"
    trials_dir = experiment_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    print("=============================================")
    print("Multi-Trial MoE Evaluation")
    print("=============================================")
    print(f"Experiment:   {args.experiment_name}")
    print(f"Checkpoint:   {args.checkpoint_dir}")
    print(f"Config:       {args.config_file}")
    print(f"Preset:       {args.preset}")
    print(f"Trials:       {args.num_trials}")
    print(f"Skip lm_eval: {skip_lm_eval}")
    print(f"Output dir:   {experiment_dir}")
    print("=============================================")
    print()

    # Run trials
    succeeded = 0
    t_start = time.monotonic()
    for i in range(1, args.num_trials + 1):
        trial_output = trials_dir / f"trial_{i:03d}.json"

        ok = run_single_trial(
            trial_num=i,
            total=args.num_trials,
            checkpoint_dir=args.checkpoint_dir,
            config_file=args.config_file,
            preset=args.preset,
            skip_lm_eval=skip_lm_eval,
            output_path=trial_output,
            script_dir=script_dir,
            repo_root=repo_root,
        )
        if ok:
            succeeded += 1
        print()

    elapsed = time.monotonic() - t_start
    avg_elapsed = elapsed / args.num_trials

    print("=============================================")
    print(f"Trials complete: {succeeded}/{args.num_trials} succeeded")
    print(f"Total elapsed:   {elapsed:.1f}s ({elapsed/3600:.1f}h)")
    print(f"Avg per trial:   {avg_elapsed:.1f}s")
    print("=============================================")

    if succeeded == 0:
        print("Error: All trials failed.", file=sys.stderr)
        sys.exit(1)

    # Aggregate results
    print("Aggregating results...")
    trials = load_trials(trials_dir)
    if not trials:
        print("Error: No valid trial JSONs to aggregate.", file=sys.stderr)
        sys.exit(1)

    summary = build_summary(trials, args.experiment_name)
    summary["trial_config"] = {
        "num_trials": args.num_trials,
        "preset": args.preset,
        "skip_lm_eval": skip_lm_eval,
        "checkpoint_dir": args.checkpoint_dir,
        "config_file": args.config_file,
        "elapsed_seconds": elapsed,
        "avg_elapsed_per_trial": avg_elapsed,
    }
    summary_path = experiment_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary written to {summary_path}")
    print_summary(summary)

    print(f"Results saved to:")
    print(f"  Trials:  {trials_dir}/")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
