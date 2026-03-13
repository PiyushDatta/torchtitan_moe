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
import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Ensure sibling modules are importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate_trials import build_summary, load_trials, print_summary

# ---------------------------------------------------------------------------
# Process cleanup infrastructure
# ---------------------------------------------------------------------------
_child_procs: list[subprocess.Popen] = []
_current_proc: subprocess.Popen | None = None  # catch race with _register
_shutting_down = False
_TORCHRUN_DEFAULT_PORT = 29500


def _register_child(proc: subprocess.Popen) -> None:
    _child_procs.append(proc)


def _unregister_child(proc: subprocess.Popen) -> None:
    try:
        _child_procs.remove(proc)
    except ValueError:
        pass


def _send_sig_to_proc_group(proc: subprocess.Popen, sig: int) -> None:
    """Send a signal to a process's entire process group. Never waits."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, OSError):
        pass


def _kill_proc_group(proc: subprocess.Popen) -> None:
    """Kill a process group: SIGTERM, wait, then SIGKILL. Main thread only."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return

    # Graceful SIGTERM first
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        return
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass

    # Force SIGKILL
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass


def _cleanup_all_children() -> None:
    """Kill all tracked child process groups. Main thread only."""
    for proc in list(_child_procs):
        _kill_proc_group(proc)


def _is_owned_by_current_user(pid: int) -> bool:
    """Check if a process belongs to the current user via /proc."""
    try:
        stat = Path(f"/proc/{pid}/status").read_text()
        for line in stat.splitlines():
            if line.startswith("Uid:"):
                real_uid = int(line.split()[1])
                return real_uid == os.getuid()
    except (OSError, ValueError, IndexError):
        pass
    return False


def _cleanup_stale_port_holders(port: int = _TORCHRUN_DEFAULT_PORT) -> None:
    """Kill orphaned processes *owned by this user* holding the torchrun port.

    Safety net for cases where process-group cleanup didn't fully work
    (e.g., after a SIGKILL of this script).
    """
    my_pid = os.getpid()
    pids_to_kill: list[int] = []

    # Try fuser first, then lsof as fallback
    for cmd in (
        ["fuser", f"{port}/tcp"],
        ["lsof", "-ti", f":{port}"],
    ):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5,
            )
            raw = (result.stdout.strip() or result.stderr.strip())
            if result.returncode == 0 and raw:
                for token in raw.split():
                    # fuser appends access modifiers (e.g., "12345e", "12346m")
                    token = token.strip().rstrip("eFfmr")
                    if token.isdigit():
                        pid = int(token)
                        if pid != my_pid and _is_owned_by_current_user(pid):
                            pids_to_kill.append(pid)
            if pids_to_kill:
                break
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            continue

    for pid in pids_to_kill:
        try:
            os.kill(pid, signal.SIGKILL)
            print(f"  Killed stale process {pid} on port {port}")
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _signal_handler(signum, _frame):
    """Handle termination signals by sending SIGTERM to all child groups.

    Only sends signals — never calls proc.wait() — to stay safe when
    the main thread is already inside proc.wait() (reentrant waitpid
    would corrupt the child's exit status).
    """
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True
    sig_name = signal.Signals(signum).name
    # os.write is async-signal-safe; print() is not (it acquires a lock
    # that the interrupted main-thread code may already hold).
    os.write(sys.stderr.fileno(),
             f"\n[run_eval_trials] Received {sig_name}, shutting down...\n".encode())
    for proc in list(_child_procs):
        _send_sig_to_proc_group(proc, signal.SIGTERM)
    # Catch the race where Popen returned but _register_child hasn't run yet
    cur = _current_proc
    if cur is not None and cur not in _child_procs:
        _send_sig_to_proc_group(cur, signal.SIGTERM)


def _atexit_cleanup():
    """Fallback cleanup on normal/abnormal exit."""
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True
    _cleanup_all_children()


# ---------------------------------------------------------------------------


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
    lm_eval_tasks: list[str] | None = None,
    lm_eval_limit: int | None = None,
) -> bool:
    """Run a single eval trial. Returns True on success."""
    global _current_proc

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
        if lm_eval_tasks:
            cmd += ["--lm_eval_tasks"] + lm_eval_tasks
        if lm_eval_limit is not None:
            cmd += ["--lm_eval_limit", str(lm_eval_limit)]

        proc = subprocess.Popen(
            cmd, cwd=str(repo_root), start_new_session=True,
        )
        _current_proc = proc
        _register_child(proc)
        try:
            proc.wait()
        finally:
            _unregister_child(proc)
            _current_proc = None

        # If the signal handler fired, the child was SIGTERM'd.  Do a
        # full SIGKILL follow-up now that we're back in the main thread
        # (safe to wait here — proc.wait() above already returned).
        if _shutting_down:
            _kill_proc_group(proc)
            return False

        if proc.returncode != 0:
            print(f"[Trial {trial_num}/{total}] FAILED (exit code {proc.returncode})")
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
    parser.add_argument("--lm_eval_tasks", nargs="*", default=None, help="Override lm_eval tasks (e.g. hellaswag arc_easy mmlu)")
    parser.add_argument("--lm_eval_limit", type=int, default=None, help="Limit examples per lm_eval task")
    args = parser.parse_args()

    skip_lm_eval = args.skip_lm_eval

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    results_dir = Path(args.results_dir) if args.results_dir else repo_root / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = results_dir / f"{args.experiment_name}_{timestamp}"
    trials_dir = experiment_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    # --- Register cleanup handlers ---
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _signal_handler)
        except (OSError, ValueError):
            pass  # SIGHUP unavailable on some platforms
    atexit.register(_atexit_cleanup)

    # Clean up stale processes from a previous interrupted run
    _cleanup_stale_port_holders()

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
    trials_attempted = 0
    t_start = time.monotonic()
    try:
        for i in range(1, args.num_trials + 1):
            if _shutting_down:
                print("[run_eval_trials] Shutdown requested, stopping trials.")
                break

            trials_attempted += 1
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
                lm_eval_tasks=args.lm_eval_tasks,
                lm_eval_limit=args.lm_eval_limit,
            )
            if ok:
                succeeded += 1
            print()
    finally:
        # Always clean up ports, even on abnormal exit
        _cleanup_stale_port_holders()
        if _shutting_down:
            print("[run_eval_trials] Cleanup complete.")

    elapsed = time.monotonic() - t_start
    avg_elapsed = elapsed / max(1, trials_attempted)

    print("=============================================")
    print(f"Trials complete: {succeeded}/{trials_attempted} succeeded"
          + (f" ({args.num_trials - trials_attempted} skipped)" if _shutting_down else ""))
    print(f"Total elapsed:   {elapsed:.1f}s ({elapsed/3600:.1f}h)")
    print(f"Avg per trial:   {avg_elapsed:.1f}s")
    print("=============================================")

    if succeeded == 0:
        if _shutting_down:
            print("Interrupted before any trials completed.")
            sys.exit(130)
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
        "trials_attempted": trials_attempted,
        "preset": args.preset,
        "skip_lm_eval": skip_lm_eval,
        "checkpoint_dir": args.checkpoint_dir,
        "config_file": args.config_file,
        "elapsed_seconds": elapsed,
        "avg_elapsed_per_trial": avg_elapsed,
        "interrupted": _shutting_down,
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
