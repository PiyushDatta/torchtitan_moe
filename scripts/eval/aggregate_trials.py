#!/usr/bin/env python3
"""
Aggregate multiple eval trial JSONs into a single summary with per-trial data and averages.

Usage:
    python scripts/eval/aggregate_trials.py \
        --trials_dir ./results/baseline_20260309/trials \
        --output ./results/baseline_20260309/summary.json \
        --experiment_name baseline
"""

import argparse
import json
import statistics
import sys
from pathlib import Path


def load_trials(trials_dir: Path) -> list[dict]:
    """Load all trial JSON files from directory, sorted by name."""
    trials = []
    for f in sorted(trials_dir.glob("trial_*.json")):
        try:
            with open(f) as fh:
                trials.append(json.load(fh))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skipping {f.name}: {e}", file=sys.stderr)
    return trials


def safe_mean(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.mean(clean) if clean else None


def safe_stdev(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.stdev(clean) if len(clean) >= 2 else None


def aggregate_routing(trials: list[dict]) -> dict:
    """Aggregate routing stats across trials."""
    # Collect per-layer stats across trials
    layer_keys = set()
    for t in trials:
        if t.get("routing_stats"):
            layer_keys.update(
                k for k in t["routing_stats"] if k.startswith("layer_")
            )

    layer_keys = sorted(layer_keys, key=lambda x: int(x.split("_")[1]))

    per_layer = {}
    for lk in layer_keys:
        cvs = []
        ginis = []
        utils = []
        for t in trials:
            stats = (t.get("routing_stats") or {}).get(lk)
            if stats:
                cvs.append(stats.get("coefficient_of_variation"))
                ginis.append(stats.get("gini_coefficient"))
                utils.append(stats.get("expert_utilization_rate"))

        per_layer[lk] = {
            "cv_mean": safe_mean(cvs),
            "cv_stdev": safe_stdev(cvs),
            "gini_mean": safe_mean(ginis),
            "gini_stdev": safe_stdev(ginis),
            "utilization_mean": safe_mean(utils),
            "utilization_stdev": safe_stdev(utils),
            "n_trials": len(cvs),
        }

    # Overall averages (average of per-layer means across all trials)
    all_cvs = []
    all_ginis = []
    all_utils = []
    for t in trials:
        if not t.get("routing_stats"):
            continue
        t_cvs = []
        t_ginis = []
        t_utils = []
        for lk in layer_keys:
            stats = t["routing_stats"].get(lk)
            if stats:
                t_cvs.append(stats.get("coefficient_of_variation"))
                t_ginis.append(stats.get("gini_coefficient"))
                t_utils.append(stats.get("expert_utilization_rate"))
        if t_cvs:
            all_cvs.append(safe_mean(t_cvs))
        if t_ginis:
            all_ginis.append(safe_mean(t_ginis))
        if t_utils:
            all_utils.append(safe_mean(t_utils))

    return {
        "overall": {
            "avg_cv_mean": safe_mean(all_cvs),
            "avg_cv_stdev": safe_stdev(all_cvs),
            "avg_gini_mean": safe_mean(all_ginis),
            "avg_gini_stdev": safe_stdev(all_ginis),
            "avg_utilization_mean": safe_mean(all_utils),
            "avg_utilization_stdev": safe_stdev(all_utils),
        },
        "per_layer": per_layer,
    }


def aggregate_inference(trials: list[dict]) -> dict:
    """Aggregate inference performance across trials."""
    latencies = []
    throughputs = []
    mem_alloc = []
    mem_reserved = []

    for t in trials:
        perf = t.get("inference_performance") or {}
        latencies.append(perf.get("latency_ms"))
        throughputs.append(perf.get("throughput_tokens_per_sec"))
        mem_alloc.append(perf.get("memory_allocated_gb"))
        mem_reserved.append(perf.get("memory_reserved_gb"))

    return {
        "latency_ms": {
            "mean": safe_mean(latencies),
            "stdev": safe_stdev(latencies),
        },
        "throughput_tokens_per_sec": {
            "mean": safe_mean(throughputs),
            "stdev": safe_stdev(throughputs),
        },
        "memory_allocated_gb": {
            "mean": safe_mean(mem_alloc),
            "stdev": safe_stdev(mem_alloc),
        },
        "memory_reserved_gb": {
            "mean": safe_mean(mem_reserved),
            "stdev": safe_stdev(mem_reserved),
        },
    }


def aggregate_compute(trials: list[dict]) -> dict:
    """Aggregate computational cost (should be constant across trials)."""
    flops = []
    active_params = []
    flops_per_tok = []

    for t in trials:
        cost = t.get("computational_cost") or {}
        flops.append(cost.get("total_flops"))
        active_params.append(cost.get("active_params_billions"))
        flops_per_tok.append(cost.get("num_flops_per_token"))

    return {
        "total_flops": safe_mean(flops),
        "active_params_billions": safe_mean(active_params),
        "num_flops_per_token": safe_mean(flops_per_tok),
    }


def aggregate_walltimes(trials: list[dict]) -> dict:
    wts = [t.get("walltime_seconds") for t in trials]
    return {
        "mean": safe_mean(wts),
        "stdev": safe_stdev(wts),
        "total": sum(w for w in wts if w),
    }


def build_summary(
    trials: list[dict], experiment_name: str
) -> dict:
    """Build the full summary dict."""
    # Per-trial compact view (no large distributions)
    per_trial = []
    for i, t in enumerate(trials):
        perf = t.get("inference_performance") or {}
        cost = t.get("computational_cost") or {}

        # Compute per-trial routing averages
        routing = t.get("routing_stats") or {}
        layer_cvs = []
        layer_ginis = []
        layer_utils = []
        for k, v in routing.items():
            if k.startswith("layer_") and isinstance(v, dict):
                if v.get("coefficient_of_variation") is not None:
                    layer_cvs.append(v["coefficient_of_variation"])
                if v.get("gini_coefficient") is not None:
                    layer_ginis.append(v["gini_coefficient"])
                if v.get("expert_utilization_rate") is not None:
                    layer_utils.append(v["expert_utilization_rate"])

        per_trial.append({
            "trial": i + 1,
            "avg_cv": safe_mean(layer_cvs),
            "avg_gini": safe_mean(layer_ginis),
            "avg_utilization": safe_mean(layer_utils),
            "latency_ms": perf.get("latency_ms"),
            "throughput_tok_s": perf.get("throughput_tokens_per_sec"),
            "memory_allocated_gb": perf.get("memory_allocated_gb"),
            "walltime_s": t.get("walltime_seconds"),
        })

    return {
        "experiment_name": experiment_name,
        "num_trials": len(trials),
        "checkpoint_dir": trials[0].get("checkpoint_dir") if trials else None,
        "config": trials[0].get("config") if trials else None,
        "model_args": trials[0].get("model_args") if trials else None,
        "averages": {
            "routing": aggregate_routing(trials),
            "inference": aggregate_inference(trials),
            "compute": aggregate_compute(trials),
            "walltime": aggregate_walltimes(trials),
        },
        "per_trial": per_trial,
    }


def print_summary(summary: dict):
    """Print a human-readable summary."""
    avg = summary["averages"]
    n = summary["num_trials"]

    print(f"\n{'='*72}")
    print(f"AGGREGATED RESULTS: {summary['experiment_name']} ({n} trials)")
    print(f"{'='*72}")

    routing = avg["routing"]["overall"]
    print(f"\n[ROUTING EFFICIENCY]")
    print(f"  Avg CV:          {routing['avg_cv_mean']:.4f} +/- {routing['avg_cv_stdev']:.4f}" if routing["avg_cv_stdev"] else f"  Avg CV:          {routing['avg_cv_mean']:.4f}")
    print(f"  Avg Gini:        {routing['avg_gini_mean']:.4f} +/- {routing['avg_gini_stdev']:.4f}" if routing["avg_gini_stdev"] else f"  Avg Gini:        {routing['avg_gini_mean']:.4f}")
    print(f"  Avg Utilization: {routing['avg_utilization_mean']:.2%} +/- {routing['avg_utilization_stdev']:.2%}" if routing["avg_utilization_stdev"] else f"  Avg Utilization: {routing['avg_utilization_mean']:.2%}")

    inf = avg["inference"]
    if inf["latency_ms"]["mean"]:
        print(f"\n[INFERENCE PERFORMANCE]")
        lat = inf["latency_ms"]
        thr = inf["throughput_tokens_per_sec"]
        mem = inf["memory_allocated_gb"]
        print(f"  Latency:         {lat['mean']:.2f} +/- {lat['stdev']:.2f} ms" if lat["stdev"] else f"  Latency:         {lat['mean']:.2f} ms")
        print(f"  Throughput:      {thr['mean']:.2f} +/- {thr['stdev']:.2f} tok/s" if thr["stdev"] else f"  Throughput:      {thr['mean']:.2f} tok/s")
        print(f"  Memory:          {mem['mean']:.2f} +/- {mem['stdev']:.2f} GB" if mem["stdev"] else f"  Memory:          {mem['mean']:.2f} GB")

    comp = avg["compute"]
    print(f"\n[COMPUTATIONAL COST]")
    print(f"  Active Params:   {comp['active_params_billions']:.2f}B")
    print(f"  FLOPs/token:     {comp['num_flops_per_token']:.2e}")

    wt = avg["walltime"]
    print(f"\n[WALLTIME]")
    print(f"  Per trial:       {wt['mean']:.1f} +/- {wt['stdev']:.1f}s" if wt["stdev"] else f"  Per trial:       {wt['mean']:.1f}s")
    print(f"  Total:           {wt['total']:.1f}s ({wt['total']/3600:.1f}h)")
    print(f"{'='*72}\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate eval trial results")
    parser.add_argument("--trials_dir", required=True, help="Directory containing trial_*.json files")
    parser.add_argument("--output", required=True, help="Output summary JSON path")
    parser.add_argument("--experiment_name", default="experiment", help="Name for this experiment")
    args = parser.parse_args()

    trials_dir = Path(args.trials_dir)
    if not trials_dir.is_dir():
        print(f"Error: {trials_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    trials = load_trials(trials_dir)
    if not trials:
        print(f"Error: no valid trial JSONs found in {trials_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(trials)} trials from {trials_dir}")

    summary = build_summary(trials, args.experiment_name)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary written to {output_path}")
    print_summary(summary)


if __name__ == "__main__":
    main()
