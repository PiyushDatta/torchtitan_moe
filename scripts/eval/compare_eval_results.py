#!/usr/bin/env python3
"""
Compare two MoE evaluation results to determine which performs better.

Supports both individual trial files and summary.json files (with averaged metrics).

Usage:
    # Compare two summary files (recommended):
    python scripts/eval/compare_eval_results.py results/exp_a/summary.json results/exp_b/summary.json

    # Compare two individual trial files:
    python scripts/eval/compare_eval_results.py results/trial_a.json results/trial_b.json

    # JSON output:
    python scripts/eval/compare_eval_results.py summary_a.json summary_b.json --json
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

# =============================================================================
# Constants
# =============================================================================

CATEGORY_ORDER = ["routing", "performance", "cost", "lm_eval", "walltime"]

CATEGORY_DISPLAY_NAMES = {
    "routing": "ROUTING EFFICIENCY",
    "performance": "INFERENCE PERFORMANCE",
    "cost": "COMPUTATIONAL COST",
    "lm_eval": "MODEL ACCURACY",
    "walltime": "EVALUATION TIME",
}

# Metric definitions: (display_name, json_path, lower_is_better)
ROUTING_METRICS = [
    ("Avg Gini Coefficient", "avg_gini_coefficient", True),
    ("Avg Coeff. of Variation", "avg_coefficient_of_variation", True),
    ("Avg Expert Utilization", "avg_expert_utilization_rate", False),
]

PERFORMANCE_METRICS = [
    ("Latency (ms)", "latency_ms", True),
    ("Throughput (tokens/sec)", "throughput_tokens_per_sec", False),
    ("Memory Allocated (GB)", "memory_allocated_gb", True),
    ("Memory Reserved (GB)", "memory_reserved_gb", True),
]

COST_METRICS = [
    ("TFLOPs", "tflops", True),
    ("Active Params (B)", "active_params_billions", True),
]

LM_EVAL_TASKS = ["mmlu", "hellaswag", "winogrande", "arc_easy", "arc_challenge"]

# Box drawing characters
BOX_TL, BOX_TR, BOX_BL, BOX_BR = "┌", "┐", "└", "┘"
BOX_H, BOX_V = "─", "│"
BOX_LT, BOX_RT, BOX_TT, BOX_BT, BOX_X = "├", "┤", "┬", "┴", "┼"


# =============================================================================
# Utility Functions
# =============================================================================


def load_results(path: str) -> dict:
    """Load eval results from JSON file."""
    with open(path) as f:
        return json.load(f)


def get_nested(data: dict, *keys, default=None):
    """Safely get nested dictionary value."""
    for key in keys:
        if data is None or not isinstance(data, dict):
            return default
        data = data.get(key, default)
    return data


def format_value(value, precision: int = 4) -> str:
    """Format a value for display."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.{precision}f}"
    return str(value)


def format_mean_std(mean, stdev, precision: int = 4) -> str:
    """Format a mean +/- stdev value for display."""
    if mean is None:
        return "N/A"
    mean_str = format_value(mean, precision)
    if stdev is not None and stdev > 0:
        stdev_str = format_value(stdev, precision)
        return f"{mean_str} ±{stdev_str}"
    return mean_str


def is_summary_format(data: dict) -> bool:
    """Detect whether data is a summary.json (vs individual trial)."""
    return "averages" in data


# =============================================================================
# Normalization: convert both formats to a common structure
# =============================================================================


def normalize_summary(data: dict, path: Path) -> dict:
    """Normalize summary.json format to common comparison structure."""
    avg = data.get("averages", {})
    routing = get_nested(avg, "routing", "overall", default={}) or {}
    inference = avg.get("inference", {}) or {}
    compute = avg.get("compute", {}) or {}
    walltime = avg.get("walltime", {}) or {}

    result = {
        "name": data.get("experiment_name", path.parent.name),
        "num_trials": data.get("num_trials"),
        "is_summary": True,
        "routing": {
            "avg_gini_coefficient": routing.get("avg_gini_mean"),
            "avg_gini_stdev": routing.get("avg_gini_stdev"),
            "avg_coefficient_of_variation": routing.get("avg_cv_mean"),
            "avg_cv_stdev": routing.get("avg_cv_stdev"),
            "avg_expert_utilization_rate": routing.get("avg_utilization_mean"),
            "avg_utilization_stdev": routing.get("avg_utilization_stdev"),
        },
        "routing_per_layer": get_nested(avg, "routing", "per_layer", default={}),
        "performance": {
            "latency_ms": get_nested(inference, "latency_ms", "mean"),
            "latency_ms_stdev": get_nested(inference, "latency_ms", "stdev"),
            "throughput_tokens_per_sec": get_nested(
                inference, "throughput_tokens_per_sec", "mean"
            ),
            "throughput_stdev": get_nested(
                inference, "throughput_tokens_per_sec", "stdev"
            ),
            "memory_allocated_gb": get_nested(
                inference, "memory_allocated_gb", "mean"
            ),
            "memory_reserved_gb": get_nested(inference, "memory_reserved_gb", "mean"),
        },
        "cost": {
            "tflops": compute.get("total_flops"),
            "active_params_billions": compute.get("active_params_billions"),
        },
        "walltime": {
            "mean": walltime.get("mean"),
            "stdev": walltime.get("stdev"),
            "total": walltime.get("total"),
        },
        "lm_eval": {},
        "model_args": data.get("model_args", {}),
    }

    # Aggregate lm_eval from trial files if available
    trials_dir = path.parent / "trials"
    if trials_dir.is_dir():
        result["lm_eval"] = aggregate_lm_eval_from_trials(trials_dir)

    return result


def aggregate_lm_eval_from_trials(trials_dir: Path) -> dict:
    """Aggregate lm_eval results across trial files, computing mean and stdev."""
    trial_files = sorted(trials_dir.glob("trial_*.json"))
    if not trial_files:
        return {}

    # Collect all task/metric values across trials
    task_metrics: dict[str, dict[str, list[float]]] = {}

    for tf in trial_files:
        try:
            trial = json.load(open(tf))
        except (json.JSONDecodeError, OSError):
            continue
        lm = trial.get("lm_eval_results", {})
        if not lm:
            continue
        for task_name, task_data in lm.items():
            if task_name.startswith("_") or not isinstance(task_data, dict):
                continue
            if task_name not in task_metrics:
                task_metrics[task_name] = {}
            for metric_key, value in task_data.items():
                if metric_key in ("alias",) or not isinstance(value, (int, float)):
                    continue
                if metric_key not in task_metrics[task_name]:
                    task_metrics[task_name][metric_key] = []
                task_metrics[task_name][metric_key].append(float(value))

    # Compute mean and stdev
    aggregated = {}
    for task_name, metrics in task_metrics.items():
        aggregated[task_name] = {}
        for metric_key, values in metrics.items():
            if len(values) > 0:
                aggregated[task_name][metric_key] = {
                    "mean": statistics.mean(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "n": len(values),
                }

    return aggregated


def normalize_trial(data: dict, path: Path) -> dict:
    """Normalize individual trial file format to common comparison structure."""
    agg = get_nested(data, "routing_stats", "aggregate", default={}) or {}
    perf = data.get("inference_performance", {}) or {}
    cost = data.get("computational_cost", {}) or {}

    result = {
        "name": path.stem,
        "num_trials": 1,
        "is_summary": False,
        "routing": {
            "avg_gini_coefficient": agg.get("avg_gini_coefficient"),
            "avg_coefficient_of_variation": agg.get("avg_coefficient_of_variation"),
            "avg_expert_utilization_rate": agg.get("avg_expert_utilization_rate"),
        },
        "routing_per_layer": {},
        "performance": {
            "latency_ms": perf.get("latency_ms"),
            "throughput_tokens_per_sec": perf.get("throughput_tokens_per_sec"),
            "memory_allocated_gb": perf.get("memory_allocated_gb"),
            "memory_reserved_gb": perf.get("memory_reserved_gb"),
        },
        "cost": {
            "tflops": cost.get("tflops"),
            "active_params_billions": cost.get("active_params_billions"),
        },
        "walltime": {
            "mean": data.get("walltime_seconds"),
        },
        "lm_eval": {},
        "model_args": data.get("model_args", {}),
    }

    # Convert lm_eval to consistent format (wrap in mean/stdev)
    lm = data.get("lm_eval_results", {})
    if lm:
        for task_name, task_data in lm.items():
            if task_name.startswith("_") or not isinstance(task_data, dict):
                continue
            result["lm_eval"][task_name] = {}
            for metric_key, value in task_data.items():
                if metric_key in ("alias",) or not isinstance(value, (int, float)):
                    continue
                result["lm_eval"][task_name][metric_key] = {
                    "mean": float(value),
                    "stdev": 0.0,
                    "n": 1,
                }

    return result


def normalize(data: dict, path: Path) -> dict:
    """Auto-detect format and normalize."""
    if is_summary_format(data):
        return normalize_summary(data, path)
    return normalize_trial(data, path)


# =============================================================================
# Comparison Logic
# =============================================================================


def compare_metric(
    name: str, val_a, val_b, lower_is_better: bool = True, stdev_a=None, stdev_b=None
) -> dict:
    """Compare a single metric between two results."""
    result = {
        "name": name,
        "a": val_a,
        "b": val_b,
        "stdev_a": stdev_a,
        "stdev_b": stdev_b,
        "diff": None,
        "pct_diff": None,
        "winner": None,
    }

    if val_a is None or val_b is None:
        return result

    try:
        float_a = float(val_a)
        float_b = float(val_b)
    except (ValueError, TypeError):
        return result

    diff = float_b - float_a
    result["diff"] = diff
    if float_a != 0:
        result["pct_diff"] = (diff / abs(float_a)) * 100

    if float_a == float_b:
        result["winner"] = "tie"
    elif lower_is_better:
        result["winner"] = "B" if float_b < float_a else "A"
    else:
        result["winner"] = "B" if float_b > float_a else "A"

    return result


def compare_routing(norm_a: dict, norm_b: dict) -> list[dict]:
    """Compare routing efficiency metrics."""
    ra = norm_a["routing"]
    rb = norm_b["routing"]
    return [
        compare_metric(
            name,
            ra.get(key),
            rb.get(key),
            lower_is_better,
            ra.get(key.replace("avg_", "avg_").rstrip("_rate").rstrip("_coefficient").rstrip("_of_variation") + "_stdev" if not key.endswith("_stdev") else None),
            rb.get(key.replace("avg_", "avg_").rstrip("_rate").rstrip("_coefficient").rstrip("_of_variation") + "_stdev" if not key.endswith("_stdev") else None),
        )
        for name, key, lower_is_better in ROUTING_METRICS
    ]


def compare_performance(norm_a: dict, norm_b: dict) -> list[dict]:
    """Compare inference performance metrics."""
    pa = norm_a["performance"]
    pb = norm_b["performance"]

    stdev_keys = {
        "latency_ms": "latency_ms_stdev",
        "throughput_tokens_per_sec": "throughput_stdev",
    }

    return [
        compare_metric(
            name,
            pa.get(key),
            pb.get(key),
            lower_is_better,
            pa.get(stdev_keys.get(key)),
            pb.get(stdev_keys.get(key)),
        )
        for name, key, lower_is_better in PERFORMANCE_METRICS
    ]


def compare_cost(norm_a: dict, norm_b: dict) -> list[dict]:
    """Compare computational cost metrics."""
    ca = norm_a["cost"]
    cb = norm_b["cost"]
    return [
        compare_metric(name, ca.get(key), cb.get(key), lower_is_better)
        for name, key, lower_is_better in COST_METRICS
    ]


def compare_lm_eval(norm_a: dict, norm_b: dict) -> list[dict]:
    """Compare lm_eval benchmark scores."""
    lm_a = norm_a.get("lm_eval", {})
    lm_b = norm_b.get("lm_eval", {})

    if not lm_a and not lm_b:
        return []

    all_tasks = set(lm_a.keys()) | set(lm_b.keys())

    # Order: known tasks first, then alphabetical extras
    tasks_to_compare = [t for t in LM_EVAL_TASKS if t in all_tasks]
    for task in sorted(all_tasks):
        if task not in tasks_to_compare:
            tasks_to_compare.append(task)

    comparisons = []
    for task in tasks_to_compare:
        task_a = lm_a.get(task, {})
        task_b = lm_b.get(task, {})

        # Check for acc and acc_norm metrics
        for metric_suffix in ["acc,none", "acc_norm,none"]:
            label = metric_suffix.split(",")[0]
            ma = task_a.get(metric_suffix, {})
            mb = task_b.get(metric_suffix, {})

            val_a = ma.get("mean") if isinstance(ma, dict) else None
            val_b = mb.get("mean") if isinstance(mb, dict) else None
            std_a = ma.get("stdev") if isinstance(ma, dict) else None
            std_b = mb.get("stdev") if isinstance(mb, dict) else None

            if val_a is not None or val_b is not None:
                comparisons.append(
                    compare_metric(
                        f"{task} ({label})",
                        val_a,
                        val_b,
                        lower_is_better=False,
                        stdev_a=std_a,
                        stdev_b=std_b,
                    )
                )

    return comparisons


def compare_walltime(norm_a: dict, norm_b: dict) -> list[dict]:
    """Compare evaluation walltime."""
    wa = norm_a["walltime"]
    wb = norm_b["walltime"]
    return [
        compare_metric(
            "Avg Walltime (seconds)",
            wa.get("mean"),
            wb.get("mean"),
            lower_is_better=True,
            stdev_a=wa.get("stdev"),
            stdev_b=wb.get("stdev"),
        )
    ]


def compare_results(norm_a: dict, norm_b: dict) -> dict:
    """Compare two normalized results across all categories."""
    return {
        "routing": compare_routing(norm_a, norm_b),
        "performance": compare_performance(norm_a, norm_b),
        "cost": compare_cost(norm_a, norm_b),
        "lm_eval": compare_lm_eval(norm_a, norm_b),
        "walltime": compare_walltime(norm_a, norm_b),
    }


def determine_overall_winner(comparisons: dict) -> tuple[str, dict, dict]:
    """Determine overall winner based on all comparisons.
    Returns (overall_winner, total_wins, per_category_wins)."""
    wins = {"A": 0, "B": 0, "tie": 0}
    per_category = {}

    for cat, metrics in comparisons.items():
        cat_wins = {"A": 0, "B": 0, "tie": 0}
        for metric in metrics:
            winner = metric.get("winner")
            if winner in wins:
                wins[winner] += 1
                cat_wins[winner] += 1
        per_category[cat] = cat_wins

    if wins["A"] > wins["B"]:
        overall = "A"
    elif wins["B"] > wins["A"]:
        overall = "B"
    else:
        overall = "tie"

    return overall, wins, per_category


# =============================================================================
# Config Diff
# =============================================================================


def find_config_diffs(norm_a: dict, norm_b: dict) -> list[tuple[str, str, str]]:
    """Find differences in model_args between two results.
    Returns list of (key_path, value_a, value_b)."""
    diffs = []

    def _compare_dicts(da, db, prefix=""):
        all_keys = set(list(da.keys()) + list(db.keys()))
        for key in sorted(all_keys):
            path = f"{prefix}.{key}" if prefix else key
            va = da.get(key)
            vb = db.get(key)
            if isinstance(va, dict) and isinstance(vb, dict):
                _compare_dicts(va, vb, path)
            elif va != vb:
                diffs.append((path, str(va), str(vb)))

    ma = norm_a.get("model_args", {})
    mb = norm_b.get("model_args", {})
    _compare_dicts(ma, mb)
    return diffs


# =============================================================================
# Table Printing
# =============================================================================


def build_table_rows(comparisons: dict) -> list[tuple[str, dict]]:
    """Build list of (category_name, metric) tuples for table display."""
    rows = []
    for cat in CATEGORY_ORDER:
        if cat in comparisons and comparisons[cat]:
            cat_name = CATEGORY_DISPLAY_NAMES.get(cat, cat.upper())
            for metric in comparisons[cat]:
                rows.append((cat_name, metric))
    return rows


def format_winner_str(winner: str | None) -> str:
    """Format the winner column string."""
    return winner or "-"


def format_diff_str(metric: dict) -> str:
    """Format the difference column string."""
    if metric["diff"] is None:
        return "N/A"
    if metric["pct_diff"] is not None and metric["pct_diff"] != 0:
        sign = "+" if metric["pct_diff"] > 0 else ""
        return f"{sign}{metric['pct_diff']:.2f}%"
    if metric["diff"] == 0:
        return "-"
    return format_value(metric["diff"])


def format_cell_value(metric: dict, side: str) -> str:
    """Format a cell value, including stdev if available."""
    val = metric[side]
    stdev = metric.get(f"stdev_{side}")
    if val is None:
        return "N/A"
    if stdev is not None and stdev > 0:
        return format_mean_std(val, stdev)
    return format_value(val)


def print_table_border(cols: tuple, char_left: str, char_mid: str, char_right: str):
    """Print a table border line."""
    parts = [f"{char_left}"]
    for i, w in enumerate(cols):
        parts.append(f"{BOX_H * (w + 2)}")
        parts.append(char_mid if i < len(cols) - 1 else char_right)
    print("".join(parts))


def print_table_row(cols: tuple, values: tuple):
    """Print a table data row."""
    parts = []
    for i, (w, v) in enumerate(zip(cols, values)):
        if i == 0:
            parts.append(f"{BOX_V} {v:<{w}} ")
        elif i == len(cols) - 1:
            parts.append(f"{BOX_V} {v:^{w}} {BOX_V}")
        else:
            parts.append(f"{BOX_V} {v:>{w}} ")
    print("".join(parts))


def print_legend(name_a: str, name_b: str, trials_a: int, trials_b: int):
    """Print the legend showing which result is A and B."""
    print()
    width = 80
    print(BOX_TL + BOX_H * width + BOX_TR)
    print(f"{BOX_V} {'MODEL COMPARISON':^{width}} {BOX_V}")
    print(BOX_LT + BOX_H * width + BOX_BT)

    suffix_a = f"  ({trials_a} trials)" if trials_a and trials_a > 1 else ""
    suffix_b = f"  ({trials_b} trials)" if trials_b and trials_b > 1 else ""

    line_a = f"  Model A:  {name_a}{suffix_a}"
    line_b = f"  Model B:  {name_b}{suffix_b}"

    # Truncate if needed
    line_a = line_a[: width - 1]
    line_b = line_b[: width - 1]

    print(f"  {line_a}")
    print(f"  {line_b}")


def print_config_diffs(diffs: list[tuple[str, str, str]]):
    """Print config differences between the two models."""
    if not diffs:
        return
    print()
    print(f"  Config differences:")
    for path, va, vb in diffs:
        print(f"    {path}: {va} → {vb}")


def print_comparison_table(comparisons: dict, has_stdev: bool):
    """Print a comparison table with box-drawing characters."""
    rows = build_table_rows(comparisons)
    if not rows:
        print("No metrics to compare.")
        return

    # Column widths - wider for stdev values
    if has_stdev:
        cols = (30, 20, 20, 10, 8)
    else:
        cols = (30, 14, 14, 10, 8)

    # Print header
    print()
    print_table_border(cols, BOX_TL, BOX_TT, BOX_TR)
    print_table_row(cols, ("Metric", "Model A", "Model B", "% Diff", "Winner"))
    print_table_border(cols, BOX_LT, BOX_X, BOX_RT)

    # Print rows grouped by category
    current_category = None
    for category, metric in rows:
        if category != current_category:
            if current_category is not None:
                print_table_border(cols, BOX_LT, BOX_X, BOX_RT)
            current_category = category
            print_table_row(cols, (f"  {category}", "", "", "", ""))
            print_table_border(cols, BOX_LT, BOX_X, BOX_RT)

        print_table_row(
            cols,
            (
                f"    {metric['name'][:26]}",
                format_cell_value(metric, "a"),
                format_cell_value(metric, "b"),
                format_diff_str(metric),
                format_winner_str(metric.get("winner")),
            ),
        )

    # Print footer
    print_table_border(cols, BOX_BL, BOX_BT, BOX_BR)


def _short_name(name: str, max_len: int = 20) -> str:
    """Shorten an experiment name for display."""
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


def _category_winner_str(cat_wins: dict, short_a: str, short_b: str) -> str:
    """Return a short string describing who won a category."""
    if cat_wins["A"] > cat_wins["B"]:
        return f"A ({short_a}) wins {cat_wins['A']}-{cat_wins['B']}"
    elif cat_wins["B"] > cat_wins["A"]:
        return f"B ({short_b}) wins {cat_wins['B']}-{cat_wins['A']}"
    elif cat_wins["A"] == 0 and cat_wins["B"] == 0:
        return "tied (all same)"
    else:
        return f"tied ({cat_wins['A']}-{cat_wins['B']})"


def print_summary_box(
    name_a: str,
    name_b: str,
    wins: dict,
    per_category: dict,
    overall_winner: str,
):
    """Print the summary box with per-category breakdown."""
    short_a = _short_name(name_a, 25)
    short_b = _short_name(name_b, 25)

    print()
    width = 72
    print(BOX_TL + BOX_H * width + BOX_TR)
    print(f"{BOX_V} {'SUMMARY':^{width}} {BOX_V}")
    print(BOX_LT + BOX_H * width + BOX_RT)

    # Legend reminder
    print(f"{BOX_V} {'  A = ' + short_a:<{width - 1}}{BOX_V}")
    print(f"{BOX_V} {'  B = ' + short_b:<{width - 1}}{BOX_V}")
    print(BOX_LT + BOX_H * width + BOX_RT)

    # Per-category breakdown
    for cat in CATEGORY_ORDER:
        if cat not in per_category:
            continue
        cw = per_category[cat]
        # Skip categories with no comparisons
        if cw["A"] + cw["B"] + cw["tie"] == 0:
            continue
        cat_display = CATEGORY_DISPLAY_NAMES.get(cat, cat.upper())
        winner_str = _category_winner_str(cw, short_a, short_b)
        line = f"  {cat_display + ':':<26} {winner_str}"
        line = line[: width - 1]
        print(f"{BOX_V} {line:<{width - 1}}{BOX_V}")

    print(BOX_LT + BOX_H * width + BOX_RT)

    # Total score
    score_line = f"  Total:  A: {wins['A']} wins  |  B: {wins['B']} wins  |  Ties: {wins['tie']}"
    print(f"{BOX_V} {score_line:<{width - 1}}{BOX_V}")
    print(BOX_LT + BOX_H * width + BOX_RT)

    # Overall winner
    if overall_winner == "A":
        winner_line = f"  OVERALL WINNER: {name_a}"
    elif overall_winner == "B":
        winner_line = f"  OVERALL WINNER: {name_b}"
    else:
        winner_line = "  RESULT: TIE (no clear winner)"

    winner_line = winner_line[: width - 1]
    print(f"{BOX_V} {winner_line:<{width - 1}}{BOX_V}")
    print(BOX_BL + BOX_H * width + BOX_BR)


def print_interpretation_guide():
    """Print the interpretation guide."""
    print()
    width = 68
    print(BOX_TL + BOX_H * width + BOX_TR)
    print(f"{BOX_V} {'INTERPRETATION GUIDE':^{width - 2}} {BOX_V}")
    print(BOX_LT + BOX_H * width + BOX_RT)
    print(f"{BOX_V} {'Winner: A = Model A is better  |  B = Model B is better':<{width-1}}{BOX_V}")
    print(BOX_LT + BOX_H * width + BOX_RT)
    print(f"{BOX_V} {'ROUTING EFFICIENCY':<{width-1}}{BOX_V}")
    print(f"{BOX_V} {'  Gini Coefficient:        Lower is better (0=perfect)':<{width-1}}{BOX_V}")
    print(f"{BOX_V} {'  Coeff. of Variation:     Lower is better':<{width-1}}{BOX_V}")
    print(f"{BOX_V} {'  Expert Utilization:      Higher is better (1.0=100%)':<{width-1}}{BOX_V}")
    print(BOX_LT + BOX_H * width + BOX_RT)
    print(f"{BOX_V} {'INFERENCE PERFORMANCE':<{width-1}}{BOX_V}")
    print(f"{BOX_V} {'  Latency:                 Lower is better':<{width-1}}{BOX_V}")
    print(f"{BOX_V} {'  Throughput:              Higher is better':<{width-1}}{BOX_V}")
    print(f"{BOX_V} {'  Memory:                  Lower is better':<{width-1}}{BOX_V}")
    print(BOX_LT + BOX_H * width + BOX_RT)
    print(f"{BOX_V} {'COMPUTATIONAL COST':<{width-1}}{BOX_V}")
    print(f"{BOX_V} {'  TFLOPs:                  Lower = more efficient':<{width-1}}{BOX_V}")
    print(f"{BOX_V} {'  Active Params:           Lower = more efficient':<{width-1}}{BOX_V}")
    print(BOX_LT + BOX_H * width + BOX_RT)
    print(f"{BOX_V} {'MODEL ACCURACY (lm_eval)':<{width-1}}{BOX_V}")
    print(f"{BOX_V} {'  acc / acc_norm:          Higher is better':<{width-1}}{BOX_V}")
    print(BOX_BL + BOX_H * width + BOX_BR)
    print()


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare two MoE evaluation results (summary.json or trial files)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/eval/compare_eval_results.py results/exp_a/summary.json results/exp_b/summary.json
    python scripts/eval/compare_eval_results.py trial_a.json trial_b.json
    python scripts/eval/compare_eval_results.py summary_a.json summary_b.json --json
        """,
    )
    parser.add_argument(
        "result_a",
        type=str,
        help="Path to first result file (summary.json or trial file, labeled as Model A)",
    )
    parser.add_argument(
        "result_b",
        type=str,
        help="Path to second result file (summary.json or trial file, labeled as Model B)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON instead of formatted table"
    )
    return parser.parse_args()


def validate_paths(path_a: Path, path_b: Path):
    """Validate that input paths exist."""
    if not path_a.exists():
        print(f"Error: File not found: {path_a}", file=sys.stderr)
        sys.exit(1)
    if not path_b.exists():
        print(f"Error: File not found: {path_b}", file=sys.stderr)
        sys.exit(1)


def output_json(
    norm_a: dict, norm_b: dict, comparisons: dict, winner: str, wins: dict, per_category: dict
):
    """Output results as JSON."""
    output = {
        "model_a": norm_a["name"],
        "model_b": norm_b["name"],
        "comparisons": comparisons,
        "overall_winner": winner,
        "wins": wins,
        "per_category_wins": per_category,
        "config_diffs": find_config_diffs(norm_a, norm_b),
    }
    print(json.dumps(output, indent=2, default=str))


def output_table(
    norm_a: dict, norm_b: dict, comparisons: dict, winner: str, wins: dict, per_category: dict
):
    """Output results as formatted table."""
    has_stdev = norm_a["is_summary"] or norm_b["is_summary"]

    print("\n" + "=" * 82)
    print(" MoE EVALUATION COMPARISON".center(82))
    print("=" * 82)

    print_legend(norm_a["name"], norm_b["name"], norm_a["num_trials"], norm_b["num_trials"])

    diffs = find_config_diffs(norm_a, norm_b)
    print_config_diffs(diffs)

    print_comparison_table(comparisons, has_stdev)
    print_summary_box(norm_a["name"], norm_b["name"], wins, per_category, winner)
    print_interpretation_guide()


def main():
    args = parse_args()

    path_a = Path(args.result_a)
    path_b = Path(args.result_b)
    validate_paths(path_a, path_b)

    raw_a = load_results(path_a)
    raw_b = load_results(path_b)

    norm_a = normalize(raw_a, path_a)
    norm_b = normalize(raw_b, path_b)

    comparisons = compare_results(norm_a, norm_b)
    overall_winner, wins, per_category = determine_overall_winner(comparisons)

    if args.json:
        output_json(norm_a, norm_b, comparisons, overall_winner, wins, per_category)
    else:
        output_table(norm_a, norm_b, comparisons, overall_winner, wins, per_category)


if __name__ == "__main__":
    main()
