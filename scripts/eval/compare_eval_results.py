#!/usr/bin/env python3
"""
Compare two MoE evaluation result files to determine which performs better.

Usage:
    python scripts/eval/compare_eval_results.py results_a.json results_b.json
    python scripts/eval/compare_eval_results.py results_a.json results_b.json --json
"""

import argparse
import json
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


# =============================================================================
# Comparison Logic
# =============================================================================


def compare_metric(name: str, val_a, val_b, lower_is_better: bool = True) -> dict:
    """Compare a single metric between two results."""
    result = {
        "name": name,
        "a": val_a,
        "b": val_b,
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

    diff = abs(float_b - float_a)
    result["diff"] = diff
    if float_a != 0:
        result["pct_diff"] = (diff / abs(float_a)) * 100

    # Determine winner
    if float_a == float_b:
        result["winner"] = "tie"
    elif lower_is_better:
        result["winner"] = "B" if float_b < float_a else "A"
    else:
        result["winner"] = "B" if float_b > float_a else "A"

    return result


def compare_routing(result_a: dict, result_b: dict) -> list[dict]:
    """Compare routing efficiency metrics."""
    agg_a = get_nested(result_a, "routing_stats", "aggregate", default={})
    agg_b = get_nested(result_b, "routing_stats", "aggregate", default={})

    # Handle None cases (when routing eval was skipped)
    if agg_a is None:
        agg_a = {}
    if agg_b is None:
        agg_b = {}

    return [
        compare_metric(name, agg_a.get(key), agg_b.get(key), lower_is_better)
        for name, key, lower_is_better in ROUTING_METRICS
    ]


def compare_performance(result_a: dict, result_b: dict) -> list[dict]:
    """Compare inference performance metrics."""
    perf_a = get_nested(result_a, "inference_performance", default={})
    perf_b = get_nested(result_b, "inference_performance", default={})

    # Handle None cases (when performance eval was skipped)
    if perf_a is None:
        perf_a = {}
    if perf_b is None:
        perf_b = {}

    return [
        compare_metric(name, perf_a.get(key), perf_b.get(key), lower_is_better)
        for name, key, lower_is_better in PERFORMANCE_METRICS
    ]


def compare_cost(result_a: dict, result_b: dict) -> list[dict]:
    """Compare computational cost metrics."""
    cost_a = get_nested(result_a, "computational_cost", default={})
    cost_b = get_nested(result_b, "computational_cost", default={})

    # Handle None cases (when cost eval was skipped)
    if cost_a is None:
        cost_a = {}
    if cost_b is None:
        cost_b = {}

    return [
        compare_metric(name, cost_a.get(key), cost_b.get(key), lower_is_better)
        for name, key, lower_is_better in COST_METRICS
    ]


def compare_lm_eval(result_a: dict, result_b: dict) -> list[dict]:
    """Compare lm_eval benchmark scores."""
    # lm_eval results are nested under "results" key
    lm_a = get_nested(result_a, "lm_eval_results", "results", default={})
    lm_b = get_nested(result_b, "lm_eval_results", "results", default={})

    # If results key doesn't exist, try direct access (older format)
    if not lm_a:
        lm_a = get_nested(result_a, "lm_eval_results", default={})
    if not lm_b:
        lm_b = get_nested(result_b, "lm_eval_results", default={})

    # Handle None cases (when lm_eval was skipped)
    if lm_a is None:
        lm_a = {}
    if lm_b is None:
        lm_b = {}

    comparisons = []

    # Get all tasks from both results
    all_tasks = set()
    all_tasks.update(lm_a.keys())
    all_tasks.update(lm_b.keys())

    # Filter to known tasks or use all found tasks
    tasks_to_compare = [t for t in LM_EVAL_TASKS if t in all_tasks]
    # Also add any other tasks found that aren't in our predefined list
    for task in sorted(all_tasks):
        if task not in tasks_to_compare and not task.startswith("_"):
            tasks_to_compare.append(task)

    for task in tasks_to_compare:
        task_a = lm_a.get(task, {})
        task_b = lm_b.get(task, {})

        # Try different metric key formats
        # Format 1: "acc,none" and "acc_norm,none" (standard lm_eval format)
        # Format 2: "acc" and "acc_norm" (simplified format)
        acc_a = task_a.get("acc,none") or task_a.get("acc")
        acc_b = task_b.get("acc,none") or task_b.get("acc")
        acc_norm_a = task_a.get("acc_norm,none") or task_a.get("acc_norm")
        acc_norm_b = task_b.get("acc_norm,none") or task_b.get("acc_norm")

        # Add accuracy comparison if available
        if acc_a is not None or acc_b is not None:
            comparisons.append(
                compare_metric(f"{task} (acc)", acc_a, acc_b, lower_is_better=False)
            )

        # Add normalized accuracy comparison if available (this is usually the reported metric)
        if acc_norm_a is not None or acc_norm_b is not None:
            comparisons.append(
                compare_metric(
                    f"{task} (acc_norm)", acc_norm_a, acc_norm_b, lower_is_better=False
                )
            )

    return comparisons


def compare_walltime(result_a: dict, result_b: dict) -> list[dict]:
    """Compare evaluation walltime."""
    return [
        compare_metric(
            "Walltime (seconds)",
            result_a.get("walltime_seconds"),
            result_b.get("walltime_seconds"),
            lower_is_better=True,
        )
    ]


def compare_results(result_a: dict, result_b: dict) -> dict:
    """Compare two evaluation results across all categories."""
    return {
        "routing": compare_routing(result_a, result_b),
        "performance": compare_performance(result_a, result_b),
        "cost": compare_cost(result_a, result_b),
        "lm_eval": compare_lm_eval(result_a, result_b),
        "walltime": compare_walltime(result_a, result_b),
    }


def determine_overall_winner(comparisons: dict) -> tuple[str, dict]:
    """Determine overall winner based on all comparisons."""
    wins = {"A": 0, "B": 0, "tie": 0}

    for metrics in comparisons.values():
        for metric in metrics:
            winner = metric.get("winner")
            if winner in wins:
                wins[winner] += 1

    if wins["A"] > wins["B"]:
        overall = "A"
    elif wins["B"] > wins["A"]:
        overall = "B"
    else:
        overall = "tie"

    return overall, wins


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
    """Format the winner column string with visual indicator."""
    winner = winner or "-"
    return winner


def format_diff_str(metric: dict) -> str:
    """Format the difference column string."""
    if metric["diff"] is None:
        return "N/A"
    if metric["diff"] == 0.00:
        return "-"
    if metric["pct_diff"] is not None:
        return f"{metric['pct_diff']:.2f}%"
    return format_value(metric["diff"])


def print_table_border(cols: tuple, char_left: str, char_mid: str, char_right: str):
    """Print a table border line."""
    col_stat, col_a, col_b, col_diff, col_winner = cols
    print(
        f"{char_left}{BOX_H * (col_stat + 2)}{char_mid}"
        f"{BOX_H * (col_a + 2)}{char_mid}"
        f"{BOX_H * (col_b + 2)}{char_mid}"
        f"{BOX_H * (col_diff + 2)}{char_mid}"
        f"{BOX_H * (col_winner + 2)}{char_right}"
    )


def print_table_row(cols: tuple, values: tuple, highlight: bool = False):
    """Print a table data row."""
    col_stat, col_a, col_b, col_diff, col_winner = cols
    stat, va, vb, diff, winner = values
    print(
        f"{BOX_V} {stat:<{col_stat}} {BOX_V} {va:>{col_a}} {BOX_V} "
        f"{vb:>{col_b}} {BOX_V} {diff:>{col_diff}} {BOX_V} {winner:^{col_winner}} {BOX_V}"
    )


def print_legend(path_a: Path, path_b: Path):
    """Print the legend showing which file is A and B."""
    print()
    width = 78
    print("┌" + "─" * width + "┐")
    print(f"│ {'MODEL COMPARISON LEGEND':^{width - 2}} │")
    print("├" + "─" * width + "┤")

    # Truncate long paths for display
    name_a = path_a.name
    name_b = path_b.name

    line_a = f"  Model A:  {name_a}"
    line_b = f"  Model B:  {name_b}"

    print(f"│ {line_a:<{width - 1}}│")
    print(f"│ {line_b:<{width - 1}}│")
    print("└" + "─" * width + "┘")


def print_comparison_table(comparisons: dict):
    """Print a comparison table with box-drawing characters."""
    rows = build_table_rows(comparisons)
    if not rows:
        print("No metrics to compare.")
        return

    # Column widths
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
                format_value(metric["a"]),
                format_value(metric["b"]),
                format_diff_str(metric),
                format_winner_str(metric.get("winner")),
            ),
        )

    # Print footer
    print_table_border(cols, BOX_BL, BOX_BT, BOX_BR)


def print_summary_box(path_a: Path, path_b: Path, wins: dict, overall_winner: str):
    """Print the summary box."""
    print()
    width = 60
    print("┌" + "─" * width + "┐")
    print(f"│ {'SUMMARY':^{width - 2}} │")
    print("├" + "─" * width + "┤")
    # Score display
    score_line = f"  Model A: {wins['A']} wins  |  Model B: {wins['B']} wins  |  Ties: {wins['tie']}"
    print(f"│ {score_line:<{width - 1}}│")
    print("├" + "─" * width + "┤")
    # Overall winner with clear indication
    if overall_winner == "A":
        winner_line = f"  WINNER: Model A  ({path_a.name[:35]})"
    elif overall_winner == "B":
        winner_line = f"  WINNER: Model B  ({path_b.name[:35]})"
    else:
        winner_line = "  RESULT: TIE (no clear winner)"
    # Truncate if too long
    if len(winner_line) > width - 2:
        winner_line = winner_line[: width - 5] + "..."

    print(f"│ {winner_line:<{width - 1}}│")
    print("└" + "─" * width + "┘")


def print_interpretation_guide():
    """Print the interpretation guide."""
    print()
    width = 60
    print("┌" + "─" * width + "┐")
    print(f"│ {'INTERPRETATION GUIDE':^{width - 2}} │")
    print("├" + "─" * width + "┤")
    print(f"│ {'A = Model A is better  |  B = Model B is better':<{width-1}}│")
    print(f"│ {'tie = Same value':<{width-1}}│")
    print("├" + "─" * width + "┤")
    print(f"│ {'Routing: Lower Gini/CV = better load balance':<{width-1}}│")
    print(f"│ {'Performance: Lower latency, higher throughput = better':<{width - 1}}│")
    print(f"│ {'Cost: Lower = more efficient':<{width - 1}}│")
    print(f"│ {'Accuracy: Higher = better':<{width - 1}}│")
    print("└" + "─" * width + "┘")
    print()


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare two MoE evaluation result files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/eval/compare_eval_results.py results_a.json results_b.json
    python scripts/eval/compare_eval_results.py results_a.json results_b.json --json
        """,
    )
    parser.add_argument(
        "result_a",
        type=str,
        help="Path to first eval_results.json (labeled as Model A)",
    )
    parser.add_argument(
        "result_b",
        type=str,
        help="Path to second eval_results.json (labeled as Model B)",
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


def output_json(path_a: Path, path_b: Path, comparisons: dict, winner: str, wins: dict):
    """Output results as JSON."""
    output = {
        "file_a": str(path_a),
        "file_b": str(path_b),
        "comparisons": comparisons,
        "overall_winner": winner,
        "wins": wins,
    }
    print(json.dumps(output, indent=2))


def output_table(
    path_a: Path, path_b: Path, comparisons: dict, winner: str, wins: dict
):
    """Output results as formatted table."""
    print("\n" + "=" * 80)
    print(" MoE EVALUATION COMPARISON".center(80))
    print("=" * 80)

    print_legend(path_a, path_b)
    print_comparison_table(comparisons)
    print_summary_box(path_a, path_b, wins, winner)
    print_interpretation_guide()


def main():
    args = parse_args()

    path_a = Path(args.result_a)
    path_b = Path(args.result_b)
    validate_paths(path_a, path_b)

    result_a = load_results(path_a)
    result_b = load_results(path_b)

    comparisons = compare_results(result_a, result_b)
    overall_winner, wins = determine_overall_winner(comparisons)

    if args.json:
        output_json(path_a, path_b, comparisons, overall_winner, wins)
    else:
        output_table(path_a, path_b, comparisons, overall_winner, wins)


if __name__ == "__main__":
    main()
