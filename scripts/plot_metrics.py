import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def load_metrics_from_jsonl(file_path: Path) -> List[Dict[str, float]]:
    """Loads alignment metrics from a JSONL file."""
    metrics_list = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "metrics" in data and isinstance(data["metrics"], dict):
                    metrics_list.append(data["metrics"])
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from line in {file_path}")
    return metrics_list


def calculate_stats(data: List[float]) -> Dict[str, float]:
    """Calculates mean, standard error, and 95% confidence interval."""
    if not data:
        return {"mean": 0, "se": 0, "ci_lower": 0, "ci_upper": 0}

    arr = np.array(data)
    mean = np.mean(arr)
    se = stats.sem(arr)

    # 95% confidence interval
    ci = stats.t.interval(0.95, len(arr) - 1, loc=mean, scale=se)

    return {
        "mean": float(mean),
        "se": float(se),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
    }


def main():
    """Main function to parse files, calculate stats, and plot results."""
    parser = argparse.ArgumentParser(description="Analyze and plot alignment metrics from JSONL result files.")
    parser.add_argument("--jsonl_files", type=str, nargs="+", required=True, help="Paths to one or more JSONL result files.")
    parser.add_argument("--labels", type=str, nargs="+", required=True, help="Labels for each JSONL file, for use in plot legends.")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save the output plots.")
    parser.add_argument("--plot_type", type=str, default="boxplot", choices=["boxplot", "barplot"], help="Type of plot to generate.")
    parser.add_argument("--error_bar", type=str, default="ci", choices=["ci", "sd"], help="Error bar type for barplot: 'ci' (95%% confidence interval) or 'sd' (standard deviation).")
    parser.add_argument("--metric_names", type=str, nargs="+", default=["f1", "precision", "recall", "jaccard"], help="Names of the metrics to plot.")
    args = parser.parse_args()

    if len(args.jsonl_files) != len(args.labels):
        raise ValueError("The number of --jsonl_files must match the number of --labels.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = []

    for file_path_str, label in zip(args.jsonl_files, args.labels):
        file_path = Path(file_path_str)
        if not file_path.exists():
            print(f"Warning: File not found at {file_path}, skipping.")
            continue

        metrics_data = load_metrics_from_jsonl(file_path)
        if not metrics_data:
            print(f"Warning: No metrics found in {file_path}, skipping.")
            continue

        for metric_name in args.metric_names:
            values = [m.get(metric_name, 0) for m in metrics_data]
            for value in values:
                all_data.append({"label": label, "metric": metric_name, "value": value})

    df = pd.DataFrame(all_data)
    df["metric"] = df["metric"].str.capitalize()

    # --- Plotting ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))

    if args.plot_type == "boxplot":
        plot = sns.boxplot(data=df, x="metric", y="value", hue="label")
        plot_suffix = "boxplot"
    elif args.plot_type == "barplot":
        errorbar_param = ("ci", 95) if args.error_bar == "ci" else "sd"
        plot = sns.barplot(data=df, x="metric", y="value", hue="label", capsize=0.1, errorbar=errorbar_param)
        plot_suffix = f"barplot_{args.error_bar}"
    else:
        raise ValueError(f"Unknown plot type: {args.plot_type}")

    plot.set_title("Alignment Metrics Comparison", fontsize=16)
    plot.set_xlabel("Metric", fontsize=12)
    plot.set_ylabel("Value", fontsize=12)
    plot.set_ylim(0, 1.05)
    plot.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, title="Experiment")

    plt.tight_layout()

    output_plot_path = output_dir / f"alignment_metrics_{plot_suffix}.svg"
    plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
    print(f"[ok] Plot saved to {output_plot_path}")

    # --- Print and Save Summary Statistics ---
    summary = df.groupby(["label", "metric"])["value"].agg(["mean", "sem", "std"]).reset_index()
    summary["ci_95_lower"] = summary["mean"] - 1.96 * summary["sem"]
    summary["ci_95_upper"] = summary["mean"] + 1.96 * summary["sem"]

    print("\n--- Summary Statistics ---")
    for label, group in summary.groupby("label"):
        print(f"\nLabel: {label}")
        print(group[["metric", "mean", "std", "sem", "ci_95_lower", "ci_95_upper"]].to_string(index=False))

    output_summary_path = output_dir / "alignment_metrics_summary.csv"
    summary.to_csv(output_summary_path, index=False)
    print(f"\n[ok] Summary statistics saved to {output_summary_path}")


if __name__ == "__main__":
    main()
