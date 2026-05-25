"""
Overlay ROC and PR curves across all methods that have produced
`out/results/ecod30_hard/<model>/roc_pr_metrics.json`.

Outputs:
  out/plots/ecod30_hard/roc_comparison.png
  out/plots/ecod30_hard/pr_comparison.png
  out/plots/ecod30_hard/auc_summary.csv
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def load_run(metrics_path: Path):
    with open(metrics_path, "r") as f:
        m = json.load(f)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="out/results/ecod30_hard")
    ap.add_argument("--plots_dir", default="out/plots/ecod30_hard")
    ap.add_argument(
        "--config",
        default="configs/benchmark_config.yaml",
        help="Used for human-readable labels and colors",
    )
    args = ap.parse_args()

    results_root = Path(args.results_root)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    label_map = {}
    color_map = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.get("models", {}).items():
            label_map[k] = v.get("label", k)
            color_map[k] = v.get("color", None)

    if not results_root.exists():
        print(f"No results yet at {results_root}")
        return

    runs = []
    for model_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        # Convention: rename a dir to "<name>.skip" to exclude it from the comparison
        # without deleting the underlying files.
        if model_dir.name.endswith(".skip"):
            continue
        # Prefer our evaluator's output name, but accept the standalone runners' name too.
        metrics_path = model_dir / "roc_pr_metrics.json"
        if not metrics_path.exists():
            metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        m = load_run(metrics_path)
        runs.append((model_dir.name, m))

    if not runs:
        print(f"No roc_pr_metrics.json found under {results_root}")
        return

    # ROC overlay (1:1 aspect)
    fig, ax = plt.subplots(figsize=(7, 7))
    for key, m in runs:
        label = label_map.get(key, key)
        color = color_map.get(key)
        ax.plot(
            m["fpr"],
            m["tpr"],
            lw=2,
            color=color,
            label=f"{label} (AUC={m['roc_auc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ECOD30 Hard - ROC Comparison")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        out = plots_dir / f"roc_comparison{ext}"
        fig.savefig(out, dpi=200)
        print(f"Wrote {out}")
    plt.close(fig)

    # PR overlay (1:1 aspect)
    fig, ax = plt.subplots(figsize=(7, 7))
    for key, m in runs:
        label = label_map.get(key, key)
        color = color_map.get(key)
        ax.plot(
            m["recall"],
            m["precision"],
            lw=2,
            color=color,
            label=f"{label} (AP={m['pr_auc']:.3f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("ECOD30 Hard - PR Comparison")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        out = plots_dir / f"pr_comparison{ext}"
        fig.savefig(out, dpi=200)
        print(f"Wrote {out}")
    plt.close(fig)

    # AUC summary table
    rows = []
    for key, m in runs:
        rows.append(
            {
                "model": key,
                "label": label_map.get(key, key),
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "n_positive": m.get("n_positive"),
                "n_negative": m.get("n_negative"),
                "num_pairs": m.get("num_pairs"),
                "mode": m.get("mode"),
            }
        )
    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    out = plots_dir / "auc_summary.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
