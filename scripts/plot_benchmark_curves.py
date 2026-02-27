"""
Plot PR and ROC (TP vs FP) curves for ECOD30 hard benchmark methods.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")

# Method configs: (dir_name, display_name, color, linestyle)
METHODS = [
    ("ecod30_hard_otalign_norm_dp_minlen_global", "OTalign", "#e6194b", "-"),
    ("ecod30_hard_plmblast_paper_global", "pLM-BLAST", "#4363d8", "-"),
    ("ecod30_hard_eba", "EBA", "#f58231", "-"),
]


def load_metrics(dir_name):
    with open(RESULTS_DIR / dir_name / "metrics.json") as f:
        return json.load(f)


def main():
    plt.rcParams.update({"font.size": 14})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.set_aspect("equal", adjustable="box")
    ax2.set_aspect("auto")
    ax2.set_box_aspect(1)

    for dir_name, label, color, ls in METHODS:
        m = load_metrics(dir_name)

        recall = np.array(m["recall"])
        precision = np.array(m["precision"])
        fpr = np.array(m["fpr"])
        tpr = np.array(m["tpr"])
        n_pos = m["n_positive"]
        n_neg = m["n_negative"]

        # TP vs FP (absolute counts)
        tp = tpr * n_pos
        fp = fpr * n_neg

        # Left: PR curve
        ax1.plot(recall, precision, color=color, linestyle=ls, linewidth=2, label=f"{label}")

        # Right: TP vs FP
        ax2.plot(fp, tp, color=color, linestyle=ls, linewidth=2, label=f"{label}")

    # --- Random baseline ---
    ax1.plot([0.01, 0.01, 1.0], [1.0, 0.012, 0.012], color="gray", linestyle="-", linewidth=2.5, label="Random")

    fp_rand = np.logspace(0, np.log10(n_neg), 500)
    tp_rand = fp_rand * (n_pos / n_neg)
    ax2.plot(fp_rand, tp_rand, color="gray", linestyle="-", linewidth=2, label="Random")

    # --- Left panel: PR curve ---
    ax1.set_xlabel("Recall  TP/(TP+FN)")
    ax1.set_ylabel("Precision  TP/(TP+FP)")
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax1.grid(False)

    # --- Right panel: TP vs FP (log-scale x) ---
    ax2.set_xlabel("FP")
    ax2.set_ylabel("TP")
    ax2.set_xscale("log")
    ax2.set_xlim(1, n_neg)
    ax2.set_ylim(0, n_pos + 100)
    ax2.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax2.grid(False)

    plt.suptitle(r"Homology detection benchmark ($\leq$30% sequence identity)", fontsize=15, fontweight="bold")
    plt.tight_layout(w_pad=0.1)

    out_path = RESULTS_DIR / "ecod30_hard_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out_path}")

    # --- Save metrics summary CSV ---
    rows = []
    for dir_name, label, _, _ in METHODS:
        m = load_metrics(dir_name)
        rows.append(
            {
                "Method": label,
                "ROC-AUC": m["roc_auc"],
                "PR-AUC": m["pr_auc"],
                "n_positive": m["n_positive"],
                "n_negative": m["n_negative"],
                "n_evaluated": m["n_evaluated"],
            }
        )
    csv_path = RESULTS_DIR / "ecod30_hard_comparison.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")

    # --- Save TP at FP thresholds CSV ---
    fp_thresholds = [1, 5, 10, 50, 100, 500, 1000]
    tp_rows = []
    for dir_name, label, _, _ in METHODS:
        m = load_metrics(dir_name)
        fpr = np.array(m["fpr"])
        tpr = np.array(m["tpr"])
        n_pos = m["n_positive"]
        n_neg = m["n_negative"]
        fp = fpr * n_neg
        tp = tpr * n_pos
        row = {"Method": label}
        for ft in fp_thresholds:
            mask = fp <= ft
            max_tp = int(tp[mask].max()) if mask.any() else 0
            row[f"TP@FP<={ft}"] = max_tp
        tp_rows.append(row)
    tp_csv_path = RESULTS_DIR / "ecod30_hard_tp_at_fp.csv"
    pd.DataFrame(tp_rows).to_csv(tp_csv_path, index=False)
    print(f"Saved to {tp_csv_path}")


if __name__ == "__main__":
    main()
