"""
Reviewer 2 (Q3) response: HHalign local vs global on alignment-quality benchmarks.

For each dataset (MALIDUP, MALISAM, SABMARK_sup, SABMARK_twi):
  * Unconditional F1 (HHalign output as-is; pred_size=0 counts as F1=0)
  * Conditional F1 (only pairs where HHalign produced a non-empty alignment)
  * Rejection rate (% of pairs with pred_size=0)

This directly addresses the reviewer's two hypotheses:
  (1) F1=0 inflation from rejected / empty alignments
  (2) Mode mismatch: HHalign default is local but the metric expects global coverage
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ["malidup", "malisam", "sabmark_sup", "sabmark_twi"]
NICE = {
    "malidup": "MALIDUP",
    "malisam": "MALISAM",
    "sabmark_sup": "SABMARK\n(superfamily)",
    "sabmark_twi": "SABMARK\n(twilight)",
}
METHODS = [
    ("HHalign (global)", "hhalign", "#666666"),
    ("HHalign (local)", "hhalign_local", "#bdbdbd"),
    ("OTalign (Ankh-Large)", "otalign_ankh_large", "#d62728"),
    ("pLM-BLAST (ProtT5)", "plmblast_prott5", "#1f77b4"),
    ("EBA (ProtT5)", "eba_prott5", "#ffbf00"),
]


def load_jsonl(p: Path):
    rows = []
    if not p.exists():
        return rows
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            rows.append(r.get("metrics", {}))
    return rows


def summarise(rows):
    n = len(rows)
    if n == 0:
        return None
    f1 = np.array([r.get("f1", np.nan) for r in rows], dtype=float)
    pred0 = np.array([r.get("pred_size", 0) == 0 for r in rows])
    nan_mask = np.isnan(f1)
    f1_zeroed = np.where(nan_mask, 0.0, f1)  # NaN -> 0 (unconditional)
    f1_cond = f1[(~pred0) & (~nan_mask)]  # only pairs with non-empty pred
    return {
        "n": n,
        "rejection_rate": pred0.mean(),
        "f1_uncond_mean": float(f1_zeroed.mean()),
        "f1_uncond_median": float(np.median(f1_zeroed)),
        "f1_cond_mean": float(f1_cond.mean()) if len(f1_cond) else float("nan"),
        "f1_cond_median": float(np.median(f1_cond)) if len(f1_cond) else float("nan"),
        "n_cond": int(len(f1_cond)),
        "f1_uncond_array": f1_zeroed,
        "f1_cond_array": f1_cond,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="out/results")
    ap.add_argument("--out_dir", default="out/plots/hhalign_local_vs_global")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect summaries
    rows = []
    cache = {}  # (dataset, method_dir) -> summary
    for ds in DATASETS:
        for label, mdir, color in METHODS:
            p = Path(args.results_root) / ds / mdir / "results.jsonl"
            s = summarise(load_jsonl(p))
            cache[(ds, mdir)] = s
            if s is None:
                continue
            rows.append(
                {
                    "dataset": ds,
                    "method": label,
                    "n": s["n"],
                    "rejection_rate_pct": 100 * s["rejection_rate"],
                    "f1_uncond_mean": s["f1_uncond_mean"],
                    "f1_uncond_median": s["f1_uncond_median"],
                    "f1_cond_mean": s["f1_cond_mean"],
                    "f1_cond_median": s["f1_cond_median"],
                    "n_cond": s["n_cond"],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    print(df.to_string(index=False))
    print()

    # ---- Figure: two-row grouped bar chart ----
    # Row 1: unconditional vs conditional mean F1 per method per dataset
    # Row 2: rejection rate per method per dataset

    fig, axes = plt.subplots(2, len(DATASETS), figsize=(4.0 * len(DATASETS), 8), sharey="row")

    for j, ds in enumerate(DATASETS):
        # ---- Top row: mean F1 (uncond solid bar, cond hatched bar) ----
        ax = axes[0, j]
        present = [m for m in METHODS if cache.get((ds, m[1])) is not None]
        x = np.arange(len(present))
        u = [cache[(ds, m[1])]["f1_uncond_mean"] for m in present]
        c = [cache[(ds, m[1])]["f1_cond_mean"] for m in present]
        colors = [m[2] for m in present]
        w = 0.4
        ax.bar(x - w / 2, u, w, color=colors, label="Unconditional", edgecolor="black", linewidth=0.6)
        ax.bar(x + w / 2, c, w, color=colors, alpha=0.55, label="Conditional", edgecolor="black", linewidth=0.6, hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in present], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(NICE[ds], fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        if j == 0:
            ax.set_ylabel("Mean F1")
            ax.legend(loc="upper left", fontsize=8)

        # ---- Bottom row: rejection rate ----
        ax = axes[1, j]
        rj = [100 * cache[(ds, m[1])]["rejection_rate"] for m in present]
        ax.bar(x, rj, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in present], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis="y")
        if j == 0:
            ax.set_ylabel("Rejection rate (%)")

    plt.suptitle(
        "HHalign local vs. global on alignment-quality benchmarks\n(unconditional F1 counts rejected/empty alignments as 0; conditional excludes them)",
        fontsize=12,
    )
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        out = out_dir / f"hhalign_mode_comparison{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
