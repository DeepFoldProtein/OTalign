"""
Reviewer 2 (Q2) response: Show that OTalign scores systematically distinguish
homologous (ECOD30 same-H-group) from analogous (MALISAM) pairs.

Loads OTalign norm_dp_minlen scores from:
  - ECOD30-hard (TP: same H-group, FP: different X-group)  [pre-computed]
  - MALISAM (analogous pairs)  [computed as score/min(len1,len2) from results.jsonl]

All three sets use the SAME OTalign config (Ankh_Large, global DP, reg=0.1).

Outputs:
  out/plots/homology_vs_analogy/score_distributions.png   (violin + histogram)
  out/plots/homology_vs_analogy/score_summary.csv
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_malisam_norm_dp_minlen(jsonl: Path) -> pd.DataFrame:
    rows = []
    with open(jsonl) as f:
        for line in f:
            r = json.loads(line)
            if "score" not in r or "len1" not in r:
                continue
            rows.append(
                {
                    "pair_id": r["pair_id"],
                    "raw_dp": r["score"],
                    "len1": r["len1"],
                    "len2": r["len2"],
                    "score": r["score"] / min(r["len1"], r["len2"]),
                }
            )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecod_csv", default="out/results/ecod30_hard/otalign_norm_dp_minlen_global.skip/search_results.csv")
    ap.add_argument("--malisam_jsonl", default="out/results/malisam/otalign_ankh_large/results.jsonl")
    ap.add_argument("--out_dir", default="out/plots/homology_vs_analogy")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load ECOD30 TP/FP
    edf = pd.read_csv(args.ecod_csv)
    tp = edf[edf["label"] == 1]["score"].values
    fp = edf[edf["label"] == 0]["score"].values

    # Load MALISAM (analogous), normalize to match
    mdf = load_malisam_norm_dp_minlen(Path(args.malisam_jsonl))
    analog = mdf["score"].values

    # Summary table
    summary = pd.DataFrame(
        [
            {
                "group": "ECOD30 TP (same H, homolog)",
                "n": len(tp),
                "median": np.median(tp),
                "p10": np.quantile(tp, 0.10),
                "p25": np.quantile(tp, 0.25),
                "p75": np.quantile(tp, 0.75),
                "p90": np.quantile(tp, 0.90),
            },
            {
                "group": "MALISAM (analogous)",
                "n": len(analog),
                "median": np.median(analog),
                "p10": np.quantile(analog, 0.10),
                "p25": np.quantile(analog, 0.25),
                "p75": np.quantile(analog, 0.75),
                "p90": np.quantile(analog, 0.90),
            },
            {
                "group": "ECOD30 FP (diff X, unrelated)",
                "n": len(fp),
                "median": np.median(fp),
                "p10": np.quantile(fp, 0.10),
                "p25": np.quantile(fp, 0.25),
                "p75": np.quantile(fp, 0.75),
                "p90": np.quantile(fp, 0.90),
            },
        ]
    )
    summary["median_rank_vs_TP_pct"] = [100, 100 * (tp <= np.median(analog)).mean(), 100 * (tp <= np.median(fp)).mean()]
    summary["median_rank_vs_FP_pct"] = [100 * (fp <= np.median(tp)).mean(), 100 * (fp <= np.median(analog)).mean(), 100]
    summary.to_csv(out_dir / "score_summary.csv", index=False)
    print("Summary table:")
    print(summary.to_string(index=False))

    # Figure: side-by-side violin + horizontal range strip
    fig, (ax_v, ax_h) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.4]})

    # ---- Violin (square plot, 3 categories) ----
    data = [tp, analog, fp]
    labels = [
        f"ECOD30 TP\n(homolog)\nn={len(tp):,}",
        f"MALISAM\n(analogous)\nn={len(analog)}",
        f"ECOD30 FP\n(unrelated)\nn={len(fp):,}",
    ]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]

    parts = ax_v.violinplot(data, showmeans=False, showmedians=True, widths=0.85)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.6)
        body.set_edgecolor("black")
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)
    ax_v.set_xticks([1, 2, 3])
    ax_v.set_xticklabels(labels, fontsize=10)
    ax_v.set_ylabel("OTalign score", fontsize=11)
    ax_v.set_title("Score distribution by pair type", fontsize=12)
    ax_v.grid(True, alpha=0.3, axis="y")
    ax_v.axhline(0, color="grey", lw=0.6, ls="--")

    # Annotate medians next to each violin
    medians = [np.median(d) for d in data]
    for i, med in enumerate(medians):
        ax_v.annotate(
            f"median\n{med:.3f}",
            xy=(i + 1, med),
            xytext=(12, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=colors[i],
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": colors[i], "alpha": 0.85},
        )

    # ---- Right panel: histogram overlay ----
    bins = np.linspace(min(fp.min(), analog.min(), tp.min()), max(fp.max(), analog.max(), tp.max()), 80)
    ax_h.hist(fp, bins=bins, density=True, alpha=0.55, color=colors[2], label=f"ECOD30 FP\n(n={len(fp):,})", edgecolor="none")
    ax_h.hist(analog, bins=bins, density=True, alpha=0.85, color=colors[1], label=f"MALISAM analog\n(n={len(analog)})", edgecolor="black", linewidth=0.5)
    ax_h.hist(tp, bins=bins, density=True, alpha=0.55, color=colors[0], label=f"ECOD30 TP\n(n={len(tp):,})", edgecolor="none")
    for q, c in zip([np.median(fp), np.median(analog), np.median(tp)], colors):
        ax_h.axvline(q, color=c, lw=2, ls="--", alpha=0.9)
    ax_h.set_xlabel("OTalign score", fontsize=11)
    ax_h.set_ylabel("Density", fontsize=11)
    ax_h.set_title("Density overlay (medians shown dashed)", fontsize=12)
    ax_h.legend(loc="upper left", fontsize=9)
    ax_h.grid(True, alpha=0.3)

    plt.suptitle(
        "OTalign discriminates homology from analogy\n"
        f"MALISAM analog median sits at the {100 * (tp <= np.median(analog)).mean():.0f}th percentile of ECOD30 homologs "
        f"and the {100 * (fp <= np.median(analog)).mean():.0f}th percentile of ECOD30 non-homologs",
        fontsize=12,
    )
    plt.tight_layout()
    for ext in (".png", ".pdf"):
        out = out_dir / f"score_distributions{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
