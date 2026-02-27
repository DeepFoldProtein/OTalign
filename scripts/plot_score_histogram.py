"""
Plot score distribution histogram for ECOD30 hard benchmark:
homolog (same H-group) vs analog (different X-group).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")


def main():
    # Load search results
    df = pd.read_csv(RESULTS_DIR / "ecod30_hard_otalign_norm_dp_minlen_global" / "search_results.csv")

    homolog = df[df["label"] == 1]["score"].values
    analog = df[df["label"] == 0]["score"].values

    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(7, 5))

    # Determine common bin range
    lo = min(analog.min(), homolog.min())
    hi = max(analog.max(), homolog.max())
    bins = np.linspace(lo, hi, 120)

    ax.hist(analog, bins=bins, density=True, alpha=0.55, color="#4363d8", label=f"Analog (n={len(analog):,})")
    ax.hist(homolog, bins=bins, density=True, alpha=0.55, color="#e6194b", label=f"Homolog (n={len(homolog):,})")

    ax.set_xlabel("OTalign score (DP / min(L))")
    ax.set_ylabel("Density")
    ax.set_title(r"ECOD30 $\leq$30% seq. id.", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(False)

    plt.tight_layout()
    out_path = RESULTS_DIR / "ecod30_hard_score_histogram.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
