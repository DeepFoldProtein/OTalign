"""
Rebuild Figure S1 (remote homology detection on ECOD30 hard) with HHalign added.

Left panel  : Precision-Recall curves
Right panel : Cumulative TP vs FP on a semi-log x-axis

Colors:
  OTalign (Global)  red    (otalign_norm_dp_minlen_global)
  pLM-BLAST         blue   (plmblast_paper_global)
  EBA               gold   (eba)
  HHalign (Global)  purple (ecod_hhalign_global)
  HHalign (Local)   brown  (ecod_hhalign_local, if present)
  Random            grey
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCES: Dict[str, Dict] = {
    "OTalign (Global)": {
        "dir": "otalign_norm_dp_global",
        "color": "#d62728",  # red
        "lw": 2.2,
    },
    "pLM-BLAST": {
        "dir": "plmblast_paper_global",
        "color": "#1f77b4",  # blue
        "lw": 2.2,
    },
    "EBA": {
        "dir": "eba",
        "color": "#ffbf00",  # amber/yellow
        "lw": 2.2,
    },
    "HHalign (Global)": {
        "dir": "ecod_hhalign_global",
        "color": "#7f3fbf",  # purple
        "lw": 2.0,
    },
    "HHalign (Local)": {
        "dir": "ecod_hhalign_local",
        "color": "#8c564b",  # brown
        "lw": 2.0,
    },
}


def find_results_dir(root: Path, slug: str) -> Path | None:
    """Locate a result directory, transparently handling the .skip rename convention."""
    for cand in (root / slug, root / f"{slug}.skip"):
        if cand.is_dir():
            return cand
    return None


def load_metrics(rd: Path) -> dict | None:
    for name in ("roc_pr_metrics.json", "metrics.json"):
        p = rd / name
        if p.exists():
            return json.loads(p.read_text())
    return None


def load_scores(rd: Path) -> Tuple[np.ndarray, np.ndarray] | None:
    csv = rd / "search_results.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df = df[df["label"] != -1]
    return df["score"].values.astype(np.float64), df["label"].values.astype(np.int8)


def tp_vs_fp_curve(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort by score descending; cumulative FP on x, cumulative TP on y."""
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    cum_tp = np.cumsum(y == 1)
    cum_fp = np.cumsum(y == 0)
    return cum_fp, cum_tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="out/results/ecod30_hard")
    ap.add_argument("--out_dir", default="out/plots/ecod30_hard")
    ap.add_argument("--out_name", default="figure_s1_with_hhalign.png")
    ap.add_argument("--fp_max", type=float, default=1e6)
    args = ap.parse_args()

    root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    n_pos = None
    n_neg = None
    rows = []

    for label, spec in SOURCES.items():
        rd = find_results_dir(root, spec["dir"])
        if rd is None:
            print(f"[skip] {label}: no directory '{spec['dir']}' (or .skip) under {root}")
            continue

        m = load_metrics(rd)
        sl = load_scores(rd)
        if m is None or sl is None:
            print(f"[skip] {label}: missing metrics or search_results.csv in {rd}")
            continue
        scores, labels = sl

        if n_pos is None:
            n_pos = int((labels == 1).sum())
            n_neg = int((labels == 0).sum())

        # ---- PR curve (left) ----
        axL.plot(
            m["recall"],
            m["precision"],
            color=spec["color"],
            lw=spec["lw"],
            label=f"{label} (AP={m['pr_auc']:.3f})",
        )

        # ---- TP-vs-FP (right) ----
        cum_fp, cum_tp = tp_vs_fp_curve(scores, labels)
        axR.plot(cum_fp, cum_tp, color=spec["color"], lw=spec["lw"], label=label)

        rows.append(
            {
                "method": label,
                "dir": rd.name.removesuffix(".skip"),
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "tp_at_fp_1": int(cum_tp[(cum_fp <= 1).sum() - 1] if (cum_fp <= 1).any() else 0),
                "tp_at_fp_10": int(cum_tp[(cum_fp <= 10).sum() - 1] if (cum_fp <= 10).any() else 0),
                "tp_at_fp_100": int(cum_tp[(cum_fp <= 100).sum() - 1] if (cum_fp <= 100).any() else 0),
                "tp_at_fp_1000": int(cum_tp[(cum_fp <= 1000).sum() - 1] if (cum_fp <= 1000).any() else 0),
            }
        )

    # Random baseline (right panel only). Random = positives × (FP / negatives)
    fp_x = np.geomspace(1, args.fp_max, 200)
    axR.plot(fp_x, n_pos * (fp_x / n_neg), color="grey", lw=1.5, ls="--", label="Random")

    # ---- styling ----
    axL.set_xlabel("Recall")
    axL.set_ylabel("Precision")
    axL.set_title(f"Precision-Recall  (n_pos={n_pos:,}, n_neg={n_neg:,})")
    axL.set_xlim(0, 1)
    axL.set_ylim(0, 1)
    axL.set_aspect("equal", adjustable="box")
    axL.grid(True, alpha=0.3)
    axL.legend(loc="best", fontsize=9, framealpha=0.9)

    axR.set_xlabel("False positives (cumulative)")
    axR.set_ylabel("True positives (cumulative)")
    axR.set_title("TP vs FP")
    axR.set_xscale("log")
    axR.set_xlim(1, args.fp_max)
    axR.set_ylim(0, n_pos)
    axR.grid(True, which="both", alpha=0.3)
    axR.legend(loc="lower right", fontsize=9, framealpha=0.9)

    plt.suptitle("ECOD30 hard — remote homology detection (with HHalign)", fontsize=13)
    plt.tight_layout()
    stem = Path(args.out_name).with_suffix("")
    for ext in (".png", ".pdf"):
        out = out_dir / f"{stem}{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)

    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    out_csv = out_dir / f"{stem}_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
