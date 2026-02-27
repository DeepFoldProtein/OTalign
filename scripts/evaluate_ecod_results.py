"""
Compute ROC and PR curves for homolog detection benchmark.

This module provides utilities to compute ROC (Receiver Operating Characteristic)
and PR (Precision-Recall) curves from search results with binary labels.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve


@dataclass
class ROCPRMetrics:
    """Container for ROC and PR curve metrics."""

    # ROC curve
    fpr: np.ndarray  # False Positive Rate
    tpr: np.ndarray  # True Positive Rate (Recall)
    roc_thresholds: np.ndarray
    roc_auc: float  # Area Under ROC Curve

    # PR curve
    precision: np.ndarray
    recall: np.ndarray
    pr_thresholds: np.ndarray
    pr_auc: float  # Area Under PR Curve (Average Precision)

    # Additional metrics
    num_positives: int
    num_negatives: int
    total_predictions: int


def compute_roc_pr_curves(
    scores: np.ndarray,
    labels: np.ndarray,
) -> ROCPRMetrics:
    """
    Compute ROC and PR curves from scores and binary labels.

    Args:
        scores: Array of prediction scores (higher = more likely positive)
        labels: Binary labels (1 = positive, 0 = negative)

    Returns:
        ROCPRMetrics object with computed curves and AUC values
    """
    if len(scores) != len(labels):
        raise ValueError(f"Score and label arrays must have same length: {len(scores)} vs {len(labels)}")

    if len(scores) == 0:
        raise ValueError("Cannot compute curves with empty input")

    # Compute ROC curve
    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)

    # Compute PR curve
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    # Statistics
    num_positives = int(labels.sum())
    num_negatives = int(len(labels) - num_positives)

    return ROCPRMetrics(
        fpr=fpr,
        tpr=tpr,
        roc_thresholds=roc_thresholds,
        roc_auc=roc_auc,
        precision=precision,
        recall=recall,
        pr_thresholds=pr_thresholds,
        pr_auc=pr_auc,
        num_positives=num_positives,
        num_negatives=num_negatives,
        total_predictions=len(labels),
    )


def load_search_results(results_file: Path, exclude_neutral: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load search results from JSONL file and extract scores/labels.

    Args:
        results_file: Path to results JSONL file from run_ecod_plmblast_search.py
        exclude_neutral: If True, skip hits with label == -1 (Neutral; same X, diff H)
            so ROC/PR use only TP vs FP (paper-style evaluation).

    Returns:
        (scores, labels, query_ids) tuple
    """
    all_scores = []
    all_labels = []
    all_query_ids = []

    with open(results_file, "r") as f:
        for line in f:
            record = json.loads(line)
            query_id = record["query_id"]

            for hit in record["hits"]:
                label = hit["label"]
                if exclude_neutral and label == -1:
                    continue
                all_scores.append(hit["score"])
                all_labels.append(label)
                all_query_ids.append(query_id)

    return np.array(all_scores), np.array(all_labels), all_query_ids


def compute_per_query_metrics(results_file: Path, output_file: Optional[Path] = None) -> pd.DataFrame:
    """
    Compute per-query ROC AUC and PR AUC.

    Args:
        results_file: Path to results JSONL
        output_file: Optional path to save per-query metrics CSV

    Returns:
        DataFrame with per-query metrics
    """
    per_query_metrics = []

    with open(results_file, "r") as f:
        for line in f:
            record = json.loads(line)
            query_id = record["query_id"]

            if len(record["hits"]) == 0:
                continue

            # Exclude neutral (label==-1) for ROC/PR
            hits_eval = [(h["score"], h["label"]) for h in record["hits"] if h["label"] != -1]
            if not hits_eval:
                continue
            scores = np.array([x[0] for x in hits_eval])
            labels = np.array([x[1] for x in hits_eval])

            # Need at least one positive and one negative for meaningful AUC
            if labels.sum() == 0 or labels.sum() == len(labels):
                per_query_metrics.append({"query_id": query_id, "num_hits": len(labels), "num_positives": int(labels.sum()), "roc_auc": np.nan, "pr_auc": np.nan})
                continue

            try:
                metrics = compute_roc_pr_curves(scores, labels)
                per_query_metrics.append(
                    {
                        "query_id": query_id,
                        "num_hits": len(labels),
                        "num_positives": metrics.num_positives,
                        "num_negatives": metrics.num_negatives,
                        "roc_auc": metrics.roc_auc,
                        "pr_auc": metrics.pr_auc,
                    }
                )
            except Exception as e:
                print(f"Warning: Failed to compute metrics for {query_id}: {e}")
                per_query_metrics.append({"query_id": query_id, "num_hits": len(labels), "num_positives": int(labels.sum()), "roc_auc": np.nan, "pr_auc": np.nan})

    df = pd.DataFrame(per_query_metrics)

    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Per-query metrics saved to {output_file}")

    return df


def plot_roc_pr_curves(metrics: ROCPRMetrics, output_dir: Path, title_prefix: str = "ECOD Homolog Detection"):
    """
    Plot ROC and PR curves.

    Args:
        metrics: ROCPRMetrics object
        output_dir: Directory to save plots
        title_prefix: Prefix for plot titles
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # ROC Curve
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(metrics.fpr, metrics.tpr, "b-", linewidth=2, label=f"ROC (AUC = {metrics.roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=14)
    ax.set_title(f"{title_prefix} - ROC Curve", fontsize=16)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)

    roc_file = output_dir / "roc_curve.png"
    plt.savefig(roc_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved to {roc_file}")

    # PR Curve
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(metrics.recall, metrics.precision, "b-", linewidth=2, label=f"PR (AUC = {metrics.pr_auc:.3f})")

    # Baseline (random classifier)
    baseline = metrics.num_positives / metrics.total_predictions
    ax.plot([0, 1], [baseline, baseline], "r--", linewidth=1, label=f"Random (P={baseline:.3f})")

    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.set_title(f"{title_prefix} - Precision-Recall Curve", fontsize=16)
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    pr_file = output_dir / "pr_curve.png"
    plt.savefig(pr_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"PR curve saved to {pr_file}")


def plot_score_distribution(scores: np.ndarray, labels: np.ndarray, output_file: Path, title: str = "Score Distribution"):
    """
    Plot score distribution for positives vs negatives.

    Args:
        scores: Array of scores
        labels: Binary labels
        output_file: Output file path
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    ax.hist(neg_scores, bins=50, alpha=0.5, label=f"Negatives (n={len(neg_scores)})", color="red")
    ax.hist(pos_scores, bins=50, alpha=0.5, label=f"Positives (n={len(pos_scores)})", color="green")

    ax.set_xlabel("Score", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Score distribution saved to {output_file}")


def evaluate_search_results(results_file: Path, output_dir: Path, title_prefix: str = "ECOD Homolog Detection"):
    """
    Complete evaluation: compute metrics and generate plots.

    Args:
        results_file: Path to search results JSONL
        output_dir: Output directory for plots and metrics
        title_prefix: Prefix for plot titles
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_file}")
    scores, labels, query_ids = load_search_results(results_file)

    print(f"Loaded {len(scores)} hits from {len(set(query_ids))} queries")
    print(f"  - Positives: {labels.sum()} ({100 * labels.sum() / len(labels):.1f}%)")
    print(f"  - Negatives: {len(labels) - labels.sum()} ({100 * (1 - labels.sum() / len(labels)):.1f}%)")

    # Compute overall metrics
    print("\nComputing ROC/PR curves...")
    metrics = compute_roc_pr_curves(scores, labels)

    print(f"  - ROC AUC: {metrics.roc_auc:.4f}")
    print(f"  - PR AUC: {metrics.pr_auc:.4f}")

    # Save metrics
    metrics_dict = {
        "roc_auc": float(metrics.roc_auc),
        "pr_auc": float(metrics.pr_auc),
        "num_positives": int(metrics.num_positives),
        "num_negatives": int(metrics.num_negatives),
        "total_predictions": int(metrics.total_predictions),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # Plot curves
    print("\nGenerating plots...")
    plot_roc_pr_curves(metrics, output_dir, title_prefix)
    plot_score_distribution(scores, labels, output_dir / "score_distribution.png", f"{title_prefix} - Score Distribution")

    # Per-query metrics
    print("\nComputing per-query metrics...")
    per_query_df = compute_per_query_metrics(results_file, output_dir / "per_query_metrics.csv")

    # Print summary
    valid_queries = per_query_df.dropna(subset=["roc_auc"])
    if len(valid_queries) > 0:
        print(f"\nPer-query statistics ({len(valid_queries)} queries with valid AUC):")
        print(f"  - Mean ROC AUC: {valid_queries['roc_auc'].mean():.4f} ± {valid_queries['roc_auc'].std():.4f}")
        print(f"  - Mean PR AUC: {valid_queries['pr_auc'].mean():.4f} ± {valid_queries['pr_auc'].std():.4f}")

    print(f"\nEvaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ECOD homolog detection results")
    parser.add_argument("--results", type=str, required=True, help="Path to search results JSONL")
    parser.add_argument("--output", type=str, required=True, help="Output directory for plots")
    parser.add_argument("--title", type=str, default="ECOD Homolog Detection", help="Plot title prefix")

    args = parser.parse_args()

    evaluate_search_results(Path(args.results), Path(args.output), args.title)
