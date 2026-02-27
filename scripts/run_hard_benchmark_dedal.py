"""
Run DEDAL All-vs-All benchmark on ECOD "Hard" benchmark dataset.

This script:
1. Loads the Hard benchmark dataset
2. Loads DEDAL model from TF Hub
3. Runs all-vs-all pairwise scoring (sw_scores or homology_logits)
4. Assigns labels based on H-group/X-group criteria
5. Computes ROC/PR curves and metrics

Score methods:
  - sw_scores (default): Smith-Waterman alignment score, higher = more similar
  - homology_logits: trained homology classifier logit, higher = more similar

Note: DEDAL max sequence length is 512 aa. Pairs with either sequence > 512 are skipped.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm


# Add third_party so that "from dedal import ..." works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = PROJECT_ROOT / "third_party"
DEDAL_ROOT = THIRD_PARTY / "dedal"
if not DEDAL_ROOT.exists():
    raise FileNotFoundError(f"DEDAL not found at {DEDAL_ROOT}. Run: git submodule update --init third_party/dedal")
sys.path.insert(0, str(THIRD_PARTY))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCORE_METHODS = ["sw_scores", "homology_logits"]


def load_benchmark_data(data_dir: Path) -> pd.DataFrame:
    """Load hard benchmark dataset."""
    csv_path = data_dir / "hard_benchmark.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"H": str, "X": str, "T": str})
    logging.info(f"Loaded {len(df)} domains from {csv_path}")
    return df


def load_dedal_model(model_url: str, use_gpu: bool):
    """Load DEDAL model from TF Hub."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    if use_gpu:
        os.environ["DEDAL_USE_GPU"] = "1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import tensorflow_hub as hub

    device = "GPU" if use_gpu else "CPU"
    logging.info(f"Loading DEDAL model on {device}...")
    model = hub.load(model_url)
    logging.info("DEDAL model loaded")
    return model


def score_pairs_dedal(
    model,
    df: pd.DataFrame,
    score_method: str = "sw_scores",
    max_length: int = 512,
) -> List[Tuple[str, str, float]]:
    """
    Run all-vs-all DEDAL scoring.

    Directly extracts scores from model output (skips expensive Alignment construction).
    Returns list of (query_id, hit_id, score) tuples.
    """
    from dedal import infer

    domain_ids = df["id"].tolist()
    sequences = df["sequence"].tolist()
    seq_map = dict(zip(domain_ids, sequences))

    n = len(domain_ids)

    # Filter domains exceeding max_length
    valid_ids = [did for did in domain_ids if len(seq_map[did]) <= max_length]
    skipped = n - len(valid_ids)
    if skipped > 0:
        logging.warning(f"Skipped {skipped} domains with length > {max_length}")

    n_valid = len(valid_ids)
    total_valid_pairs = n_valid * (n_valid - 1) // 2
    logging.info(f"Running {total_valid_pairs:,} pairwise DEDAL searches ({score_method}, {n_valid} valid domains)...")

    results = []
    with tqdm(total=total_valid_pairs, desc="Scoring pairs") as pbar:
        for i in range(n_valid):
            seq_i = seq_map[valid_ids[i]]
            for j in range(i + 1, n_valid):
                seq_j = seq_map[valid_ids[j]]

                inputs = infer.preprocess(seq_i, seq_j, max_length)
                output = model(inputs)

                if isinstance(output, dict) and score_method in output:
                    score = float(output[score_method].numpy().squeeze())
                else:
                    # Fallback: use expand -> postprocess for sw_scores
                    output_tuple = infer.expand(output)
                    score = float(output_tuple[0].numpy().squeeze())

                results.append((valid_ids[i], valid_ids[j], score))
                pbar.update(1)

    return results


def assign_labels(results: List[Tuple[str, str, float]], df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign labels based on H-group/X-group criteria.

    Labels:
    - 1: True Positive (same H-group)
    - 0: False Positive (different X-group)
    - -1: Neutral (same X-group, different H-group) - excluded from evaluation
    """
    id_to_h = dict(zip(df["id"], df["H"]))
    id_to_x = dict(zip(df["id"], df["X"]))

    labeled_results = []
    for query_id, hit_id, score in results:
        query_h = id_to_h.get(query_id, "")
        hit_h = id_to_h.get(hit_id, "")
        query_x = id_to_x.get(query_id, "")
        hit_x = id_to_x.get(hit_id, "")

        if query_h == hit_h:
            label = 1
        elif query_x != hit_x:
            label = 0
        else:
            label = -1

        labeled_results.append({"query_id": query_id, "hit_id": hit_id, "score": score, "label": label, "query_h": query_h, "hit_h": hit_h, "query_x": query_x, "hit_x": hit_x})

    result_df = pd.DataFrame(labeled_results)

    tp_count = (result_df["label"] == 1).sum()
    fp_count = (result_df["label"] == 0).sum()
    neutral_count = (result_df["label"] == -1).sum()

    logging.info("Label distribution:")
    logging.info(f"  - True Positives (same H): {tp_count:,} ({100 * tp_count / len(result_df):.2f}%)")
    logging.info(f"  - False Positives (diff X): {fp_count:,} ({100 * fp_count / len(result_df):.2f}%)")
    logging.info(f"  - Neutral (same X, diff H): {neutral_count:,} ({100 * neutral_count / len(result_df):.2f}%)")

    return result_df


def compute_metrics(result_df: pd.DataFrame, exclude_neutral: bool = True) -> Dict:
    """Compute ROC-AUC and PR-AUC metrics."""
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    if exclude_neutral:
        eval_df = result_df[result_df["label"] != -1].copy()
    else:
        eval_df = result_df.copy()

    scores = eval_df["score"].values
    labels = eval_df["label"].values

    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)

    logging.info(f"Metrics (exclude_neutral={exclude_neutral}):")
    logging.info(f"  - ROC-AUC: {roc_auc:.4f}")
    logging.info(f"  - PR-AUC: {pr_auc:.4f}")

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "n_evaluated": len(eval_df),
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Run DEDAL All-vs-All benchmark on ECOD Hard dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing hard_benchmark.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--score_method", type=str, default="sw_scores", choices=SCORE_METHODS, help="Score method: sw_scores (default), homology_logits")
    parser.add_argument("--model_url", type=str, default="https://tfhub.dev/google/dedal/3", help="TF Hub DEDAL model URL")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length (DEDAL supports up to 512)")
    parser.add_argument("--use_gpu", action="store_true", default=False, help="Use GPU (default: CPU to avoid PTX issues)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_benchmark_data(data_dir)

    # Load model
    model = load_dedal_model(args.model_url, use_gpu=args.use_gpu)

    # Run scoring
    start_time = time.time()
    results = score_pairs_dedal(
        model,
        df,
        score_method=args.score_method,
        max_length=args.max_length,
    )
    search_time = time.time() - start_time
    logging.info(f"Scoring completed in {search_time:.2f} seconds")

    # Assign labels
    result_df = assign_labels(results, df)

    # Save results
    result_df.to_csv(output_dir / "search_results.csv", index=False)
    logging.info(f"Saved results to {output_dir / 'search_results.csv'}")

    # Compute metrics
    metrics = compute_metrics(result_df, exclude_neutral=True)
    metrics["search_time_seconds"] = search_time
    metrics["score_method"] = args.score_method
    metrics["model_url"] = args.model_url
    metrics["max_length"] = args.max_length
    metrics["num_domains"] = len(df)
    metrics["num_valid_domains"] = len({r[0] for r in results} | {r[1] for r in results})
    metrics["num_pairs"] = len(results)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Saved metrics to {output_dir / 'metrics.json'}")

    # Generate plots
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(metrics["fpr"], metrics["tpr"], "b-", linewidth=2)
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title(f"ROC Curve (AUC = {metrics['roc_auc']:.4f})")
        ax1.grid(True, alpha=0.3)

        ax2.plot(metrics["recall"], metrics["precision"], "r-", linewidth=2)
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title(f"PR Curve (AUC = {metrics['pr_auc']:.4f})")
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f"DEDAL ({args.score_method})")
        plt.tight_layout()
        plt.savefig(output_dir / "curves.png", dpi=150)
        plt.close()
        logging.info(f"Saved plots to {output_dir / 'curves.png'}")

    except ImportError:
        logging.warning("matplotlib not available, skipping plot generation")

    logging.info("Benchmark complete!")
    logging.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    logging.info(f"  PR-AUC: {metrics['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
