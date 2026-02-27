"""
Run EBA All-vs-All benchmark on ECOD "Hard" benchmark dataset.

This script:
1. Loads the Hard benchmark dataset
2. Computes PLM embeddings for all domains (ProtT5 direct, avoids esm dependency)
3. Runs all-vs-all EBA scoring (similarity matrix + DTW alignment)
4. Assigns labels based on H-group/X-group criteria
5. Computes ROC/PR curves and metrics

Score methods:
  - EBA_min (default): EBA_raw / max(len1, len2), higher = more similar
  - EBA_max: EBA_raw / min(len1, len2), higher = more similar
  - EBA_raw: unnormalized DTW alignment score, higher = more similar
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# Add EBA to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EBA_ROOT = PROJECT_ROOT / "third_party" / "eba"
if not EBA_ROOT.exists():
    raise FileNotFoundError(f"EBA not found at {EBA_ROOT}. Run: git submodule update --init third_party/eba")
sys.path.insert(0, str(EBA_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCORE_METHODS = ["EBA_min", "EBA_max", "EBA_raw"]


def load_benchmark_data(data_dir: Path) -> pd.DataFrame:
    """Load hard benchmark dataset."""
    csv_path = data_dir / "hard_benchmark.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"H": str, "X": str, "T": str})
    logging.info(f"Loaded {len(df)} domains from {csv_path}")
    return df


def _load_prott5(device: torch.device):
    """Load ProtT5 model and tokenizer."""
    from transformers import T5EncoderModel, T5Tokenizer
    from transformers import logging as tf_logging

    tf_logging.set_verbosity_error()
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    model = model.to(device)
    model.eval()
    return model, tokenizer


def _extract_prott5(seq: str, model, tokenizer, device: torch.device) -> torch.Tensor:
    """Extract per-residue ProtT5 embeddings for a single sequence."""
    import re

    seq_spaced = " ".join(list(seq))
    seq_spaced = re.sub(r"[UZOB]", "X", seq_spaced)
    ids = tokenizer.batch_encode_plus([seq_spaced], add_special_tokens=True)
    input_ids = torch.tensor(ids["input_ids"]).to(device)
    attention_mask = torch.tensor(ids["attention_mask"]).to(device)

    with torch.no_grad():
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)[0]

    # Remove special tokens: [1:len(seq)+1]
    return embedding[0][1 : len(seq) + 1]


def compute_embeddings(
    df: pd.DataFrame,
    output_dir: Path,
    plm: str = "ProtT5",
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """
    Compute PLM embeddings for all domains.

    Returns dict mapping domain_id -> numpy array (shape: [seq_len, D]).
    """
    embeddings_cache = output_dir / f"embeddings_eba_{plm}.pt"

    if embeddings_cache.exists():
        logging.info(f"Loading cached embeddings from {embeddings_cache}")
        data = torch.load(embeddings_cache, map_location="cpu", weights_only=False)
        # Convert to numpy for efficient multiprocessing (fork COW)
        return {k: v.numpy() for k, v in data.items()}

    logging.info(f"Computing embeddings with {plm}...")

    torch_device = torch.device(device)

    if plm == "ProtT5":
        model, tokenizer = _load_prott5(torch_device)
    else:
        raise ValueError(f"PLM '{plm}' not supported without esm. Install fair-esm or use ProtT5.")

    embeddings = {}
    ids = df["id"].tolist()
    sequences = df["sequence"].tolist()

    for seq_id, seq in tqdm(zip(ids, sequences), total=len(ids), desc="Computing embeddings"):
        emb = _extract_prott5(seq.strip().upper(), model, tokenizer, torch_device)
        embeddings[seq_id] = emb.cpu()

    # Clean up
    del model, tokenizer
    torch.cuda.empty_cache()

    # Cache
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, embeddings_cache)
    logging.info(f"Cached embeddings to {embeddings_cache}")

    # Convert to numpy
    return {k: v.numpy() for k, v in embeddings.items()}


def eba_score_numpy(
    emb1: np.ndarray,
    emb2: np.ndarray,
    score_method: str = "EBA_min",
    l_sim: float = 1.0,
) -> float:
    """
    Compute EBA alignment score - pure numpy (no torch overhead).

    Pipeline: cdist -> exp -> Z-score -> DTW (Numba)
    """
    from eba import alignments as alg
    from scipy.spatial.distance import cdist as scipy_cdist

    # Similarity matrix (pure numpy, no torch)
    dist = scipy_cdist(emb1, emb2, metric="euclidean")
    sm = np.exp(-l_sim * dist)

    # Z-score normalization (row + column)
    row_avg = sm.mean(axis=1, keepdims=True)
    col_avg = sm.mean(axis=0, keepdims=True)
    row_std = sm.std(axis=1, keepdims=True)
    col_std = sm.std(axis=0, keepdims=True)

    row_std = np.where(row_std < 1e-12, 1.0, row_std)
    col_std = np.where(col_std < 1e-12, 1.0, col_std)

    z_rows = (sm - row_avg) / row_std
    z_cols = (sm - col_avg) / col_std
    sm_zscore = (z_rows + z_cols) / 2

    # DTW alignment (Numba JIT)
    _, _, eba_raw = alg.dtw_align(
        sm_zscore.astype(np.float64),
        gap_open_penalty=0.0,
        gap_extend_penalty=0.0,
    )

    l_min = min(emb1.shape[0], emb2.shape[0])
    l_max = max(emb1.shape[0], emb2.shape[0])

    if score_method == "EBA_raw":
        return float(eba_raw)
    elif score_method == "EBA_min":
        return float(eba_raw / l_max)
    else:  # EBA_max
        return float(eba_raw / l_min)


# Module-level globals for multiprocessing (set BEFORE fork, true COW)
_G_EMB: Dict[str, np.ndarray] = {}
_G_METHOD: str = "EBA_min"
_G_LSIM: float = 1.0


def _score_row(row_args):
    """Worker: process all pairs for row i (i vs j for all j > i)."""
    id_i, j_ids = row_args
    emb_i = _G_EMB[id_i]
    results = []
    for id_j in j_ids:
        score = eba_score_numpy(emb_i, _G_EMB[id_j], score_method=_G_METHOD, l_sim=_G_LSIM)
        results.append((id_i, id_j, score))
    return results


def _warmup_numba():
    """Pre-compile Numba JIT functions with a tiny dummy call."""
    from eba import alignments as alg

    dummy = np.random.randn(3, 3)
    alg.dtw_align(dummy, 0.0, 0.0)
    logging.info("Numba JIT warmup done")


def score_pairs_eba(
    embeddings: Dict[str, np.ndarray],
    domain_ids: List[str],
    score_method: str = "EBA_min",
    l_sim: float = 1.0,
    num_workers: int = 1,
) -> List[Tuple[str, str, float]]:
    """
    Run all-vs-all EBA scoring.

    Returns list of (query_id, hit_id, score) tuples.
    Score direction: higher = more similar for all methods.
    """
    global _G_EMB, _G_METHOD, _G_LSIM

    n = len(domain_ids)
    total_pairs = n * (n - 1) // 2

    logging.info(f"Running {total_pairs:,} pairwise EBA searches ({score_method}, l={l_sim}, workers={num_workers})...")

    # Warmup Numba JIT before forking (compile once, share across workers)
    _warmup_numba()

    if num_workers <= 1:
        # Single-threaded
        results = []
        with tqdm(total=total_pairs, desc="Scoring pairs") as pbar:
            for i in range(n):
                for j in range(i + 1, n):
                    score = eba_score_numpy(
                        embeddings[domain_ids[i]],
                        embeddings[domain_ids[j]],
                        score_method=score_method,
                        l_sim=l_sim,
                    )
                    results.append((domain_ids[i], domain_ids[j], score))
                    pbar.update(1)
        return results

    # Set globals BEFORE fork (true COW - no pickling of embeddings)
    _G_EMB = embeddings
    _G_METHOD = score_method
    _G_LSIM = l_sim

    import multiprocessing as mp

    # Row tasks: (domain_id_i, [domain_id_j, ...])
    row_tasks = []
    for i in range(n):
        j_ids = [domain_ids[j] for j in range(i + 1, n)]
        if j_ids:
            row_tasks.append((domain_ids[i], j_ids))

    results = []
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=total_pairs, desc="Scoring pairs") as pbar:
            for row_results in pool.imap_unordered(_score_row, row_tasks):
                results.extend(row_results)
                pbar.update(len(row_results))

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
    parser = argparse.ArgumentParser(description="Run EBA All-vs-All benchmark on ECOD Hard dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing hard_benchmark.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--plm", type=str, default="ProtT5", choices=["ProtT5", "ESMb1", "ESM2", "ProstT5"], help="PLM model for embeddings")
    parser.add_argument("--score_method", type=str, default="EBA_min", choices=SCORE_METHODS, help="Score method: EBA_min (default), EBA_max, EBA_raw")
    parser.add_argument("--l_sim", type=float, default=1.0, help="Similarity matrix regularization parameter")
    parser.add_argument("--device", type=str, default="cuda", help="Device for embedding computation")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers for pairwise search (default: 1)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_benchmark_data(data_dir)

    # Compute embeddings
    embeddings = compute_embeddings(
        df,
        output_dir,
        plm=args.plm,
        device=args.device,
    )

    # Run scoring
    start_time = time.time()
    results = score_pairs_eba(
        embeddings,
        df["id"].tolist(),
        score_method=args.score_method,
        l_sim=args.l_sim,
        num_workers=args.num_workers,
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
    metrics["plm"] = args.plm
    metrics["l_sim"] = args.l_sim
    metrics["num_domains"] = len(df)
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

        plt.suptitle(f"EBA ({args.plm}, {args.score_method})")
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
