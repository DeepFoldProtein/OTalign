"""
Run pLM-BLAST All-vs-All benchmark on ECOD "Hard" benchmark dataset.

Implements the full pLM-BLAST scoring pipeline from the paper:
1. L2-normalized cosine similarity matrix (densitymap)
2. Z-score normalization → modified Smith-Waterman DP (no truncation at 0)
3. Traceback from border points of the score matrix
4. Moving average smoothing (window=20) + sigma threshold filtering (2σ)
5. Final score = mean of cosine similarities at aligned positions

This script:
1. Loads the Hard benchmark dataset
2. Computes ProtT5 embeddings for all domains (or loads cached)
3. Runs all-vs-all pLM-BLAST searches using the full paper pipeline
4. Assigns labels based on H-group/X-group criteria
5. Computes ROC/PR curves and metrics
"""

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# Add third_party/plmblast to path for alntools
PLMBLAST_PATH = Path(__file__).parent.parent / "third_party" / "plmblast"
sys.path.insert(0, str(PLMBLAST_PATH))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ── Global state for fork COW multiprocessing ──
_G_EMB: Dict[str, np.ndarray] = {}
_G_WINDOW: int = 20
_G_MIN_SPAN: int = 20
_G_SIGMA: float = 2.0
_G_GAP: float = 0.0
_G_BFACTOR = 2  # int or "global"


def load_benchmark_data(data_dir: Path) -> pd.DataFrame:
    """Load hard benchmark dataset."""
    csv_path = data_dir / "hard_benchmark.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")

    # Force H, X, T columns to be read as strings (numerical IDs like "907.1"
    # would otherwise be read as floats, causing comparison issues)
    df = pd.read_csv(csv_path, dtype={"H": str, "X": str, "T": str})
    logging.info(f"Loaded {len(df)} domains from {csv_path}")
    return df


def compute_embeddings(
    df: pd.DataFrame, output_dir: Path, model_name: str = "Rostlab/prot_t5_xl_uniref50", device: str = "cuda", batch_size: int = 8, half_precision: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute ProtT5 embeddings for all domains.

    Returns dict mapping domain_id -> embedding array (shape: [seq_len, 1024]).
    """
    embeddings_cache = output_dir / "embeddings.npz"

    if embeddings_cache.exists():
        logging.info(f"Loading cached embeddings from {embeddings_cache}")
        data = np.load(embeddings_cache, allow_pickle=True)
        return dict(data["embeddings"].item())

    logging.info("Computing ProtT5 embeddings...")

    import torch
    from transformers import T5EncoderModel, T5Tokenizer

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model
    logging.info(f"Loading model: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)

    if half_precision and device.type == "cuda":
        model = model.half()
    model = model.to(device)
    model.eval()

    embeddings = {}
    sequences = df["sequence"].tolist()
    ids = df["id"].tolist()

    with torch.no_grad():
        for i, (seq, domain_id) in enumerate(tqdm(zip(sequences, ids), total=len(sequences), desc="Computing embeddings")):
            # Add spaces between amino acids (ProtT5 requirement)
            seq_spaced = " ".join(list(seq))

            # Tokenize
            inputs = tokenizer(seq_spaced, return_tensors="pt", add_special_tokens=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get embeddings - use full 1024D (64D pooling is only for pre-screening)
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[0, :-1, :].cpu().float().numpy()  # [L, 1024] float32

            embeddings[domain_id] = emb

    # Cache embeddings
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(embeddings_cache, embeddings=embeddings)
    logging.info(f"Cached embeddings to {embeddings_cache}")

    # Clean up
    del model, tokenizer
    torch.cuda.empty_cache()

    return embeddings


def plmblast_score(
    emb1: np.ndarray,
    emb2: np.ndarray,
    gap_penalty: float = 0.0,
    bfactor=2,
    window_size: int = 20,
    min_span: int = 20,
    sigma_factor: float = 2.0,
) -> float:
    """
    Compute pLM-BLAST alignment score using the full paper pipeline.

    Pipeline:
    1. Cosine similarity matrix (L2-norm + dot product)
    2. Z-score normalization → modified Smith-Waterman (no truncation at 0)
    3. Traceback from border points of the score matrix
    4. Moving average smoothing + sigma threshold filtering
    5. Score = mean of cosine similarities at aligned positions

    Returns the best (highest) alignment score, or 0.0 if no valid alignment found.
    """
    from alntools.alignment import gather_all_paths
    from alntools.numeric import embedding_local_similarity
    from alntools.prepare import search_paths

    emb1 = emb1.astype(np.float32)
    emb2 = emb2.astype(np.float32)

    # Step 1: Cosine similarity matrix (raw, unnormalized)
    densitymap = embedding_local_similarity(emb1, emb2)

    # Step 2-3: Z-score norm → SW DP → traceback from border points
    # gather_all_paths internally does: norm → fill_score_matrix → border_argmaxpool → traceback
    paths = gather_all_paths(
        densitymap,
        norm=True,  # Z-score normalize for DP
        minlen=min_span,
        bfactor=bfactor,  # argmax pooling factor for border points
        gap_penalty=gap_penalty,
    )

    if not paths:
        return 0.0

    # Step 4-5: Moving average + sigma filter → score = mean(cosine_sim at aligned positions)
    globalmode = bfactor == "global"
    spans = search_paths(
        densitymap,  # Raw cosine similarity (NOT normalized)
        paths=paths,
        window=window_size,
        min_span=min_span,
        sigma_factor=sigma_factor,
        globalmode=globalmode,
    )

    if not spans:
        return 0.0

    # Return the best alignment score
    best_score = max(v["score"] for v in spans.values())
    return float(best_score)


def _score_row(row_args):
    """Worker: score one query against all its targets using fork COW globals."""
    id_i, j_ids = row_args
    emb_i = _G_EMB[id_i]
    results = []
    for id_j in j_ids:
        score = plmblast_score(
            emb_i,
            _G_EMB[id_j],
            gap_penalty=_G_GAP,
            bfactor=_G_BFACTOR,
            window_size=_G_WINDOW,
            min_span=_G_MIN_SPAN,
            sigma_factor=_G_SIGMA,
        )
        results.append((id_i, id_j, score))
    return results


def run_pairwise_search(
    embeddings: Dict[str, np.ndarray],
    domain_ids: List[str],
    gap_penalty: float = 0.0,
    bfactor: int = 2,
    window_size: int = 20,
    min_span: int = 20,
    sigma_factor: float = 2.0,
    num_workers: int = 1,
) -> List[Tuple[str, str, float]]:
    """
    Run all-vs-all pLM-BLAST search using the full paper pipeline.

    Uses fork COW multiprocessing: globals are set BEFORE fork so child
    processes share memory via copy-on-write (no pickling overhead).
    """
    global _G_EMB, _G_WINDOW, _G_MIN_SPAN, _G_SIGMA, _G_GAP, _G_BFACTOR

    n = len(domain_ids)
    total_pairs = n * (n - 1) // 2

    logging.info(f"Running {total_pairs:,} pairwise searches (paper pipeline, {num_workers} workers, window={window_size}, min_span={min_span}, sigma={sigma_factor})...")

    # Warm up Numba JIT before fork
    logging.info("Warming up Numba JIT...")
    _dummy = np.random.randn(10, 1024).astype(np.float32)
    # Warmup both local and global DP paths for Numba JIT
    _ = plmblast_score(_dummy, _dummy, gap_penalty=gap_penalty, bfactor=2, window_size=window_size, min_span=5, sigma_factor=sigma_factor)
    _ = plmblast_score(_dummy, _dummy, gap_penalty=gap_penalty, bfactor="global", window_size=window_size, min_span=5, sigma_factor=sigma_factor)
    logging.info("JIT warmup done")

    # Build row tasks: each row = (id_i, [list of j_ids])
    row_tasks = []
    for i in range(n):
        j_ids = [domain_ids[j] for j in range(i + 1, n)]
        if j_ids:
            row_tasks.append((domain_ids[i], j_ids))

    if num_workers <= 1:
        results = []
        with tqdm(total=total_pairs, desc="Searching") as pbar:
            for id_i, j_ids in row_tasks:
                emb_i = embeddings[id_i]
                for id_j in j_ids:
                    score = plmblast_score(
                        emb_i,
                        embeddings[id_j],
                        gap_penalty=gap_penalty,
                        bfactor=bfactor,
                        window_size=window_size,
                        min_span=min_span,
                        sigma_factor=sigma_factor,
                    )
                    results.append((id_i, id_j, float(score)))
                    pbar.update(1)
        return results

    # Set globals BEFORE fork (true COW, no pickling)
    _G_EMB = embeddings
    _G_WINDOW = window_size
    _G_MIN_SPAN = min_span
    _G_SIGMA = sigma_factor
    _G_GAP = gap_penalty
    _G_BFACTOR = bfactor

    results = []
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=total_pairs, desc="Searching") as pbar:
            for row_results in pool.imap_unordered(_score_row, row_tasks):
                results.extend(row_results)
                pbar.update(len(row_results))

    return results


def assign_labels(results: List[Tuple[str, str, float]], df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign labels based on H-group/X-group criteria.

    Returns DataFrame with columns: query_id, hit_id, score, label, query_h, hit_h, query_x, hit_x

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
            label = 1  # Same H-group = True Positive
        elif query_x != hit_x:
            label = 0  # Different X-group = False Positive
        else:
            label = -1  # Same X, different H = Neutral

        labeled_results.append({"query_id": query_id, "hit_id": hit_id, "score": score, "label": label, "query_h": query_h, "hit_h": hit_h, "query_x": query_x, "hit_x": hit_x})

    result_df = pd.DataFrame(labeled_results)

    # Statistics
    tp_count = (result_df["label"] == 1).sum()
    fp_count = (result_df["label"] == 0).sum()
    neutral_count = (result_df["label"] == -1).sum()

    logging.info("Label distribution:")
    logging.info(f"  - True Positives (same H): {tp_count:,} ({100 * tp_count / len(result_df):.2f}%)")
    logging.info(f"  - False Positives (diff X): {fp_count:,} ({100 * fp_count / len(result_df):.2f}%)")
    logging.info(f"  - Neutral (same X, diff H): {neutral_count:,} ({100 * neutral_count / len(result_df):.2f}%)")

    return result_df


def compute_metrics(result_df: pd.DataFrame, exclude_neutral: bool = True) -> Dict:
    """
    Compute ROC-AUC and PR-AUC metrics.
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    if exclude_neutral:
        eval_df = result_df[result_df["label"] != -1].copy()
    else:
        eval_df = result_df.copy()

    scores = eval_df["score"].values
    labels = eval_df["label"].values

    # ROC-AUC
    roc_auc = roc_auc_score(labels, scores)

    # PR-AUC
    pr_auc = average_precision_score(labels, scores)

    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(labels, scores)

    # PR curve
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)

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
    parser = argparse.ArgumentParser(description="Run pLM-BLAST All-vs-All benchmark on ECOD Hard dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing hard_benchmark.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device for embedding computation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for embedding computation")
    parser.add_argument("--gap_penalty", type=float, default=0.0, help="Gap penalty for SW alignment (paper default: 0.0)")
    parser.add_argument("--bfactor", type=int, default=2, help="Argmax pooling factor for border points (paper default: 2)")
    parser.add_argument("--window_size", type=int, default=20, help="Moving average window size (paper default: 20)")
    parser.add_argument("--min_span", type=int, default=20, help="Minimum alignment span length (paper default: 20)")
    parser.add_argument("--sigma_factor", type=float, default=2.0, help="Sigma factor for threshold filtering (paper default: 2.0)")
    parser.add_argument("--model_name", type=str, default="Rostlab/prot_t5_xl_uniref50", help="ProtT5 model name")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers for pairwise search")
    parser.add_argument("--mode", type=str, default="local", choices=["local", "global"], help="Alignment mode: local (SW, default) or global (NW)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map mode to bfactor: "global" string triggers NW in gather_all_paths
    effective_bfactor = "global" if args.mode == "global" else args.bfactor

    # Load data
    df = load_benchmark_data(data_dir)

    # Compute embeddings
    embeddings = compute_embeddings(df, output_dir, model_name=args.model_name, device=args.device, batch_size=args.batch_size)

    # Run search
    start_time = time.time()
    results = run_pairwise_search(
        embeddings,
        df["id"].tolist(),
        gap_penalty=args.gap_penalty,
        bfactor=effective_bfactor,
        window_size=args.window_size,
        min_span=args.min_span,
        sigma_factor=args.sigma_factor,
        num_workers=args.num_workers,
    )
    search_time = time.time() - start_time
    logging.info(f"Search completed in {search_time:.2f} seconds")

    # Assign labels
    result_df = assign_labels(results, df)

    # Save results
    result_df.to_csv(output_dir / "search_results.csv", index=False)
    logging.info(f"Saved results to {output_dir / 'search_results.csv'}")

    # Compute metrics
    metrics = compute_metrics(result_df, exclude_neutral=True)
    metrics["search_time_seconds"] = search_time
    metrics["scoring"] = "paper_pipeline"
    metrics["mode"] = args.mode
    metrics["gap_penalty"] = args.gap_penalty
    metrics["bfactor"] = str(effective_bfactor)
    metrics["window_size"] = args.window_size
    metrics["min_span"] = args.min_span
    metrics["sigma_factor"] = args.sigma_factor
    metrics["model_name"] = args.model_name
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

        plt.suptitle(f"pLM-BLAST (paper pipeline, {args.mode})")
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
