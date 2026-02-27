"""
Run OTalign All-vs-All benchmark on ECOD "Hard" benchmark dataset.

This script:
1. Loads the Hard benchmark dataset
2. Computes PLM embeddings for all domains (via OTalign's embedding system)
3. Runs all-vs-all OTalign scoring using Sinkhorn UOT
4. Assigns labels based on H-group/X-group criteria
5. Computes ROC/PR curves and metrics

Score methods (soft, from transport plan):
  - mean_cosine: sum(P*(1-C)) / sum(P), direct (higher=more similar)
  - sinkhorn_divergence: J(X,Y) - 0.5*(J(X,X) + J(Y,Y)), negated (higher=more similar)
  - transport_cost: <P, C>, negated (higher=more similar)

Score methods (hard alignment from transport plan → PMI DP):
  - norm_dp (default): DP score / n_matches (higher=more similar)
  - mean_cos_hard: mean cosine similarity at hard alignment match positions
  - norm_dp_len: DP score / max(Lq, Lt) (higher=more similar)
  - norm_dp_minlen: DP score / min(Lq, Lt) (higher=more similar)
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from otalign.align.sinkhorn import SinkhornUOT
from otalign.align.uot_alignment import hard_alignment_from_transport
from otalign.models.embedding import get_embeddings_for_sequences


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCORE_METHODS = [
    "norm_dp",
    "mean_cos_hard",
    "norm_dp_len",
    "norm_dp_minlen",
    "mean_cosine",
    "sinkhorn_divergence",
    "transport_cost",
]

# Hard-alignment-based methods (require DP traceback after Sinkhorn)
_HARD_ALN_METHODS = {"norm_dp", "mean_cos_hard", "norm_dp_len", "norm_dp_minlen"}


def load_benchmark_data(data_dir: Path) -> pd.DataFrame:
    """Load hard benchmark dataset."""
    csv_path = data_dir / "hard_benchmark.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"H": str, "X": str, "T": str})
    logging.info(f"Loaded {len(df)} domains from {csv_path}")
    return df


def compute_embeddings(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str = "Ankh_Large",
    device: str = "cuda",
    dtype: str = "fp32",
    cache_dir: str = None,
    embed_batch_size: int = 4,
) -> Dict[str, torch.Tensor]:
    """
    Compute PLM embeddings for all domains using OTalign's embedding system.

    Returns dict mapping domain_id -> embedding tensor (shape: [seq_len, D]).
    Embeddings are L2-normalized (as per OTalign convention).
    """
    embeddings_cache = output_dir / f"embeddings_{model_name}.pt"

    if embeddings_cache.exists():
        logging.info(f"Loading cached embeddings from {embeddings_cache}")
        data = torch.load(embeddings_cache, map_location="cpu", weights_only=False)
        return data

    logging.info(f"Computing embeddings with {model_name} (embed_batch_size={embed_batch_size})...")

    sequences = df["sequence"].tolist()
    ids = df["id"].tolist()

    emb_list = get_embeddings_for_sequences(
        sequences=sequences,
        seq_ids=ids,
        model_name=model_name,
        cache_dir=cache_dir,
        device=device,
        batch_size=embed_batch_size,
        dtype=dtype,
    )

    embeddings = {}
    for seq_id, emb in zip(ids, emb_list):
        embeddings[seq_id] = emb.cpu()

    # Cache
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, embeddings_cache)
    logging.info(f"Cached embeddings to {embeddings_cache}")

    return embeddings


def score_pairs_otalign(
    embeddings: Dict[str, torch.Tensor],
    domain_ids: List[str],
    score_method: str = "norm_dp",
    device: str = "cuda",
    dtype: str = "fp32",
    batch_size: int = 16,
    reg: float = 0.1,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    num_iter: int = 1000,
    aln_mode: str = "glocal",
) -> List[Tuple[str, str, float]]:
    """
    Run all-vs-all OTalign scoring.

    For sinkhorn_divergence: computes AB, AA, BB alignments (3x cost).
    For other methods: computes AB only (1x cost).
    For hard alignment methods (norm_dp, mean_cos_hard, norm_dp_len):
      runs PMI-based DP + traceback on the transport plan.

    Returns list of (query_id, hit_id, score) tuples.
    Score direction: higher = more similar for all methods.
    """
    n = len(domain_ids)
    total_pairs = n * (n - 1) // 2
    need_self = score_method == "sinkhorn_divergence"

    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    torch_dtype = dtype_map[dtype]
    torch_device = torch.device(device)

    logging.info(f"Running {total_pairs:,} pairwise OTalign searches ({score_method})...")
    if need_self:
        logging.info("  Computing self-alignments (AA, BB) for Sinkhorn divergence")

    # Pre-compute self-alignment costs for sinkhorn_divergence
    self_objectives = {}
    aligner = SinkhornUOT()

    if need_self:
        logging.info("Pre-computing self-alignment objectives...")
        for domain_id in tqdm(domain_ids, desc="Self-alignments"):
            emb = embeddings[domain_id].to(device=torch_device, dtype=torch_dtype).unsqueeze(0)
            mask = torch.ones(1, emb.shape[1], dtype=torch.bool, device=torch_device)

            with torch.no_grad():
                res = aligner(emb, emb, mask, mask, num_iter, reg, lambda1, lambda2)

            mu = mask.float() / mask.sum(-1, keepdim=True).clamp(min=1.0)
            P = res["transport_plan"]
            C = res["cost_matrix"]

            # Compute UOT objective for self
            obj = _uot_objective_single(P[0], C[0], mu[0], mu[0], mask[0], mask[0], reg, lambda1, lambda2)
            self_objectives[domain_id] = obj

        # Free GPU memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Build pair list
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((domain_ids[i], domain_ids[j]))

    # Process in batches
    results = []
    with tqdm(total=total_pairs, desc="Scoring pairs") as pbar:
        for batch_start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[batch_start : batch_start + batch_size]
            batch_scores = _score_batch(
                batch_pairs,
                embeddings,
                aligner,
                score_method,
                self_objectives,
                torch_device,
                torch_dtype,
                reg,
                lambda1,
                lambda2,
                num_iter,
                aln_mode,
            )
            results.extend(batch_scores)
            pbar.update(len(batch_pairs))

    return results


def _uot_objective_single(
    P: torch.Tensor,
    C: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    reg: float,
    lambda1: float,
    lambda2: float,
    tiny: float = 1e-12,
) -> float:
    """Compute UOT objective J for a single (non-batched) pair."""
    joint_mask = mask_a.unsqueeze(-1) & mask_b.unsqueeze(-2)
    Pm = torch.where(joint_mask, P, torch.zeros_like(P))
    Cm = torch.where(joint_mask, C, torch.full_like(C, 1e6))

    cost_term = (Pm * Cm).sum()
    ent_term = reg * (Pm.clamp_min(tiny) * (Pm.clamp_min(tiny).log() - 1.0)).sum()

    r = Pm.sum(dim=-1)
    c = Pm.sum(dim=-2)

    def _kl(x, y, mask):
        x_ = (x * mask.float()).clamp_min(0.0)
        y_ = (y * mask.float()).clamp_min(tiny)
        term = x_ * (torch.log(x_.clamp_min(tiny)) - torch.log(y_)) - x_ + y_
        return term.sum()

    a_masked = torch.where(mask_a, a, torch.zeros_like(a))
    b_masked = torch.where(mask_b, b, torch.zeros_like(b))

    kl_r = _kl(r, a_masked, mask_a)
    kl_c = _kl(c, b_masked, mask_b)

    return float(cost_term + ent_term + lambda1 * kl_r + lambda2 * kl_c)


def _score_hard_alignment(
    P_np: np.ndarray,
    C_np: np.ndarray,
    score_method: str,
    f_np: np.ndarray = None,
    g_np: np.ndarray = None,
    aln_mode: str = "glocal",
) -> float:
    """Compute alignment-based score from transport plan via hard alignment DP."""
    aln = hard_alignment_from_transport(P_np, f=f_np, g=g_np, mode=aln_mode)
    match_positions = [(i - 1, j - 1) for i, j, op in aln["path"] if op == "M"]

    if not match_positions:
        return 0.0

    n_matches = len(match_positions)

    if score_method == "norm_dp":
        return aln["score"] / n_matches
    elif score_method == "mean_cos_hard":
        cosines = [1.0 - C_np[i, j] for i, j in match_positions]
        return float(np.mean(cosines))
    elif score_method == "norm_dp_len":
        Lq, Lt = P_np.shape
        return aln["score"] / max(Lq, Lt)
    elif score_method == "norm_dp_minlen":
        Lq, Lt = P_np.shape
        return aln["score"] / min(Lq, Lt)
    else:
        raise ValueError(f"Unknown hard alignment score method: {score_method}")


def _score_batch(
    batch_pairs: List[Tuple[str, str]],
    embeddings: Dict[str, torch.Tensor],
    aligner: SinkhornUOT,
    score_method: str,
    self_objectives: Dict[str, float],
    device: torch.device,
    dtype: torch.dtype,
    reg: float,
    lambda1: float,
    lambda2: float,
    num_iter: int,
    aln_mode: str = "glocal",
) -> List[Tuple[str, str, float]]:
    """Score a batch of pairs on GPU."""
    bs = len(batch_pairs)
    use_hard_aln = score_method in _HARD_ALN_METHODS

    emb1_list = [embeddings[p[0]] for p in batch_pairs]
    emb2_list = [embeddings[p[1]] for p in batch_pairs]

    max_len1 = max(e.shape[0] for e in emb1_list)
    max_len2 = max(e.shape[0] for e in emb2_list)
    emb_dim = emb1_list[0].shape[1]

    emb1_padded = torch.zeros(bs, max_len1, emb_dim, device=device, dtype=dtype)
    emb2_padded = torch.zeros(bs, max_len2, emb_dim, device=device, dtype=dtype)
    mask1 = torch.zeros(bs, max_len1, dtype=torch.bool, device=device)
    mask2 = torch.zeros(bs, max_len2, dtype=torch.bool, device=device)

    for i in range(bs):
        len1 = emb1_list[i].shape[0]
        emb1_padded[i, :len1] = emb1_list[i].to(device=device, dtype=dtype)
        mask1[i, :len1] = True

        len2 = emb2_list[i].shape[0]
        emb2_padded[i, :len2] = emb2_list[i].to(device=device, dtype=dtype)
        mask2[i, :len2] = True

    with torch.no_grad():
        res_ab = aligner(emb1_padded, emb2_padded, mask1, mask2, num_iter, reg, lambda1, lambda2)

    results = []
    for i in range(bs):
        qid, hid = batch_pairs[i]
        len1 = mask1[i].sum().item()
        len2 = mask2[i].sum().item()

        P_ab = res_ab["transport_plan"][i : i + 1, :len1, :len2]
        C_ab = res_ab["cost_matrix"][i : i + 1, :len1, :len2]
        mu_i = res_ab["mu"][i : i + 1, :len1]
        nu_i = res_ab["nu"][i : i + 1, :len2]
        mask1_i = mask1[i : i + 1, :len1]
        mask2_i = mask2[i : i + 1, :len2]

        if use_hard_aln:
            P_np = P_ab[0].cpu().numpy()
            C_np = C_ab[0].cpu().numpy()
            f_np = res_ab["f"][i, :len1].cpu().numpy() if "f" in res_ab else None
            g_np = res_ab["g"][i, :len2].cpu().numpy() if "g" in res_ab else None
            score = _score_hard_alignment(P_np, C_np, score_method, f_np, g_np, aln_mode)
        elif score_method == "mean_cosine":
            mass = P_ab.sum()
            score = float(((P_ab * (1.0 - C_ab)).sum() / (mass + 1e-12)).item())
        elif score_method == "transport_cost":
            score = -float((P_ab * C_ab).sum().item())
        elif score_method == "sinkhorn_divergence":
            obj_xy = _uot_objective_single(
                P_ab[0],
                C_ab[0],
                mu_i[0],
                nu_i[0],
                mask1_i[0],
                mask2_i[0],
                reg,
                lambda1,
                lambda2,
            )
            obj_xx = self_objectives[qid]
            obj_yy = self_objectives[hid]
            sd = obj_xy - 0.5 * (obj_xx + obj_yy)
            score = -sd
        else:
            raise ValueError(f"Unknown score method: {score_method}")

        results.append((qid, hid, score))

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
    parser = argparse.ArgumentParser(description="Run OTalign All-vs-All benchmark on ECOD Hard dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing hard_benchmark.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--model_name", type=str, default="Ankh_Large", help="PLM model name (e.g., Ankh_Large, ProtT5_XL_UniRef50, ESM2_33_650M)")
    parser.add_argument(
        "--score_method",
        type=str,
        default="norm_dp",
        choices=SCORE_METHODS,
        help="Score method (default: norm_dp). Hard alignment: norm_dp, mean_cos_hard, norm_dp_len. Soft transport: mean_cosine, sinkhorn_divergence, transport_cost",
    )
    parser.add_argument("--aln_mode", type=str, default="global", choices=["global", "glocal", "q2t", "t2q", "local"], help="Hard alignment mode (default: global)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for computation")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "fp32", "bf16"], help="Data type for computation")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for pairwise Sinkhorn alignment")
    parser.add_argument("--reg", type=float, default=0.1, help="Entropic regularization for Sinkhorn")
    parser.add_argument("--lambda1", type=float, default=1.0, help="KL penalty for source marginal")
    parser.add_argument("--lambda2", type=float, default=1.0, help="KL penalty for target marginal")
    parser.add_argument("--num_iter", type=int, default=1000, help="Number of Sinkhorn iterations")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory for embedding cache")
    parser.add_argument("--embed_batch_size", type=int, default=4, help="Batch size for PLM embedding computation (reduce if OOM)")

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
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        cache_dir=args.cache_dir,
        embed_batch_size=args.embed_batch_size,
    )

    # Run scoring
    start_time = time.time()
    results = score_pairs_otalign(
        embeddings,
        df["id"].tolist(),
        score_method=args.score_method,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        reg=args.reg,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        num_iter=args.num_iter,
        aln_mode=args.aln_mode,
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
    metrics["aln_mode"] = args.aln_mode
    metrics["model_name"] = args.model_name
    metrics["reg"] = args.reg
    metrics["lambda1"] = args.lambda1
    metrics["lambda2"] = args.lambda2
    metrics["num_iter"] = args.num_iter
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

        plt.suptitle(f"OTalign ({args.model_name}, {args.score_method})")
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
