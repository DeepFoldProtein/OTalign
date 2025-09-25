import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


# Import PLMAlign utility functions from our local copy
try:
    from otalign.utils.plmalign_utils import dot_product, draw_alignment, plmalign_gather_all_paths, plmalign_search_paths
except ImportError as e:
    raise ImportError(f"Failed to import PLMAlign utilities: {e}")

# Try to import ProtT5 from OTalign adaptors
try:
    from otalign.models.plm_adaptors import get_plm_adaptor_and_configs
except ImportError as e:
    raise ImportError(f"Failed to import OTalign adaptors: {e}")

from otalign.cache.lmdb_reader import LMDBCache
from otalign.cache.npz_reader import NPZCache
from scripts.dataset_utils import alignment_metrics, iter_pairs_from_dataset


def perform_plmalign(emb1: torch.Tensor, emb2: torch.Tensor, seq1: str, seq2: str, mode: str = "global", gap_extension: float = 1.0, norm: bool = True) -> Tuple[float, str, List[Tuple[int, int]]]:
    """
    Perform PLMAlign using pre-computed embeddings.

    Args:
        emb1: Embedding for sequence 1
        emb2: Embedding for sequence 2
        seq1: Sequence 1 string
        seq2: Sequence 2 string
        mode: Alignment mode ("global" or "local")
        gap_extension: Gap extension penalty
        norm: Whether to normalize embeddings

    Returns:
        Tuple of (score, alignment_string, alignment_pairs)
    """
    # Convert to numpy
    emb1_np = emb1.numpy().astype(np.float32)
    emb2_np = emb2.numpy().astype(np.float32)

    # Compute similarity matrix using dot product
    densitymap = dot_product(emb1_np, emb2_np)
    densitymap = densitymap.T

    # Perform PLMAlign
    path = plmalign_gather_all_paths(densitymap, norm=norm, mode=mode, gap_extension=gap_extension, with_scores=False)

    results = plmalign_search_paths(densitymap, path=path, mode=mode, as_df=True)

    if len(results) == 0:
        return 0.0, "", []

    # Get the best alignment
    best_result = results.iloc[0]
    score = float(best_result["score"])

    # Extract alignment pairs from indices
    indices = best_result["indices"]
    alignment_pairs = [(int(idx[0]), int(idx[1])) for idx in indices]

    # Generate alignment string
    alignment_str = draw_alignment(indices, seq1, seq2, output="str")

    return score, alignment_str, alignment_pairs


@torch.no_grad()
def _worker(task):
    """Worker function to process a single sequence pair."""
    (ex, args_dict) = task
    try:
        seq1 = ex["seq1"]
        seq2 = ex["seq2"]
        seq1_id = ex.get("seq1_id", "seq1")
        seq2_id = ex.get("seq2_id", "seq2")

        # Get embeddings from cache or generate them
        cache_dir = args_dict["cache_dir"]
        alignment_mode = args_dict["alignment_mode"]
        plm_adaptor = args_dict["plm_adaptor"]
        device = args_dict.get("device", "cpu")

        # Initialize cache
        cache = None
        if cache_dir:
            cache_dir_path = Path(cache_dir)
            if (cache_dir_path / "data.lmdb").exists():
                cache = LMDBCache(cache_dir)
            else:
                cache = NPZCache(cache_dir)

        # Check if embeddings exist in cache
        seq1_embedding = None
        seq2_embedding = None

        if cache:
            try:
                seq1_embedding, _, _ = cache.get(seq1_id, device="cpu", dtype=torch.float32)
            except (FileNotFoundError, KeyError):
                pass

            try:
                seq2_embedding, _, _ = cache.get(seq2_id, device="cpu", dtype=torch.float32)
            except (FileNotFoundError, KeyError):
                pass

        # Generate embeddings if not in cache using PLM adaptor
        if seq1_embedding is None or seq2_embedding is None:
            sequences = []

            if seq1_embedding is None:
                sequences.append(seq1)
            if seq2_embedding is None:
                sequences.append(seq2)

            # Get embeddings using PLM adaptor
            if hasattr(plm_adaptor, "tokenize"):
                tokenized = plm_adaptor.tokenize(sequences)
                embeddings = plm_adaptor.forward(tokenized["input_ids"], tokenized["attention_mask"])
            else:
                # Use the encode method for newer adaptors
                embedding_output = plm_adaptor.encode(sequences, device="cpu" if device == "cpu" else None)
                embeddings = embedding_output.residue_embeddings

            # Extract individual embeddings
            idx = 0
            if seq1_embedding is None:
                seq1_len = len(seq1)
                seq1_embedding = embeddings[idx, :seq1_len].cpu()
                idx += 1
            if seq2_embedding is None:
                seq2_len = len(seq2)
                seq2_embedding = embeddings[idx, :seq2_len].cpu()

        # Perform PLMAlign using our utility functions
        # Note: Original PLMAlign uses (emb2, emb1) order, so we swap here
        score, alignment_str, pairs = perform_plmalign(seq2_embedding, seq1_embedding, seq2, seq1, alignment_mode)

        met = alignment_metrics(pairs, ex.get("ref_alignment"))
        rec = {
            "pair_id": ex.get("pair_id", f"{seq1_id}-{seq2_id}"),
            "seq1_id": seq1_id,
            "seq2_id": seq2_id,
            "pred_alignment": pairs,
            "metrics": met,
            "meta": {"tool": "PLMAlign", "alignment_mode": alignment_mode, "cache_dir": args_dict.get("cache_dir", "None")},
        }
        return rec
    except Exception as e:
        seq1_id = ex.get("seq1_id", "seq1")
        seq2_id = ex.get("seq2_id", "seq2")
        return {"pair_id": ex.get("pair_id", f"{seq1_id}-{seq2_id}"), "error": str(e)}


def run_plmalign_evaluation(
    dataset: str,
    output: str,
    alignment_mode: str = "global",
    plm_model: str = "ProtT5_XL_UniRef50",
    cache_dir: Optional[str] = None,
    device: str = "cuda",
    out_dir: Optional[str] = None,
    pbar: Optional[tqdm] = None,
):
    """Runs PLMAlign evaluation on a dataset and writes results to a file."""

    # Load PLM adaptor
    plm_adaptor, _, _ = get_plm_adaptor_and_configs(plm_model, for_masked_lm=False)

    # Move to device if possible
    if hasattr(plm_adaptor, "to"):
        plm_adaptor.to(device if torch.cuda.is_available() else "cpu")
    elif hasattr(plm_adaptor, "model") and hasattr(plm_adaptor.model, "to"):
        plm_adaptor.model.to(device if torch.cuda.is_available() else "cpu")

    # Set to eval mode if possible
    if hasattr(plm_adaptor, "eval"):
        plm_adaptor.eval()
    elif hasattr(plm_adaptor, "model") and hasattr(plm_adaptor.model, "eval"):
        plm_adaptor.model.eval()

    dataset_iterator = list(iter_pairs_from_dataset(dataset))
    total_pairs = len(dataset_iterator)
    success_count = 0
    fail_count = 0

    args_dict = {
        "alignment_mode": alignment_mode,
        "cache_dir": cache_dir,
        "plm_adaptor": plm_adaptor,
        "device": device,
        "out_dir": out_dir,
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup progress bar
    local_pbar = pbar is None
    if local_pbar:
        pbar = tqdm(total=total_pairs, desc="Aligning with PLMAlign")
    else:
        pbar.total = total_pairs
        pbar.set_description("Aligning with PLMAlign")
        pbar.reset()

    with out_path.open("w", encoding="utf-8") as fout:
        tasks = ((ex, args_dict) for ex in dataset_iterator)
        for task in tasks:
            rec = _worker(task)
            if "error" in rec:
                logging.warning(f"Pair {rec['pair_id']} failed with error: {rec['error']}")
                fail_count += 1
            else:
                fout.write(json.dumps(rec) + "\n")
                success_count += 1
            pbar.update(1)

    if local_pbar:
        pbar.close()

    logging.info(f"Evaluation Summary: {success_count}/{total_pairs} pairs processed successfully.")
    if fail_count > 0:
        logging.warning(f"Failed pairs: {fail_count}")
    logging.info(f"[ok] wrote predictions -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Run PLMAlign over a dataset and export JSONL predictions.")
    ap.add_argument("--dataset", type=str, required=True, help="Path to the dataset (JSONL or HF identifier).")
    ap.add_argument("--mode", type=str, default="global", choices=["global", "local"], help="Alignment mode")
    ap.add_argument("--plm_model", type=str, default="ProtT5_XL_UniRef50", help="PLM model to use for embeddings")
    ap.add_argument("--cache_dir", type=str, default=None, help="Path to LMDB cache directory")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    run_plmalign_evaluation(dataset=args.dataset, alignment_mode=args.mode, plm_model=args.plm_model, cache_dir=args.cache_dir, output=args.output, out_dir=args.out_dir, device=args.device)


if __name__ == "__main__":
    main()
