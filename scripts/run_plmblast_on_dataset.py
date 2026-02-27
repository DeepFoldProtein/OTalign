"""
Run pLM-BLAST (https://github.com/labstructbioinf/pLM-BLAST) on a benchmark dataset.
Uses the same ProtT5 embedding cache as PLMAlign when available.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm


# Add pLM-BLAST to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLMBLAST_ROOT = PROJECT_ROOT / "third_party" / "plmblast"
if PLMBLAST_ROOT.exists():
    sys.path.insert(0, str(PLMBLAST_ROOT))
else:
    raise ImportError(f"pLM-BLAST not found at {PLMBLAST_ROOT}. Run: git submodule update --init third_party/plmblast")

import alntools as aln  # noqa: E402

from otalign.cache.lmdb_reader import LMDBCache
from otalign.cache.npz_reader import NPZCache
from scripts.dataset_utils import alignment_metrics, iter_pairs_from_dataset


def _get_plmblast_extractor(global_alignment: bool = True, **kwargs):
    """Build pLM-BLAST Extractor with optional global alignment."""
    return aln.Extractor(
        enh=False,
        norm=False,
        bfactor="global" if global_alignment else 2,
        sigma_factor=kwargs.get("sigma_factor", 2.0),
        gap_penalty=kwargs.get("gap_penalty", 0.0),
        min_spanlen=kwargs.get("min_spanlen", 20),
        window_size=kwargs.get("window_size", 20),
        filter_results=kwargs.get("filter_results", True),  # Default True per pLM-BLAST examples
    )


def _worker(task):
    (ex, args_dict) = task
    try:
        seq1 = ex["seq1"]
        seq2 = ex["seq2"]
        seq1_id = ex.get("seq1_id", "seq1")
        seq2_id = ex.get("seq2_id", "seq2")
        cache_dir = args_dict.get("cache_dir")
        extractor = args_dict["extractor"]

        cache = None
        if cache_dir:
            cache_path = Path(cache_dir)
            if (cache_path / "data.lmdb").exists():
                cache = LMDBCache(cache_dir)
            else:
                cache = NPZCache(cache_dir)

        def get_emb(seq_id, seq):
            if cache:
                try:
                    emb, _, _ = cache.get(seq_id, device="cpu", dtype=torch.float32)
                    return emb.numpy().astype(np.float32)
                except (FileNotFoundError, KeyError):
                    pass
            return None

        emb1 = get_emb(seq1_id, seq1)
        emb2 = get_emb(seq2_id, seq2)
        if emb1 is None or emb2 is None:
            return {
                "pair_id": ex.get("pair_id", f"{seq1_id}-{seq2_id}"),
                "error": "Embedding not in cache; build cache first (e.g. same as PLMAlign/ProtT5).",
            }

        # pLM-BLAST: embedding_local_similarity(X,Y) returns density with shape (Y_len, X_len) so
        # row=seq2_idx, col=seq1_idx. Path indices from search_paths are (y,x)=(seq2_idx, seq1_idx).
        # We need (seq1_idx, seq2_idx) for the benchmark -> swap to (j, i).
        results = extractor.embedding_to_span(emb1, emb2)
        if results is None or len(results) == 0:
            pairs = []
        else:
            # Apply filtering if enabled (removes overlapping/redundant hits)
            if extractor.FILTER_RESULTS:
                results = aln.postprocess.filter_result_dataframe(results)

            if len(results) == 0:
                pairs = []
            else:
                # Take best hit by score
                best = results.sort_values("score", ascending=False).iloc[0]
                indices = best["indices"]
                if hasattr(indices, "tolist"):
                    indices = indices.tolist()
                # indices from pLM-BLAST are (seq2_idx, seq1_idx); benchmark expects (seq1_idx, seq2_idx)
                raw_pairs = [(int(j), int(i)) for i, j in indices]
                # Use pLM-BLAST's coords_to_match_pairs (same rule as draw_alignment: match = both coords change)
                pairs = aln.coords_to_match_pairs(raw_pairs)

        met = alignment_metrics(pairs, ex.get("ref_alignment"))
        return {
            "pair_id": ex.get("pair_id", f"{seq1_id}-{seq2_id}"),
            "seq1_id": seq1_id,
            "seq2_id": seq2_id,
            "pred_alignment": pairs,
            "metrics": met,
            "meta": {"tool": "pLM-BLAST", "global_alignment": args_dict.get("global_alignment", True)},
        }
    except Exception as e:
        return {
            "pair_id": ex.get("pair_id", f"{ex.get('seq1_id', 'seq1')}-{ex.get('seq2_id', 'seq2')}"),
            "error": str(e),
        }


def run_plmblast_evaluation(
    dataset: str,
    output: str,
    cache_dir: Optional[str] = None,
    global_alignment: bool = True,
    device: str = "cuda",
    pbar: Optional[tqdm] = None,
    **extractor_kwargs,
):
    """Run pLM-BLAST evaluation on a dataset and write results to JSONL."""
    extractor = _get_plmblast_extractor(global_alignment=global_alignment, **extractor_kwargs)

    dataset_iterator = list(iter_pairs_from_dataset(dataset))
    total_pairs = len(dataset_iterator)
    success_count = 0
    fail_count = 0

    args_dict = {
        "cache_dir": cache_dir,
        "extractor": extractor,
        "device": device,
        "global_alignment": global_alignment,
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    local_pbar = pbar is None
    if local_pbar:
        pbar = tqdm(total=total_pairs, desc="Aligning with pLM-BLAST")
    else:
        pbar.total = total_pairs
        pbar.set_description("Aligning with pLM-BLAST")
        pbar.reset()

    with out_path.open("w", encoding="utf-8") as fout:
        for ex in dataset_iterator:
            rec = _worker((ex, args_dict))
            if "error" in rec:
                logging.warning(f"Pair {rec['pair_id']} failed: {rec['error']}")
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
    ap = argparse.ArgumentParser(description="Run pLM-BLAST on a dataset and export JSONL predictions.")
    ap.add_argument("--dataset", type=str, required=True, help="Dataset (JSONL or HF id, e.g. DeepFoldProtein/SABmark-dataset,sup,test)")
    ap.add_argument("--output", type=str, required=True, help="Output JSONL path")
    ap.add_argument("--cache_dir", type=str, default=None, help="Path to embedding cache (ProtT5, same as PLMAlign)")
    ap.add_argument("--local", action="store_true", help="Use local alignment instead of global")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--sigma_factor", type=float, default=2.0)
    ap.add_argument("--gap_penalty", type=float, default=0.0)
    ap.add_argument("--min_spanlen", type=int, default=20)
    ap.add_argument("--window_size", type=int, default=20)
    ap.add_argument("--no-filter", dest="filter_results", action="store_false", default=True, help="Disable result filtering (default: enabled)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_plmblast_evaluation(
        dataset=args.dataset,
        output=args.output,
        cache_dir=args.cache_dir,
        global_alignment=not args.local,
        device=args.device,
        sigma_factor=args.sigma_factor,
        gap_penalty=args.gap_penalty,
        min_spanlen=args.min_spanlen,
        window_size=args.window_size,
        filter_results=args.filter_results,
    )


if __name__ == "__main__":
    main()
