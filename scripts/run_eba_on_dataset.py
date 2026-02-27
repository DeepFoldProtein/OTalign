"""
Run EBA (Embedding-based alignment, https://github.com/DeepFoldProtein/EBA) on a benchmark dataset.
Uses a protein language model (ProtT5 or ESMb1) for per-residue embeddings, then DTW alignment.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = PROJECT_ROOT / "third_party"
EBA_ROOT = THIRD_PARTY / "eba"
if not EBA_ROOT.exists():
    raise FileNotFoundError(f"EBA not found at {EBA_ROOT}. Run: git submodule update --init third_party/eba")
# So that "from eba import ..." works when run from project root (EBA package is eba/eba/)
sys.path.insert(0, str(EBA_ROOT))

from scripts.dataset_utils import alignment_metrics, iter_pairs_from_dataset


def _log_progress(msg: str, pbar: Optional[tqdm] = None) -> None:
    if pbar is not None:
        pbar.write(msg)
    else:
        print(msg, flush=True)


def _alignment_to_match_pairs(aln_1: np.ndarray, aln_2: np.ndarray) -> list:
    """Convert EBA DTW alignment (aln_1, aln_2; -1 = gap) to list of (seq1_idx, seq2_idx)."""
    pairs = []
    for k in range(len(aln_1)):
        i, j = int(aln_1[k]), int(aln_2[k])
        if i >= 0 and j >= 0:
            pairs.append((i, j))
    return pairs


def run_eba_evaluation(
    dataset: str,
    output: str,
    plm: str = "ProtT5",
    pbar: Optional[tqdm] = None,
    device: Optional[str] = None,
    l_sim: float = 1.0,
):
    """Run EBA evaluation on a dataset and write results to JSONL.

    Args:
        dataset: Dataset path (e.g. DeepFoldProtein/malidup-dataset,all,test).
        output: Output JSONL path.
        plm: PLM for embeddings: ProtT5, ESMb1, ESM2, or ProstT5.
        device: torch device ('cuda', 'cpu', or None for auto).
        l_sim: Similarity matrix regularization (EBA l parameter).
    """
    import torch
    from eba import methods
    from eba import plm_extractor as plm_ext
    from eba import score_matrices as sm

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    _log_progress(f"Loading EBA PLM ({plm}) on {device}...", pbar)
    extractor = plm_ext.load_extractor(plm, "residue", device=device)
    _log_progress("Aligning pairs...", pbar)

    dataset_list = list(iter_pairs_from_dataset(dataset))
    total = len(dataset_list)
    success_count = 0
    fail_count = 0
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    local_pbar = pbar is None
    if local_pbar:
        pbar = tqdm(total=total, desc="Aligning with EBA", mininterval=0.5, file=sys.stdout)
    else:
        pbar.total = total
        pbar.set_description("Aligning with EBA")
        pbar.reset()

    with out_path.open("w", encoding="utf-8") as fout:
        for ex in dataset_list:
            try:
                seq1 = ex["seq1"].strip().upper()
                seq2 = ex["seq2"].strip().upper()
                seq1_id = ex.get("seq1_id", "seq1")
                seq2_id = ex.get("seq2_id", "seq2")
                pair_id = ex.get("pair_id", f"{seq1_id}-{seq2_id}")

                emb1 = extractor.extract(seq1)
                emb2 = extractor.extract(seq2)
                similarity_matrix = sm.compute_similarity_matrix(emb1, emb2, l=l_sim)
                eba_results = methods.compute_eba(
                    similarity_matrix,
                    extensive_output=True,
                    gap_open_penalty=0.0,
                    gap_extend_penalty=0.0,
                )
                aln_1 = eba_results["aln_1"]
                aln_2 = eba_results["aln_2"]
                pairs = _alignment_to_match_pairs(aln_1, aln_2)
                met = alignment_metrics(pairs, ex.get("ref_alignment"))
                rec = {
                    "pair_id": pair_id,
                    "seq1_id": seq1_id,
                    "seq2_id": seq2_id,
                    "pred_alignment": pairs,
                    "metrics": met,
                    "meta": {"tool": "EBA", "plm": plm},
                }
                fout.write(json.dumps(rec) + "\n")
                success_count += 1
            except Exception as e:
                rec = {
                    "pair_id": ex.get("pair_id", f"{ex.get('seq1_id', '')}-{ex.get('seq2_id', '')}"),
                    "error": str(e),
                }
                logging.warning("Pair %s failed: %s", rec["pair_id"], e)
                fail_count += 1
            pbar.update(1)

    if local_pbar:
        pbar.close()
    logging.info("Evaluation Summary: %s/%s pairs processed successfully.", success_count, total)
    if fail_count > 0:
        logging.warning("Failed pairs: %s", fail_count)
    logging.info("[ok] wrote predictions -> %s", out_path)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Run EBA on a dataset and export JSONL predictions.")
    ap.add_argument("--dataset", type=str, required=True, help="Dataset (e.g. DeepFoldProtein/malidup-dataset,all,test)")
    ap.add_argument("--output", type=str, required=True, help="Output JSONL path")
    ap.add_argument("--plm", type=str, default="ProtT5", choices=["ProtT5", "ESMb1", "ESM2", "ProstT5"], help="PLM for embeddings")
    ap.add_argument("--device", type=str, default=None, help="Device: cuda, cpu, or empty for auto")
    ap.add_argument("--l", type=float, default=1.0, help="Similarity matrix regularization (l)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_eba_evaluation(
        dataset=args.dataset,
        output=args.output,
        plm=args.plm,
        device=args.device,
        l_sim=args.l,
    )


if __name__ == "__main__":
    main()
