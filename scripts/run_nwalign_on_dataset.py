import argparse
import json
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from otalign.baselines.nwalign import run_nwalign_for_pair
from otalign.metrics.alignment import alignment_scores
from scripts.dataset_utils import iter_pairs_from_dataset


def gapped_len_from_pairs(pairs: List[Tuple[int, int]]) -> Tuple[int, int]:
    if not pairs:
        return 0, 0
    i_max = max(i for i, _ in pairs)
    j_max = max(j for _, j in pairs)
    return i_max + 1, j_max + 1


def alignment_metrics(pred: List[Tuple[int, int]], ref: Optional[List[List[int]]]) -> Dict[str, float]:
    ref_pairs = set()
    if ref:
        for x, y in ref:
            ref_pairs.add((x, y))
    pred_pairs = set()
    if pred:
        for x, y in pred:
            pred_pairs.add((x, y))
    metrics = alignment_scores(pred_pairs, ref_pairs)
    return asdict(metrics)


def _worker(task):
    (ex, args_dict) = task
    try:
        pairs, a1, a2 = run_nwalign_for_pair(
            ex["seq1_id"],
            ex["seq1"],
            ex["seq2_id"],
            ex["seq2"],
            nwalign_bin=args_dict["nwalign_bin"],
            out_dir=args_dict["out_dir"],
            infmt1=args_dict["infmt1"],
            infmt2=args_dict["infmt2"],
            glocal=args_dict["glocal"],
            extra_args=args_dict["extra_args"],
        )
        met = alignment_metrics(pairs, ex.get("ref_alignment"))
        rec = {
            "pair_id": ex.get("pair_id", f"{ex['seq1_id']}-{ex['seq2_id']}"),
            "seq1_id": ex["seq1_id"],
            "seq2_id": ex["seq2_id"],
            "pred_alignment": pairs,
            "metrics": met,
            "meta": {"tool": "NWalign", "glocal": args_dict["glocal"]},
        }
        return rec
    except Exception as e:
        return {
            "pair_id": ex.get("pair_id", f"{ex['seq1_id']}-{ex['seq2_id']}"),
            "error": str(e),
        }


def main():
    ap = argparse.ArgumentParser(description="Run NWalign over a dataset and export JSONL predictions.")
    ap.add_argument("--dataset", type=str, required=True, help="Path to the dataset (JSONL or HF identifier).")
    ap.add_argument("--nwalign_bin", type=str, default="NWalign")
    ap.add_argument("--infmt1", type=int, default=4)
    ap.add_argument("--infmt2", type=int, default=4)
    ap.add_argument("--glocal", type=int, default=0, help="0=global, 1=glocal")
    ap.add_argument("--extra_args", nargs="*", default=[])
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    it = iter_pairs_from_dataset(args.dataset)

    args_dict = {
        "nwalign_bin": args.nwalign_bin,
        "infmt1": args.infmt1,
        "infmt2": args.infmt2,
        "glocal": args.glocal,
        "extra_args": args.extra_args,
        "out_dir": args.out_dir,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        with mp.Pool(processes=args.workers) as pool:
            for rec in pool.imap_unordered(_worker, ((ex, args_dict) for ex in it), chunksize=4):
                fout.write(json.dumps(rec) + "\n")

    print(f"[ok] wrote predictions -> {out_path}")


if __name__ == "__main__":
    main()
