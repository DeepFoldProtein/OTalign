import argparse
import json
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from otalign.baselines.hhalign import run_hhalign_hhm_pair
from otalign.io.parser import gapped_to_pairs
from otalign.metrics.alignment import alignment_scores
from scripts.dataset_utils import iter_pairs_from_dataset


def metrics(pred: List[Tuple[int, int]], ref: Optional[List[List[int]]]) -> Dict[str, float]:
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
    ex, args = task
    try:
        q_hhm = Path(args["hhm_dir"]) / f"{ex['seq1_id']}.hhm"
        t_hhm = Path(args["hhm_dir"]) / f"{ex['seq2_id']}.hhm"
        if not q_hhm.exists() or not t_hhm.exists():
            return {"pair_id": ex["pair_id"], "error": "missing_hhm"}

        a1, a2, start1, start2 = run_hhalign_hhm_pair(
            q_hhm,
            t_hhm,
            hhalign_bin=args["hhalign_bin"],
            out_dir=args["out_dir"],
            mode=args["mode"],
            extra_args=args["extra_args"],
        )
        pairs = gapped_to_pairs(a1, a2, start1, start2)
        met = metrics(pairs, ex.get("ref_alignment"))
        return {
            "pair_id": ex["pair_id"],
            "seq1_id": ex["seq1_id"],
            "seq2_id": ex["seq2_id"],
            "pred_alignment": pairs,
            "metrics": met,
            "meta": {"tool": "HHalign", "mode": args["mode"]},
        }
    except Exception as e:
        return {"pair_id": ex["pair_id"], "error": str(e)}


def main():
    ap = argparse.ArgumentParser(description="Run HHalign (HMM-HMM) over a dataset using prebuilt HHM files.")
    ap.add_argument("--dataset", type=str, required=True, help="Path to the dataset (JSONL or HF identifier).")
    ap.add_argument("--hhm_dir", type=str, required=True, help="Directory containing *.hhm files")
    ap.add_argument("--hhalign_bin", type=str, default="hhalign")
    ap.add_argument("--mode", type=str, default="local", choices=["local", "global", "glocal"], help="Alignment mode. Tries a few flag patterns per mode until one works.")
    ap.add_argument("--extra_args", nargs="*", default=[], help="Additional flags passed to hhalign after mode flags")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    it = iter_pairs_from_dataset(args.dataset)

    args_dict = {
        "hhm_dir": args.hhm_dir,
        "hhalign_bin": args.hhalign_bin,
        "mode": args.mode,
        "extra_args": args.extra_args,
        "out_dir": args.out_dir,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fout:
        with mp.Pool(processes=args.workers) as pool:
            for rec in pool.imap_unordered(_worker, ((ex, args_dict) for ex in it), chunksize=4):
                fout.write(json.dumps(rec) + "\n")

    print(f"[ok] wrote predictions -> {out}")


if __name__ == "__main__":
    main()
