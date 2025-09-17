import argparse
import json
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from datasets import load_dataset
from otalign.baselines.nwalign import run_nwalign_for_pair
from otalign.metrics.alignment import alignment_scores


def load_pairs_from_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def iter_pairs_hf(dataset_name: str, name: str, split: str):
    ds = load_dataset(dataset_name, name=name, split=split)  # type: ignore
    for ex_raw in ds:
        ex = cast(dict, ex_raw)
        yield {
            "pair_id": ex.get("pair_id"),
            "seq1_id": ex["seq1_id"],
            "seq2_id": ex["seq2_id"],
            "seq1": ex["seq1"],
            "seq2": ex["seq2"],
            "ref_alignment": ex.get("ref_alignment"),
        }


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
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf_dataset", type=str, help="e.g. DeepFoldProtein/SABmark")
    ap.add_argument("--name", type=str, default=None, help="HF config name, e.g. twi")
    ap.add_argument("--split", type=str, default="test")
    src.add_argument("--jsonl", type=str, help="Path to SABmark/MALIDUP/MALISAM JSONL file")

    ap.add_argument("--nwalign_bin", type=str, default="NWalign")
    ap.add_argument("--infmt1", type=int, default=4)
    ap.add_argument("--infmt2", type=int, default=4)
    ap.add_argument("--glocal", type=int, default=0, help="0=global, 1=glocal")
    ap.add_argument("--extra_args", nargs="*", default=[])

    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if args.jsonl:
        items = load_pairs_from_jsonl(args.jsonl)
        it = (
            {
                "pair_id": ex.get("pair_id"),
                "seq1_id": ex["seq1_id"],
                "seq2_id": ex["seq2_id"],
                "seq1": ex["seq1"],
                "seq2": ex["seq2"],
                "ref_alignment": ex.get("ref_alignment"),
            }
            for ex in items
        )
    else:
        if load_dataset is None:
            raise RuntimeError("datasets is not installed; install `datasets` or use --jsonl")
        it = iter_pairs_hf(args.hf_dataset, args.name, args.split)  # type: ignore

    args_dict = {
        "nwalign_bin": args.nwalign_bin,
        "infmt1": args.infmt1,
        "infmt2": args.infmt2,
        "glocal": args.glocal,
        "extra_args": args.extra_args,
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
