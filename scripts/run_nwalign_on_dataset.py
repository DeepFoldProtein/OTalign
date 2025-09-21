import argparse
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from otalign.baselines.nwalign import run_nwalign_for_pair
from scripts.dataset_utils import alignment_metrics, iter_pairs_from_dataset


def gapped_len_from_pairs(pairs: list[tuple[int, int]]) -> tuple[int, int]:
    if not pairs:
        return 0, 0
    i_max = max(i for i, _ in pairs)
    j_max = max(j for _, j in pairs)
    return i_max + 1, j_max + 1


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


def run_nwalign_evaluation(
    dataset: str,
    nwalign_bin: str,
    output: str,
    infmt1: int = 4,
    infmt2: int = 4,
    glocal: int = 0,
    extra_args: Optional[list] = None,
    out_dir: Optional[str] = None,
    workers: int = 4,
    pbar: Optional[tqdm] = None,
):
    """Runs NWalign evaluation on a dataset and writes results to a file."""
    if extra_args is None:
        extra_args = []

    dataset_iterator = list(iter_pairs_from_dataset(dataset))
    total_pairs = len(dataset_iterator)
    success_count = 0
    fail_count = 0

    args_dict = {
        "nwalign_bin": nwalign_bin,
        "infmt1": infmt1,
        "infmt2": infmt2,
        "glocal": glocal,
        "extra_args": extra_args,
        "out_dir": out_dir,
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup progress bar
    local_pbar = pbar is None
    if local_pbar:
        pbar = tqdm(total=total_pairs, desc="Aligning with NWalign")
    else:
        pbar.total = total_pairs
        pbar.set_description("Aligning with NWalign")
        pbar.reset()

    with out_path.open("w", encoding="utf-8") as fout:
        with mp.Pool(processes=workers) as pool:
            tasks = ((ex, args_dict) for ex in dataset_iterator)
            for rec in pool.imap_unordered(_worker, tasks, chunksize=4):
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

    run_nwalign_evaluation(
        dataset=args.dataset,
        nwalign_bin=args.nwalign_bin,
        output=args.output,
        infmt1=args.infmt1,
        infmt2=args.infmt2,
        glocal=args.glocal,
        extra_args=args.extra_args,
        out_dir=args.out_dir,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
