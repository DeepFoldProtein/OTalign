import argparse
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from otalign.baselines.hhalign import run_hhalign_hhm_pair
from otalign.io.parser import gapped_to_pairs
from scripts.dataset_utils import alignment_metrics, iter_pairs_from_dataset


def _worker(task):
    ex, args = task
    try:
        q_hhm = Path(args["hhm_dir"]) / f"{ex['seq1_id']}.hhm"
        t_hhm = Path(args["hhm_dir"]) / f"{ex['seq2_id']}.hhm"
        if not q_hhm.exists() or not t_hhm.exists():
            return {"pair_id": ex["pair_id"], "error": "missing_hhm"}

        a1, a2, start1, start2, hit = run_hhalign_hhm_pair(
            q_hhm,
            t_hhm,
            hhalign_bin=args["hhalign_bin"],
            out_dir=args["out_dir"],
            mode=args["mode"],
            extra_args=args["extra_args"],
        )
        pairs = gapped_to_pairs(a1, a2, start1, start2)
        met = alignment_metrics(pairs, ex.get("ref_alignment"))
        return {
            "pair_id": ex["pair_id"],
            "seq1_id": ex["seq1_id"],
            "seq2_id": ex["seq2_id"],
            "pred_alignment": pairs,
            "metrics": met,
            "meta": {"tool": "HHalign", "mode": args["mode"], "hit": hit},
        }
    except Exception as e:
        return {"pair_id": ex["pair_id"], "error": str(e)}


def run_hhalign_evaluation(
    dataset: str,
    hhm_dir: str,
    hhalign_bin: str,
    output: str,
    mode: str = "local",
    extra_args: Optional[list] = None,
    out_dir: Optional[str] = None,
    workers: int = 4,
    pbar: Optional[tqdm] = None,
):
    """Runs HHalign evaluation on a dataset and writes results to a file."""
    if extra_args is None:
        extra_args = []

    dataset_iterator = list(iter_pairs_from_dataset(dataset))
    total_pairs = len(dataset_iterator)
    success_count = 0
    fail_count = 0

    args_dict = {
        "hhm_dir": hhm_dir,
        "hhalign_bin": hhalign_bin,
        "mode": mode,
        "extra_args": extra_args,
        "out_dir": out_dir,
    }

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Setup progress bar
    local_pbar = pbar is None
    if local_pbar:
        pbar = tqdm(total=total_pairs, desc="Aligning with HHalign")
    else:
        pbar.total = total_pairs
        pbar.set_description("Aligning with HHalign")
        pbar.reset()

    with out.open("w", encoding="utf-8") as fout:
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
    logging.info(f"[ok] wrote predictions -> {out}")


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

    run_hhalign_evaluation(
        dataset=args.dataset,
        hhm_dir=args.hhm_dir,
        hhalign_bin=args.hhalign_bin,
        output=args.output,
        mode=args.mode,
        extra_args=args.extra_args,
        out_dir=args.out_dir,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
