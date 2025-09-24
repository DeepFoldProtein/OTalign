import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm


try:
    from deepblast.dataset.utils import states2alignment as to_alignment
    from deepblast.utils import load_model as load_deepblast
except ImportError as e:
    raise e

from otalign.io.parser import gapped_to_pairs
from scripts.dataset_utils import alignment_metrics, iter_pairs_from_dataset


@torch.no_grad()
def _worker(task):
    (ex, args_dict) = task
    try:
        model = args_dict["model"]
        x = ex["seq1"]
        y = ex["seq2"]
        s = model.align(x, y)
        a1, a2 = to_alignment(s, x, y)

        pairs = gapped_to_pairs(a1, a2)
        met = alignment_metrics(pairs, ex.get("ref_alignment"))
        rec = {
            "pair_id": ex.get("pair_id", f"{ex['seq1_id']}-{ex['seq2_id']}"),
            "seq1_id": ex["seq1_id"],
            "seq2_id": ex["seq2_id"],
            "pred_alignment": pairs,
            "metrics": met,
            "meta": {"tool": "DeepBlast", "alignment_mode": args_dict["alignment_mode"], "checkpoint": args_dict["checkpoint"]},
        }
        return rec
    except Exception as e:
        return {"pair_id": ex.get("pair_id", f"{ex['seq1_id']}-{ex['seq2_id']}"), "error": str(e)}


def run_deepblast_evaluation(
    dataset: str,
    deepblast_ckpt: str,
    output: str,
    alignment_mode: str,
    device: str = "cuda",
    out_dir: Optional[str] = None,
    pbar: Optional[tqdm] = None,
):
    """Runs DeepBlast evaluation on a dataset and writes results to a file."""

    model = load_deepblast(deepblast_ckpt, pretrain_path="Rostlab/prot_t5_xl_uniref50", device=device, alignment_mode=alignment_mode)

    dataset_iterator = list(iter_pairs_from_dataset(dataset))
    total_pairs = len(dataset_iterator)
    success_count = 0
    fail_count = 0

    args_dict = {
        "model": model,
        "checkpoint": deepblast_ckpt,
        "alignment_mode": alignment_mode,
        "out_dir": out_dir,
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup progress bar
    local_pbar = pbar is None
    if local_pbar:
        pbar = tqdm(total=total_pairs, desc="Aligning with DeepBLAST")
    else:
        pbar.total = total_pairs
        pbar.set_description("Aligning with DeepBLAST")
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
    ap = argparse.ArgumentParser(description="Run NWalign over a dataset and export JSONL predictions.")
    ap.add_argument("--dataset", type=str, required=True, help="Path to the dataset (JSONL or HF identifier).")
    ap.add_argument("--deepblast_ckpt", type=str, default="deepblast-v3.ckpt")
    ap.add_argument("--mode", type=str, default="needleman-wunsch", choices=["smith-waterman", "needleman-wunsch"])
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    run_deepblast_evaluation(dataset=args.dataset, deepblast_ckpt=args.deepblast_ckpt, alignment_mode=args.mode, output=args.output, out_dir=args.out_dir, device=args.device)


if __name__ == "__main__":
    main()
