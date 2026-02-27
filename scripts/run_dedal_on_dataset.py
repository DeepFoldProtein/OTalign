"""
Run DEDAL (https://github.com/DeepFoldProtein/dedal-fork) on a benchmark dataset.
Uses TensorFlow Hub model https://tfhub.dev/google/dedal/3.

Runs on CPU only to avoid CUDA_ERROR_UNSUPPORTED_PTX_VERSION on newer GPUs (e.g. RTX Ada).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm


# Add third_party so that "from dedal import ..." works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = PROJECT_ROOT / "third_party"
DEDAL_ROOT = THIRD_PARTY / "dedal"
if not DEDAL_ROOT.exists():
    raise FileNotFoundError(f"DEDAL not found at {DEDAL_ROOT}. Run: git submodule update --init third_party/dedal")
sys.path.insert(0, str(THIRD_PARTY))

from scripts.dataset_utils import alignment_metrics, iter_pairs_from_dataset


def _log_progress(msg: str, pbar: Optional[tqdm] = None) -> None:
    """Print a progress message without breaking tqdm bar (use tqdm.write)."""
    if pbar is not None:
        pbar.write(msg)
    else:
        print(msg, flush=True)


def _alignment_to_match_pairs(alignment) -> list:
    """Extract (seq1_idx, seq2_idx) match pairs from DEDAL Alignment.
    alignment.path is (len_1, len_2, 3) with [match, gap_open, gap_extend].
    """
    path = alignment.path  # numpy array (L1, L2, 3)
    match_mask = path[:, :, 0] > 0
    ij = np.argwhere(match_mask)[:, :2]
    return [(int(i), int(j)) for i, j in ij]


def run_dedal_evaluation(
    dataset: str,
    output: str,
    model_url: str = "https://tfhub.dev/google/dedal/3",
    max_length: int = 512,
    pbar: Optional[tqdm] = None,
    use_gpu: bool = True,
):
    """Run DEDAL evaluation on a dataset and write results to JSONL.

    Args:
        use_gpu: If False, force CPU (avoids PTX/CUDA issues on some GPUs).
                 If True, use GPU if available (faster but may fail on RTX Ada with TF 2.20+).
    """
    # Suppress TF INFO/WARNING so progress bar is visible
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    if not use_gpu:
        # Force CPU (avoids CUDA_ERROR_UNSUPPORTED_PTX_VERSION on RTX Ada etc.)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import tensorflow_hub as hub
    from dedal import infer

    device = "CPU" if not use_gpu else "GPU"
    _log_progress(f"Loading DEDAL model ({device})...", pbar)
    model = hub.load(model_url)
    _log_progress(f"Model loaded on {device}. Aligning pairs...", pbar)
    dataset_list = list(iter_pairs_from_dataset(dataset))
    total = len(dataset_list)
    success_count = 0
    fail_count = 0

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    local_pbar = pbar is None
    if local_pbar:
        pbar = tqdm(total=total, desc="Aligning with DEDAL", mininterval=0.5, file=sys.stdout)
    else:
        pbar.total = total
        pbar.set_description("Aligning with DEDAL")
        pbar.reset()

    with out_path.open("w", encoding="utf-8") as fout:
        for ex in dataset_list:
            try:
                seq1 = ex["seq1"]
                seq2 = ex["seq2"]
                seq1_id = ex.get("seq1_id", "seq1")
                seq2_id = ex.get("seq2_id", "seq2")
                pair_id = ex.get("pair_id", f"{seq1_id}-{seq2_id}")

                if len(seq1) > max_length or len(seq2) > max_length:
                    rec = {
                        "pair_id": pair_id,
                        "error": f"Sequence length > {max_length}; DEDAL supports up to 512.",
                    }
                    logging.warning(f"Pair {pair_id}: skipped (length)")
                    fail_count += 1
                    pbar.update(1)
                    continue

                alignment = infer.align(model, seq1, seq2, max_length=max_length)
                pairs = _alignment_to_match_pairs(alignment)
                met = alignment_metrics(pairs, ex.get("ref_alignment"))
                rec = {
                    "pair_id": pair_id,
                    "seq1_id": seq1_id,
                    "seq2_id": seq2_id,
                    "pred_alignment": pairs,
                    "metrics": met,
                    "meta": {"tool": "DEDAL", "model": model_url},
                }
                fout.write(json.dumps(rec) + "\n")
                success_count += 1
            except Exception as e:
                rec = {
                    "pair_id": ex.get("pair_id", f"{ex.get('seq1_id', '')}-{ex.get('seq2_id', '')}"),
                    "error": str(e),
                }
                logging.warning(f"Pair {rec['pair_id']} failed: {e}")
                fail_count += 1
            pbar.update(1)

    if local_pbar:
        pbar.close()
    logging.info(f"Evaluation Summary: {success_count}/{total} pairs processed successfully.")
    if fail_count > 0:
        logging.warning(f"Failed pairs: {fail_count}")
    logging.info(f"[ok] wrote predictions -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Run DEDAL on a dataset and export JSONL predictions.")
    ap.add_argument("--dataset", type=str, required=True, help="Dataset (e.g. DeepFoldProtein/malidup-dataset,all,test)")
    ap.add_argument("--output", type=str, required=True, help="Output JSONL path")
    ap.add_argument("--model_url", type=str, default="https://tfhub.dev/google/dedal/3", help="TF Hub DEDAL model URL")
    ap.add_argument("--max_length", type=int, default=512, help="Max sequence length (DEDAL default 512)")
    ap.add_argument("--use-gpu", action="store_true", default=False, help="Use GPU (default: CPU to avoid PTX issues)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_dedal_evaluation(
        dataset=args.dataset,
        output=args.output,
        model_url=args.model_url,
        max_length=args.max_length,
        use_gpu=args.use_gpu,
    )


if __name__ == "__main__":
    main()
