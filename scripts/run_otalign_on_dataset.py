import argparse
import itertools
import json
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path
from typing import Optional, cast

import torch
from datasets import load_dataset
from tqdm.auto import tqdm

from otalign.align.sinkhorn import SinkhornUOT
from otalign.align.uot_alignment import hard_alignment_from_transport, uot_alignment_metrics_with_sinkhorn
from otalign.metrics.alignment import alignment_scores
from otalign.models.embedding import get_embeddings_for_sequences


def iter_hf(dataset: str, name: str, split: str):
    ds = load_dataset(dataset, name=name, split=split)  # type: ignore
    for ex_raw in ds:
        ex = cast(dict, ex_raw)
        yield {
            "pair_id": ex.get("pair_id", f"{ex['seq1_id']}-{ex['seq2_id']}"),
            "seq1_id": ex["seq1_id"],
            "seq2_id": ex["seq2_id"],
            "seq1": ex["seq1"],
            "seq2": ex["seq2"],
            "ref_alignment": ex.get("ref_alignment"),
        }


def batch_iterator(iterable, batch_size):
    """Yields batches of a given size from an iterable."""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, batch_size))
        if not chunk:
            return
        yield chunk


def alignment_metrics(pred: list[tuple[int, int]], ref: Optional[list[list[int]]]) -> dict[str, float]:
    """
    Computes alignment scores (precision, recall, F1) between predicted and reference alignments.
    """
    ref_pairs = {cast(tuple[int, int], tuple(p)) for p in ref} if ref else set()
    pred_pairs = set(pred) if pred else set()
    metrics = alignment_scores(pred_pairs, ref_pairs)
    return asdict(metrics)


def _process_batch(batch: list[dict], args_dict: dict) -> list[dict]:
    """
    Processes a batch of sequence pairs on a GPU.
    """
    device = torch.device(args_dict["device"])
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    torch_dtype = dtype_map[args_dict["dtype"]]

    try:
        # 1. Get embeddings for all sequences in the batch
        seqs1 = [ex["seq1"] for ex in batch]
        seqs2 = [ex["seq2"] for ex in batch]
        seq_ids1 = [ex["seq1_id"] for ex in batch]
        seq_ids2 = [ex["seq2_id"] for ex in batch]

        all_seqs = seqs1 + seqs2
        all_seq_ids = seq_ids1 + seq_ids2

        embeddings = get_embeddings_for_sequences(
            sequences=all_seqs,
            seq_ids=all_seq_ids,
            model_name=args_dict["model"],
            cache_dir=args_dict["cache_dir"],
            device=args_dict["device"],
            batch_size=args_dict["align_batch_size"] * 2,
            dtype=args_dict["dtype"],
        )

        # 2. Pad and batch embeddings
        batch_size = len(batch)
        emb1_list = [e.to(device, dtype=torch_dtype) for e in embeddings[:batch_size]]
        emb2_list = [e.to(device, dtype=torch_dtype) for e in embeddings[batch_size:]]

        max_len1 = max(e.shape[0] for e in emb1_list) if emb1_list else 0
        max_len2 = max(e.shape[0] for e in emb2_list) if emb2_list else 0

        emb1_padded = torch.zeros(batch_size, max_len1, emb1_list[0].shape[1], device=device, dtype=torch_dtype)
        emb2_padded = torch.zeros(batch_size, max_len2, emb2_list[0].shape[1], device=device, dtype=torch_dtype)
        mask1 = torch.zeros(batch_size, max_len1, dtype=torch.bool, device=device)
        mask2 = torch.zeros(batch_size, max_len2, dtype=torch.bool, device=device)

        for i in range(batch_size):
            len1 = emb1_list[i].shape[0]
            emb1_padded[i, :len1] = emb1_list[i]
            mask1[i, :len1] = True

            len2 = emb2_list[i].shape[0]
            emb2_padded[i, :len2] = emb2_list[i]
            mask2[i, :len2] = True

        # 3. Initialize aligner
        aligner = SinkhornUOT()

        # 4. Perform alignments (A:B, A:A, B:B)
        reg = args_dict["reg"]
        lambda1 = args_dict["lambda1"]
        lambda2 = args_dict["lambda2"]
        num_iter = args_dict["num_iter"]

        res_ab = aligner(emb1_padded, emb2_padded, mask1, mask2, num_iter, reg, lambda1, lambda2)
        res_aa = aligner(emb1_padded, emb1_padded, mask1, mask1, num_iter, reg, lambda1, lambda2)
        res_bb = aligner(emb2_padded, emb2_padded, mask2, mask2, num_iter, reg, lambda1, lambda2)

        # 5. Post-process each pair in the batch
        records = []
        for i in range(batch_size):
            ex = batch[i]
            len1 = mask1[i].sum().item()
            len2 = mask2[i].sum().item()

            # Extract individual results, slicing off padding
            res_ab_i = {
                "transport_plan": res_ab["transport_plan"][i : i + 1, :len1, :len2],
                "scaling_u": res_ab["scaling_u"][i : i + 1, :len1],
                "scaling_v": res_ab["scaling_v"][i : i + 1, :len2],
                "cost_matrix": res_ab["cost_matrix"][i : i + 1, :len1, :len2],
                "mu": res_ab["mu"][i : i + 1, :len1],
                "nu": res_ab["nu"][i : i + 1, :len2],
            }
            res_aa_i = {
                "transport_plan": res_aa["transport_plan"][i : i + 1, :len1, :len1],
                "cost_matrix": res_aa["cost_matrix"][i : i + 1, :len1, :len1],
            }
            res_bb_i = {
                "transport_plan": res_bb["transport_plan"][i : i + 1, :len2, :len2],
                "cost_matrix": res_bb["cost_matrix"][i : i + 1, :len2, :len2],
            }
            mask1_i = mask1[i : i + 1, :len1]
            mask2_i = mask2[i : i + 1, :len2]

            # Compute OT metrics
            ot_metrics = uot_alignment_metrics_with_sinkhorn(
                a=res_ab_i["mu"],
                b=res_ab_i["nu"],
                cost_matrix=res_ab_i["cost_matrix"],
                transport_plan=res_ab_i["transport_plan"],
                mask_a=mask1_i,
                mask_b=mask2_i,
                plan_xx=res_aa_i["transport_plan"],
                cost_xx=res_aa_i["cost_matrix"],
                plan_yy=res_bb_i["transport_plan"],
                cost_yy=res_bb_i["cost_matrix"],
                reg=reg,
                lambda1=lambda1,
                lambda2=lambda2,
            )
            ot_metrics_serializable = {k: v.item() for k, v in ot_metrics.items()}

            # Get hard alignment
            P_np = res_ab_i["transport_plan"][0].cpu().numpy()
            f_np = res_ab_i["scaling_u"][0].log().cpu().numpy()
            g_np = res_ab_i["scaling_v"][0].log().cpu().numpy()

            hard_aln = hard_alignment_from_transport(
                P=P_np,
                f=f_np,
                g=g_np,
                mode=args_dict["dp_mode"],
                score_scale=args_dict["score_scale"],
                go_base=args_dict["go_base"],
                ge_base=args_dict["ge_base"],
            )

            pred_pairs = [(r - 1, c - 1) for r, c, op in hard_aln["path"] if op == "M"]

            # Compute standard alignment metrics
            std_metrics = alignment_metrics(pred_pairs, ex.get("ref_alignment"))

            # Assemble record
            rec = {
                "pair_id": ex.get("pair_id", f"{ex['seq1_id']}-{ex['seq2_id']}"),
                "seq1_id": ex["seq1_id"],
                "seq2_id": ex["seq2_id"],
                "pred_alignment": pred_pairs,
                "metrics": std_metrics,
                "ot_metrics": ot_metrics_serializable,
                "meta": {"tool": "OTAlign", "model": args_dict["model"], "params": {k: v for k, v in args_dict.items() if k != "device"}},
            }
            records.append(rec)

        return records

    except Exception as e:
        return [
            {
                "pair_id": ex.get("pair_id", f"{ex['seq1_id']}-{ex['seq2_id']}"),
                "error": str(e),
            }
            for ex in batch
        ]


def _worker(task):
    """
    Worker function to process a single sequence pair.
    """
    (ex, args_dict) = task
    try:
        # 1. Get embeddings for the pair
        seqs = [ex["seq1"], ex["seq2"]]
        seq_ids = [ex["seq1_id"], ex["seq2_id"]]

        embeddings = get_embeddings_for_sequences(
            sequences=seqs,
            seq_ids=seq_ids,
            model_name=args_dict["model"],
            cache_dir=args_dict["cache_dir"],
            device=args_dict["device"],
            batch_size=2,
            dtype=args_dict["dtype"],
        )

        emb1_np, emb2_np = embeddings[0], embeddings[1]

        # Convert to torch tensors
        device = torch.device(args_dict["device"])
        dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
        torch_dtype = dtype_map[args_dict["dtype"]]
        emb1 = torch.from_numpy(emb1_np).to(device, dtype=torch_dtype).unsqueeze(0)
        emb2 = torch.from_numpy(emb2_np).to(device, dtype=torch_dtype).unsqueeze(0)
        mask1 = torch.ones(emb1.shape[:2], dtype=torch.bool, device=device)
        mask2 = torch.ones(emb2.shape[:2], dtype=torch.bool, device=device)

        # 2. Initialize aligner
        aligner = SinkhornUOT()

        # 3. Perform alignments (A:B, A:A, B:B)
        reg = args_dict["reg"]
        lambda1 = args_dict["lambda1"]
        lambda2 = args_dict["lambda2"]
        num_iter = args_dict["num_iter"]

        res_ab = aligner(emb1, emb2, mask1, mask2, num_iter, reg, lambda1, lambda2)
        res_aa = aligner(emb1, emb1, mask1, mask1, num_iter, reg, lambda1, lambda2)
        res_bb = aligner(emb2, emb2, mask2, mask2, num_iter, reg, lambda1, lambda2)

        # 4. Compute OT metrics
        ot_metrics = uot_alignment_metrics_with_sinkhorn(
            a=res_ab["mu"],
            b=res_ab["nu"],
            cost_matrix=res_ab["cost_matrix"],
            transport_plan=res_ab["transport_plan"],
            mask_a=mask1,
            mask_b=mask2,
            plan_xx=res_aa["transport_plan"],
            cost_xx=res_aa["cost_matrix"],
            plan_yy=res_bb["transport_plan"],
            cost_yy=res_bb["cost_matrix"],
            reg=reg,
            lambda1=lambda1,
            lambda2=lambda2,
        )
        ot_metrics_serializable = {k: v.item() for k, v in ot_metrics.items()}

        # 5. Get hard alignment
        P_np = res_ab["transport_plan"][0].cpu().numpy()
        f_np = res_ab["scaling_u"][0].log().cpu().numpy()
        g_np = res_ab["scaling_v"][0].log().cpu().numpy()

        hard_aln = hard_alignment_from_transport(
            P=P_np,
            f=f_np,
            g=g_np,
            mode=args_dict["dp_mode"],
            score_scale=args_dict["score_scale"],
            go_base=args_dict["go_base"],
            ge_base=args_dict["ge_base"],
        )

        # Convert 1-based path from DP to 0-based pairs
        pred_pairs = [(i - 1, j - 1) for i, j, op in hard_aln["path"] if op == "M"]

        # 6. Compute standard alignment metrics
        std_metrics = alignment_metrics(pred_pairs, ex.get("ref_alignment"))

        # 7. Assemble record
        rec = {
            "pair_id": ex.get("pair_id", f"{ex['seq1_id']}-{ex['seq2_id']}"),
            "seq1_id": ex["seq1_id"],
            "seq2_id": ex["seq2_id"],
            "pred_alignment": pred_pairs,
            "metrics": std_metrics,
            "ot_metrics": ot_metrics_serializable,
            "meta": {"tool": "OTAlign", "model": args_dict["model"], "params": args_dict},
        }
        return rec

    except Exception as e:
        return {
            "pair_id": ex.get("pair_id", f"{ex['seq1_id']}-{ex['seq2_id']}"),
            "error": str(e),
        }


def main():
    ap = argparse.ArgumentParser(description="Run OTAlign over a dataset and export JSONL predictions.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf_dataset", type=str, help="e.g. DeepFoldProtein/SABmark-dataset")
    ap.add_argument("--name", type=str, default=None, help="HF config name, e.g. twi")
    ap.add_argument("--split", type=str, default="test")
    src.add_argument("--jsonl", type=str, help="Path to a JSONL file with sequence pairs.")

    # Model and cache
    ap.add_argument("--model", required=True, help="Name of the PLM to use (e.g., 'ankh-base').")
    ap.add_argument("--cache_dir", required=True, help="Directory for embedding cache.")
    ap.add_argument("--device", type=str, default="cpu", help="Device to run inference on.")
    ap.add_argument("--dtype", default="fp32", choices=["fp16", "fp32", "bf16"])

    # OTAlign parameters
    ap.add_argument("--reg", type=float, default=0.1)
    ap.add_argument("--lambda1", type=float, default=1.0)
    ap.add_argument("--lambda2", type=float, default=1.0)
    ap.add_argument("--num_iter", type=int, default=1000)

    # DP parameters
    ap.add_argument("--dp_mode", type=str, default="global", choices=["global", "glocal", "local"])
    ap.add_argument("--score_scale", type=float, default=1.0)
    ap.add_argument("--go_base", type=float, default=8.0, help="Base gap open penalty.")
    ap.add_argument("--ge_base", type=float, default=1.0, help="Base gap extend penalty.")

    # Output and processing
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--align_batch_size", type=int, default=16, help="Batch size for GPU alignment.")
    ap.add_argument("--output", type=str, required=True, help="Path to write output JSONL file.")
    args = ap.parse_args()

    # Load dataset iterator
    if args.jsonl:
        items = [json.loads(line) for line in open(args.jsonl, "r", encoding="utf-8")]
        it = (
            {
                "pair_id": ex.get("pair_id", f"{ex['seq1_id']}-{ex['seq2_id']}"),
                "seq1_id": ex["seq1_id"],
                "seq2_id": ex["seq2_id"],
                "ref_alignment": ex.get("ref_alignment"),
            }
            for ex in items
        )
    else:
        if load_dataset is None:
            raise RuntimeError("datasets is not installed; install `datasets` or use --jsonl")
        it = iter_hf(args.hf_dataset, args.name, args.split)  # type: ignore

    args_dict = vars(args)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = ((ex, args_dict) for ex in it)

    with out_path.open("w", encoding="utf-8") as fout:
        if args.device == "cpu":
            # Use multiprocessing for CPU-bound tasks
            tasks = ((ex, args_dict) for ex in it)
            # Get total for tqdm
            total = len(items) if args.jsonl else len(load_dataset(args.hf_dataset, args.name, split=args.split))  # type: ignore
            with mp.Pool(processes=args.workers) as pool:
                for rec in tqdm(pool.imap_unordered(_worker, tasks, chunksize=4), total=total, desc="Aligning pairs (CPU)"):
                    fout.write(json.dumps(rec) + "\n")
        else:
            # Use batching for GPU-bound tasks
            batch_size = args.align_batch_size
            batched_it = batch_iterator(it, batch_size)

            # Determine total number of items for tqdm
            if args.jsonl:
                total = len(items)
            else:
                # Re-load dataset to get length for progress bar
                ds = load_dataset(args.hf_dataset, args.name, split=args.split)
                total = len(ds)  # type: ignore

            with tqdm(total=total, desc=f"Aligning pairs (GPU, batch_size={batch_size})") as pbar:
                for batch in batched_it:
                    records = _process_batch(list(batch), args_dict)
                    for rec in records:
                        fout.write(json.dumps(rec) + "\n")
                    pbar.update(len(batch))

    print(f"[ok] wrote predictions -> {out_path}")


if __name__ == "__main__":
    main()
