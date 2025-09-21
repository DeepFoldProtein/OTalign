import argparse
import itertools
import json
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union, cast

import numpy as np
import torch
from tqdm.auto import tqdm

from otalign.align.sinkhorn import SinkhornUOT
from otalign.align.uot_alignment import hard_alignment_from_transport, uot_alignment_metrics_with_sinkhorn
from otalign.cache.lmdb_reader import LMDBCache
from otalign.cache.npz_reader import NPZCache
from otalign.io.fasta_utils import reconstruct_alignment
from otalign.metrics.alignment import alignment_scores, in_band_mass, recall_in_band
from otalign.models.embedding import get_embeddings_for_sequences
from otalign.models.plm_adaptors import get_plm_adaptor_and_configs
from otalign.quantize import quantize
from otalign.utils.checkpointing import load_peft_model_from_checkpoint
from scripts.dataset_utils import iter_pairs_from_dataset


# Global cache for multiprocessing workers
AnyCache = Union[NPZCache, LMDBCache, None]
worker_cache: AnyCache = None
worker_model_objects: dict = {}


def init_worker(cache_dir: Optional[str], cache_type: Optional[str], args_dict: dict):
    """Initializer for multiprocessing pool to create a shared cache object and model."""
    global worker_cache, worker_model_objects
    if cache_dir and cache_type:
        if cache_type == "lmdb":
            worker_cache = LMDBCache(cache_dir)
        else:
            worker_cache = NPZCache(cache_dir)
    else:
        worker_cache = None

    device = torch.device(args_dict["device"])
    model_path = Path(args_dict["model"])
    model = None
    model_name_for_adaptor = args_dict["model"]
    adaptor = None
    if model_path.is_dir():
        base_model_name = args_dict["base_model_for_checkpoint"]
        plm_adaptor, _, _ = get_plm_adaptor_and_configs(base_model_name, for_masked_lm=True)
        if plm_adaptor:
            model = load_peft_model_from_checkpoint(plm_adaptor.model, str(model_path)).to(device)
            plm_adaptor.model = model  # Use the loaded PEFT model in the adaptor
            model_name_for_adaptor = base_model_name
            adaptor = plm_adaptor

    worker_model_objects["model"] = model
    worker_model_objects["model_name_for_adaptor"] = model_name_for_adaptor
    worker_model_objects["adaptor"] = adaptor


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


def _process_batch(
    batch: list[dict],
    args_dict: dict,
    cache: AnyCache,
    model: Optional[torch.nn.Module],
    model_name_for_adaptor: str,
    adaptor: Optional[torch.nn.Module] = None,
) -> list[dict]:
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
            model_name=model_name_for_adaptor,
            cache_dir=args_dict["cache_dir"],
            device=args_dict["device"],
            batch_size=args_dict["align_batch_size"] * 2,
            dtype=args_dict["dtype"],
            cache=cache,
            model=model,
            adaptor=adaptor,
        )

        # 3. Pad and batch embeddings
        batch_size = len(batch)
        emb1_list = embeddings[:batch_size]
        emb2_list = embeddings[batch_size:]

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

            # Save transport plan if requested
            save_transport_plan_dir_str = args_dict.get("save_transport_plan_dir")
            if save_transport_plan_dir_str:
                Path(save_transport_plan_dir_str).mkdir(parents=True, exist_ok=True)
                plan_dtype = np.uint8 if args_dict["plan_dtype"] == "uint8" else np.uint16
                quantized_data = quantize(P_np, dtype=plan_dtype)
                pair_id = ex.get("pair_id", f"{ex['seq1_id']}-{ex['seq2_id']}")
                output_path = Path(save_transport_plan_dir_str) / f"{pair_id}.npz"
                np.savez_compressed(output_path, **quantized_data, f=f_np, g=g_np)

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

            # Compute new metrics if reference alignment is available
            new_metrics = {}
            ref_alignment = ex.get("ref_alignment")
            if ref_alignment:
                true_plan = torch.zeros_like(res_ab_i["transport_plan"][0], device=device)
                for r, c in ref_alignment:
                    if r < true_plan.shape[0] and c < true_plan.shape[1]:
                        true_plan[r, c] = 1.0

                # Normalize transport plan for in-band mass
                plan_sum = res_ab_i["transport_plan"][0].sum().clamp(min=1e-8)
                normalized_plan = res_ab_i["transport_plan"][0] / plan_sum
                precision_like_score = in_band_mass(normalized_plan, true_plan, band_width=args_dict["eval_band_width"])
                recall_like_score = recall_in_band(normalized_plan, true_plan, band_width=args_dict["eval_band_width"])
                new_metrics = {f"in_band_mass_w_{args_dict['eval_band_width']}": precision_like_score, f"recall_in_band_w_{args_dict['eval_band_width']}": recall_like_score}

            # Assemble record
            rec = {
                "pair_id": ex.get("pair_id", f"{ex['seq1_id']}-{ex['seq2_id']}"),
                "seq1_id": ex["seq1_id"],
                "seq2_id": ex["seq2_id"],
                "pred_alignment": pred_pairs,
                "metrics": std_metrics,
                "new_metrics": new_metrics,
                "ot_metrics": ot_metrics_serializable,
                "meta": {"tool": "OTAlign", "model": args_dict["model"], "params": {k: v for k, v in args_dict.items() if k != "device"}},
            }

            # Export FASTA if requested
            fasta_export_dir_str = args_dict.get("export_fasta_dir")
            if fasta_export_dir_str and "pred_alignment" in rec:
                aligned_seq1, aligned_seq2 = reconstruct_alignment(ex["seq1"], ex["seq2"], rec["pred_alignment"])
                fasta_content = f">{rec['seq1_id']}\n{aligned_seq1}\n>{rec['seq2_id']}\n{aligned_seq2}\n"
                output_path = Path(fasta_export_dir_str) / f"{rec['pair_id']}.fasta"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(fasta_content)

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
        # 1. Get embeddings
        device = torch.device(args_dict["device"])
        dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
        torch_dtype = dtype_map[args_dict["dtype"]]

        seqs = [ex["seq1"], ex["seq2"]]
        seq_ids = [ex["seq1_id"], ex["seq2_id"]]

        embeddings = get_embeddings_for_sequences(
            sequences=seqs,
            seq_ids=seq_ids,
            model_name=worker_model_objects["model_name_for_adaptor"],
            cache_dir=args_dict["cache_dir"],
            device=args_dict["device"],
            batch_size=2,
            dtype=args_dict["dtype"],
            cache=worker_cache,
            model=worker_model_objects["model"],
            adaptor=worker_model_objects.get("adaptor"),
        )
        emb1_tensor, emb2_tensor = embeddings[0], embeddings[1]

        # Convert to torch tensors
        device = torch.device(args_dict["device"])
        dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
        torch_dtype = dtype_map[args_dict["dtype"]]
        emb1 = emb1_tensor.to(device, dtype=torch_dtype).unsqueeze(0)
        emb2 = emb2_tensor.to(device, dtype=torch_dtype).unsqueeze(0)
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

        # Save transport plan if requested
        save_transport_plan_dir_str = args_dict.get("save_transport_plan_dir")
        if save_transport_plan_dir_str:
            Path(save_transport_plan_dir_str).mkdir(parents=True, exist_ok=True)
            plan_dtype = np.uint8 if args_dict["plan_dtype"] == "uint8" else np.uint16
            quantized_data = quantize(P_np, dtype=plan_dtype)
            pair_id = ex.get("pair_id", f"{ex['seq1_id']}-{ex['seq2_id']}")
            output_path = Path(save_transport_plan_dir_str) / f"{pair_id}.npz"
            np.savez_compressed(output_path, **quantized_data, f=f_np, g=g_np)

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

        # 7. Compute new metrics
        new_metrics = {}
        ref_alignment = ex.get("ref_alignment")
        if ref_alignment:
            true_plan = torch.zeros_like(res_ab["transport_plan"][0], device=device)
            for r, c in ref_alignment:
                if r < true_plan.shape[0] and c < true_plan.shape[1]:
                    true_plan[r, c] = 1.0

            plan_sum = res_ab["transport_plan"][0].sum().clamp(min=1e-8)
            normalized_plan = res_ab["transport_plan"][0] / plan_sum
            precision_like_score = in_band_mass(normalized_plan, true_plan, band_width=args_dict["eval_band_width"])
            recall_like_score = recall_in_band(normalized_plan, true_plan, band_width=args_dict["eval_band_width"])
            new_metrics = {f"in_band_mass_w_{args_dict['eval_band_width']}": precision_like_score, f"recall_in_band_w_{args_dict['eval_band_width']}": recall_like_score}

        # 8. Assemble record
        rec = {
            "pair_id": ex.get("pair_id", f"{ex['seq1_id']}-{ex['seq2_id']}"),
            "seq1_id": ex["seq1_id"],
            "seq2_id": ex["seq2_id"],
            "pred_alignment": pred_pairs,
            "metrics": std_metrics,
            "new_metrics": new_metrics,
            "ot_metrics": ot_metrics_serializable,
            "meta": {
                "tool": "OTAlign",
                "model": args_dict["model"],
                "params": {k: v for k, v in args_dict.items() if k != "device"},
            },
        }

        # Export FASTA if requested
        fasta_export_dir_str = args_dict.get("export_fasta_dir")
        if fasta_export_dir_str and "pred_alignment" in rec:
            aligned_seq1, aligned_seq2 = reconstruct_alignment(ex["seq1"], ex["seq2"], rec["pred_alignment"])
            fasta_content = f">{rec['seq1_id']}\n{aligned_seq1}\n>{rec['seq2_id']}\n{aligned_seq2}\n"
            output_path = Path(fasta_export_dir_str) / f"{rec['pair_id']}.fasta"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(fasta_content)

        return rec

    except Exception as e:
        return {
            "pair_id": ex.get("pair_id", f"{ex['seq1_id']}-{ex['seq2_id']}"),
            "error": str(e),
        }


def main():
    ap = argparse.ArgumentParser(description="Run OTAlign over a dataset and export JSONL predictions.")
    ap.add_argument("--dataset", type=str, required=True, help="Path to the dataset (JSONL or HF identifier).")
    # Model and cache
    ap.add_argument("--model", required=True, help="Name of the PLM to use (e.g., 'ankh-base') or path to a PEFT checkpoint directory.")
    ap.add_argument("--base_model_for_checkpoint", type=str, help="Base model name if --model is a checkpoint path.")
    ap.add_argument("--cache_dir", help="Directory for embedding cache.")
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

    # New metric parameters
    ap.add_argument("--eval_band_width", type=int, default=2, help="Band width for In-band mass.")

    # Output and processing
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--align_batch_size", type=int, default=16, help="Batch size for GPU alignment.")
    ap.add_argument("--output", type=str, required=True, help="Path to write output JSONL file.")
    ap.add_argument("--export_fasta_dir", type=str, default=None, help="If provided, export alignments as FASTA files to this directory.")
    ap.add_argument("--save_transport_plan_dir", type=str, default=None, help="If provided, save transport plans to this directory.")
    ap.add_argument("--plan_dtype", type=str, default="uint8", choices=["uint8", "uint16"], help="Data type for quantizing transport plans.")
    ap.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars.")
    args = ap.parse_args()

    # Add logging about the model being used
    model_path = Path(args.model)
    if model_path.is_dir():
        print(f"INFO: Loading fine-tuned checkpoint from: {args.model}")
        if not args.base_model_for_checkpoint:
            raise ValueError("--base_model_for_checkpoint is required when --model is a directory.")
        print(f"INFO: Using base model for checkpoint: {args.base_model_for_checkpoint}")
    else:
        print(f"INFO: Using base model: {args.model}")

    # Load dataset iterator
    items = list(iter_pairs_from_dataset(args.dataset))
    total_pairs = len(items)
    it = iter(items)

    args_dict = vars(args)
    success_count = 0
    fail_count = 0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Create FASTA export directory if specified
    if args.export_fasta_dir:
        Path(args.export_fasta_dir).mkdir(parents=True, exist_ok=True)

    # Create transport plan export directory if specified
    if args.save_transport_plan_dir:
        Path(args.save_transport_plan_dir).mkdir(parents=True, exist_ok=True)

    # Auto-detect cache type and initialize
    if args.cache_dir is not None:
        cache_dir_path = Path(args.cache_dir)
        if (cache_dir_path / "data.lmdb").exists():
            cache = LMDBCache(args.cache_dir)
            cache_type = "lmdb"
            print("Using LMDBCache")
        else:
            cache = NPZCache(args.cache_dir)
            cache_type = "npz"
            print("Using NPZCache")
    else:
        cache = None
        cache_type = None

    with out_path.open("w", encoding="utf-8") as fout:
        model_path = Path(args.model)
        is_dl_model = model_path.is_dir()

        if args.device == "cpu" and is_dl_model:
            # Process pairs sequentially for DL models on CPU
            print("INFO: Using deep learning model on CPU, processing sequentially.")
            init_worker(args.cache_dir, cache_type, args_dict)
            with tqdm(items, total=total_pairs, desc="Aligning pairs (CPU, sequential)", disable=args.no_tqdm) as pbar:
                for ex in pbar:
                    pair_id = ex.get("pair_id", "N/A")
                    tqdm.write(f"INFO: Processing pair: {pair_id}")
                    rec = _worker((ex, args_dict))
                    if "error" in rec:
                        tqdm.write(f"ERROR: Pair {pair_id} failed with error: {rec['error']}")
                        fail_count += 1
                    else:
                        fout.write(json.dumps(rec) + "\n")
                        success_count += 1

        elif args.device == "cpu" and not is_dl_model:
            # Use multiprocessing for non-DL CPU-bound tasks
            print("INFO: Using multiprocessing for CPU-bound tasks.")
            tasks = ((ex, args_dict) for ex in it)
            with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(args.cache_dir, cache_type, args_dict)) as pool:
                for rec in tqdm(
                    pool.imap_unordered(_worker, tasks, chunksize=32),
                    total=total_pairs,
                    desc="Aligning pairs (CPU)",
                    disable=args.no_tqdm,
                ):
                    if "error" in rec:
                        fail_count += 1
                    else:
                        fout.write(json.dumps(rec) + "\n")
                        success_count += 1
        else:
            # Use batching for GPU-bound tasks
            print("INFO: Using GPU for batch processing.")
            device = torch.device(args.device)
            model = None
            model_name_for_adaptor = args.model
            adaptor = None
            if is_dl_model:
                base_model_name = args.base_model_for_checkpoint
                plm_adaptor, _, _ = get_plm_adaptor_and_configs(base_model_name, for_masked_lm=bool(args.base_model_for_checkpoint))
                if plm_adaptor:
                    model = load_peft_model_from_checkpoint(plm_adaptor.model, str(model_path)).to(device)
                    plm_adaptor.model = model
                    model_name_for_adaptor = base_model_name
                    adaptor = plm_adaptor

            if is_dl_model and not adaptor:
                raise ValueError("Failed to initialize PLM adaptor for GPU processing.")

            batch_size = args.align_batch_size
            batched_it = batch_iterator(it, batch_size)

            with tqdm(total=total_pairs, desc=f"Aligning pairs (GPU, batch_size={batch_size})", disable=args.no_tqdm) as pbar:
                for batch in batched_it:
                    records = _process_batch(list(batch), args_dict, cache, model, model_name_for_adaptor, adaptor)
                    for rec in records:
                        if "error" in rec:
                            fail_count += 1
                        else:
                            fout.write(json.dumps(rec) + "\n")
                            success_count += 1
                    pbar.update(len(batch))

    print(f"  - Evaluation Summary: {success_count}/{total_pairs} pairs processed successfully.")
    if fail_count > 0:
        print(f"  - Failed pairs: {fail_count}")
    print(f"[ok] wrote predictions -> {out_path}")


if __name__ == "__main__":
    main()
