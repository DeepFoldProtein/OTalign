import argparse
import itertools
import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from otalign.align.cost import pairwise_cosine
from otalign.align.crf import uot_alignment_path
from otalign.align.soft_dp import compute_logZ_and_der1_batched, der1_bisect, forward_backward_batched
from otalign.cache.lmdb_reader import LMDBCache
from otalign.cache.npz_reader import NPZCache
from otalign.functional.sinkhorn_uot import unbalanced_sinkhorn
from otalign.io.fasta_utils import reconstruct_alignment
from otalign.metrics.alignment import alignment_scores, build_ref_matrix, soft_alignment_scores
from otalign.models.embedding import get_embeddings_for_sequences
from otalign.models.plm_adaptors import get_plm_adaptor_and_configs
from otalign.utils.checkpointing import load_peft_model_from_checkpoint
from scripts.dataset_utils import iter_pairs_from_dataset


AnyCache = Union[NPZCache, LMDBCache, None]


def batch_iterator(iterable, batch_size):
    """Yields batches of a given size from an iterable."""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, batch_size))
        if not chunk:
            return
        yield chunk


def _process_batch(
    batch: list[dict],
    args_dict: dict,
    cache: AnyCache,
    model: Optional[torch.nn.Module],
    model_name_for_adaptor: str,
    adaptor: Optional[torch.nn.Module] = None,
) -> list[dict]:
    """
    Processes a batch of sequence pairs on a GPU using the progressive Sinkhorn method.
    """
    device = torch.device(args_dict["device"])
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    torch_dtype = dtype_map[args_dict["dtype"]]

    try:
        # 1. Get embeddings
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

        # 2. Pad and batch embeddings
        batch_size = len(batch)
        emb1_list = embeddings[:batch_size]
        emb2_list = embeddings[batch_size:]
        max_len1 = max(e.shape[0] for e in emb1_list) if emb1_list else 0
        max_len2 = max(e.shape[0] for e in emb2_list) if emb2_list else 0

        emb1_padded = torch.zeros(batch_size, max_len1, emb1_list[0].shape[1], device=device, dtype=torch_dtype)
        emb2_padded = torch.zeros(batch_size, max_len2, emb2_list[0].shape[1], device=device, dtype=torch_dtype)
        lens1 = torch.zeros(batch_size, dtype=torch.int32, device=device)
        lens2 = torch.zeros(batch_size, dtype=torch.int32, device=device)

        for i in range(batch_size):
            len1, len2 = emb1_list[i].shape[0], emb2_list[i].shape[0]
            emb1_padded[i, :len1] = emb1_list[i]
            emb2_padded[i, :len2] = emb2_list[i]
            lens1[i], lens2[i] = len1, len2

        # 3. Calculate cost matrix and marginals
        cost_matrix = pairwise_cosine(emb1_padded, emb2_padded)
        # Assuming cost_matrix uses -cosine_similarity (low cost = good match), set invalid/padded to high cost (e.g., 2.0 > max possible 1.0)
        cost_matrix = torch.nan_to_num(cost_matrix, nan=2.0, posinf=2.0, neginf=-2.0)
        B, N, M = cost_matrix.shape
        mask1 = torch.arange(N, device=device)[None, :] < lens1[:, None]
        mask2 = torch.arange(M, device=device)[None, :] < lens2[:, None]
        a = mask1.float() / lens1[:, None].clamp(min=1).float()
        b = mask2.float() / lens2[:, None].clamp(min=1).float()

        # 4. Iterative Sinkhorn
        reg_m = args_dict["reg_m"]
        num_iter = args_dict["num_iter"]
        reg_schedule = np.geomspace(args_dict["reg_init"], args_dict["reg_final"], args_dict["reg_steps"])
        final_reg = reg_schedule[-1]
        u, v = None, None
        for reg in reg_schedule:
            transport_plan, u, v = unbalanced_sinkhorn(cost_matrix, a, b, num_iter, reg, reg_m, reg_m, mask_a=mask1, mask_b=mask2, u_init=u, v_init=v)

        # 5. Batched Post-processing
        P_masked = transport_plan * mask1[:, :, None] * mask2[:, None, :]
        phi = -reg_m * torch.log(P_masked.sum(2) / a.clamp(min=1e-8) + 1e-8) * mask1
        psi = -reg_m * torch.log(P_masked.sum(1) / b.clamp(min=1e-8) + 1e-8) * mask2
        S = (phi[:, :, None] + psi[:, None, :] - cost_matrix) / final_reg
        S = S * mask1[:, :, None] * mask2[:, None, :]

        rho_D = torch.exp(-phi / reg_m)
        rho_I = torch.exp(-psi / reg_m)
        rho_D = rho_D / (1 + rho_D)
        rho_I = rho_I / (1 + rho_I)
        geD, goD = torch.log(rho_D), torch.log1p(-rho_D)
        geI, goI = torch.log(rho_I), torch.log1p(-rho_I)

        # Use batched functions
        mu_0 = S.new_full((B,), args_dict["mu_lower"])
        target_moment = P_masked.sum((-1, -2))
        logZ0_b, ENM0_b = compute_logZ_and_der1_batched(S, goD, geD, goI, geI, mu_0, lens1, lens2)
        mu_1 = der1_bisect(
            target_moment,
            S,
            goD,
            geD,
            goI,
            geI,
            lens1,
            lens2,
            initial_low=args_dict["mu_lower"],
            initial_high=args_dict["mu_upper"],
            max_iters=args_dict["bisect_iter"],
            tol=args_dict["bisect_tol"],
        )
        logZ_1, ENM_1 = compute_logZ_and_der1_batched(S, goD, geD, goI, geI, mu_1, lens1, lens2)

        mu_align = mu_1 + args_dict["tilt_scale"]
        dp_dic = forward_backward_batched(S, goD, geD, goI, geI, mu_align, lens1, lens2)
        F_M, B_M, logZ_b = dp_dic["F_M"], dp_dic["B_M"], dp_dic["logZ"]
        P_M_b = torch.exp(F_M + B_M - logZ_b[:, None, None])

        # Homology
        null_length = args_dict["null_len_frac"] * (lens1 * (lens1 <= lens2) + lens2 * (lens1 > lens2))
        S_null = (phi[:, :, None] + psi[:, None, :]) / final_reg
        # S_null = torch.zeros_like(S)
        S_null = S_null * mask1[:, :, None] * mask2[:, None, :]
        mu_null = der1_bisect(
            null_length,
            S_null,
            goD,
            geD,
            goI,
            geI,
            lens1,
            lens2,
            initial_low=args_dict["mu_null_lower"],
            initial_high=args_dict["mu_null_upper"],
            max_iters=args_dict["bisect_iter"],
            tol=args_dict["bisect_tol"],
        )
        logZ_null, ENM_null = compute_logZ_and_der1_batched(S_null, goD, geD, goI, geI, mu_null, lens1, lens2)

        # 6. Sequential post-processing (for parts that are hard to batch)
        records = []
        for i in range(batch_size):
            ex = batch[i]
            len1_i, len2_i = int(lens1[i].item()), int(lens2[i].item())
            K_eff = len1_i + len2_i - 1

            P_M = P_M_b[i, :len1_i, :len2_i].cpu().numpy()
            path, meta = uot_alignment_path(
                S[i, :len1_i, :len2_i].cpu().numpy() + mu_align[i].item(),
                goD[i, :len1_i].cpu().numpy(),
                geD[i, :len1_i].cpu().numpy(),
                goI[i, :len2_i].cpu().numpy(),
                geI[i, :len2_i].cpu().numpy(),
            )
            pred_pairs = [(int(x), int(y)) for x, y, m in path if m == "M"]

            metrics = {}
            if ex.get("ref_alignment"):
                ref_set = {tuple(item) for item in ex["ref_alignment"]}
                scores = alignment_scores(set(pred_pairs), ref_set, tolerance=args_dict["eval_band_width"])
                metrics = asdict(scores)
                T = build_ref_matrix(ref_set, len1_i, len2_i)
                metrics.update(soft_alignment_scores(P_M, T))

            rec = {
                "pair_id": ex.get("pair_id", f"{ex['seq1_id']}-{ex['seq2_id']}"),
                "seq1_id": ex["seq1_id"],
                "seq2_id": ex["seq2_id"],
                "pred_alignment": pred_pairs,
                "metrics": metrics,
                "mu_align": mu_align[i].item(),
                "ENM": float(np.sum(P_M)),
                "logZ": logZ_b[i].item(),
                "ENM_1": ENM_1[i].item(),
                "logZ_1": logZ_1[i].item(),
                "mu_1": mu_1[i].item(),
                "logZ_0": logZ0_b[i].item(),
                "ENM_0": ENM0_b[i].item(),
                "mu_0": mu_0[i].item(),
                "ENM_null": ENM_null[i].item(),
                "logZ_null": logZ_null[i].item(),
                "mu_null": mu_null[i].item(),
                "len1": len1_i,
                "len2": len2_i,
                "K_eff": K_eff,
                **meta,
                "meta": {"tool": "OTAlign-CRF", "model": args_dict["model"], "params": {k: v for k, v in args_dict.items() if k not in ["device", "pbar"]}},
            }

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
        print(f"Error processing batch: {e}")
        traceback.print_exc()
        return [{"pair_id": ex.get("pair_id", f"{ex['seq1_id']}-{ex['seq2_id']}"), "error": str(e)} for ex in batch]


def run_otalign_evaluation(
    dataset: str,
    model: str,
    output: str,
    base_model_for_checkpoint: Optional[str] = None,
    cache_dir: Optional[str] = None,
    device: str = "cpu",
    dtype: str = "fp32",
    reg_init: float = 1.0,
    reg_final: float = 0.05,
    reg_steps: int = 5,
    reg_m: float = 2.0,
    tilt_scale: float = 3.0,
    bisect_iter: int = 10,
    bisect_tol: float = 0.05,
    mu_lower: float = 0.0,
    mu_upper: float = 30.0,
    null_len_frac: float = 0.02,
    mu_null_lower: float = -20.0,
    mu_null_upper: float = 10.0,
    num_iter: int = 10_000,
    eval_band_width: int = 0,
    align_batch_size: int = 64,
    export_fasta_dir: Optional[str] = None,
    no_tqdm: bool = False,
    pbar: Optional[tqdm] = None,
):
    """Runs OTAlign evaluation on a dataset and writes results to a file."""
    args_dict = {k: v for k, v in locals().items() if k not in ("export_fasta_dir", "no_tqdm", "pbar")}

    model_path = Path(model)
    if model_path.is_dir():
        print(f"INFO: Loading fine-tuned checkpoint from: {model}")
        if not base_model_for_checkpoint:
            raise ValueError("--base_model_for_checkpoint is required when --model is a directory.")
        print(f"INFO: Using base model for checkpoint: {base_model_for_checkpoint}")
    else:
        print(f"INFO: Using base model: {model}")

    items = list(iter_pairs_from_dataset(dataset))
    total_pairs = len(items)
    it = iter(items)

    success_count = 0
    fail_count = 0

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if export_fasta_dir:
        Path(export_fasta_dir).mkdir(parents=True, exist_ok=True)

    cache: AnyCache = None
    if cache_dir is not None:
        cache_dir_path = Path(cache_dir)
        if (cache_dir_path / "data.lmdb").exists():
            cache = LMDBCache(cache_dir)
            print("Using LMDBCache")

        else:
            cache = NPZCache(cache_dir)
            print("Using NPZCache")

    with out_path.open("w", encoding="utf-8") as fout:
        is_dl_model = model_path.is_dir()

        local_pbar = pbar is None
        if local_pbar:
            pbar = tqdm(total=total_pairs, desc="Aligning pairs", disable=no_tqdm)
        else:
            pbar.total = total_pairs
            pbar.set_description("Aligning pairs")
            pbar.reset()

        print(f"INFO: Using {device} for batch processing.")
        target_device = torch.device(device)
        loaded_model = None
        model_name_for_adaptor = model
        adaptor = None
        if is_dl_model:
            if not base_model_for_checkpoint:
                raise ValueError("base_model_for_checkpoint must be provided for deep learning models")
            plm_adaptor, _, _ = get_plm_adaptor_and_configs(base_model_for_checkpoint, for_masked_lm=bool(base_model_for_checkpoint))
            if plm_adaptor:
                loaded_model = load_peft_model_from_checkpoint(plm_adaptor.model, str(model_path)).to(target_device)
                plm_adaptor.model = loaded_model
                model_name_for_adaptor = base_model_for_checkpoint
                adaptor = plm_adaptor
        if is_dl_model and not adaptor:
            raise ValueError(f"Failed to initialize PLM adaptor for {device} processing.")

        batched_it = batch_iterator(it, align_batch_size)
        pbar.set_description(f"Aligning pairs ({device}, batch_size={align_batch_size})")
        for batch in batched_it:
            records = _process_batch(list(batch), args_dict, cache, loaded_model, model_name_for_adaptor, adaptor)
            for rec in records:
                if "error" in rec:
                    fail_count += 1
                    print(f"Error for pair {rec.get('pair_id')}: {rec.get('error')}")
                else:
                    fout.write(json.dumps(rec) + "\n")
                    success_count += 1
            pbar.update(len(batch))

        if local_pbar:
            pbar.close()

    print(f"  - Evaluation Summary: {success_count}/{total_pairs} pairs processed successfully.")
    if fail_count > 0:
        print(f"  - Failed pairs: {fail_count}")
    print(f"[ok] wrote predictions -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Run OTAlign (Progressive Sinkhorn version) over a dataset.")
    ap.add_argument("--dataset", type=str, required=True, help="Path to the dataset (JSONL or HF identifier).")
    ap.add_argument("--model", required=True, help="Name of the PLM to use (e.g., 'ankh-base') or path to a PEFT checkpoint directory.")
    ap.add_argument("--base_model_for_checkpoint", type=str, help="Base model name if --model is a checkpoint path.")
    ap.add_argument("--cache_dir", help="Directory for embedding cache.")
    ap.add_argument("--device", type=str, default="cpu", help="Device to run inference on.")
    ap.add_argument("--dtype", default="fp32", choices=["fp16", "fp32", "bf16"])

    # Arguments based on the notebook's progressive method
    ap.add_argument("--reg_init", type=float, default=1.0, help="Initial Sinkhorn regularization.")
    ap.add_argument("--reg_final", type=float, default=0.1, help="Final Sinkhorn regularization.")
    ap.add_argument("--reg_steps", type=int, default=5, help="Number of steps for regularization annealing.")
    ap.add_argument("--reg_m", type=float, default=1.0, help="Marginal relaxation term for UOT.")
    ap.add_argument("--num_iter", type=int, default=10_000, help="Max Sinkhorn iterations per step.")
    ap.add_argument("--eval_band_width", type=int, default=0, help="Tolerance for alignment score evaluation (e.g., 0, 1, 2, 4).")
    # ap.add_argument("--tilt_scale", type=float, default=0.0, help="")
    # ap.add_argument("--bisect_iter", type=int, default=50, help="")
    # ap.add_argument("--bisect_tol", type=float, default=0.01, help="")
    # ap.add_argument("--mu_lower", type=float, default=0.0, help="")
    # ap.add_argument("--mu_upper", type=float, default=50.0, help="")

    # IO and batching
    ap.add_argument("--align_batch_size", type=int, default=64, help="Batch size for alignment on both CPU and GPU.")
    ap.add_argument("--output", type=str, required=True, help="Path to write output JSONL file.")
    ap.add_argument("--export_fasta_dir", type=str, default=None, help="If provided, export alignments as FASTA files to this directory.")
    ap.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bars.")
    args = ap.parse_args()

    run_otalign_evaluation(
        dataset=args.dataset,
        model=args.model,
        output=args.output,
        base_model_for_checkpoint=args.base_model_for_checkpoint,
        cache_dir=args.cache_dir,
        device=args.device,
        dtype=args.dtype,
        reg_init=args.reg_init,
        reg_final=args.reg_final,
        reg_steps=args.reg_steps,
        reg_m=args.reg_m,
        num_iter=args.num_iter,
        eval_band_width=args.eval_band_width,
        align_batch_size=args.align_batch_size,
        export_fasta_dir=args.export_fasta_dir,
        no_tqdm=args.no_tqdm,
    )


if __name__ == "__main__":
    main()
