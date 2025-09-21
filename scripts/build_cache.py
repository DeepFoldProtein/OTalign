import argparse
import typing
from pathlib import Path

import torch
from tqdm.auto import tqdm

from otalign.cache.config import CacheConfig
from otalign.cache.lmdb_writer import LMDBCacheWriter
from otalign.cache.npz_writer import NPZCacheWriter
from otalign.models.plm_adaptors import get_plm_adaptor_and_configs
from otalign.utils.checkpointing import load_peft_model_from_checkpoint
from scripts.dataset_utils import iter_pairs_from_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to the dataset (JSONL or HF identifier).")
    ap.add_argument("--model", required=True, help="Name of the PLM to use (e.g., 'ankh-base') or path to a PEFT checkpoint directory.")
    ap.add_argument("--base_model_for_checkpoint", type=str, help="Base model name if --model is a checkpoint path.")
    ap.add_argument("--dtype", default="fp32", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--shard_size", type=int, default=2000)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--cache_type", type=str, default="lmdb", choices=["npz", "lmdb"])
    ap.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars.")
    args = ap.parse_args()

    device = torch.device(args.device)
    model_path = Path(args.model)
    model_name_for_config: str

    if model_path.is_dir():
        print(f"INFO: Loading fine-tuned checkpoint from: {args.model}")
        if not args.base_model_for_checkpoint:
            raise ValueError("--base_model_for_checkpoint is required when --model is a directory.")
        print(f"INFO: Using base model for checkpoint: {args.base_model_for_checkpoint}")

        model_name_for_config = args.base_model_for_checkpoint
        adaptor, policy, adaptor_name = get_plm_adaptor_and_configs(args.base_model_for_checkpoint, for_masked_lm=True)
        model = load_peft_model_from_checkpoint(adaptor.model, str(model_path))
        adaptor.model = model
    else:
        print(f"INFO: Using base model: {args.model}")
        model_name_for_config = args.model
        adaptor, policy, adaptor_name = get_plm_adaptor_and_configs(args.model)

    torch.set_grad_enabled(False)

    dataset_name = args.dataset.split(",")[0].split("/")[-1]
    cfg = CacheConfig(
        dataset_name=dataset_name,
        model_name=model_name_for_config,
        adaptor_name=adaptor_name,
        adaptor_version="1",
        dtype=args.dtype,
        policy=policy,
        tokenizer_pretok=None,
        shard_size=args.shard_size,
        extra={},
    )
    if args.cache_type == "npz":
        writer = NPZCacheWriter(args.output_root, cfg)
    elif args.cache_type == "lmdb":
        writer = LMDBCacheWriter(args.output_root, cfg)
    else:
        raise ValueError(f"Unknown cache type: {args.cache_type}")

    ds_iterator = iter_pairs_from_dataset(args.dataset)

    id_set = set()
    pair_set = set()
    for ex_raw in ds_iterator:
        ex = typing.cast(dict, ex_raw)
        id_set.add(ex["seq1_id"])
        pair_set.add((ex["seq1_id"], ex["seq1"]))
        id_set.add(ex["seq2_id"])
        pair_set.add((ex["seq2_id"], ex["seq2"]))

    if len(id_set) != len(pair_set):
        raise ValueError("More than two sequences are matched to a single id.")

    adaptor.model.to(device)

    pairs = list(pair_set)
    # batch over sequences
    pbar = tqdm(total=len(pairs), disable=args.no_tqdm)
    for i in range(0, len(pairs), args.batch_size):
        batch = pairs[i : i + args.batch_size]
        seqs = [s for _, s in batch]
        out = adaptor.encode(seqs, batch_size=len(batch), device=device, fp16=args.dtype == "fp16")
        ids = [i for i, _ in batch]
        writer.append_batch(ids, out.residue_embeddings, out.attention_mask, out.per_sequence_lengths)
        pbar.update(len(batch))
    pbar.close()

    writer.close()
    print("Cache built:", writer.root)


if __name__ == "__main__":
    main()
