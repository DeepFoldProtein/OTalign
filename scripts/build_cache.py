import argparse

import torch
from tqdm.auto import tqdm

from datasets import load_dataset
from otalign.cache.config import CacheConfig
from otalign.cache.npz_writer import NPZCacheWriter
from otalign.models.plm_adaptors import build_ankhcl_adaptor, build_esm_adaptor, build_prott5_adaptor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)  # e.g., DeepFoldProtein/SABmark-dataset
    ap.add_argument("--name", required=True)  # e.g., twi
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="fp32", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--shard_size", type=int, default=2000)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    if args.model == "ESM2":
        adaptor = build_esm_adaptor("facebook/esm2_t33_650M_UR50D")
        policy, adaptor_name = "drop_first_last_active", "ESM-2"
        tokenizer_pretok = None
    elif args.model == "ESM1b":
        adaptor = build_esm_adaptor("facebook/esm1b_t33_650M_UR50S")
        policy, adaptor_name = "drop_first_last_active", "ESM-1b"
        tokenizer_pretok = None
    elif args.model == "AnkhCL":
        adaptor = build_ankhcl_adaptor()
        policy, adaptor_name = "drop_last_active", "AnkhCL"
        tokenizer_pretok = None
    elif args.model == "ProtT5":
        adaptor = build_prott5_adaptor("Rostlab/prot_t5_xl_uniref50")
        policy, adaptor_name = "drop_last_active", "ProtT5"
        tokenizer_pretok = None
    else:
        raise ValueError(f"Not valid model name: '{args.model}'")

    torch.set_grad_enabled(False)
    cfg = CacheConfig(
        dataset_name=f"{args.dataset.split('/')[-1]}-{args.name}",
        model_name=adaptor.model.name_or_path if hasattr(adaptor.model, "name_or_path") else str(adaptor.model),
        adaptor_name=adaptor_name,
        adaptor_version="1",
        dtype=args.dtype,
        policy=policy,
        tokenizer_pretok=tokenizer_pretok,
        shard_size=args.shard_size,
        extra={},
    )
    writer = NPZCacheWriter(args.output_root, cfg)

    ds = load_dataset(args.dataset, name=args.name, split=args.split)

    id_set = set()
    pair_set = set()
    for ex in ds:
        id_set.add(ex["seq1_id"])
        pair_set.add((ex["seq1_id"], ex["seq1"]))
        id_set.add(ex["seq2_id"])
        pair_set.add((ex["seq2_id"], ex["seq2"]))

    if len(id_set) != len(pair_set):
        raise ValueError("More than two sequences are matched to a single id.")

    device = torch.device(args.device)
    adaptor.model.to(device)

    pairs = list(pair_set)
    # batch over sequences
    pbar = tqdm(total=len(pairs))
    for i in range(0, len(pairs), args.batch_size):
        batch = pairs[i : i + 8]
        seqs = [s for _, s in batch]
        out = adaptor.encode(seqs, batch_size=len(batch), device=device, fp16=True)
        ids = [i for i, _ in batch]
        writer.append_batch(ids, out.residue_embeddings, out.attention_mask, out.per_sequence_lengths)
        pbar.update(args.batch_size)
    pbar.close()

    writer.close()
    print("Cache built:", writer.root)


if __name__ == "__main__":
    main()
