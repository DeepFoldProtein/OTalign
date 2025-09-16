import argparse
import json
import os
from pathlib import Path

import torch

from datasets import load_dataset
from otalign.cache.config import CacheConfig
from otalign.cache.npz_writer import NPZCacheWriter
from otalign.models.plm_adaptors import build_ankhcl_adaptor, build_esm_adaptor, build_prott5_adaptor


def get_rank_and_world() -> tuple[int, int]:
    """Read rank/world from SLURM or env; fallback to single-process."""
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1")))
    return rank, world


def shard_prefix(rank: int) -> str:
    return f"r{rank:02d}_"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="fp32", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--shard_size", type=int, default=2000)
    args = ap.parse_args()

    rank, world = get_rank_and_world()
    print(f"[rank {rank}/{world}] starting")

    # Build adaptor
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

    # Load HF dataset split
    ds = load_dataset(args.dataset, name=args.name, split=args.split)

    # Build flat sequence/id lists (both seq1 & seq2)
    all_seqs: list[str] = []
    all_ids: list[str] = []
    for ex in ds:
        all_seqs.append(ex["seq1"])
        all_ids.append(ex["seq1_id"])
        all_seqs.append(ex["seq2"])
        all_ids.append(ex["seq2_id"])

    # Rank-strided split
    idxs = list(range(rank, len(all_seqs), world))
    seqs = [all_seqs[i] for i in idxs]
    ids = [all_ids[i] for i in idxs]
    print(f"[rank {rank}] assigned {len(seqs)} sequences")

    # Build a rank-prefixed cache dir OR rank-prefixed shards in a shared dir
    # Here we choose rank-prefixed shards in a shared dir.
    cfg = CacheConfig(
        dataset_name=f"{args.dataset.split('/')[-1]}-{args.name}",
        model_name=args.model_name,
        adaptor_name=adaptor_name,
        adaptor_version="1",
        dtype=args.dtype,
        policy=policy,
        tokenizer_pretok=tokenizer_pretok,
        shard_size=args.shard_size,
    )

    # We will write into a rank-specific subdir and then move shard files up with prefix.
    root = Path(args.output_root) / f"{cfg.dataset_name}__{args.model_name.split('/')[-1]}__{args.dtype}_{_short_key(cfg)}__v1"
    tmp_rank_dir = root / f"__tmp_rank_{rank:02d}"
    tmp_rank_dir.mkdir(parents=True, exist_ok=True)

    writer = NPZCacheWriter(tmp_rank_dir, cfg)

    # Device & dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adaptor.model.to(device)
    use_fp16 = args.dtype == "fp16"

    # Encode in batches
    for i in range(0, len(seqs), args.batch_size):
        batch = seqs[i : i + args.batch_size]
        out = adaptor.encode(batch, batch_size=len(batch), device=device, fp16=use_fp16)
        batch_ids = ids[i : i + args.batch_size]
        writer.append_batch(batch_ids, out.residue_embeddings, out.attention_mask, out.per_sequence_lengths)

    writer.close()
    print(f"[rank {rank}] local shards written to {tmp_rank_dir}")

    # Promote rank-local shards to shared root with rank-prefixed names
    # Move shard_000.npz -> shard_r{rank}_{k:03d}.npz and append manifest rows
    root.mkdir(parents=True, exist_ok=True)
    # Ensure shared manifest exists
    shared_manifest = root / "manifest.jsonl"
    shared_meta = root / "_cache_meta.json"
    # If first writer, copy meta
    if not shared_meta.exists():
        (tmp_rank_dir / "_cache_meta.json").replace(shared_meta)

    # Merge manifests
    with (tmp_rank_dir / "manifest.jsonl").open("r", encoding="utf-8") as fin, open(shared_manifest, "a", encoding="utf-8") as fout:
        for line in fin:
            o = json.loads(line)
            # rewrite shard filename
            # find all local shard files
            # Our NPZCacheWriter emits shard_XXX.npz names; prefix with rank
            shard_old = o["shard"]
            k = int(shard_old.split("_")[-1].split(".")[0])  # XXX from shard_XXX.npz
            shard_new = f"shard_{shard_prefix(rank)}{k:03d}.npz"
            o["shard"] = shard_new
            fout.write(json.dumps(o) + "\n")

    # Move .npz files up with new names
    for npz in sorted(tmp_rank_dir.glob("shard_*.npz")):
        k = int(npz.stem.split("_")[-1])  # XXX
        target = root / f"shard_{shard_prefix(rank)}{k:03d}.npz"
        npz.replace(target)

    # Cleanup temp rank dir
    for leftover in tmp_rank_dir.glob("*"):
        try:
            leftover.unlink()
        except IsADirectoryError:
            pass
    try:
        tmp_rank_dir.rmdir()
    except OSError:
        pass

    print(f"[rank {rank}] done -> {root}")


def _short_key(cfg: CacheConfig) -> str:
    import hashlib

    payload = json.dumps(
        {
            "dataset_name": cfg.dataset_name,
            "model_name": cfg.model_name,
            "adaptor_name": cfg.adaptor_name,
            "adaptor_version": cfg.adaptor_version,
            "dtype": cfg.dtype,
            "policy": cfg.policy,
            "tokenizer_pretok": cfg.tokenizer_pretok,
            "shard_size": cfg.shard_size,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


if __name__ == "__main__":
    main()
