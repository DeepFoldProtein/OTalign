import argparse

import torch

from otalign.cache.npz_reader import NPZCache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--seq_id", required=True)
    args = ap.parse_args()

    cache = NPZCache(args.cache_dir)
    emb, mask, L = cache.get(args.seq_id, device="cpu", dtype=torch.float32)
    print(emb.shape, mask.shape, L)


if __name__ == "__main__":
    main()
