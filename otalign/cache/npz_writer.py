import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import CacheConfig


class NPZCacheWriter:
    """
    Buffer rows in memory, then flush as a single .npz shard padded to that shard's Lmax.
    Simpler than Zarr; great for read-mostly benchmarks.
    """

    def __init__(self, root: str | Path, cfg: CacheConfig):
        self.cfg = cfg
        key = _short_key(asdict(cfg))
        self.root = Path(root) / f"{cfg.dataset_name}__{cfg.model_name.split('/')[-1]}__{cfg.dtype}_{key}__v1"
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "_cache_meta.json").write_text(json.dumps({"config": asdict(cfg)}, indent=2))
        self.manifest_f = (self.root / "manifest.jsonl").open("w", encoding="utf-8")

        # in‑memory buffers for current shard
        self.cur_ids: List[str] = []
        self.cur_embs: List[np.ndarray] = []  # each [L_i, D]
        self.cur_masks: List[np.ndarray] = []  # each [L_i]
        self.cur_lens: List[int] = []
        self.cur_D: Optional[int] = None
        self.shard_idx = -1

    def append_batch(self, seq_ids: List[str], embeds: torch.Tensor, mask: torch.Tensor, lengths: List[int]):
        """
        embeds: [B, Lmax, D] torch
        mask:   [B, Lmax] bool/int
        lengths:[B] int
        """
        B, Lmax, D = embeds.shape
        np_emb = embeds.detach().to("cpu").numpy()
        np_msk = mask.to("cpu").numpy().astype(np.uint8)
        lens = [int(x) for x in lengths]

        # If D changes mid-shard, flush the current shard first (rare, but safe)
        if self.cur_D is not None and self.cur_D != D:
            self._flush()

        if self.cur_D is None:
            self.cur_D = D

        for i in range(B):
            L = lens[i]
            self.cur_ids.append(seq_ids[i])
            self.cur_embs.append(np_emb[i, :L, :])  # store cropped [L, D]
            self.cur_masks.append(np_msk[i, :L])  # [L]
            self.cur_lens.append(L)

            if len(self.cur_ids) >= self.cfg.shard_size:
                self._flush()

    def _flush(self):
        if not self.cur_ids:
            return
        self.shard_idx += 1
        shard_name = f"shard_{self.shard_idx:03d}.npz"
        path = self.root / shard_name

        N = len(self.cur_ids)
        Lmax = max(self.cur_lens) if self.cur_lens else 0
        D = int(self.cur_D or 0)

        # allocate padded arrays
        dtype_map = {"fp16": np.float16, "fp32": np.float32, "bf16": np.float16}
        emb_arr = np.zeros((N, Lmax, D), dtype=dtype_map.get(self.cfg.dtype, np.float16))
        msk_arr = np.zeros((N, Lmax), dtype=np.uint8)
        len_arr = np.asarray(self.cur_lens, dtype=np.int32)
        ids_arr = np.array(self.cur_ids, dtype=object)

        for i, (e, m) in enumerate(zip(self.cur_embs, self.cur_masks)):
            Li = e.shape[0]
            emb_arr[i, :Li, :] = e
            msk_arr[i, :Li] = m

        # write npz (compressed)
        meta = {
            "model_name": self.cfg.model_name,
            "adaptor_name": self.cfg.adaptor_name,
            "adaptor_version": self.cfg.adaptor_version,
            "dtype": self.cfg.dtype,
            "policy": self.cfg.policy,
            "tokenizer_pretok": self.cfg.tokenizer_pretok,
            "Lmax": int(Lmax),
            "D": int(D),
        }
        np.savez_compressed(
            path,
            emb=emb_arr,
            mask=msk_arr,
            lengths=len_arr,
            ids=ids_arr,
            meta=json.dumps(meta),
        )

        # update manifest
        for i, sid in enumerate(self.cur_ids):
            rec = {"seq_id": sid, "shard": shard_name, "index": i, "length": int(self.cur_lens[i])}
            self.manifest_f.write(json.dumps(rec) + "\n")

        # reset buffers
        self.cur_ids.clear()
        self.cur_embs.clear()
        self.cur_masks.clear()
        self.cur_lens.clear()
        self.cur_D = None

    def close(self):
        self._flush()
        if self.manifest_f:
            self.manifest_f.close()


def _short_key(cfg_dict: Dict) -> str:
    import hashlib

    payload = json.dumps(cfg_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()[:8]
