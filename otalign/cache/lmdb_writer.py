import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import lmdb
import numpy as np
import torch

from .config import CacheConfig


class LMDBCacheWriter:
    def __init__(self, root: str | Path, cfg: CacheConfig, map_size: int = 1024**4):  # 1TB
        self.cfg = cfg
        key = _short_key(asdict(cfg))
        self.root = Path(root) / f"{cfg.dataset_name}__{cfg.model_name.split('/')[-1]}__{cfg.dtype}_{key}__v2_lmdb"
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "_cache_meta.json").write_text(json.dumps({"config": asdict(cfg)}, indent=2))

        self.env = lmdb.open(str(self.root / "data.lmdb"), map_size=map_size, writemap=True)
        self.txn = self.env.begin(write=True)
        self.write_count = 0

    def append_batch(self, seq_ids: List[str], embeds: torch.Tensor, mask: torch.Tensor, lengths: List[int]):
        B, _, D = embeds.shape
        np_emb = embeds.detach().to("cpu").numpy()
        np_msk = mask.to("cpu").numpy().astype(np.uint8)
        lens = [int(x) for x in lengths]

        for i in range(B):
            L = lens[i]
            seq_id = seq_ids[i]
            emb = np_emb[i, :L, :]
            msk = np_msk[i, :L]

            # Use pickle to serialize the data tuple
            value = pickle.dumps((emb, msk, L))
            self.txn.put(seq_id.encode("utf-8"), value)
            self.write_count += 1

        # Commit periodically to avoid large transactions
        if self.write_count > 1000:
            self.txn.commit()
            self.txn = self.env.begin(write=True)
            self.write_count = 0

    def close(self):
        if self.txn:
            self.txn.commit()
        self.env.close()


def _short_key(cfg_dict: Dict) -> str:
    import hashlib

    payload = json.dumps(cfg_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()[:8]
