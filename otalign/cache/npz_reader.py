import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


@dataclass
class IndexEntry:
    shard: str
    index: int
    length: int


class NPZCache:
    def __init__(self, cache_dir: str | Path):
        self.root = Path(cache_dir)
        self._index: Dict[str, IndexEntry] = {}
        with (self.root / "manifest.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                self._index[o["seq_id"]] = IndexEntry(o["shard"], int(o["index"]), int(o["length"]))
        self._open_npz: Dict[str, dict] = {}  # shard_name -> dict of arrays

    def _open_shard(self, shard_name: str):
        if shard_name not in self._open_npz:
            path = self.root / shard_name
            data = np.load(path, allow_pickle=True)
            self._open_npz[shard_name] = data
        return self._open_npz[shard_name]

    def get(self, seq_id: str, device="cpu", dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor, int]:
        ent = self._index[seq_id]
        data = self._open_shard(ent.shard)
        emb = data["emb"][ent.index]  # [Lmax_shard, D]
        msk = data["mask"][ent.index].astype(bool)  # [Lmax_shard]
        L = ent.length
        emb_t = torch.from_numpy(emb[:L]).to(device=device, dtype=dtype)
        msk_t = torch.from_numpy(msk[:L]).to(device=device)
        return emb_t, msk_t, L
