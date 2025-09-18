import pickle
from pathlib import Path
from typing import Tuple

import lmdb
import torch


class LMDBCache:
    def __init__(self, cache_dir: str | Path):
        self.root = Path(cache_dir)
        self.env = lmdb.open(str(self.root / "data.lmdb"), readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)

    def get(self, seq_id: str, device="cpu", dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor, int]:
        value = self.txn.get(seq_id.encode("utf-8"))
        if value is None:
            raise KeyError(f"Sequence ID not found: {seq_id}")

        emb, msk, L = pickle.loads(value)

        emb_t = torch.from_numpy(emb).to(device=device, dtype=dtype)
        msk_t = torch.from_numpy(msk).to(device=device)
        return emb_t, msk_t, L

    def __del__(self):
        if hasattr(self, "env"):
            self.env.close()
