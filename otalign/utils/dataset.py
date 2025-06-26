# datasets.py
import functools
import pathlib

import h5py
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, get_worker_info


def get_embedding(name: str) -> torch.Tensor:
    path = pathlib.Path(f"/dev/shm/esm1b_t33_650M_UR50S/{name}.h5")
    if not path.exists():
        path = pathlib.Path(f"esm1b_t33_650M_UR50S/{name}.h5")
    with h5py.File(str(path), "r") as fp:
        if "embedding" not in fp:
            raise KeyError(f"'embedding' dataset not found in file {path}")
        dset = fp["embedding"]
        arr = dset[()]  # type: ignore
        return torch.from_numpy(arr)


@functools.lru_cache(maxsize=16)
def _cached_embedding(name: str):
    emb = get_embedding(name)
    return emb


class EmbeddingDataset(IterableDataset):
    def __init__(self, target_df, max_len, device):
        self.device = device

        if max_len is None:
            mask = target_df["length"] > 0
        else:
            mask = target_df["length"] <= max_len
        self.ids = target_df["rcsb_id"][mask].tolist()

    def __iter__(self):
        # Re-install cache in each worker process to ensure independent caching
        # for each worker.
        _cached_embedding.cache_clear()  # Clear cache for each new iteration/worker
        _cached_embedding_func = functools.lru_cache(maxsize=16)(_cached_embedding)

        worker_info = get_worker_info()

        # Determine the start and end indices for the current worker.
        if worker_info is None:  # Single-process data loading (num_workers = 0)
            iter_start = 0
            iter_end = len(self.ids)
        else:  # Multi-process data loading (num_workers > 0)
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            # Calculate the number of items per worker.
            per_worker = int(len(self.ids) / num_workers)

            # Determine the start and end indices for this specific worker.
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.ids))

            # Ensure the last worker picks up any remaining items.
            if worker_id == num_workers - 1:
                iter_end = len(self.ids)

        # Iterate only over the assigned portion of IDs for this worker.
        for i in range(iter_start, iter_end):
            name = self.ids[i]
            tgt = _cached_embedding_func(name)  # Use the cached function specific to this worker
            yield name, tgt

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    """
    Returns names, padded tensor (B, L, D), and masks (B, L)
    """
    names, emb = zip(*batch)
    # Flatten into [B] list for pad_sequence
    flat = list(emb)
    padded = pad_sequence(flat, batch_first=True, padding_value=0.0)  # (2B, L, D)
    # Build masks
    lengths = [seq.size(0) for seq in flat]
    mask = torch.arange(padded.size(1), device=padded.device).unsqueeze(0) < torch.tensor(
        lengths, device=padded.device
    ).unsqueeze(1)  # (2B, L)
    # Reshape back to (B, 2, L, D) and (B, 2, L).
    B = len(batch)
    L, D = padded.size(1), padded.size(2)
    padded = padded.view(B, L, D)
    mask = mask.view(B, L)
    return list(names), padded, mask
