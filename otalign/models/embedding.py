from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from otalign.cache.lmdb_reader import LMDBCache
from otalign.cache.npz_reader import NPZCache
from otalign.models.plm_adaptors import get_plm_adaptor_and_configs


AnyCache = Union[NPZCache, LMDBCache]


def l2_normalize(X: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """
    L2 normalize an array along a given axis.
    """
    n = np.linalg.norm(X, axis=axis, keepdims=True)
    return X / np.maximum(n, eps)


def l2_normalize_torch(X: torch.Tensor, axis: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """
    L2 normalize a tensor along a given axis.
    """
    n = torch.linalg.norm(X, ord=2, dim=axis, keepdim=True)
    return X / n.clamp(min=eps)


def get_embeddings_for_sequences(
    sequences: list[str],
    seq_ids: list[str],
    model_name: str,
    cache_dir: Optional[str],
    device: str = "cpu",
    batch_size: int = 4,
    dtype: str = "fp32",
    cache: Optional[AnyCache] = None,
    model: Optional[torch.nn.Module] = None,
    adaptor: Optional[torch.nn.Module] = None,
) -> list[torch.Tensor]:
    """
    Get per-residue embeddings for a list of sequences, using a cache if possible.

    Args:
        sequences (list[str]): A list of protein sequences.
        seq_ids (list[str]): A list of unique IDs for the sequences.
        model_name (str): The name of the pre-trained language model to use.
        cache_dir (Optional[str]): The directory where the embedding cache is stored.
        device (str): The device to run the model on (e.g., "cpu", "cuda:0").
        batch_size (int): The batch size for embedding generation.
        dtype (str): The data type for the embeddings ("fp16", "fp32", "bf16").
        cache (Optional[AnyCache]): An optional pre-initialized cache object.

    Returns:
        A list of tensors, where each array is the per-residue embedding for a sequence.
    """
    if cache is None and cache_dir:
        cache_dir_path = Path(cache_dir)
        if (cache_dir_path / "data.lmdb").exists():
            cache = LMDBCache(cache_dir)
        else:
            cache = NPZCache(cache_dir)

    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    torch_dtype = dtype_map[dtype]

    # 1) Probe cache to avoid recomputation
    embeddings: list[torch.Tensor | None] = [None] * len(sequences)
    to_compute_indices: list[int] = []

    if cache:
        for i, seq_id in enumerate(seq_ids):
            try:
                emb, _, _ = cache.get(seq_id, device=device, dtype=torch_dtype)
                embeddings[i] = emb
            except (FileNotFoundError, KeyError):
                to_compute_indices.append(i)
    else:
        to_compute_indices = list(range(len(sequences)))

    if not to_compute_indices:
        return [emb for emb in embeddings if emb is not None]

    # 2) Load model and embed sequences that are not in the cache
    if adaptor is None:
        adaptor, _, _ = get_plm_adaptor_and_configs(model_name)

    if adaptor is None:
        raise ValueError(f"Could not load PLM adaptor for model '{model_name}'.")
    model_device = torch.device(device)

    if model is not None:
        adaptor.model = model
    adaptor.model.to(model_device)

    seqs_to_embed = [sequences[i] for i in to_compute_indices]

    with torch.no_grad():
        out = adaptor.encode(
            seqs_to_embed,
            batch_size=batch_size,
            device=model_device,
            fp16=(dtype == "fp16"),
        )

    # 3) Process and store embeddings
    for i, original_index in enumerate(to_compute_indices):
        length = out.per_sequence_lengths[i]
        # Remove padding and CLS/EOS tokens if any
        residue_emb = out.residue_embeddings[i, :length, :].to(device=model_device, dtype=torch_dtype)

        # The notebook normalizes embeddings, so we do it here as well.
        normalized_emb = l2_normalize_torch(residue_emb, axis=-1)
        embeddings[original_index] = normalized_emb

    # The cache writer is not used here because this function is for reading/on-the-fly generation.
    # The cache should be pre-built using a script like `scripts/build_cache.py`.

    return [emb for emb in embeddings if emb is not None]
