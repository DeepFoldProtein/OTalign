from typing import List, Sequence, Tuple

import torch


def trim_first_last(tokens: torch.Tensor) -> torch.Tensor:
    """
    Remove the first and last token per sequence.
    tokens: [B, T]
    """
    if tokens.size(1) <= 2:
        return tokens[:, :0]  # empty
    return tokens[:, 1:-1]


def trim_last(tokens: torch.Tensor) -> torch.Tensor:
    """
    Remove only the last token per sequence.
    """
    if tokens.size(1) <= 1:
        return tokens[:, :0]
    return tokens[:, :-1]


def drop_token_ids(tokens: torch.Tensor, drop_ids: Sequence[int]) -> Tuple[torch.Tensor, List[int]]:
    """
    Remove all occurrences of any id in drop_ids from each sequence.
    Returns a NEW padded tensor (ragged -> padded) and new lengths.
    """
    B, T = tokens.shape
    kept: List[List[int]] = []
    lengths: List[int] = []
    for b in range(B):
        row = [int(x) for x in tokens[b].tolist() if int(x) not in drop_ids]
        kept.append(row)
        lengths.append(len(row))
    max_len = max(lengths) if lengths else 0
    out = torch.full((B, max_len), 0, dtype=tokens.dtype, device=tokens.device)
    for b, row in enumerate(kept):
        if row:
            out[b, : len(row)] = torch.tensor(row, dtype=tokens.dtype, device=tokens.device)
    return out, lengths


def make_mask_from_lengths(lengths: List[int], device, dtype=torch.bool) -> torch.Tensor:
    """
    Build a right-padded mask from per-sequence lengths.
    """
    B = len(lengths)
    L = max(lengths) if lengths else 0
    mask = torch.zeros((B, L), dtype=dtype, device=device)
    for i, ln in enumerate(lengths):
        if ln > 0:
            mask[i, :ln] = True
    return mask
