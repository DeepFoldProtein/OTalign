from typing import List, Sequence, Tuple

import torch


def pack_left_aligned(src: torch.Tensor, keep_mask: torch.Tensor, maxL: int) -> torch.Tensor:
    """Left-align the kept positions of ``src`` into a zero-padded width-``maxL`` tensor.

    ``src`` is ``[B, T]`` (token ids) or ``[B, T, D]`` (embeddings); ``keep_mask``
    is ``[B, T]`` selecting the tokens to keep, in original order. Returns
    ``[B, maxL]`` / ``[B, maxL, D]`` with kept tokens packed to the front of each
    row and the rest zero. Shared by the trim policies (ids) and the encoder
    adaptor's embedding packing so both left-align identically.
    """
    B = keep_mask.size(0)
    out = src.new_zeros((B, maxL, *src.shape[2:]))
    if maxL > 0:
        dest_col = keep_mask.cumsum(dim=1) - 1  # destination column per kept token (its rank)
        r, c = keep_mask.nonzero(as_tuple=True)
        out[r, dest_col[r, c]] = src[r, c]
    return out


def trim_active_edges_vec(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    drop_first: int,
    drop_last: int,
) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
    """Vectorized "drop N leading + M trailing active tokens" trim policy.

    Generalizes the per-row Python-loop trim policies used by the T5/Ankh
    (``drop_first=0, drop_last=1`` → drop EOS) and ESM (``drop_first=1,
    drop_last=1`` → drop BOS/EOS) adaptors into a single tensor-only kernel.
    "Active" means ``attention_mask == 1`` (non-pad). Returns a
    ``(trimmed_ids, lengths, keep_mask)`` triple.
    """
    active = attention_mask.to(torch.bool)
    B = active.size(0)
    total = active.sum(dim=1, keepdim=True)  # [B, 1] active count per row
    csum = active.cumsum(dim=1)  # [B, T] 1-indexed rank among active tokens
    # Keep active tokens whose rank is strictly past the leading `drop_first`
    # and within the last `drop_last` from the end. Rows too short to survive
    # both cuts keep nothing (the two conditions become disjoint).
    keep_mask = active & (csum > drop_first) & (csum <= (total - drop_last))
    lengths_t = keep_mask.sum(dim=1)  # [B]
    maxL = int(lengths_t.max().item()) if B > 0 else 0
    trimmed = pack_left_aligned(input_ids, keep_mask, maxL)
    lengths = [int(x) for x in lengths_t.tolist()]
    return trimmed, lengths, keep_mask


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
