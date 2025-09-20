from typing import List, Tuple

import torch


def sparse_to_dense_alignment(ref_alignment: List[Tuple[int, int]], len1: int, len2: int) -> torch.Tensor:
    """
    Converts a sparse list of aligned residue pairs into a dense
    binary alignment matrix.

    Args:
        ref_alignment (List[Tuple[int, int]]): List of (idx1, idx2) pairs.
        len1 (int): Length of the first sequence.
        len2 (int): Length of the second sequence.

    Returns:
        torch.Tensor: A [len1, len2] binary tensor where T[i, j] = 1 if
                      residue i and j are aligned.
    """
    dense_alignment = torch.zeros(len1, len2, dtype=torch.float32)
    if not ref_alignment:
        return dense_alignment

    # Unzip the pairs into two tensors of indices
    indices1, indices2 = zip(*ref_alignment)

    # Use the indices to set the corresponding elements in the dense tensor to 1
    dense_alignment[torch.tensor(indices1), torch.tensor(indices2)] = 1.0

    return dense_alignment
