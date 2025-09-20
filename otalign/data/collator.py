from typing import Any, Dict, List

import torch

from .alignment_utils import sparse_to_dense_alignment


class OTAlignCollator:
    def __init__(self):
        pass

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collates a batch of data from the CATHDataset.

        - Converts sparse ground truth alignments to dense tensors.
        - Does NOT handle tokenization, as this is now managed by the PLM adaptors.
        - Pads sequences and alignments to the max length in the batch.
        """

        seqs1 = [item["seq1"] for item in batch]
        seqs2 = [item["seq2"] for item in batch]
        is_positive = torch.tensor([item["is_positive"] for item in batch], dtype=torch.bool)
        lens1 = torch.tensor([item["len1"] for item in batch], dtype=torch.long)
        lens2 = torch.tensor([item["len2"] for item in batch], dtype=torch.long)

        # Handle ground truth alignments
        gt_alignments = []
        if is_positive.any():
            max_len1 = max(item["len1"] for item in batch)
            max_len2 = max(item["len2"] for item in batch)

            for item in batch:
                if item["is_positive"]:
                    dense_align = sparse_to_dense_alignment(item["ref_alignment"], item["len1"], item["len2"])
                    # Pad the dense alignment tensor to the max size in the batch
                    padded_align = torch.nn.functional.pad(
                        dense_align,
                        (0, max_len2 - item["len2"], 0, max_len1 - item["len1"]),
                        "constant",
                        0,
                    )
                    gt_alignments.append(padded_align)
                else:
                    # For negative samples, we can append an empty tensor of the correct size
                    gt_alignments.append(torch.zeros(max_len1, max_len2, dtype=torch.float32))

        return {
            "seqs1": seqs1,
            "seqs2": seqs2,
            "gt_alignments": torch.stack(gt_alignments) if gt_alignments else None,
            "is_positive": is_positive,
            "lens1": lens1,
            "lens2": lens2,
        }
