from typing import Any, Dict, List, Optional

import torch

from .alignment_utils import sparse_to_dense_alignment
from .mlm_collator import MLMCollator


class OTAlignCollator:
    def __init__(
        self,
        mlm_collator: Optional[MLMCollator] = None,
        max_len1: Optional[int] = None,
        max_len2: Optional[int] = None,
    ):
        self.mlm_collator = mlm_collator
        self.max_len1 = max_len1
        self.max_len2 = max_len2

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
        max_len1 = self.max_len1 if self.max_len1 is not None else (max(item["len1"] for item in batch) if batch else 0)
        max_len2 = self.max_len2 if self.max_len2 is not None else (max(item["len2"] for item in batch) if batch else 0)

        if is_positive.any():
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
                    # For negative samples, we can append a tensor of the correct size
                    gt_alignments.append(torch.zeros(max_len1, max_len2, dtype=torch.float32))

        final_gt_alignments = torch.stack(gt_alignments) if gt_alignments else torch.zeros(len(batch), max_len1, max_len2)

        output = {
            "seqs1": seqs1,
            "seqs2": seqs2,
            "gt_alignments": final_gt_alignments,
            "is_positive": is_positive,
            "lens1": lens1,
            "lens2": lens2,
        }

        if self.mlm_collator:
            mlm_inputs1 = self.mlm_collator(seqs1)
            mlm_inputs2 = self.mlm_collator(seqs2)
            output["mlm_input_ids1"] = mlm_inputs1["input_ids"]
            output["mlm_labels1"] = mlm_inputs1["labels"]
            output["mlm_attention_mask1"] = mlm_inputs1["attention_mask"]
            output["mlm_input_ids2"] = mlm_inputs2["input_ids"]
            output["mlm_labels2"] = mlm_inputs2["labels"]
            output["mlm_attention_mask2"] = mlm_inputs2["attention_mask"]

        return output
