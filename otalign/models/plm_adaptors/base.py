from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch


@dataclass(frozen=True)
class EmbeddingOutput:
    """
    residue_embeddings: [B, L, D] tensor after removing special tokens (per sequence length L_i)
    attention_mask:     [B, L] bool tensor, True for valid residues
    per_sequence_lengths: list[int] original residue lengths (after trimming specials)
    extras: optional dict for model-specific info (e.g., CLS, EOS, logits)
    """

    residue_embeddings: torch.Tensor
    attention_mask: torch.Tensor
    per_sequence_lengths: List[int]
    extras: Dict[str, Any]


class BasePLMAdaptor(ABC):
    """
    Minimal contract for all PLM adaptors.
    """

    @abstractmethod
    def encode(
        self,
        sequences: Sequence[str],
        *,
        batch_size: int = 8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        fp16: bool = False,
        disable_grad: bool = True,
    ) -> EmbeddingOutput:
        """
        Convert a batch of AA sequences (strings over 20-letter alphabet) into residue-level embeddings.
        Implementations are responsible for:
          - tokenization
          - model forward
          - trimming special tokens (CLS/EOS/SEP/PAD) to return ONLY residue embeddings
          - building a tight attention_mask over residues
          - padding to max length inside the batch for tensorization
        """
        raise NotImplementedError
