import torch
from transformers import AutoModel, AutoTokenizer, EsmForMaskedLM

from .hf_adaptor import HFEncoderAdaptor


def esm_trim_first_last_policy(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """
    Drop the first and last *active* tokens (BOS/EOS) using attention_mask (1=active, 0=pad).
    """
    B, T = input_ids.shape
    keep_mask = torch.zeros((B, T), dtype=torch.bool, device=input_ids.device)
    lengths: list[int] = []
    for b in range(B):
        active_idx = torch.nonzero(attention_mask[b].to(torch.bool), as_tuple=False).flatten()
        if active_idx.numel() >= 2:
            keep_idx = active_idx[1:-1]  # drop BOS/EOS
        else:
            keep_idx = active_idx[:0]
        keep_mask[b, keep_idx] = True
        lengths.append(int(keep_idx.numel()))
    maxL = max(lengths) if lengths else 0
    trimmed = torch.zeros((B, maxL), dtype=input_ids.dtype, device=input_ids.device)
    for b in range(B):
        kept = input_ids[b][keep_mask[b]]
        if kept.numel():
            trimmed[b, : kept.numel()] = kept
    return trimmed, lengths, keep_mask


def build_esm_adaptor(model_name: str = "facebook/esm2_t33_650M_UR50D", for_masked_lm: bool = False) -> HFEncoderAdaptor:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if for_masked_lm:
        model = EsmForMaskedLM.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
    return HFEncoderAdaptor(
        tokenizer=tok,
        model=model,
        trim_policy=esm_trim_first_last_policy,
        token_field="input_ids",
        attention_mask_field="attention_mask",
        model_output_field="last_hidden_state",
        pad_to_multiple_of=8,
    )
