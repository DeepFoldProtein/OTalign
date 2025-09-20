from typing import List, Optional, Sequence

import torch
from transformers import AutoModel, AutoTokenizer

from .base import BasePLMAdaptor, EmbeddingOutput
from .policies import make_mask_from_lengths


def proteinglm_trim_last_policy(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """
    Drop the final *active* token (EOS) for each row using attention_mask.
    ProteinGLM uses EOS token at the end that needs to be removed.
    """
    B, T = input_ids.shape
    keep_mask = torch.zeros((B, T), dtype=torch.bool, device=input_ids.device)
    lengths: list[int] = []
    for b in range(B):
        active_idx = torch.nonzero(attention_mask[b].to(torch.bool), as_tuple=False).flatten()
        if active_idx.numel() >= 1:
            keep_idx = active_idx[:-1]  # drop EOS
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


def _split_into_batches(xs: Sequence[str], batch_size: int) -> List[List[str]]:
    return [list(xs[i : i + batch_size]) for i in range(0, len(xs), batch_size)]


class ProteinGLMAdaptor(BasePLMAdaptor):
    """
    Custom adaptor for ProteinGLM that handles its specific output structure.
    ProteinGLM requires output_hidden_states=True and uses hidden_states[-1] for embeddings.
    """

    def __init__(self, model_name: str = "Bo1015/proteinglm-100b-int4") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.half)

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
        model = self.model
        tok = self.tokenizer
        device = device or next(model.parameters()).device
        # autocast_dtype = torch.float16 if fp16 else None
        all_embeds: List[torch.Tensor] = []
        all_masks: List[torch.Tensor] = []
        all_lengths: List[int] = []

        if dtype is None:
            dtype = next(model.parameters()).dtype

        was_training = model.training
        model.eval()

        cm = torch.no_grad() if disable_grad else torch.enable_grad()
        with cm:
            for chunk in _split_into_batches(sequences, batch_size):
                seq_batch = list(chunk)

                enc = tok(seq_batch, return_tensors="pt", padding=True, truncation=False, add_special_tokens=True, pad_to_multiple_of=8)
                input_ids: torch.Tensor = enc["input_ids"].to(device)
                attn_mask: torch.Tensor = enc["attention_mask"].to(device)

                for seq in seq_batch:
                    print(len(seq))
                # Forward with ProteinGLM specific parameters
                with torch.inference_mode():
                    outputs = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True, return_last_hidden_state=True)
                    # Use the last layer hidden states
                    hidden: torch.Tensor = outputs.hidden_states.transpose(0, 1)  # [B, T, D] drop the <eos> token
                    

                # Trim EOS tokens using our policy
                trimmed_ids, lengths, keep_mask = proteinglm_trim_last_policy(enc["input_ids"].to(device), enc["attention_mask"].to(device))
                # keep_mask: [B, T] True where original token should be kept as a residue
                # Select hidden states
                B, _, D = hidden.shape
                maxL = keep_mask.sum(dim=1).max().item() if B > 0 else 0
                # pack kept positions per row into a padded [B, maxL, D]
                kept_embeds = []
                for b in range(B):
                    sel = hidden[b][keep_mask[b]]  # [L_b, D]
                    if sel.numel() == 0:
                        pad = torch.zeros((0, D), device=hidden.device, dtype=hidden.dtype)
                    else:
                        pad = sel
                    if sel.size(0) < maxL:
                        pad = torch.cat([pad, hidden.new_zeros((maxL - sel.size(0), D))], dim=0)
                    kept_embeds.append(pad.unsqueeze(0))
                kept_embeds = torch.cat(kept_embeds, dim=0) if kept_embeds else torch.empty((B, 0, D), device=hidden.device, dtype=hidden.dtype)
                all_embeds.append(kept_embeds)
                all_masks.append(make_mask_from_lengths(lengths, device=hidden.device))
                all_lengths.extend([int(x) for x in lengths])

        if was_training:
            model.train()

        # Concatenate
        residue_embeddings = torch.cat(all_embeds, dim=0) if all_embeds else torch.empty((0, 0, 0), device=device, dtype=dtype)
        attention_mask = torch.cat(all_masks, dim=0) if all_masks else torch.empty((0, 0), device=device, dtype=torch.bool)

        return EmbeddingOutput(residue_embeddings=residue_embeddings.to(dtype=dtype), attention_mask=attention_mask, per_sequence_lengths=all_lengths, extras={})


def build_proteinglm_int4_adaptor(model_name: str = "Bo1015/proteinglm-100b-int4") -> ProteinGLMAdaptor:
    """
    Build ProteinGLM adaptor with trust_remote_code and appropriate configuration.
    """
    return ProteinGLMAdaptor(model_name)
