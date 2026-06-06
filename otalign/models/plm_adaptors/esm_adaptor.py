import torch
from transformers import AutoModel, AutoTokenizer, EsmForMaskedLM

from .hf_adaptor import HFEncoderAdaptor
from .policies import trim_active_edges_vec


def esm_trim_first_last_policy_vec(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """Trim policy for ESM encoders: drop the first and last active tokens (BOS/EOS).

    Thin wrapper over :func:`policies.trim_active_edges_vec` with
    ``drop_first=1, drop_last=1`` (vectorized; no per-row Python loop).
    """
    return trim_active_edges_vec(input_ids, attention_mask, drop_first=1, drop_last=1)


def build_esm_adaptor(model_name: str = "facebook/esm2_t33_650M_UR50D", for_masked_lm: bool = False) -> HFEncoderAdaptor:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if for_masked_lm:
        model = EsmForMaskedLM.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
    return HFEncoderAdaptor(
        tokenizer=tok,
        model=model,
        trim_policy=esm_trim_first_last_policy_vec,
        token_field="input_ids",
        attention_mask_field="attention_mask",
        model_output_field="last_hidden_state",
        pad_to_multiple_of=8,
    )
