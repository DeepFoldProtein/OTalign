import torch
from transformers import T5EncoderModel, T5Tokenizer

from .hf_adaptor import HFEncoderAdaptor


def t5_trim_last_policy(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """
    Drop the final *active* token (EOS) for each row using attention_mask.
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


def build_prott5_adaptor(model_name="Rostlab/prot_t5_xl_uniref50"):
    tok = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, legacy=True)
    model = T5EncoderModel.from_pretrained(model_name)

    def preproc_batch(seqs: list[str]) -> list[str]:
        # Replace uncommon AA with X (common in ProtT5 recipes)
        table = str.maketrans({"U": "X", "O": "X", "B": "X", "Z": "X"})
        return [" ".join(list(s.translate(table))) for s in seqs]

    adaptor = HFEncoderAdaptor(
        tokenizer=tok,
        model=model,
        trim_policy=t5_trim_last_policy,  # now takes (input_ids, attention_mask)
        pad_to_multiple_of=8,
    )
    setattr(adaptor, "preproc_batch", preproc_batch)
    return adaptor
