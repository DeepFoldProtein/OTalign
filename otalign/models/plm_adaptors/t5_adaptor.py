import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from ._t5_backend import load_t5_encoder
from .hf_adaptor import HFEncoderAdaptor
from .policies import trim_active_edges_vec


def t5_trim_last_policy_vec(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """Trim policy for T5/Ankh encoders: drop the final active token (EOS) per row.

    Thin wrapper over :func:`policies.trim_active_edges_vec` with
    ``drop_first=0, drop_last=1`` (vectorized; no per-row Python loop).
    """
    return trim_active_edges_vec(input_ids, attention_mask, drop_first=0, drop_last=1)


def build_prott5_adaptor(model_name="Rostlab/prot_t5_xl_uniref50", for_masked_lm: bool = False):
    tok = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, legacy=True)
    if for_masked_lm:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        model = load_t5_encoder(model_name)

    def preproc_batch(seqs: list[str]) -> list[str]:
        # Replace uncommon AA with X (common in ProtT5 recipes)
        table = str.maketrans({"U": "X", "O": "X", "B": "X", "Z": "X"})
        return [" ".join(list(s.translate(table))) for s in seqs]

    adaptor = HFEncoderAdaptor(
        tokenizer=tok,
        model=model,
        trim_policy=t5_trim_last_policy_vec,  # vectorized; takes (input_ids, attention_mask)
        pad_to_multiple_of=8,
        fp16_unsafe=True,  # ProtT5 overflows to NaN in fp16
    )
    setattr(adaptor, "preproc_batch", preproc_batch)
    return adaptor
