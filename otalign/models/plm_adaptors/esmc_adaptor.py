from otalign.models.esmc import build_esmc_tokenizer, load_esmc

from .esm_adaptor import esm_trim_first_last_policy_vec
from .hf_adaptor import HFEncoderAdaptor


def build_esmc_adaptor(size: str = "300m", for_masked_lm: bool = False) -> HFEncoderAdaptor:
    """Build an ESM-C adaptor over the self-contained encoder + official weights.

    ESM-C tokenization wraps the sequence in ``<cls> ... <eos>``, so residue
    trimming drops the first and last active tokens (same policy as ESM).
    """
    if for_masked_lm:
        raise NotImplementedError("Training/MLM fine-tuning of ESMC is not supported here.")
    tok = build_esmc_tokenizer()
    model = load_esmc(size)
    return HFEncoderAdaptor(
        tokenizer=tok,
        model=model,
        trim_policy=esm_trim_first_last_policy_vec,
        pad_to_multiple_of=8,
    )
