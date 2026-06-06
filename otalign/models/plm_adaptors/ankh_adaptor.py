from transformers import AutoTokenizer

from ._t5_backend import load_t5_encoder
from .hf_adaptor import HFEncoderAdaptor
from .t5_adaptor import t5_trim_last_policy_vec


ANKH_BASE = "ElnaggarLab/ankh-base"
ANKH_LARGE = "ElnaggarLab/ankh-large"
ANKH3_LARGE = "ElnaggarLab/ankh3-large"
ANKH3_XL = "ElnaggarLab/ankh3-xl"


def build_ankh_adaptor(model_name: str = ANKH_LARGE, output_attentions: bool = False, for_masked_lm: bool = False):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = load_t5_encoder(model_name, output_attentions=output_attentions)
    adaptor = HFEncoderAdaptor(
        tokenizer=tok,
        model=model,
        trim_policy=t5_trim_last_policy_vec,  # vectorized; takes (input_ids, attention_mask)
        pad_to_multiple_of=8,
        fp16_unsafe=True,  # Ankh (T5) overflows to NaN in fp16
    )
    return adaptor
