from transformers import AutoTokenizer, T5EncoderModel

from .hf_adaptor import HFEncoderAdaptor
from .t5_adaptor import t5_trim_last_policy


ANKH_BASE = "ElnaggarLab/ankh-base"
ANKH_LARGE = "ElnaggarLab/ankh-large"


def build_ankh_adaptor(model_name: str = ANKH_LARGE, output_attentions: bool = False):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name, output_attentions=output_attentions)
    adaptor = HFEncoderAdaptor(
        tokenizer=tok,
        model=model,
        trim_policy=t5_trim_last_policy,  # now takes (input_ids, attention_mask)
        pad_to_multiple_of=8,
    )
    return adaptor
