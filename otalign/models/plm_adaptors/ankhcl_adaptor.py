from transformers import AutoTokenizer

from otalign.procl.model.ankh import AnkhCL

from ._t5_backend import load_t5_pretrained
from .hf_adaptor import HFEncoderAdaptor
from .t5_adaptor import t5_trim_last_policy_vec


def build_ankhcl_adaptor(for_masked_lm: bool = False):
    model_name = "DeepFoldProtein/Ankh-Large-Contrastive"
    tok = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)

    if for_masked_lm:
        raise NotImplementedError("Training AnkhCL model is not supported.")
    else:
        model = load_t5_pretrained(AnkhCL, model_name, freeze_base=True, is_scratch=False)

    def preproc_batch(seqs: list[str]) -> list[str]:
        # Replace uncommon AA with X (common in ProtT5 recipes)
        table = str.maketrans({"U": "X", "O": "X", "B": "X", "Z": "X"})
        return [s.translate(table) for s in seqs]

    adaptor = HFEncoderAdaptor(
        tokenizer=tok,
        model=model,
        trim_policy=t5_trim_last_policy_vec,  # vectorized; takes (input_ids, attention_mask)
        pad_to_multiple_of=8,
        fp16_unsafe=True,  # AnkhCL (Ankh-Large T5 backbone) overflows to NaN in fp16
    )
    setattr(adaptor, "preproc_batch", preproc_batch)
    return adaptor
