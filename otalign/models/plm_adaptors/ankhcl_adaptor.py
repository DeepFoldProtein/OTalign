from transformers import AutoTokenizer

from otalign.procl.model.ankh import AnkhCL

from .hf_adaptor import HFEncoderAdaptor
from .t5_adaptor import t5_trim_last_policy


def build_ankhcl_adaptor():
    tok = AutoTokenizer.from_pretrained("DeepFoldProtein/Ankh-Large-Contrastive", do_lower_case=False)
    model = AnkhCL.from_pretrained("DeepFoldProtein/Ankh-Large-Contrastive", freeze_base=True, is_scratch=False)

    def preproc_batch(seqs: list[str]) -> list[str]:
        # Replace uncommon AA with X (common in ProtT5 recipes)
        table = str.maketrans({"U": "X", "O": "X", "B": "X", "Z": "X"})
        return ["".join(list(s.translate(table))) for s in seqs]

    adaptor = HFEncoderAdaptor(
        tokenizer=tok,
        model=model,
        trim_policy=t5_trim_last_policy,  # now takes (input_ids, attention_mask)
        pad_to_multiple_of=8,
    )
    setattr(adaptor, "preproc_batch", preproc_batch)
    return adaptor
