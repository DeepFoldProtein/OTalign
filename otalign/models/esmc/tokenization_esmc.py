"""ESM-C sequence tokenizer, rebuilt from the official vocabulary.

Mirrors EvolutionaryScale's ``EsmSequenceTokenizer`` (a character-level BPE with
no merges plus a ``<cls> $A <eos>`` post-processor). Building it from the vocab
here keeps token ids aligned with the embedding rows of the
``biohub/esmc-*-2024-12`` checkpoints without depending on any external tokenizer
files or the ``esm`` SDK.
"""

from __future__ import annotations

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


# Order is significant: token id == index, matching the checkpoint's embedding
# rows (e.g. <cls>=0, <pad>=1, <eos>=2, <mask>=32).
SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
    "O", ".", "-", "|",
    "<mask>",
]


def build_esmc_tokenizer() -> PreTrainedTokenizerFast:
    """Construct the ESM-C sequence tokenizer (cls/eos added around the sequence)."""
    token_to_id = {tok: i for i, tok in enumerate(SEQUENCE_VOCAB)}
    bpe = BPE(token_to_id, merges=[], unk_token="<unk>")
    tokenizer = Tokenizer(bpe)
    tokenizer.add_special_tokens(["<cls>", "<pad>", "<mask>", "<eos>", "|"])
    tokenizer.post_processor = TemplateProcessing(
        single="<cls> $A <eos>",
        special_tokens=[
            ("<cls>", token_to_id["<cls>"]),
            ("<eos>", token_to_id["<eos>"]),
        ],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        eos_token="<eos>",
        additional_special_tokens=["|"],
    )
