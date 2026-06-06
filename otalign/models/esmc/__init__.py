"""Self-contained ESM-C (ESMC) encoder for OTalign.

A clean-room reimplementation of the EvolutionaryScale ESM-C architecture
(reference: github.com/evolutionaryscale/esm — ``esm/models/esmc.py`` and
``esm/layers/*``), written so the official esm-SDK-format checkpoints hosted at
``biohub/esmc-{300m,600m,6b}-2024-12`` load with ``strict=True``.

This avoids both (a) the EvolutionaryScale ``esm`` SDK, which pins
``transformers<4.48.2`` and would conflict with OTalign's ``transformers>=4.56.1``
requirement, and (b) any non-mainline transformers ``esmc`` modeling code — ESMC
is not part of mainline transformers (4.x or 5.x).
"""

from .modeling_esmc import ESMC_CONFIGS, ESMCEncoder, ESMCEncoderOutput, load_esmc
from .tokenization_esmc import SEQUENCE_VOCAB, build_esmc_tokenizer


__all__ = [
    "ESMCEncoder",
    "ESMCEncoderOutput",
    "ESMC_CONFIGS",
    "load_esmc",
    "build_esmc_tokenizer",
    "SEQUENCE_VOCAB",
]
